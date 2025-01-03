// Copyright 2024 Francesco Biscani
//
// This file is part of the mizuba library.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/align.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include "detail/atomic_minmax.hpp"
#include "detail/file_utils.hpp"
#include "logging.hpp"
#include "polyjectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

namespace detail
{

struct polyjectory_impl {
    using traj_offset = polyjectory::traj_offset;

    // The path to the temp dir containing the polyjectory data.
    boost::filesystem::path m_temp_dir_path;
    // Offsets for the trajectory data.
    std::vector<traj_offset> m_traj_offset_vec;
    // Offsets for the time data.
    std::vector<std::size_t> m_time_offset_vec;
    // Polynomial order + 1 for the trajectory data.
    std::uint32_t m_poly_op1 = 0;
    // The duration of the longest trajectory.
    double m_maxT = 0;
    // The initial epoch.
    double m_epoch = 0;
    // The second component of the initial epoch.
    double m_epoch2 = 0;
    // Vector of trajectory statuses.
    std::vector<std::int32_t> m_status;
    // The memory-mapped file for the trajectory data.
    boost::iostreams::mapped_file_source m_traj_file;
    // The memory-mapped file for the time data.
    boost::iostreams::mapped_file_source m_time_file;
    // Pointer to the beginning of m_traj_file, cast to double.
    const double *m_traj_ptr = nullptr;
    // Pointer to the beginning of m_time_file, cast to double.
    const double *m_time_ptr = nullptr;

    explicit polyjectory_impl(boost::filesystem::path temp_dir_path, std::vector<traj_offset> traj_offset_vec,
                              std::vector<std::size_t> time_offset_vec, std::uint32_t poly_op1, double maxT,
                              std::vector<std::int32_t> status, double epoch, double epoch2)
        : m_temp_dir_path(std::move(temp_dir_path)), m_traj_offset_vec(std::move(traj_offset_vec)),
          m_time_offset_vec(std::move(time_offset_vec)), m_poly_op1(poly_op1), m_maxT(maxT), m_epoch(epoch),
          m_epoch2(epoch2), m_status(std::move(status)), m_traj_file((m_temp_dir_path / "traj").string()),
          m_time_file((m_temp_dir_path / "time").string())
    {
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        m_traj_ptr = reinterpret_cast<const double *>(m_traj_file.data());
        assert(boost::alignment::is_aligned(m_traj_ptr, alignof(double)));

        m_time_ptr = reinterpret_cast<const double *>(m_time_file.data());
        assert(boost::alignment::is_aligned(m_time_ptr, alignof(double)));
    }

    [[nodiscard]] bool is_open() noexcept
    {
        return m_traj_file.is_open();
    }

    void close() noexcept
    {
        // NOTE: a polyjectory is not supposed to be closed
        // more than once.
        assert(is_open());

        // Close all memory-mapped files.
        m_traj_file.close();
        m_time_file.close();

        // Remove the temp dir and everything within.
        boost::filesystem::remove_all(m_temp_dir_path);
    }

    polyjectory_impl(polyjectory_impl &&) noexcept = delete;
    polyjectory_impl(const polyjectory_impl &) = delete;
    polyjectory_impl &operator=(const polyjectory_impl &) = delete;
    polyjectory_impl &operator=(polyjectory_impl &&) noexcept = delete;
    ~polyjectory_impl()
    {
        if (is_open()) {
            close();
        }
    }
};

// LCOV_EXCL_START

void close_pj(std::shared_ptr<polyjectory_impl> &pj) noexcept
{
    pj->close();
}

// LCOV_EXCL_STOP

const std::shared_ptr<polyjectory_impl> &fetch_pj_impl(const polyjectory &pj) noexcept
{
    return pj.m_impl;
}

} // namespace detail

polyjectory::polyjectory(ptag,
                         std::tuple<std::vector<traj_span_t>, std::vector<time_span_t>, std::vector<std::int32_t>> tup,
                         double epoch, double epoch2)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    auto &[traj_spans, time_spans, status] = tup;

    // Cache the total number of objects.
    const auto n_objs = traj_spans.size();

    if (n_objs == 0u) [[unlikely]] {
        throw std::invalid_argument("Cannot initialise a polyjectory object from an empty list of trajectories");
    }

    if (n_objs != time_spans.size()) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "In the construction of a polyjectory, the number of objects deduced from the list of trajectories "
            "({}) is inconsistent with the number of objects deduced from the list of times ({})",
            n_objs, time_spans.size()));
    }

    if (n_objs != status.size()) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "In the construction of a polyjectory, the number of objects deduced from the list of trajectories "
            "({}) is inconsistent with the number of objects deduced from the status list ({})",
            n_objs, status.size()));
    }

    // Check epoch and epoch2.
    if (!std::isfinite(epoch)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The initial epoch of a polyjectory must be finite, but instead a value of {} was provided", epoch));
    }

    if (!std::isfinite(epoch2)) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The second component of the initial epoch of a polyjectory must be "
                                                "finite, but instead a value of {} was provided",
                                                epoch2));
    }

    // Assemble a "unique" dir path into the system temp dir.
    auto tmp_dir_path = detail::create_temp_dir("mizuba_polyjectory-%%%%-%%%%-%%%%-%%%%");

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the temp dir in case of exceptions.
    try {
        // Change the permissions so that only the owner has access.
        boost::filesystem::permissions(tmp_dir_path, boost::filesystem::owner_all);

        // Do a first single-threaded pass on the spans to determine:
        // - the offsets of traj and time data,
        // - the polynomial order,
        // - the duration of the longest trajectory.

        // Init the trajectories offset vector.
        std::vector<traj_offset> traj_offset_vec;
        traj_offset_vec.reserve(n_objs);

        // Keep track of the current offset into the traj file.
        safe_size_t cur_traj_offset(0);

        // Init the poly order + 1.
        std::uint32_t poly_op1 = 0;

        // Build the traj offset vector, determine poly_op1 and
        // run checks on the traj spans' dimensions.
        for (decltype(traj_spans.size()) i = 0; i < n_objs; ++i) {
            // Fetch the traj data for the current object.
            const auto cur_traj = traj_spans[i];

            // Set/check the order + 1.
            const auto op1 = cur_traj.extent(2);
            if (i == 0u) {
                if (op1 < 3u) [[unlikely]] {
                    throw std::invalid_argument("The trajectory polynomial order for the first object "
                                                "is less than 2 - this is not allowed");
                }

                poly_op1 = boost::numeric_cast<std::uint32_t>(op1);
            } else {
                if (op1 != poly_op1) [[unlikely]] {
                    throw std::invalid_argument(
                        fmt::format("The trajectory polynomial order for the object at index "
                                    "{} is inconsistent with the polynomial order deduced from the first object ({})",
                                    i, poly_op1 - 1u));
                }
            }

            // Compute the total data size (in number of floating-point values).
            const auto traj_size = safe_size_t(cur_traj.extent(0)) * cur_traj.extent(1) * op1;

            // Add entry to the offset vector.
            traj_offset_vec.emplace_back(cur_traj_offset, cur_traj.extent(0));

            // Update cur_offset.
            cur_traj_offset += traj_size;
        }

        if (cur_traj_offset == 0u) [[unlikely]] {
            throw std::invalid_argument(
                "All the trajectories in a polyjectory have a number of steps equal to zero: this is not allowed");
        }

        // Offset vector for the time data.
        std::vector<std::size_t> time_offset_vec;
        time_offset_vec.reserve(n_objs);

        // Keep track of the current offset into the time file.
        safe_size_t cur_time_offset(0);

        // Init maxT.
        double maxT = 0;

        // Build the time offset vector, determine maxT and
        // run checks on the time spans' dimensions.
        for (decltype(time_spans.size()) i = 0; i < n_objs; ++i) {
            // Fetch the current traj and time spans.
            const auto cur_traj = traj_spans[i];
            const auto cur_time = time_spans[i];

            // The number of times must be consistent with the number of steps.
            if (cur_traj.extent(0) != cur_time.extent(0)) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("The number of steps for the trajectory of the object at index {} is {}, but the "
                                "number of times is {} - the two numbers must be equal",
                                i, cur_traj.extent(0), cur_time.extent(0)));
            }

            // Compute the total data size (in number of floating-point values).
            const auto time_size = cur_time.extent(0);

            // Update maxT.
            // NOTE: we do this only if the current trajectory is non-empty. Note that maxT might
            // end up begin non-finite or incorrect here, but this is fine as we will check in
            // detail the time values in the parallel loop below. Tentatively computing maxT here
            // allows us to avoid having to mess around with atomics in the parallel loop below.
            if (cur_time.extent(0) > 0u) {
                const auto curT = cur_time(cur_time.extent(0) - 1u);
                maxT = (curT > maxT) ? curT : maxT;
            }

            // Add entry to the offset vector.
            time_offset_vec.emplace_back(cur_time_offset);

            // Update cur_offset.
            cur_time_offset += time_size;
        }

        // Time the data copy from the spans into the file.
        stopwatch sw;

        // NOTE: at this point maxT could contain bogus/incorrect values because
        // we have not checked the time data yet. We will do this below.

        // Init the storage files.
        const auto traj_path = tmp_dir_path / "traj";
        const auto time_path = tmp_dir_path / "time";
        detail::create_sized_file(traj_path, cur_traj_offset * sizeof(double));
        detail::create_sized_file(time_path, cur_time_offset * sizeof(double));

        // Memory-map them.
        // NOTE: we should consider the pros and cons of memory-mapping here.
        // Perhaps pwrite() would be better?
        boost::iostreams::mapped_file_sink traj_file(traj_path.string());
        boost::iostreams::mapped_file_sink time_file(time_path.string());

        // Fetch pointers to the beginning of the data.
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        auto *traj_base_ptr = reinterpret_cast<double *>(traj_file.data());
        assert(boost::alignment::is_aligned(traj_base_ptr, alignof(double)));
        auto *time_base_ptr = reinterpret_cast<double *>(time_file.data());
        assert(boost::alignment::is_aligned(time_base_ptr, alignof(double)));

        // Check and copy over the data from the spans.
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<decltype(traj_spans.size())>(0, n_objs),
            [traj_base_ptr, time_base_ptr, poly_op1, &traj_spans, &traj_offset_vec, &time_spans,
             &time_offset_vec](const auto &range) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    // Trajectory data.
                    const auto cur_traj = traj_spans[i];

                    // Check for non-finite data.
                    // NOTE: at one point we should probably investigate here if it is
                    // better to copy the data while it is being checked, instead of
                    // checking first and then doing a bulk copy later.
                    for (std::size_t j = 0; j < cur_traj.extent(0); ++j) {
                        for (std::size_t k = 0; k < cur_traj.extent(1); ++k) {
                            for (std::size_t l = 0; l < poly_op1; ++l) {
                                if (!std::isfinite(cur_traj(j, k, l))) [[unlikely]] {
                                    throw std::invalid_argument(
                                        fmt::format("A non-finite value was found in the trajectory at index {}", i));
                                }
                            }
                        }
                    }

                    // Compute the total data size (in number of floating-point values).
                    const auto traj_size = safe_size_t(cur_traj.extent(0)) * cur_traj.extent(1) * poly_op1;

                    // Copy the data into the file.
                    std::ranges::copy(cur_traj.data_handle(),
                                      cur_traj.data_handle() + static_cast<std::size_t>(traj_size),
                                      traj_base_ptr + traj_offset_vec[i].offset);

                    // Time data.
                    const auto cur_time = time_spans[i];

                    // Check data.
                    // NOTE: at one point we should probably investigate here if it is
                    // better to copy the data while it is being checked, instead of
                    // checking first and then doing a bulk copy later.
                    for (std::size_t j = 0; j < cur_time.extent(0); ++j) {
                        if (!std::isfinite(cur_time(j))) [[unlikely]] {
                            throw std::invalid_argument(
                                fmt::format("A non-finite time coordinate was found for the object at index {}", i));
                        }

                        if (cur_time(j) <= 0) [[unlikely]] {
                            throw std::invalid_argument(
                                fmt::format("A non-positive time coordinate was found for the object at index {}", i));
                        }

                        if (j > 0u && !(cur_time(j) > cur_time(j - 1u))) [[unlikely]] {
                            throw std::invalid_argument(fmt::format(
                                "The sequence of times for the object at index {} is not monotonically increasing", i));
                        }
                    }

                    // Compute the total data size (in number of floating-point values).
                    const auto time_size = cur_time.extent(0);

                    // Copy the data into the file.
                    std::ranges::copy(cur_time.data_handle(), cur_time.data_handle() + time_size,
                                      time_base_ptr + time_offset_vec[i]);
                }
            });

        // We can now assert that maxT must not be zero: the time values have all been validated
        // and we checked earlier that at least one trajectory has a nonzero number of steps.
        assert(maxT != 0);

        // Close the storage files.
        traj_file.close();
        time_file.close();

        // Mark them as read-only.
        detail::mark_file_read_only(traj_path);
        detail::mark_file_read_only(time_path);

        log_trace("polyjectory copy from spans time: {}", sw);

        // Create the impl.
        // NOTE: here make_shared() first allocates, and then constructs. If there are no exceptions, the assignment
        // to m_impl is noexcept and the dtor of impl takes charge of cleaning up the tmp_dir_path upon destruction.
        // If an exception is thrown (e.g., from memory allocation or from the impl ctor throwing), the impl has not
        // been fully constructed and thus its dtor will not be invoked, and the cleanup of tmp_dir_path will be
        // performed in the catch block below.
        m_impl = std::make_shared<detail::polyjectory_impl>(std::move(tmp_dir_path), std::move(traj_offset_vec),
                                                            std::move(time_offset_vec), poly_op1, maxT,
                                                            std::move(status), epoch, epoch2);
    } catch (...) {
        boost::filesystem::remove_all(tmp_dir_path);
        throw;
    }
}

// Constructor from a traj and time data files at the locations 'orig_traj_file_path'
// and 'orig_time_file_path'. The original files will be moved into the polyjectory's data dir.
//
// The layout of the trajectory data into the data file is described by traj_offsets, from which we
// also deduce the layout of the time data. 'order' it the polynomial order of the polyjectory,
// 'status' the vector of object statuses.
polyjectory::polyjectory(const std::filesystem::path &orig_traj_file_path,
                         const std::filesystem::path &orig_time_file_path, std::uint32_t order,
                         std::vector<traj_offset> traj_offsets, std::vector<std::int32_t> status, double epoch,
                         double epoch2)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Check the polynomial order and compute order + 1.
    if (order < 2u || order == std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid polynomial order {} specified during the construction of a polyjectory", order));
    }
    const auto op1 = order + 1u;

    // Initial check on traj_offsets.
    if (traj_offsets.empty()) [[unlikely]] {
        throw std::invalid_argument(
            "Invalid trajectory offsets vector passed to the constructor of a polyjectory: the vector cannot be empty");
    }

    // Check the status vector.
    if (status.size() != traj_offsets.size()) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid status vector passed to the constructor of a polyjectory: the "
                                                "expected size is {}, but the actual size is {}",
                                                traj_offsets.size(), status.size()));
    }

    // Check epoch.
    if (!std::isfinite(epoch)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The initial epoch of a polyjectory must be finite, but instead a value of {} was provided", epoch));
    }

    if (!std::isfinite(epoch2)) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The second component of the initial epoch of a polyjectory must be "
                                                "finite, but instead a value of {} was provided",
                                                epoch2));
    }

    // Canonicalise the file paths and turn them into Boost fs paths.
    boost::filesystem::path traj_file_path, time_file_path;
    try {
        traj_file_path = std::filesystem::canonical(orig_traj_file_path);
        time_file_path = std::filesystem::canonical(orig_time_file_path);
        // LCOV_EXCL_START
    } catch (...) {
        throw std::invalid_argument(
            "Invalid data file(s) passed to the constructor of a polyjectory: the path(s) could not be canonicalised");
    }
    // LCOV_EXCL_STOP

    // Iterate over the trajectory offsets, sanity-checking them and computing
    // the total number of floating-point values expected in the files.
    safe_size_t tot_num_traj_values = 0, tot_num_time_values = 0;

    for (decltype(traj_offsets.size()) i = 0; i < traj_offsets.size(); ++i) {
        const auto [offset, n_steps] = traj_offsets[i];

        // Check the offset value.
        if (i == 0u) {
            // The offset at index 0 must be zero.
            if (offset != 0u) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("Invalid trajectory offsets vector passed to the constructor of a polyjectory: "
                                "the initial offset value must be zero but it is {} instead",
                                offset));
            }
        } else {
            // An offset at index i > 0 must be consistent with the previous offset and n_steps.
            const auto [prev_offset, prev_n_steps] = traj_offsets[i - 1u];

            if (!(offset > prev_offset)) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "Invalid trajectory offsets vector passed to the constructor of a polyjectory: "
                    "the offset of the object at index {} is not greater than the offset of the previous object",
                    i));
            }

            if (offset - prev_offset != safe_size_t(prev_n_steps) * 7u * op1) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "Inconsistent data detected in the trajectory offsets vector passed to the constructor of "
                    "a polyjectory: the offset of the object at index {} is inconsistent with the offset and "
                    "number of steps of the previous object",
                    i));
            }
        }

        // Update tot_num_traj_values and tot_num_time_values.
        tot_num_traj_values += safe_size_t(n_steps) * 7u * op1;
        tot_num_time_values += n_steps;
    }

    if (tot_num_traj_values == 0u) [[unlikely]] {
        throw std::invalid_argument(
            "All the trajectories in a polyjectory have a number of steps equal to zero: this is not allowed");
    }

    // Build the time offsets vector from the trajectory offsets.
    // NOTE: all computations here are safe because we managed to compute
    // expected_tot_num_values without overflow.
    std::vector<std::size_t> time_offsets;
    time_offsets.reserve(traj_offsets.size());
    for (decltype(traj_offsets.size()) i = 0; i < traj_offsets.size(); ++i) {
        if (i == 0u) {
            time_offsets.push_back(0);
        } else {
            const auto [_, prev_nsteps] = traj_offsets[i - 1u];
            time_offsets.push_back(time_offsets.back() + prev_nsteps);
        }
    }

    // Assemble a "unique" dir path into the system temp dir.
    auto tmp_dir_path = detail::create_temp_dir("mizuba_polyjectory-%%%%-%%%%-%%%%-%%%%");

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the temp dir in case of exceptions.
    try {
        // Change the permissions so that only the owner has access.
        boost::filesystem::permissions(tmp_dir_path, boost::filesystem::owner_all);

        // Init the storages file paths and check that they do not exist already.
        const auto traj_path = tmp_dir_path / "traj";
        if (boost::filesystem::exists(traj_path)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(
                fmt::format("Cannot create the storage file '{}': the file exists already", traj_path.string()));
            // LCOV_EXCL_STOP
        }
        const auto time_path = tmp_dir_path / "time";
        if (boost::filesystem::exists(time_path)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(
                fmt::format("Cannot create the storage file '{}': the file exists already", time_path.string()));
            // LCOV_EXCL_STOP
        }

        // Move the original files.
        boost::filesystem::rename(traj_file_path, traj_path);
        boost::filesystem::rename(time_file_path, time_path);

        // Check the file sizes. Do it now, after moving the files into the private directory.
        if (boost::filesystem::file_size(traj_path) != tot_num_traj_values * sizeof(double)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid trajectory data file passed to the constructor of a polyjectory: the "
                            "expected size is bytes is '{}' but the actual size is {} instead",
                            static_cast<std::size_t>(tot_num_traj_values * sizeof(double)),
                            boost::filesystem::file_size(traj_path)));
        }
        if (boost::filesystem::file_size(time_path) != tot_num_time_values * sizeof(double)) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid time data file passed to the constructor of a polyjectory: the "
                            "expected size is bytes is '{}' but the actual size is {} instead",
                            static_cast<std::size_t>(tot_num_time_values * sizeof(double)),
                            boost::filesystem::file_size(time_path)));
        }

        // Memory-map the files.
        boost::iostreams::mapped_file_source traj_file(traj_path);
        const auto *traj_file_base_ptr = reinterpret_cast<const double *>(traj_file.data());
        assert(boost::alignment::is_aligned(traj_file_base_ptr, alignof(double)));

        boost::iostreams::mapped_file_source time_file(time_path);
        const auto *time_file_base_ptr = reinterpret_cast<const double *>(time_file.data());
        assert(boost::alignment::is_aligned(time_file_base_ptr, alignof(double)));

        // Atomic variable to compute maxT.
        std::atomic maxT = -std::numeric_limits<double>::infinity();

        // Check the data and compute maxT.
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<decltype(traj_offsets.size())>(0, traj_offsets.size()),
            [traj_file_base_ptr, time_file_base_ptr, &maxT, &traj_offsets, &time_offsets, op1](const auto &range) {
                // Local version of maxT.
                auto local_maxT = -std::numeric_limits<double>::infinity();

                for (auto i = range.begin(); i != range.end(); ++i) {
                    const auto [t_offset, n_steps] = traj_offsets[i];
                    const auto time_offset = time_offsets[i];

                    // Build a trajectory span and check the data.
                    const traj_span_t cur_traj{traj_file_base_ptr + t_offset, n_steps, op1};

                    // Check for non-finite trajectory data.
                    for (std::size_t j = 0; j < cur_traj.extent(0); ++j) {
                        for (std::size_t k = 0; k < cur_traj.extent(1); ++k) {
                            for (std::size_t l = 0; l < op1; ++l) {
                                if (!std::isfinite(cur_traj(j, k, l))) [[unlikely]] {
                                    throw std::invalid_argument(
                                        fmt::format("A non-finite value was found in the trajectory at index {}", i));
                                }
                            }
                        }
                    }

                    // Build a time span and check the data.
                    const time_span_t cur_time{time_file_base_ptr + time_offset, n_steps};

                    for (std::size_t j = 0; j < cur_time.extent(0); ++j) {
                        if (!std::isfinite(cur_time(j))) [[unlikely]] {
                            throw std::invalid_argument(
                                fmt::format("A non-finite time coordinate was found for the object at index {}", i));
                        }

                        if (cur_time(j) <= 0) [[unlikely]] {
                            throw std::invalid_argument(
                                fmt::format("A non-positive time coordinate was found for the object at index {}", i));
                        }

                        if (j > 0u && !(cur_time(j) > cur_time(j - 1u))) [[unlikely]] {
                            throw std::invalid_argument(fmt::format(
                                "The sequence of times for the object at index {} is not monotonically increasing", i));
                        }
                    }

                    // Update local_maxT.
                    // NOTE: we do this only if the current trajectory is non-empty.
                    if (cur_time.extent(0) > 0u) {
                        local_maxT = std::max(local_maxT, cur_time(cur_time.extent(0) - 1u));
                    }
                }

                // Atomicaly update maxT.
                // NOTE: atomic_max() usage here is safe because we checked that
                // the time values are finite.
                detail::atomic_max(maxT, local_maxT);
            });

        // We can now assert that maxT must be strictly positive: the time values have all been validated
        // and we checked earlier that at least one trajectory has a nonzero number of steps.
        assert(maxT.load() > 0);

        // Close the storage files.
        traj_file.close();
        time_file.close();

        // Mark them as read-only.
        detail::mark_file_read_only(traj_path);
        detail::mark_file_read_only(time_path);

        // Construct the implementation.
        m_impl = std::make_shared<detail::polyjectory_impl>(std::move(tmp_dir_path), std::move(traj_offsets),
                                                            std::move(time_offsets), op1, maxT.load(),
                                                            std::move(status), epoch, epoch2);

        // LCOV_EXCL_START
    } catch (...) {
        boost::filesystem::remove_all(tmp_dir_path);
        throw;
    }
    // LCOV_EXCL_STOP
}

// NOTE: the polyjectory class will have shallow copy semantics - this is ok
// as the public API is immutable and thus there is no point in making deep copies.
polyjectory::polyjectory(const polyjectory &) = default;

polyjectory::polyjectory(polyjectory &&) noexcept = default;

polyjectory &polyjectory::operator=(const polyjectory &) = default;

polyjectory &polyjectory::operator=(polyjectory &&) noexcept = default;

polyjectory::~polyjectory() = default;

// NOTE: using std::size_t for indexing is ok because we used std::size_t-based
// spans on construction - thus, std::size_t can always represent the number
// of objects in the polyjectory.
std::tuple<polyjectory::traj_span_t, polyjectory::time_span_t, std::int32_t>
polyjectory::operator[](std::size_t i) const
{
    if (i >= m_impl->m_traj_offset_vec.size()) [[unlikely]] {
        throw std::out_of_range(
            fmt::format("Invalid object index {} specified - the total number of objects is only {}", i,
                        m_impl->m_traj_offset_vec.size()));
    }

    // Fetch the base pointers.
    const auto *traj_base_ptr = m_impl->m_traj_ptr;
    const auto *time_base_ptr = m_impl->m_time_ptr;

    // Fetch the traj offset and nsteps.
    const auto [t_offset, nsteps] = m_impl->m_traj_offset_vec[i];

    // Compute the pointers.
    const auto *traj_ptr = traj_base_ptr + t_offset;
    const auto *time_ptr = time_base_ptr + m_impl->m_time_offset_vec[i];

    // Return the spans.
    return {traj_span_t{traj_ptr, nsteps,
                        // NOTE: static_cast is ok, m_poly_op1 was originally a std::size_t.
                        static_cast<std::size_t>(m_impl->m_poly_op1)},
            time_span_t{time_ptr, nsteps}, m_impl->m_status[i]};
}

// NOTE: using std::size_t as a size type is ok because we used std::size_t-based
// spans on construction - thus, std::size_t can always represent the number
// of objects in the polyjectory.
std::size_t polyjectory::get_nobjs() const noexcept
{
    return static_cast<std::size_t>(m_impl->m_traj_offset_vec.size());
}

double polyjectory::get_maxT() const noexcept
{
    return m_impl->m_maxT;
}

std::pair<double, double> polyjectory::get_epoch() const noexcept
{
    return {m_impl->m_epoch, m_impl->m_epoch2};
}

std::uint32_t polyjectory::get_poly_order() const noexcept
{
    assert(m_impl->m_poly_op1 > 0u);
    return m_impl->m_poly_op1 - 1u;
}

polyjectory::status_span_t polyjectory::get_status() const noexcept
{
    // NOTE: static_cast is ok, we know that we can represent the total number of objects
    // in the polyjectory as a std::size_t.
    return status_span_t{m_impl->m_status.data(), static_cast<std::size_t>(m_impl->m_status.size())};
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
