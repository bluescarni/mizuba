// Copyright 2024-2025 Francesco Biscani
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
#include <fstream>
#include <ios>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>

#include "detail/atomic_minmax.hpp"
#include "detail/dfloat_utils.hpp"
#include "detail/file_utils.hpp"
#include "detail/poly_utils.hpp"
#include "logging.hpp"
#include "mdspan.hpp"
#include "polyjectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

namespace detail
{

namespace
{

// The current version of the polyjectory data structure.
constexpr unsigned cur_pj_version = 0;

} // namespace

struct polyjectory_impl {
    using traj_offset = polyjectory::traj_offset;

    // NOTE: the descriptor contains several metadata about the polyjectory.
    // It is stored into a file like the rest of the data so that polyjectories
    // can easily be persisted to disk.
    struct descriptor {
        // The version of the polyjectory.
        unsigned version = cur_pj_version;
        // The total number of objects.
        std::size_t n_objs = 0;
        // Polynomial order + 1 for the trajectory data.
        std::uint32_t poly_op1 = 0;
        // The duration of the longest trajectory.
        double maxT = 0;
        // The initial epoch.
        double epoch = 0;
        // The second component of the initial epoch.
        double epoch2 = 0;
    };
    static_assert(std::is_trivially_copyable_v<descriptor>);

    // The path to the dir containing the polyjectory data.
    boost::filesystem::path m_data_dir_path;

    // The memory-mapped files.
    //
    // Descriptor.
    boost::iostreams::mapped_file_source m_desc_file;
    // Offsets of the trajectory data.
    boost::iostreams::mapped_file_source m_traj_offsets_file;
    // Offsets of the time data.
    boost::iostreams::mapped_file_source m_time_offsets_file;
    // Trajectory data.
    boost::iostreams::mapped_file_source m_traj_file;
    // Time data.
    boost::iostreams::mapped_file_source m_time_file;
    // Statuses.
    boost::iostreams::mapped_file_source m_status_file;

    // Pointers to the memory-mapped data.
    const descriptor *m_desc_ptr = nullptr;
    const traj_offset *m_traj_offsets_ptr = nullptr;
    const std::size_t *m_time_offsets_ptr = nullptr;
    const double *m_traj_ptr = nullptr;
    const double *m_time_ptr = nullptr;
    const std::int32_t *m_status_ptr = nullptr;

    explicit polyjectory_impl(boost::filesystem::path data_dir_path, std::size_t n_objs, std::uint32_t poly_op1,
                              double maxT, double epoch, double epoch2)
        : m_data_dir_path(std::move(data_dir_path)), m_traj_offsets_file((m_data_dir_path / "traj_offsets").string()),
          m_time_offsets_file((m_data_dir_path / "time_offsets").string()),
          m_traj_file((m_data_dir_path / "traj").string()), m_time_file((m_data_dir_path / "time").string()),
          m_status_file((m_data_dir_path / "status").string())
    {
        // Build the descriptor and dump it to file.
        {
            const descriptor desc{
                .n_objs = n_objs, .poly_op1 = poly_op1, .maxT = maxT, .epoch = epoch, .epoch2 = epoch2};

            // Create the file.
            const auto file_path = m_data_dir_path / "desc";
            assert(!boost::filesystem::exists(file_path));
            std::ofstream file(file_path.string(), std::ios::binary | std::ios::out);

            // Write.
            file.write(reinterpret_cast<const char *>(&desc), boost::numeric_cast<std::streamsize>(sizeof(descriptor)));

            // Close and mark as read-only.
            file.close();
            mark_file_read_only(file_path);
        }

        // Memory-map the descriptor file.
        m_desc_file.open((m_data_dir_path / "desc").string());

        // Assign the pointers to the memory-mapped data.
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        m_desc_ptr = reinterpret_cast<const descriptor *>(m_desc_file.data());
        m_traj_offsets_ptr = reinterpret_cast<const traj_offset *>(m_traj_offsets_file.data());
        m_time_offsets_ptr = reinterpret_cast<const std::size_t *>(m_time_offsets_file.data());
        m_traj_ptr = reinterpret_cast<const double *>(m_traj_file.data());
        m_time_ptr = reinterpret_cast<const double *>(m_time_file.data());
        m_status_ptr = reinterpret_cast<const std::int32_t *>(m_status_file.data());
    }

    [[nodiscard]] bool is_open() noexcept
    {
        return m_desc_file.is_open();
    }

    void close() noexcept
    {
        // NOTE: a polyjectory is not supposed to be closed
        // more than once.
        assert(is_open());

        // Close all memory-mapped files.
        m_desc_file.close();
        m_traj_offsets_file.close();
        m_time_offsets_file.close();
        m_traj_file.close();
        m_time_file.close();
        m_status_file.close();

        // Remove the data dir and everything within.
        boost::filesystem::remove_all(m_data_dir_path);
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

namespace
{

// Small helper to dump a std::vector to a file. The file must not exist already.
// The file will be marked as read-only after writing.
template <typename T>
void dump_vector_to_file(const std::vector<T> &vec, const boost::filesystem::path &file_path)
{
    // Create the file.
    assert(!boost::filesystem::exists(file_path));
    std::ofstream file(file_path.string(), std::ios::binary | std::ios::out);
    file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    // Write.
    using safe_streamsize_t = boost::safe_numerics::safe<std::streamsize>;
    file.write(reinterpret_cast<const char *>(vec.data()), safe_streamsize_t(vec.size()) * sizeof(T));

    // Close and mark as read-only.
    file.close();
    mark_file_read_only(file_path);
};

} // namespace

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

    // Check epoch and epoch2, and compute the normalised double-length epoch.
    if (!std::isfinite(epoch)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The initial epoch of a polyjectory must be finite, but instead a value of {} was provided", epoch));
    }
    if (!std::isfinite(epoch2)) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The second component of the initial epoch of a polyjectory must be "
                                                "finite, but instead a value of {} was provided",
                                                epoch2));
    }
    const auto dl_epoch = detail::hilo_to_dfloat(epoch, epoch2);

    // Assemble a "unique" dir path into the system temp dir.
    auto data_dir_path = detail::create_temp_dir("mizuba_polyjectory-%%%%-%%%%-%%%%-%%%%");

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the data dir in case of exceptions.
    try {
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
            const auto op1 = cur_traj.extent(1);
            if (i == 0u) {
                if (op1 < 3u) [[unlikely]] {
                    throw std::invalid_argument("The trajectory polynomial order for the first object "
                                                "is less than 2 - this is not allowed");
                }

                poly_op1 = boost::numeric_cast<std::uint32_t>(op1);
            } else if (op1 != poly_op1) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("The trajectory polynomial order for the object at index " // LCOV_EXCL_LINE
                                "{} is inconsistent with the polynomial order deduced from the first object ({})",
                                i, poly_op1 - 1u));
            }

            // Compute the total data size (in number of floating-point values).
            const auto traj_size = safe_size_t(cur_traj.extent(0)) * op1 * cur_traj.extent(2);

            // Add entry to the offset vector.
            traj_offset_vec.emplace_back(cur_traj_offset, cur_traj.extent(0));

            // Update cur_offset.
            cur_traj_offset += traj_size;
        }

        // Check that we have some trajectory data.
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
            if (cur_time.extent(0) == 0u) {
                // No time data, this must be an empty trajectory.
                if (cur_traj.extent(0) != 0u) [[unlikely]] {
                    throw std::invalid_argument(fmt::format("The trajectory of the object at index {} has a nonzero "
                                                            "number of steps but no associated time data",
                                                            i));
                }
            } else if (cur_time.extent(0) - 1u != cur_traj.extent(0)) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("The number of steps for the trajectory of the object at index {} is {}, but the "
                                "number of times is {} (the number of times must be equal to the number of steps + 1)",
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

        // NOTE: at this point maxT could contain bogus/incorrect values because
        // we have not checked the time data yet. We will do this below.

        // Measure the data copying time.
        stopwatch sw;

        // Concurrently fill in the data files and the offsets/status files.
        oneapi::tbb::parallel_invoke(
            // Trajectory offsets.
            [&data_dir_path, &traj_offset_vec]() {
                detail::dump_vector_to_file(traj_offset_vec, data_dir_path / "traj_offsets");
            },
            // Time offsets.
            [&data_dir_path, &time_offset_vec]() {
                detail::dump_vector_to_file(time_offset_vec, data_dir_path / "time_offsets");
            },
            // Statuses.
            [&data_dir_path, &status]() { detail::dump_vector_to_file(status, data_dir_path / "status"); },
            // Data files.
            [n_objs, &traj_spans, &traj_offset_vec, &time_spans, &time_offset_vec, &data_dir_path, cur_traj_offset,
             cur_time_offset]() {
                // Init the storage data files.
                const auto traj_path = data_dir_path / "traj";
                const auto time_path = data_dir_path / "time";
                detail::create_sized_file(traj_path, cur_traj_offset * sizeof(double));
                detail::create_sized_file(time_path, cur_time_offset * sizeof(double));

                // Prepare the file_pwrite instances for parallel writing into the datafiles.
                detail::file_pwrite traj_file(traj_path);
                detail::file_pwrite time_file(time_path);

                // Check and copy over the data from the spans.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<decltype(traj_spans.size())>(0, n_objs),
                    [&traj_spans, &traj_offset_vec, &time_spans, &time_offset_vec, &traj_file,
                     &time_file](const auto &range) {
                        for (auto i = range.begin(); i != range.end(); ++i) {
                            // Trajectory data.
                            const auto cur_traj = traj_spans[i];

                            // Check for non-finite data.
                            for (std::size_t j = 0; j < cur_traj.extent(0); ++j) {
                                for (std::size_t k = 0; k < cur_traj.extent(1); ++k) {
                                    for (std::size_t l = 0; l < cur_traj.extent(2); ++l) {
                                        if (!std::isfinite(cur_traj[j, k, l])) [[unlikely]] {
                                            throw std::invalid_argument(fmt::format(
                                                "A non-finite value was found in the trajectory at index {}", i));
                                        }
                                    }
                                }
                            }

                            // Compute the total data size (in number of floating-point values).
                            const auto traj_size = cur_traj.size();

                            // Copy the data into the file.
                            // NOTE: computations of size/offset here are safe as we managed to create
                            // the datafiles with std::size_t as a size type.
                            traj_file.pwrite(cur_traj.data_handle(), traj_size * sizeof(double),
                                             traj_offset_vec[i].offset * sizeof(double));

                            // Time data.
                            const auto cur_time = time_spans[i];

                            // Check data.
                            for (std::size_t j = 0; j < cur_time.extent(0); ++j) {
                                if (!std::isfinite(cur_time(j))) [[unlikely]] {
                                    throw std::invalid_argument(fmt::format(
                                        "A non-finite time coordinate was found for the object at index {}", i));
                                }

                                if (cur_time(j) < 0) [[unlikely]] {
                                    throw std::invalid_argument(fmt::format(
                                        "A negative time coordinate was found for the object at index {}", i));
                                }

                                if (j > 0u && !(cur_time(j) > cur_time(j - 1u))) [[unlikely]] {
                                    throw std::invalid_argument(fmt::format("The sequence of times for the object at "
                                                                            "index {} is not monotonically increasing",
                                                                            i));
                                }
                            }

                            // Compute the total data size (in number of floating-point values).
                            const auto time_size = cur_time.extent(0);

                            // Copy the data into the file.
                            // NOTE: computations of size/offset here are safe as we managed to create
                            // the datafiles with std::size_t as a size type.
                            time_file.pwrite(cur_time.data_handle(), time_size * sizeof(double),
                                             time_offset_vec[i] * sizeof(double));
                        }
                    });

                // Close the data files.
                traj_file.close();
                time_file.close();

                // Mark them as read-only.
                detail::mark_file_read_only(traj_path);
                detail::mark_file_read_only(time_path);
            });

        // We can now assert that maxT must not be zero: the time values have all been validated
        // and we checked earlier that at least one trajectory has a nonzero number of steps.
        assert(maxT != 0);

        log_trace("polyjectory copy from spans time: {}", sw);

        // Create the impl.
        // NOTE: here make_shared() first allocates, and then constructs. If there are no exceptions, the assignment
        // to m_impl is noexcept and the dtor of impl takes charge of cleaning up the data_dir_path upon destruction.
        // If an exception is thrown (e.g., from memory allocation or from the impl ctor throwing), the impl has not
        // been fully constructed and thus its dtor will not be invoked, and the cleanup of data_dir_path will be
        // performed in the catch block below.
        m_impl = std::make_shared<detail::polyjectory_impl>(std::move(data_dir_path),
                                                            boost::numeric_cast<std::size_t>(n_objs), poly_op1, maxT,
                                                            dl_epoch.hi, dl_epoch.lo);
    } catch (...) {
        boost::filesystem::remove_all(data_dir_path);
        throw;
    }
}

// Constructor from a traj and time data files at the locations 'orig_traj_file_path'
// and 'orig_time_file_path'. The original files will be moved into the polyjectory's data dir.
//
// The layout of the trajectory data into the data file is described by traj_offsets, from which we
// also deduce the layout of the time data. 'order' is the polynomial order of the polyjectory,
// 'status' the vector of object statuses.
//
// NOTE: although we try hard here to secure operations against a malicious user, I am not 100%
// sure that file handling here is completely safe. Thus, in the documentation, we should emphasise that
// users should not call this constructor on files not owned by them. And, clearly, they are not supposed
// to write to the files during or after the invocation of the constructor.
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
    const std::uint32_t op1 = order + 1u;

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

    // Check epoch and epoch2, and compute the normalised double-length epoch.
    if (!std::isfinite(epoch)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The initial epoch of a polyjectory must be finite, but instead a value of {} was provided", epoch));
    }
    if (!std::isfinite(epoch2)) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The second component of the initial epoch of a polyjectory must be "
                                                "finite, but instead a value of {} was provided",
                                                epoch2));
    }
    const auto dl_epoch = detail::hilo_to_dfloat(epoch, epoch2);

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

    // Check that the two paths are not the same.
    if (traj_file_path == time_file_path) [[unlikely]] {
        throw std::invalid_argument("Invalid data file(s) passed to the constructor of a polyjectory: the trajectory "
                                    "data file and the time data file are the same file");
    }

    // Iterate over the trajectory offsets, sanity-checking them and computing
    // the total number of floating-point values expected in the files.
    safe_size_t tot_num_traj_values = 0, tot_num_time_values = 0;

    // NOTE: this should be parallelisable with some effort, if needed. The only complication
    // would be the handling of safe arithmetics.
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

            // NOTE: the two offsets will be equal if the previous trajectory is empty, thus we check with '<'.
            if (offset < prev_offset) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("Invalid trajectory offsets vector passed to the constructor of a polyjectory: "
                                "the offset of the object at index {} is less than the offset of the previous object",
                                i));
            }

            if (offset - prev_offset != safe_size_t(prev_n_steps) * op1 * 7u) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "Inconsistent data detected in the trajectory offsets vector passed to the constructor of "
                    "a polyjectory: the offset of the object at index {} is inconsistent with the offset and "
                    "number of steps of the previous object",
                    i));
            }
        }

        // Update tot_num_traj_values and tot_num_time_values.
        tot_num_traj_values += safe_size_t(n_steps) * op1 * 7u;
        // NOTE: the total number of time values for the current trajectory is zero if n_steps == 0,
        // otherwise it is n_steps + 1. n_steps + 1 is safe to compute as we computed n_steps * 7
        // the line before.
        tot_num_time_values += n_steps + static_cast<unsigned>(n_steps != 0u);
    }

    // Check that we have some trajectory data.
    if (tot_num_traj_values == 0u) [[unlikely]] {
        throw std::invalid_argument(
            "All the trajectories in a polyjectory have a number of steps equal to zero: this is not allowed");
    }

    // Build the time offsets vector from the trajectory offsets.
    // NOTE: all computations here are safe because we managed to compute
    // tot_num_time_values without overflow.
    std::vector<std::size_t> time_offsets;
    time_offsets.reserve(traj_offsets.size());
    for (decltype(traj_offsets.size()) i = 0; i < traj_offsets.size(); ++i) {
        if (i == 0u) {
            time_offsets.push_back(0);
        } else {
            const auto [_, prev_nsteps] = traj_offsets[i - 1u];
            // NOTE: as usual, we need to add +1 to prev_nsteps if the previous trajectory is not empty.
            time_offsets.push_back(time_offsets.back() + prev_nsteps + static_cast<unsigned>(prev_nsteps != 0u));
        }
    }

    // Assemble a "unique" dir path into the system temp dir.
    auto data_dir_path = detail::create_temp_dir("mizuba_polyjectory-%%%%-%%%%-%%%%-%%%%");

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the data dir in case of exceptions.
    try {
        // Atomic variable to compute maxT.
        std::atomic maxT = -std::numeric_limits<double>::infinity();

        // Concurrently move/check the data files and dump the offsets/status files.
        oneapi::tbb::parallel_invoke(
            // Trajectory offsets.
            [&data_dir_path, &traj_offsets]() {
                detail::dump_vector_to_file(traj_offsets, data_dir_path / "traj_offsets");
            },
            // Time offsets.
            [&data_dir_path, &time_offsets]() {
                detail::dump_vector_to_file(time_offsets, data_dir_path / "time_offsets");
            },
            // Statuses.
            [&data_dir_path, &status]() { detail::dump_vector_to_file(status, data_dir_path / "status"); },
            [&data_dir_path, &traj_file_path, &time_file_path, tot_num_traj_values, tot_num_time_values, &maxT,
             &traj_offsets, &time_offsets, op1]() {
                // Init the data file paths.
                const auto traj_path = data_dir_path / "traj";
                assert(!boost::filesystem::exists(traj_path));
                const auto time_path = data_dir_path / "time";
                assert(!boost::filesystem::exists(time_path));

                // Move the original files.
                boost::filesystem::rename(traj_file_path, traj_path);
                boost::filesystem::rename(time_file_path, time_path);

                // NOTE: now that we have moved the original files, we run checks on them.

                // The files must be regular files.
                if (!boost::filesystem::is_regular_file(traj_path)) [[unlikely]] {
                    throw std::invalid_argument(
                        "Invalid trajectory data file passed to the constructor of a polyjectory: the "
                        "file is not a regular file");
                }
                if (!boost::filesystem::is_regular_file(time_path)) [[unlikely]] {
                    throw std::invalid_argument(
                        "Invalid time data file passed to the constructor of a polyjectory: the "
                        "file is not a regular file");
                }

                // Mark them as read-only.
                // NOTE: this also acts as a (partial) check on the ownership of the data files - in general we should
                // not be able to set them as read-only if we do not own them.
                detail::mark_file_read_only(traj_path);
                detail::mark_file_read_only(time_path);

                // Check the file sizes.
                if (boost::filesystem::file_size(traj_path) != tot_num_traj_values * sizeof(double)) [[unlikely]] {
                    throw std::invalid_argument(
                        // LCOV_EXCL_START
                        fmt::format("Invalid trajectory data file passed to the constructor of a polyjectory: the "
                                    "expected size in bytes is {} but the actual size is {} instead",
                                    // LCOV_EXCL_STOP
                                    static_cast<std::size_t>(tot_num_traj_values * sizeof(double)),
                                    boost::filesystem::file_size(traj_path)));
                }
                if (boost::filesystem::file_size(time_path) != tot_num_time_values * sizeof(double)) [[unlikely]] {
                    throw std::invalid_argument(
                        // LCOV_EXCL_START
                        fmt::format("Invalid time data file passed to the constructor of a polyjectory: the "
                                    "expected size in bytes is {} but the actual size is {} instead",
                                    // LCOV_EXCL_STOP
                                    static_cast<std::size_t>(tot_num_time_values * sizeof(double)),
                                    boost::filesystem::file_size(time_path)));
                }

                // Memory-map the files.
                boost::iostreams::mapped_file_source traj_file(traj_path);
                const auto *traj_file_base_ptr = reinterpret_cast<const double *>(traj_file.data());

                boost::iostreams::mapped_file_source time_file(time_path);
                const auto *time_file_base_ptr = reinterpret_cast<const double *>(time_file.data());

                // Check the data and compute maxT.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<decltype(traj_offsets.size())>(0, traj_offsets.size()),
                    [traj_file_base_ptr, time_file_base_ptr, &maxT, &traj_offsets, &time_offsets,
                     op1](const auto &range) {
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
                                    for (std::size_t l = 0; l < cur_traj.extent(2); ++l) {
                                        if (!std::isfinite(cur_traj[j, k, l])) [[unlikely]] {
                                            throw std::invalid_argument(fmt::format(
                                                "A non-finite value was found in the trajectory at index {}", i));
                                        }
                                    }
                                }
                            }

                            // Build a time span and check the data.
                            const time_span_t cur_time{time_file_base_ptr + time_offset,
                                                       // NOTE: as usual, the total number of time data points is
                                                       // n_steps + 1 if n_steps > 0, zero otherwise.
                                                       n_steps + static_cast<unsigned>(n_steps != 0u)};

                            for (std::size_t j = 0; j < cur_time.extent(0); ++j) {
                                if (!std::isfinite(cur_time(j))) [[unlikely]] {
                                    throw std::invalid_argument(fmt::format(
                                        "A non-finite time coordinate was found for the object at index {}", i));
                                }

                                if (cur_time(j) < 0) [[unlikely]] {
                                    throw std::invalid_argument(fmt::format(
                                        "A negative time coordinate was found for the object at index {}", i));
                                }

                                if (j > 0u && !(cur_time(j) > cur_time(j - 1u))) [[unlikely]] {
                                    throw std::invalid_argument(fmt::format("The sequence of times for the object at "
                                                                            "index {} is not monotonically increasing",
                                                                            i));
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
            });

        // We can now assert that maxT must be strictly positive: the time values have all been validated
        // and we checked earlier that at least one trajectory has a nonzero number of steps.
        assert(maxT.load() > 0);

        // Construct the implementation.
        m_impl = std::make_shared<detail::polyjectory_impl>(std::move(data_dir_path),
                                                            boost::numeric_cast<std::size_t>(traj_offsets.size()), op1,
                                                            maxT.load(), dl_epoch.hi, dl_epoch.lo);
    } catch (...) {
        boost::filesystem::remove_all(data_dir_path);
        throw;
    }
}

// NOTE: the polyjectory class will have shallow copy semantics - this is ok
// as the public API is immutable and thus there is no point in making deep copies.
polyjectory::polyjectory(const polyjectory &) = default;

polyjectory::polyjectory(polyjectory &&) noexcept = default;

polyjectory &polyjectory::operator=(const polyjectory &) = default;

polyjectory &polyjectory::operator=(polyjectory &&) noexcept = default;

polyjectory::~polyjectory() = default;

std::tuple<polyjectory::traj_span_t, polyjectory::time_span_t, std::int32_t>
polyjectory::operator[](std::size_t i) const
{
    if (i >= get_nobjs()) [[unlikely]] {
        throw std::out_of_range(
            fmt::format("Invalid object index {} specified - the total number of objects in the polyjectory is only {}",
                        i, get_nobjs()));
    }

    // Fetch the base pointers.
    const auto *traj_base_ptr = m_impl->m_traj_ptr;
    const auto *time_base_ptr = m_impl->m_time_ptr;

    // Fetch the traj offset and nsteps.
    const auto [t_offset, nsteps] = m_impl->m_traj_offsets_ptr[i];

    // Compute the pointers.
    const auto *traj_ptr = traj_base_ptr + t_offset;
    const auto *time_ptr = time_base_ptr + m_impl->m_time_offsets_ptr[i];

    // Return the spans.
    return {traj_span_t{traj_ptr, nsteps,
                        // NOTE: static_cast is ok, because the number of trajectory data
                        // points for a single object, which is n_steps * op1 * 7, is guaranteed
                        // to be representable as a std::size_t and we know that there is at least
                        // one object with n_steps > 0.
                        static_cast<std::size_t>(m_impl->m_desc_ptr->poly_op1)},
            time_span_t{time_ptr, nsteps + static_cast<unsigned>(nsteps != 0u)}, m_impl->m_status_ptr[i]};
}

std::size_t polyjectory::get_nobjs() const noexcept
{
    return m_impl->m_desc_ptr->n_objs;
}

double polyjectory::get_maxT() const noexcept
{
    return m_impl->m_desc_ptr->maxT;
}

std::pair<double, double> polyjectory::get_epoch() const noexcept
{
    return {m_impl->m_desc_ptr->epoch, m_impl->m_desc_ptr->epoch2};
}

std::uint32_t polyjectory::get_poly_order() const noexcept
{
    assert(m_impl->m_desc_ptr->poly_op1 > 0u);
    return m_impl->m_desc_ptr->poly_op1 - 1u;
}

std::filesystem::path polyjectory::get_data_dir() const
{
    // NOTE: we made sure on construction that the dir path is canonicalised.
    return std::filesystem::path(m_impl->m_data_dir_path.c_str());
}

dspan_1d<const std::int32_t> polyjectory::get_status() const noexcept
{
    return dspan_1d<const std::int32_t>{m_impl->m_status_ptr, get_nobjs()};
}

namespace detail
{

namespace
{

// Evaluate the state of an object in a polyjectory at time tm.
//
// pj is the polyjectory, obj_idx the object's index in the polyjectory, tm the evaluation
// time (measured from the polyjectory's epoch), out_ptr the pointer into which the result of
// the evaluation will be written.
void pj_eval_obj_state(const polyjectory &pj, std::size_t obj_idx, double tm, double *out_ptr)
{
    // Check the desired evaluation time.
    if (!std::isfinite(tm)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("An non-finite evaluation time of {} was detected during the evaluation of a polyjectory", tm));
    }

    // Fetch the traj and time spans for the current object.
    const auto [traj_span, time_span, _] = pj[obj_idx];

    // Fetch the total number of steps for the current object.
    const auto nsteps = traj_span.extent(0);

    // If either:
    //
    // - no trajectory data is available for this object, or
    // - the evaluation time is *before* the begin time of the trajectory, or
    // - the evaluation time is *at or after* the end time of the trajectory,
    //
    // then we cannot compute any evaluation for the current object, and we
    // fill up the output with nans instead.
    if (nsteps == 0u || tm < time_span(0) || tm >= time_span(nsteps)) {
        std::ranges::fill(out_ptr, out_ptr + 7, std::numeric_limits<double>::quiet_NaN());
        return;
    }

    // Make sure that nsteps + 1 (i.e., the number of time datapoints) is representable as std::ptrdiff_t.
    // This ensures that we can safely calculate pointer subtractions in the time span data,
    // which allows us to determine the index of a trajectory timestep (see the code
    // below computing step_idx).
    try {
        static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(nsteps + 1u));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error(
            "Overflow detected in the trajectory data during evaluation: the number of steps is too large");
    }
    // LCOV_EXCL_STOP

    // Fetch begin/end iterators to the time span.
    const auto t_begin = time_span.data_handle();
    const auto t_end = t_begin + (nsteps + 1u);

    // Look for the first trajectory step that ends *after* the evaluation time.
    const auto tm_it = std::ranges::upper_bound(t_begin + 1, t_end, tm);
    assert(tm_it != t_end);

    // Compute the time coordinate needed for polynomial evaluation.
    const auto h = tm - *(tm_it - 1);

    // Compute the step index.
    const auto step_idx = static_cast<std::size_t>(tm_it - (t_begin + 1));

    // Fetch the order of the polyjectory.
    const auto order = pj.get_poly_order();

    // Run the polynomial evaluations and write the results into the output span.
    for (auto i = 0u; i < 7u; ++i) {
        // NOTE: this can easily be vectorised.
        out_ptr[i] = detail::horner_eval(&traj_span[step_idx, 0, i], order, h, static_cast<std::size_t>(7));
    }
}

} // namespace

} // namespace detail

// Implementation of state_eval(). tm is either a scalar value or a 1D span.
template <typename Time>
void polyjectory::state_eval_impl(single_eval_span_t out, Time tm,
                                  std::optional<dspan_1d<const std::size_t>> selector) const
{
    if (selector) {
        // Fetch the selector.
        const auto sel = *selector;

        // Cache the number of selected objects.
        const auto n_sel_objs = sel.extent(0);

        // Check the out span.
        if (out.extent(0) != n_sel_objs) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "Invalid output array passed to state_eval(): the number of objects selected for evaluation is {} but "
                "the size of the first dimension of the array is {} (the two numbers must be equal)",
                n_sel_objs, out.extent(0)));
        }

        if constexpr (!std::same_as<double, Time>) {
            // Check the time span.
            if (tm.extent(0) != n_sel_objs) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "Invalid time array passed to state_eval(): the number of selected objects is {} but the "
                    "size of the array is {} (the two numbers must be equal)",
                    n_sel_objs, tm.extent(0)));
            }
        }

        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<std::size_t>(0, n_sel_objs), [this, tm, out, sel](const auto &range) {
                for (auto sel_idx = range.begin(); sel_idx != range.end(); ++sel_idx) {
                    if constexpr (std::same_as<double, Time>) {
                        detail::pj_eval_obj_state(*this, sel[sel_idx], tm, &out[sel_idx, 0]);
                    } else {
                        detail::pj_eval_obj_state(*this, sel[sel_idx], tm[sel_idx], &out[sel_idx, 0]);
                    }
                }
            });
    } else {
        // Cache the number of objects.
        const auto nobjs = get_nobjs();

        // Check the out span.
        if (out.extent(0) != nobjs) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid output array passed to state_eval(): the number of objects is {} but the size of "
                            "the first dimension of the array is {} (the two numbers must be equal)",
                            nobjs, out.extent(0)));
        }

        if constexpr (!std::same_as<double, Time>) {
            // Check the time span.
            if (tm.extent(0) != nobjs) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "Invalid time array passed to state_eval(): the number of objects is {} but the size of "
                    "the array is {} (the two numbers must be equal)",
                    nobjs, tm.extent(0)));
            }
        }

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                                  [this, tm, out](const auto &range) {
                                      for (auto obj_idx = range.begin(); obj_idx != range.end(); ++obj_idx) {
                                          if constexpr (std::same_as<double, Time>) {
                                              detail::pj_eval_obj_state(*this, obj_idx, tm, &out[obj_idx, 0]);
                                          } else {
                                              detail::pj_eval_obj_state(*this, obj_idx, tm[obj_idx], &out[obj_idx, 0]);
                                          }
                                      }
                                  });
    }
}

void polyjectory::state_eval(single_eval_span_t out, double tm,
                             std::optional<dspan_1d<const std::size_t>> selector) const
{
    state_eval_impl(out, tm, selector);
}

void polyjectory::state_eval(single_eval_span_t out, dspan_1d<const double> tm_arr,
                             std::optional<dspan_1d<const std::size_t>> selector) const
{
    state_eval_impl(out, tm_arr, selector);
}

// Implementation of state_meval(). tm is either a 1D or 2D span.
template <typename Time>
void polyjectory::state_meval_impl(multi_eval_span_t out, Time tm,
                                   std::optional<dspan_1d<const std::size_t>> selector) const
{
    // Cache the number of time evaluations per object.
    const auto n_time_evals = [&tm]() {
        if constexpr (std::same_as<dspan_1d<const double>, Time>) {
            return tm.extent(0);
        } else {
            return tm.extent(1);
        }
    }();

    // The second dimension of out must match the number
    // of time evaluations per object.
    if (out.extent(1) != n_time_evals) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid output array passed to state_meval(): the number of time evaluations per object is {} but the "
            "size of the second dimension of the array is {} (the two numbers must be equal)",
            n_time_evals, out.extent(1)));
    }

    if (selector) {
        // Cache the selector.
        const auto sel = *selector;

        // Cache the number of selected objects.
        const auto n_sel_objs = sel.extent(0);

        // Check the first dimension of the output span.
        if (out.extent(0) != n_sel_objs) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid output array passed to state_meval(): the number of selected objects is {} but "
                            "the size of the first dimension of the array is {} (the two numbers must be equal)",
                            n_sel_objs, out.extent(0)));
        }

        if constexpr (std::same_as<dspan_2d<const double>, Time>) {
            // Check the first dimension of the time array.
            if (tm.extent(0) != n_sel_objs) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("Invalid time array passed to state_meval(): the number of selected objects is {} but "
                                "the size of the first dimension of the array is {} (the two numbers must be equal)",
                                n_sel_objs, tm.extent(0)));
            }
        }

        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<std::size_t>(0, n_sel_objs, 0, n_time_evals),
            [this, sel, tm, out](const auto &range) {
                for (auto sel_idx = range.rows().begin(); sel_idx != range.rows().end(); ++sel_idx) {
                    for (auto tm_idx = range.cols().begin(); tm_idx != range.cols().end(); ++tm_idx) {
                        if constexpr (std::same_as<dspan_1d<const double>, Time>) {
                            detail::pj_eval_obj_state(*this, sel[sel_idx], tm[tm_idx], &out[sel_idx, tm_idx, 0]);
                        } else {
                            detail::pj_eval_obj_state(*this, sel[sel_idx], tm[sel_idx, tm_idx],
                                                      &out[sel_idx, tm_idx, 0]);
                        }
                    }
                }
            });
    } else {
        // Cache the number of objects.
        const auto nobjs = get_nobjs();

        // Check the first dimension of the output span.
        if (out.extent(0) != nobjs) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid output array passed to state_meval(): the number of objects is {} but the size of "
                            "the first dimension of the array is {} (the two numbers must be equal)",
                            nobjs, out.extent(0)));
        }

        if constexpr (std::same_as<dspan_2d<const double>, Time>) {
            // Check the first dimension of the time array.
            if (tm.extent(0) != nobjs) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "Invalid time array passed to state_meval(): the number of objects is {} but the size of "
                    "the first dimension of the array is {} (the two numbers must be equal)",
                    nobjs, tm.extent(0)));
            }
        }

        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range2d<std::size_t>(0, nobjs, 0, n_time_evals), [this, tm, out](const auto &range) {
                for (auto obj_idx = range.rows().begin(); obj_idx != range.rows().end(); ++obj_idx) {
                    for (auto tm_idx = range.cols().begin(); tm_idx != range.cols().end(); ++tm_idx) {
                        if constexpr (std::same_as<dspan_1d<const double>, Time>) {
                            detail::pj_eval_obj_state(*this, obj_idx, tm[tm_idx], &out[obj_idx, tm_idx, 0]);
                        } else {
                            detail::pj_eval_obj_state(*this, obj_idx, tm[obj_idx, tm_idx], &out[obj_idx, tm_idx, 0]);
                        }
                    }
                }
            });
    }
}

void polyjectory::state_meval(multi_eval_span_t out, dspan_1d<const double> tm_arr,
                              std::optional<dspan_1d<const std::size_t>> selector) const
{
    state_meval_impl(out, tm_arr, selector);
}

void polyjectory::state_meval(multi_eval_span_t out, dspan_2d<const double> tm_arr,
                              std::optional<dspan_1d<const std::size_t>> selector) const
{
    state_meval_impl(out, tm_arr, selector);
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
