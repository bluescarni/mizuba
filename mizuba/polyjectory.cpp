// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
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

#include "detail/file_utils.hpp"
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
    // This is a vector that will contain:
    // - the offset (in number of double-precision values) in the mmap buffer
    //   at which the trajectory data for an object begins,
    // - the total number of steps in the trajectory data.
    using traj_offset_vec_t = std::vector<std::tuple<std::size_t, std::size_t>>;

    // Path to the memory-mapped file.
    boost::filesystem::path m_file_path;
    // Offsets for the trajectory data.
    traj_offset_vec_t m_traj_offset_vec;
    // Offsets for the time data.
    std::vector<std::size_t> m_time_offset_vec;
    // Polynomial order + 1 for the trajectory data.
    std::uint32_t m_poly_op1 = 0;
    // The duration of the longest trajectory.
    double m_maxT = 0;
    // Vector of trajectory statuses.
    std::vector<std::int32_t> m_status;
    // The memory-mapped file.
    boost::iostreams::mapped_file_source m_file;
    // Pointer to the beginning of m_file, cast to double.
    const double *m_base_ptr = nullptr;

    explicit polyjectory_impl(boost::filesystem::path file_path, traj_offset_vec_t traj_offset_vec,
                              std::vector<std::size_t> time_offset_vec, std::uint32_t poly_op1, double maxT,
                              std::vector<std::int32_t> status)
        : m_file_path(std::move(file_path)), m_traj_offset_vec(std::move(traj_offset_vec)),
          m_time_offset_vec(std::move(time_offset_vec)), m_poly_op1(poly_op1), m_maxT(maxT),
          m_status(std::move(status)), m_file(m_file_path.string())
    {
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        m_base_ptr = reinterpret_cast<const double *>(m_file.data());
        assert(boost::alignment::is_aligned(m_base_ptr, alignof(double)));
    }

    [[nodiscard]] bool is_open() noexcept
    {
        return m_file.is_open();
    }

    void close() noexcept
    {
        // NOTE: a polyjectory is not supposed to be closed
        // more than once.
        assert(is_open());

        // Close the memory mapped file.
        m_file.close();

        // Remove the temp dir and everything within.
        boost::filesystem::remove_all(m_file_path.parent_path());
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

void close_pj(std::shared_ptr<polyjectory_impl> &pj) noexcept
{
    pj->close();
}

const std::shared_ptr<polyjectory_impl> &fetch_pj_impl(const polyjectory &pj) noexcept
{
    return pj.m_impl;
}

} // namespace detail

polyjectory::polyjectory(ptag,
                         std::tuple<std::vector<traj_span_t>, std::vector<time_span_t>, std::vector<std::int32_t>> tup)
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

    // Assemble a "unique" dir path into the system temp dir.
    const auto tmp_dir_path = detail::create_temp_dir("mizuba_polyjectory-%%%%-%%%%-%%%%-%%%%");

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
        detail::polyjectory_impl::traj_offset_vec_t traj_offset_vec;
        traj_offset_vec.reserve(n_objs);

        // Keep track of the current offset into the file.
        safe_size_t cur_offset(0);

        // Init the poly order + 1.
        std::uint32_t poly_op1 = 0;

        // Build the traj offset vector, determine poly_op1 and
        // run checks on the traj spans' dimensions.
        for (decltype(traj_spans.size()) i = 0; i < n_objs; ++i) {
            // Fetch the traj data for the current object.
            const auto cur_traj = traj_spans[i];

            // Check the number of steps.
            if (cur_traj.extent(0) == 0u) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "The trajectory for the object at index {} consists of zero steps - this is not allowed", i));
            }

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
            traj_offset_vec.emplace_back(cur_offset, cur_traj.extent(0));

            // Update cur_offset.
            cur_offset += traj_size;
        }

        // Offset vector for the time data.
        std::vector<std::size_t> time_offset_vec;
        time_offset_vec.reserve(n_objs);

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
            const auto time_size = safe_size_t(cur_time.extent(0));

            // Update maxT.
            const auto curT = cur_time(cur_time.extent(0) - 1u);
            maxT = (curT > maxT) ? curT : maxT;

            // Add entry to the offset vector.
            time_offset_vec.emplace_back(cur_offset);

            // Update cur_offset.
            cur_offset += time_size;
        }

        // NOTE: at this point maxT could contain bogus/incorrect values because
        // we have not checked the time data yet. We will do this below.

        // Init the storage file.
        auto storage_path = tmp_dir_path / "storage";
        detail::create_sized_file(storage_path, cur_offset * sizeof(double));

        // Memory-map it.
        boost::iostreams::mapped_file_sink file(storage_path.string());

        // Fetch a pointer to the beginning of the data.
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        auto *base_ptr = reinterpret_cast<double *>(file.data());
        assert(boost::alignment::is_aligned(base_ptr, alignof(double)));

        // Check and copy over the data from the spans.
        oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<decltype(traj_spans.size())>(0, n_objs),
            [base_ptr, poly_op1, &traj_spans, &traj_offset_vec, &time_spans, &time_offset_vec](const auto &range) {
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
                                      base_ptr + std::get<0>(traj_offset_vec[i]));

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
                    const auto time_size = safe_size_t(cur_time.extent(0));

                    // Copy the data into the file.
                    std::ranges::copy(cur_time.data_handle(),
                                      cur_time.data_handle() + static_cast<std::size_t>(time_size),
                                      base_ptr + time_offset_vec[i]);
                }
            });

        // We can now assert that maxT must not be zero.
        assert(maxT != 0);

        // Close the storage file.
        file.close();

        // Mark it as read-only.
        detail::mark_file_read_only(storage_path);

        // Create the impl.
        // NOTE: here make_shared() first allocates, and then constructs. If there are no exceptions, the assignment
        // to m_impl is noexcept and the dtor of impl takes charge of cleaning up the tmp_dir_path upon destruction.
        // If an exception is thrown (e.g., from memory allocation or from the impl ctor throwing), the impl has not
        // been fully constructed and thus its dtor will not be invoked, and the cleanup of tmp_dir_path will be
        // performed in the catch block below.
        m_impl
            = std::make_shared<detail::polyjectory_impl>(std::move(storage_path), std::move(traj_offset_vec),
                                                         std::move(time_offset_vec), poly_op1, maxT, std::move(status));

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

    // Fetch the base pointer.
    const auto *base_ptr = m_impl->m_base_ptr;

    // Fetch the traj offset and nsteps.
    const auto [traj_offset, nsteps] = m_impl->m_traj_offset_vec[i];

    // Compute the pointers.
    const auto *traj_ptr = base_ptr + traj_offset;
    const auto *time_ptr = base_ptr + m_impl->m_time_offset_vec[i];

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

std::filesystem::path polyjectory::get_file_path() const
{
    // NOTE: need to convert from Boost::filesystem to std::filesystem.
    // NOTE: m_impl->m_file_path should already be canonical, since the
    // path is somewhere inside a temp dir created via create_temp_dir().
    return std::filesystem::path(m_impl->m_file_path.string());
}

double polyjectory::get_maxT() const noexcept
{
    return m_impl->m_maxT;
}

std::uint32_t polyjectory::get_poly_order() const noexcept
{
    assert(m_impl->m_poly_op1 > 0u);
    return m_impl->m_poly_op1 - 1u;
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
