// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/align.hpp>
#include <boost/filesystem/file_status.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include "poly_trajectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

struct poly_trajectory::impl {
    // This is a vector that will contain:
    // - the offset in the buffer at which the trajectory data for an object begins,
    // - the total number of steps in the trajectory data,
    // - the polynomial order of the trajectory data.
    using traj_offset_vec_t = std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>;

    explicit impl(boost::filesystem::path file_path, traj_offset_vec_t traj_offset_vec,
                  std::vector<std::size_t> time_offset_vec)
        : m_file_path(std::move(file_path)), m_traj_offset_vec(std::move(traj_offset_vec)),
          m_time_offset_vec(std::move(time_offset_vec))
    {
        m_file.open(m_file_path.string());

        // LCOV_EXCL_START
        if (static_cast<unsigned>(m_file.alignment()) < alignof(double)) [[unlikely]] {
            throw std::runtime_error(fmt::format("Invalid alignment detected in a memory mapped file: the alignment of "
                                                 "the file is {}, but an alignment of {} is required instead",
                                                 m_file.alignment(), alignof(double)));
        }
        // LCOV_EXCL_STOP
    }

    [[nodiscard]] const double *base_ptr() const noexcept
    {
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        const auto *base_ptr = reinterpret_cast<const double *>(m_file.data());
        assert(boost::alignment::is_aligned(base_ptr, alignof(double)));

        return base_ptr;
    }

    impl(impl &&) noexcept = delete;
    impl(const impl &) = delete;
    impl &operator=(const impl &) = delete;
    impl &operator=(impl &&) noexcept = delete;
    ~impl()
    {
        // Close the memory mapped file.
        m_file.close();

        // Remove the temp dir and everything within.
        boost::filesystem::remove_all(m_file_path.parent_path());
    }

    boost::filesystem::path m_file_path;
    traj_offset_vec_t m_traj_offset_vec;
    std::vector<std::size_t> m_time_offset_vec;
    boost::iostreams::mapped_file_source m_file;
};

poly_trajectory::poly_trajectory(ptag, std::tuple<std::vector<traj_span_t>, std::vector<time_span_t>> tup)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    const auto &[traj_spans, time_spans] = tup;

    if (traj_spans.empty()) [[unlikely]] {
        throw std::invalid_argument("Cannot initialise a poly_trajectory object from an empty list of trajectories");
    }

    if (time_spans.empty()) [[unlikely]] {
        throw std::invalid_argument("Cannot initialise a poly_trajectory object from an empty list of times");
    }

    if (traj_spans.size() != time_spans.size()) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "In the construction of a poly_trajectory, the number of objects deduced from the list of trajectories "
            "({}) is inconsistent with the number of objects deduced from the list of times ({})",
            traj_spans.size(), time_spans.size()));
    }

    // Cache the total number of objects.
    const auto n_objs = traj_spans.size();

    // Assemble a "unique" dir path into the system temp dir.
    const auto tmp_dir_path
        = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path("mizuba-%%%%-%%%%-%%%%-%%%%");

    // Attempt to create it.
    // LCOV_EXCL_START
    if (!boost::filesystem::create_directory(tmp_dir_path)) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("Error while creating a unique temporary directory: the directory '{}' already exists",
                        tmp_dir_path.string()));
    }
    // LCOV_EXCL_STOP

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the temp dir in case of exceptions.
    try {
        // Change the permissions so that only the owner has access.
        boost::filesystem::permissions(tmp_dir_path, boost::filesystem::owner_all);

        // Init the storage file.
        auto storage_path = tmp_dir_path / "storage";
        // LCOV_EXCL_START
        if (boost::filesystem::exists(storage_path)) [[unlikely]] {
            throw std::runtime_error(
                fmt::format("Cannot create the storage file '{}', as it exists already", storage_path.string()));
        }
        // LCOV_EXCL_STOP
        std::ofstream storage_file(storage_path.string(), std::ios::binary | std::ios::out);
        // Make sure we throw on errors.
        storage_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        // Init the trajectories offset vector.
        impl::traj_offset_vec_t traj_offset_vec;
        traj_offset_vec.reserve(n_objs);

        // Keep track of the current offset into the file.
        safe_size_t cur_offset(0);

        // Check and write the trajectory data.
        // NOTE: we could investigate if computing and pre-allocating the file size
        // leads to better performance. For now, let us keep it simple.
        for (decltype(traj_spans.size()) i = 0; i < n_objs; ++i) {
            // Fetch the traj data for the current object.
            const auto cur_traj = traj_spans[i];

            // Check the number of steps.
            if (cur_traj.extent(0) == 0u) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "The trajectory for the object at index {} consists of zero steps - this is not allowed", i));
            }

            // Check the order + 1.
            if (cur_traj.extent(2) < 3u) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "The trajectory polynomial order for the object at index {} is less than 2 - this is not allowed",
                    i));
            }

            // Compute the total data size (in number of floating-point values).
            const auto traj_size = safe_size_t(cur_traj.extent(0)) * cur_traj.extent(1) * cur_traj.extent(2);

            // Check for non-finite data.
            for (std::size_t j = 0; j < cur_traj.extent(0); ++j) {
                for (std::size_t k = 0; k < cur_traj.extent(1); ++k) {
                    for (std::size_t l = 0; l < cur_traj.extent(2); ++l) {
                        if (!std::isfinite(cur_traj(j, k, l))) [[unlikely]] {
                            throw std::invalid_argument(
                                fmt::format("A non-finite value was found in the trajectory at index {}", i));
                        }
                    }
                }
            }

            // Bulk write into the file.
            storage_file.write(reinterpret_cast<const char *>(cur_traj.data_handle()),
                               static_cast<std::streamsize>(traj_size * sizeof(double)));

            // Add entry to the offset vector.
            traj_offset_vec.emplace_back(cur_offset, cur_traj.extent(0), cur_traj.extent(2));

            // Update cur_offset.
            cur_offset += traj_size;
        }

        // Offset vector for the time data.
        std::vector<std::size_t> time_offset_vec;
        time_offset_vec.reserve(n_objs);

        // Write the time data.
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

            // Check data.
            for (std::size_t j = 0; j < cur_time.extent(0); ++j) {
                if (!std::isfinite(cur_time(j))) [[unlikely]] {
                    throw std::invalid_argument(
                        fmt::format("A non-finite time coordinate was found for the object at index {}", i));
                }

                if (j > 0u && !(cur_time(j) > cur_time(j - 1u))) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "The sequence of times for the object at index {} is not monotonically increasing", i));
                }
            }

            // Bulk write into the file.
            storage_file.write(reinterpret_cast<const char *>(cur_time.data_handle()),
                               static_cast<std::streamsize>(time_size * sizeof(double)));

            // Add entry to the offset vector.
            time_offset_vec.emplace_back(cur_offset);

            // Update cur_offset.
            cur_offset += time_size;
        }

        // Close the storage file.
        storage_file.close();

        // Create the impl.
        // NOTE: here make_shared first allocates, and then constructs. If there are no exceptions, the assignment
        // to m_impl is noexcept and the dtor of impl takes charge of cleaning up the tmp_dir_path upon destruction.
        // If an exception is thrown (e.g., from memory allocation or from the impl ctor throwing), the impl has not
        // been fully constructed and thus its dtor will not be invoked, and the cleanup of tmp_dir_path will be
        // performed in the catch block below.
        m_impl
            = std::make_shared<impl>(std::move(storage_path), std::move(traj_offset_vec), std::move(time_offset_vec));
    } catch (...) {
        boost::filesystem::remove_all(tmp_dir_path);
        throw;
    }
}

poly_trajectory::poly_trajectory(const poly_trajectory &) = default;

poly_trajectory::poly_trajectory(poly_trajectory &&other) noexcept = default;

poly_trajectory &poly_trajectory::operator=(const poly_trajectory &) = default;

poly_trajectory &poly_trajectory::operator=(poly_trajectory &&) noexcept = default;

poly_trajectory::~poly_trajectory() = default;

std::pair<poly_trajectory::traj_span_t, poly_trajectory::time_span_t> poly_trajectory::operator[](std::size_t i) const
{
    if (i >= m_impl->m_traj_offset_vec.size()) [[unlikely]] {
        throw std::out_of_range(
            fmt::format("Invalid object index {} specified - the total number of objects is only {}", i,
                        m_impl->m_traj_offset_vec.size()));
    }

    // Fetch the base pointer.
    const auto *base_ptr = m_impl->base_ptr();

    // Fetch the traj offset, nsteps and order.
    const auto [traj_offset, nsteps, order] = m_impl->m_traj_offset_vec[i];

    // Compute the pointers.
    const auto *traj_ptr = base_ptr + traj_offset;
    const auto *time_ptr = base_ptr + m_impl->m_time_offset_vec[i];

    // Return the spans.
    return {traj_span_t{traj_ptr, nsteps, order}, time_span_t{time_ptr, nsteps}};
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
