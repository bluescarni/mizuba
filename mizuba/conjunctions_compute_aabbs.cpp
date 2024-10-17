// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include <boost/align.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include "conjunctions.hpp"
#include "detail/atomic_minmax.hpp"
#include "detail/file_utils.hpp"
#include "detail/ival.hpp"
#include "detail/poly_utils.hpp"
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

// Helper to compute the AABB for a single object within a conjunction timestep.
//
// obj_idx is the object index in the polyjectory pj, cd_begin/end the begin/end times
// of the conjunction timestep, conj_thresh the conjunction threshold.
auto compute_object_aabb(const polyjectory &pj, std::size_t obj_idx, double cd_begin, double cd_end, double conj_thresh)
{
    // Cache the polynomial order.
    const auto order = pj.get_poly_order();

    // Init the return values.
    constexpr auto finf = std::numeric_limits<float>::infinity();
    std::array<float, 4> lb{{finf, finf, finf, finf}};
    std::array<float, 4> ub{{-finf, -finf, -finf, -finf}};

    // Fetch the traj and time spans from pj.
    const auto [traj_span, time_span, _] = pj[obj_idx];

    // Fetch the number of trajectory steps.
    const auto nsteps = time_span.extent(0);

    // Make sure that nsteps is representable as std::ptrdiff_t. This ensures
    // that we can safely calculate pointer subtractions in the time span data,
    // which allows us to determine the index of a trajectory timestep (see the code
    // below computing ss_idx).
    try {
        static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(nsteps));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in the trajectory data: the number of steps is too large");
    }
    // LCOV_EXCL_STOP

    // Compute the conjunction radius.
    const auto conj_radius = conj_thresh / 2;

    // Fetch begin/end iterators to the time span.
    const auto t_begin = time_span.data_handle();
    const auto t_end = t_begin + nsteps;

    // We need to locate the range in the trajectory data
    // that fully includes the current conjunction step.
    // First we locate the first trajectory step whose end is strictly
    // *greater* than the begin of the conjunction step.
    const auto ts_begin = std::upper_bound(t_begin, t_end, cd_begin);
    // Then, we locate the first trajectory step whose end is *greater than or
    // equal to* the end of the conjunction step.
    // NOTE: instead of this, perhaps we can just iterate below until
    // t_end or until the first trajectory step whose end is *greater than or
    // equal to* the end of the conjunction step, whichever comes first.
    auto ts_end = std::lower_bound(ts_begin, t_end, cd_end);
    // Bump it up by one to define a half-open range.
    // NOTE: don't bump it if it is already at the end.
    // This could happen for instance if an object does not
    // have trajectory data for the current conjunction step,
    // or if the trajectory data ends before the end of the
    // conjunction step.
    ts_end += (ts_end != t_end);

    // Iterate over all trajectory steps and update the bounding box
    // for the current object.
    // NOTE: if the object has no steps covering the current conjunction step,
    // then this loop will never be entered, and the AABB for the object
    // in the current conjunction step will remain inited with infinities. This is
    // fine as we will deal with infinite AABBs later on.
    for (auto it = ts_begin; it != ts_end; ++it) {
        // it points to the end time of a trajectory step which overlaps
        // with the current conjunction step. The polynomial evaluation
        // interval is the intersection between the trajectory step and
        // the conjunction step.

        // Determine the initial time coordinate of the trajectory step.
        // If it is t_begin, ss_start will be zero, otherwise
        // ss_start is given by the end time of the previous step.
        const auto ss_start = (it == t_begin) ? 0. : *(it - 1);

        // Determine absolute lower/upper bounds of the evaluation interval.
        // NOTE: min/max is fine here: values in time_span are always checked
        // for finiteness, cd_begin/end are also checked in
        // get_cd_begin_end().
        const auto ev_lb = std::max(cd_begin, ss_start);
        const auto ev_ub = std::min(cd_end, *it);

        // Create the actual evaluation interval, referring
        // it to the beginning of the trajectory step.
        const auto h_int_lb = ev_lb - ss_start;
        const auto h_int_ub = ev_ub - ss_start;

        // Determine the index of the trajectory step.
        // NOTE: we checked earlier in this function that it - t_begin
        // can be computed safely.
        const auto ss_idx = static_cast<std::size_t>(it - t_begin);

        // Compute the pointers to the polynomial coefficients for the
        // trajectory step.
        const auto *cf_ptr_x = &traj_span(ss_idx, 0, 0);
        const auto *cf_ptr_y = &traj_span(ss_idx, 1, 0);
        const auto *cf_ptr_z = &traj_span(ss_idx, 2, 0);
        const auto *cf_ptr_r = &traj_span(ss_idx, 6, 0);

        // Run the polynomial evaluations with interval arithmetics.
        // NOTE: jit for performance? If so, we can do all 4 coordinates
        // in a single JIT compiled function. Possibly also the update
        // with the conjunction radius? Note also that cfunc requires
        // input data stored in contiguous order, thus we would need
        // a 7-arguments cfunc which ignores arguments at indices 3,4,5.
        // NOTE: by using interval arithmetic here, we are producing intervals
        // in the form [a, b] (i.e., closed intervals), even if originally
        // the time intervals of the trajectory steps are meant to be half-open [a, b).
        // This is fine, as the end result is a slight enlargement of the aabb,
        // which is not problematic as the resulting aabb is still guaranteed
        // to contain the position of the object.
        const auto h_int = ival(h_int_lb, h_int_ub);
        std::array xyzr_int{detail::horner_eval(cf_ptr_x, order, h_int), detail::horner_eval(cf_ptr_y, order, h_int),
                            detail::horner_eval(cf_ptr_z, order, h_int), detail::horner_eval(cf_ptr_r, order, h_int)};

        // Adjust the intervals accounting for conjunction tracking.
        for (auto &val : xyzr_int) {
            val.lower -= conj_radius;
            val.upper += conj_radius;
        }

        // A couple of helpers to cast lower/upper bounds from double to float. After
        // the cast, we will also move slightly the bounds to add a safety margin to account
        // for possible truncation in the conversion.
        auto lb_make_float = [&](double lb) {
            auto ret = std::nextafter(static_cast<float>(lb), -finf);

            if (!std::isfinite(ret)) [[unlikely]] {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("The computation of the bounding box for the object at index "
                                                        "{} produced the non-finite lower bound {}",
                                                        obj_idx, ret));
                // LCOV_EXCL_STOP
            }

            return ret;
        };
        auto ub_make_float = [&](double ub) {
            auto ret = std::nextafter(static_cast<float>(ub), finf);

            if (!std::isfinite(ret)) [[unlikely]] {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("The computation of the bounding box for the object at index "
                                                        "{} produced the non-finite upper bound {}",
                                                        obj_idx, ret));
                // LCOV_EXCL_STOP
            }

            return ret;
        };

        // Update the bounding box for the current object.
        // NOTE: min/max is fine: the make_float() helpers check for finiteness,
        // and the other operand is never NaN.
        for (auto i = 0u; i < 4u; ++i) {
            lb[i] = std::min(lb[i], lb_make_float(xyzr_int[i].lower));
            ub[i] = std::max(ub[i], ub_make_float(xyzr_int[i].upper));
        }
    }

    return std::make_pair(lb, ub);
}

#if !defined(NDEBUG)

// Helper to validate the global aabbs computed during a conjunction step.
//
// nobjs is the total number of objects, base_ptr the pointer to the beginning of aabb data for
// the current conjunction step.
void validate_global_aabbs(auto nobjs, auto base_ptr)
{
    constexpr auto finf = std::numeric_limits<float>::infinity();

    std::array lb = {finf, finf, finf, finf};
    std::array ub = {-finf, -finf, -finf, -finf};

    for (decltype(nobjs) i = 0; i < nobjs; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            lb[j] = std::min(lb[j], base_ptr[i * 8u + j]);
            ub[j] = std::max(ub[j], base_ptr[i * 8u + 4u + j]);
        }
    }

    for (auto j = 0u; j < 4u; ++j) {
        assert(lb[j] == base_ptr[nobjs * 8u + j]);
        assert(ub[j] == base_ptr[nobjs * 8u + 4u + j]);
    }
}

#endif

} // namespace

} // namespace detail

// Function to compute the AABBs for all objects in all conjunction steps.
//
// pj is the polyjectory, tmp_dir_path the temporary dir storing all conjunction data,
// n_cd_steps the number of conjunction steps, conj_thresh the conjunction threshold,
// conj_det_interval the conjunction detection interval.
//
// NOTE: if an object has no trajectory data during a conjunction step, its AABB for
// that step will be set to infinities.
std::vector<double> conjunctions::compute_aabbs(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                                                std::size_t n_cd_steps, double conj_thresh,
                                                double conj_det_interval) const
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Cache the total number of objects.
    const auto nobjs = pj.get_nobjs();

    // Cache maxT.
    const auto maxT = pj.get_maxT();

    // The total number of aabbs we need to compute and store.
    // NOTE: the +1 is the global AABB to be computed for each conjunction step.
    const auto n_tot_aabbs = (safe_size_t(nobjs) + 1) * n_cd_steps;

    // The total required size in bytes.
    const auto tot_size = static_cast<std::size_t>(n_tot_aabbs * sizeof(float) * 8u);

    // Init the storage file.
    auto storage_path = tmp_dir_path / "aabbs";
    detail::create_sized_file(storage_path, tot_size);

    // Construct the vector to store the end times of the conjunction steps.
    std::vector<double> cd_end_times;
    cd_end_times.resize(boost::numeric_cast<decltype(cd_end_times.size())>(n_cd_steps));

    // Compute the AABBs in parallel over all the conjunction steps.
    // NOTE: consider inverting the parallel for loop nesting order here, or even using the 2d blocked range.
    // The rationale is that typically the polyjectory data will be much larger than the aabb data, and for
    // locality reasons it might be better to process the trajectory data first by object and then by
    // conjunction step.
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps),
        [this, maxT, conj_det_interval, n_cd_steps, nobjs, &pj, conj_thresh, &cd_end_times,
         &storage_path](const auto &cd_range) {
            for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
                // Memory map the storage file for the current conjunction step.
                const auto mmap_size = (nobjs + 1u) * 8u * sizeof(float);
                const auto mmap_offset = cd_idx * mmap_size;
                auto [map_ptr, st_file] = detail::mmap_at_offset_rw(storage_path, mmap_size, mmap_offset);
                auto *cd_base_ptr = reinterpret_cast<float *>(map_ptr);

                // Prepare the global AABB for the current conjunction step.
                constexpr auto finf = std::numeric_limits<float>::infinity();
                // NOTE: the global AABB needs to be updated atomically.
                std::array<std::atomic<float>, 4> cur_global_lb{{finf, finf, finf, finf}};
                std::array<std::atomic<float>, 4> cur_global_ub{{-finf, -finf, -finf, -finf}};

                // Fetch the begin/end times for the current conjunction step.
                const auto [cd_begin, cd_end] = get_cd_begin_end(maxT, cd_idx, conj_det_interval, n_cd_steps);

                // Iterate over all objects to determine their AABBs for the current conjunction step.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                    [cd_idx, cd_base_ptr, nobjs, &pj, cd_begin, cd_end, conj_thresh, &cur_global_lb,
                     &cur_global_ub](const auto &obj_range) {
                        // Init the local AABB for the current obj range.
                        std::array<float, 4> cur_local_lb{{finf, finf, finf, finf}};
                        std::array<float, 4> cur_local_ub{{-finf, -finf, -finf, -finf}};

                        for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                            // Compute the AABB for the current object.
                            const auto [lb, ub]
                                = detail::compute_object_aabb(pj, obj_idx, cd_begin, cd_end, conj_thresh);

                            // Update the local AABB.
                            // NOTE: min/max usage is safe, because compute_object_aabb()
                            // ensures that the bounding boxes are finite.
                            for (auto i = 0u; i < 4u; ++i) {
                                cur_local_lb[i] = std::min(cur_local_lb[i], lb[i]);
                                cur_local_ub[i] = std::max(cur_local_ub[i], ub[i]);
                            }

                            // Compute the pointer on disk to the AABB for the current conjunction step and object.
                            auto *aabb_ptr = cd_base_ptr + obj_idx * 8u;

                            // Write the AABB to disk.
                            std::ranges::copy(lb, aabb_ptr);
                            std::ranges::copy(ub, aabb_ptr + 4);
                        }

                        // Atomically update the global AABB for the current conjunction step.
                        // NOTE: atomic_min/max() usage here is safe because we checked that
                        // all lb/ub values are finite.
                        for (auto i = 0u; i < 4u; ++i) {
                            detail::atomic_min(cur_global_lb[i], cur_local_lb[i]);
                            detail::atomic_max(cur_global_ub[i], cur_local_ub[i]);
                        }
                    });

                // Write out the global AABB for the current conjunction step to disk.
                auto *aabb_ptr = cd_base_ptr + nobjs * 8u;
                for (auto i = 0u; i < 4u; ++i) {
                    const auto cur_lb = cur_global_lb[i].load(std::memory_order_relaxed);
                    const auto cur_ub = cur_global_ub[i].load(std::memory_order_relaxed);

                    // NOTE: this is ensured by the safety margins we added when converting
                    // the double-precision AABB to single-precision. That is, even if the original
                    // double-precision AABB has a size of zero in any dimension, the conversion
                    // to single precision resulted in a small but nonzero size in every dimension.
                    assert(cur_ub > cur_lb);

                    // Check that we can safely compute the difference between ub and lb. This is
                    // needed when computing morton codes.
                    // LCOV_EXCL_START
                    if (!std::isfinite(cur_ub - cur_lb)) [[unlikely]] {
                        throw std::invalid_argument("A global bounding box with non-finite size was generated");
                    }
                    // LCOV_EXCL_STOP

                    aabb_ptr[i] = cur_lb;
                    aabb_ptr[i + 4u] = cur_ub;
                }

                // Write cd_end into cd_end_times.
                cd_end_times[cd_idx] = cd_end;

#if !defined(NDEBUG)

                // Validate the global AABBs in debug mode.
                detail::validate_global_aabbs(nobjs, cd_base_ptr);

#endif
            }
        });

    // Mark the storage file as read-only.
    detail::mark_file_read_only(storage_path);

    // Return the end times of the conjunction steps.
    assert(cd_end_times.size() == n_cd_steps);
    return cd_end_times;
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
