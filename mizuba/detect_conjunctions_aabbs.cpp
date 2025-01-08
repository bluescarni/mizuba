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
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <heyoka/mdspan.hpp>

#include "conjunctions.hpp"
#include "detail/atomic_minmax.hpp"
#include "detail/conjunctions_jit.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

namespace detail
{

namespace
{

// Helper to compute the AABB for a single object within a conjunction timestep.
//
// obj_idx is the object index in the polyjectory pj, cd_begin/end the begin/end times
// of the conjunction timestep, conj_thresh the conjunction threshold, cjd the data
// structure containing the JIT-compiled function used to compute the aabb of the object.
auto compute_object_aabb(const polyjectory &pj, std::size_t obj_idx, double cd_begin, double cd_end, double conj_thresh,
                         const detail::conj_jit_data &cjd)
{
    // Init the return values.
    const auto finf = std::numeric_limits<float>::infinity();
    std::array<float, 4> lb{{finf, finf, finf, finf}};
    std::array<float, 4> ub{{-finf, -finf, -finf, -finf}};

    // Fetch the traj and time spans from pj.
    const auto [traj_span, time_span, _] = pj[obj_idx];

    // Fetch the number of trajectory steps.
    const auto nsteps = traj_span.extent(0);
    assert((nsteps == 0u && time_span.extent(0) == 0u) || nsteps + 1u == time_span.extent(0));

    // Exit early if there are no trajectory steps. The AABB will be filled with infinities.
    // NOTE: we special case early so that we do not have to bother with complex logic in the
    // rest of the function.
    if (nsteps == 0u) {
        return std::make_pair(lb, ub);
    }

    // Also exit early if the trajectory does not begin strictly *before*
    // the end of the conjunction step, or does not end strictly *after*
    // the begin of the conjunction step. This means there is no overlap at all
    // between the trajectory and the conjunction step.
    if (!(time_span[0] < cd_end) || !(time_span[nsteps] > cd_begin)) {
        return std::make_pair(lb, ub);
    }

    // Fetch the compiled function for the computation of the aabb.
    auto *aabb_cs_cfunc = cjd.aabb_cs_cfunc;

    // Prepare the output and the parameters array for the compiled function.
    std::array<double, 8> xyzr_int{};
    std::array<double, 3> cs_pars{};

    // Make sure that nsteps + 1 (i.e., the number of time datapoints) is representable as std::ptrdiff_t.
    // This ensures that we can safely calculate pointer subtractions in the time span data,
    // which allows us to determine the index of a trajectory timestep (see the code
    // below computing ss_idx).
    try {
        static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(nsteps + 1u));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in the trajectory data: the number of steps is too large");
    }
    // LCOV_EXCL_STOP

    // Compute the conjunction radius.
    const auto conj_radius = conj_thresh / 2;

    // Fetch begin/end iterators to the time span.
    const auto t_begin = time_span.data_handle();
    const auto t_end = t_begin + (nsteps + 1u);

    // We need to locate the range in the trajectory data
    // that temporally overlaps with the current conjunction step.

    // First we locate the first trajectory step whose end is strictly
    // *greater* than the begin of the conjunction step. This will be
    // the chronologically first trajectory step that spans over the
    // conjunction step.
    const auto ts_begin = std::upper_bound(t_begin + 1, t_end, cd_begin);
    // Then, we locate the first trajectory step whose end is *greater than or
    // equal to* the end of the conjunction step. This will be
    // the chronologically last trajectory step that spans over the
    // conjunction step.
    // NOTE: instead of doing another binary search here, we could alternatively
    // start iterating from ts_begin and decide as-we-iterate when to stop.
    auto ts_end = std::lower_bound(ts_begin, t_end, cd_end);
    // Bump ts_end up by one in order to define a half-open iterator range.
    // NOTE: don't bump it if it is already at the end.
    // This happens if the trajectory data ends before the end of the
    // conjunction step.
    ts_end += (ts_end != t_end);

    // Iterate over all trajectory steps which overlap with the conjunction
    // step and update the bounding box for the current object.
    for (auto it = ts_begin; it != ts_end; ++it) {
        // it points to the end time of a trajectory step which overlaps
        // with the current conjunction step. The polynomial evaluation
        // interval is the intersection between the trajectory step and
        // the conjunction step.

        // Determine the initial time coordinate of the trajectory step.
        // This will be either the end of the previous step, or, if it is
        // t_begin + 1, the initial time of the (entire) trajectory.
        assert(it != t_begin);
        const auto ss_start = *(it - 1);

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
        // NOTE: we checked earlier in this function that it - (t_begin + 1)
        // can be computed safely.
        const auto ss_idx = static_cast<std::size_t>(it - (t_begin + 1));

        // Fetch a pointer to the polynomial coefficients for the
        // trajectory step.
        const auto *cf_ptr = &traj_span(ss_idx, 0, 0);

        // Prepare the parameters array for the invocation of the compiled function.
        cs_pars[0] = h_int_lb;
        cs_pars[1] = h_int_ub;
        cs_pars[2] = conj_radius;

        // Compute the aabb via the Cargo-Shisha algorithm.
        // NOTE: by using the Cargo-Shisha algorithm here, we are producing intervals
        // in the form [a, b] (i.e., closed intervals), even if originally
        // the time intervals of the trajectory steps are meant to be half-open [a, b).
        // This is fine, as the end result is a slight enlargement of the aabb,
        // which is not problematic as the resulting aabb is still guaranteed
        // to contain the position of the object.
        aabb_cs_cfunc(xyzr_int.data(), cf_ptr, cs_pars.data(), nullptr);

        // A couple of helpers to cast lower/upper bounds from double to float. After
        // the cast, we will also move slightly the bounds to add a safety margin to account
        // for possible truncation in the conversion.
        auto lb_make_float = [finf, obj_idx](double lb) {
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
        auto ub_make_float = [finf, obj_idx](double ub) {
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
            lb[i] = std::min(lb[i], lb_make_float(xyzr_int[i * 2u]));
            ub[i] = std::max(ub[i], ub_make_float(xyzr_int[i * 2u + 1u]));
        }
    }

    // NOTE: lb/ub must all be finite as we made sure early on that there is an overlap between the
    // trajectory and the conjunction step, and in the loop we checked for finiteness.
    assert(std::ranges::all_of(lb, [](auto x) { return std::isfinite(x); }));
    assert(std::ranges::all_of(ub, [](auto x) { return std::isfinite(x); }));

    return std::make_pair(lb, ub);
}

#if !defined(NDEBUG)

// Helper to validate the global aabbs computed during a conjunction step.
//
// cd_aabbs_span is the span containing the aabbs for all objects (plus the global aabb)
// for the conjunction step.
void validate_global_aabbs(auto cd_aabbs_span)
{
    constexpr auto finf = std::numeric_limits<float>::infinity();

    std::array lb = {finf, finf, finf, finf};
    std::array ub = {-finf, -finf, -finf, -finf};

    assert(cd_aabbs_span.extent(0) > 0u);
    const auto nobjs = cd_aabbs_span.extent(0) - 1u;

    for (decltype(cd_aabbs_span.extent(0)) i = 0; i < nobjs; ++i) {
        for (auto j = 0u; j < 4u; ++j) {
            lb[j] = std::min(lb[j], cd_aabbs_span(i, 0, j));
            ub[j] = std::max(ub[j], cd_aabbs_span(i, 1, j));
        }
    }

    // Check the global aabb.
    for (auto j = 0u; j < 4u; ++j) {
        assert(lb[j] == cd_aabbs_span(nobjs, 0, j));
        assert(ub[j] == cd_aabbs_span(nobjs, 1, j));
    }
}

#endif

} // namespace

} // namespace detail

// Compute the aabbs for all objects in a conjunction step.
//
// cd_idx is the index of the conjunction step, cd_aabbs the data buffer into which
// the aabbs will be written, pj the polyjectory, conj_thresh the conjunction threshold,
// conj_det_interval the conjunction detection interval, n_cd_steps the total number of
// conjunction steps, cd_end_times the vector of end times for the conjunction steps, cjd the data
// structure containing the JIT-compiled function used to compute the aabbs of the objects.
void conjunctions::detect_conjunctions_aabbs(std::size_t cd_idx, std::vector<float> &cd_aabbs, const polyjectory &pj,
                                             double conj_thresh, double conj_det_interval, std::size_t n_cd_steps,
                                             std::vector<double> &cd_end_times, const detail::conj_jit_data &cjd)
{
    assert(cd_idx < cd_end_times.size());

    // Cache the total number of objects.
    const auto nobjs = pj.get_nobjs();
    assert(cd_aabbs.size() == (nobjs + 1u) * 8u);

    // Cache maxT.
    const auto maxT = pj.get_maxT();

    // Prepare the global AABB for the current conjunction step.
    constexpr auto finf = std::numeric_limits<float>::infinity();
    // NOTE: the global AABB needs to be updated atomically.
    std::array<std::atomic<float>, 4> cur_global_lb{{finf, finf, finf, finf}};
    std::array<std::atomic<float>, 4> cur_global_ub{{-finf, -finf, -finf, -finf}};

    // Fetch the begin/end times for the current conjunction step.
    const auto [cd_begin, cd_end] = get_cd_begin_end(maxT, cd_idx, conj_det_interval, n_cd_steps);

    // Create a mutable span into cd_aabbs.
    using mut_aabbs_span_t = heyoka::mdspan<float, heyoka::extents<std::size_t, std::dynamic_extent, 2, 4>>;
    const mut_aabbs_span_t cd_aabbs_span{cd_aabbs.data(), nobjs + 1u};

    // Iterate over all objects to determine their AABBs for the current conjunction step.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                              [&pj, cd_begin, cd_end, conj_thresh, &cur_global_lb, &cur_global_ub, cd_aabbs_span,
                               &cjd](const auto &obj_range) {
                                  // Init the local AABB for the current obj range.
                                  std::array<float, 4> cur_local_lb{{finf, finf, finf, finf}};
                                  std::array<float, 4> cur_local_ub{{-finf, -finf, -finf, -finf}};

                                  for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                                      // Compute the AABB for the current object.
                                      const auto [lb, ub] = detail::compute_object_aabb(pj, obj_idx, cd_begin, cd_end,
                                                                                        conj_thresh, cjd);

                                      // Update the local AABB and write to cd_aabbs_span the aabb of the
                                      // current object.
                                      // NOTE: min/max usage is safe, because compute_object_aabb()
                                      // either ensures that the bounding boxes are finite, or it leaves
                                      // them inited with infinities.
                                      for (auto i = 0u; i < 4u; ++i) {
                                          cur_local_lb[i] = std::min(cur_local_lb[i], lb[i]);
                                          cur_local_ub[i] = std::max(cur_local_ub[i], ub[i]);
                                          cd_aabbs_span(obj_idx, 0, i) = lb[i];
                                          cd_aabbs_span(obj_idx, 1, i) = ub[i];
                                      }
                                  }

                                  // Atomically update the global AABB for the current conjunction step.
                                  // NOTE: atomic_min/max() usage here is safe because we either checked that
                                  // all lb/ub values are finite, or we left all lb/ub values in their default
                                  // infinity state.
                                  for (auto i = 0u; i < 4u; ++i) {
                                      detail::atomic_min(cur_global_lb[i], cur_local_lb[i]);
                                      detail::atomic_max(cur_global_ub[i], cur_local_ub[i]);
                                  }
                              });

    // Write out the global AABB for the current conjunction step to cd_aabbs_span.
    for (auto i = 0u; i < 4u; ++i) {
        const auto cur_lb = cur_global_lb[i].load();
        const auto cur_ub = cur_global_ub[i].load();

        // NOTE: run these checks only on finite global AABBs.
        if (std::isfinite(cur_lb)) {
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
        }

        cd_aabbs_span(nobjs, 0, i) = cur_lb;
        cd_aabbs_span(nobjs, 1, i) = cur_ub;
    }

#if !defined(NDEBUG)

    // Validate the global AABBs in debug mode.
    detail::validate_global_aabbs(cd_aabbs_span);

#endif

    // Write cd_end into cd_end_times.
    cd_end_times[cd_idx] = cd_end;
}

} // namespace mizuba
