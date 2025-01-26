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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>

#include <heyoka/mdspan.hpp>

#include "conjunctions.hpp"
#include "detail/conjunctions_jit.hpp"
#include "detail/poly_utils.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

namespace detail
{

namespace
{

// Function to transpose the positional polynomials stored in polys_i/j into the diff_input buffer.
//
// polys_i and polys_j contain the polynomial coefficients for the state vectors of i and j stored in
// row-major format (positions first, then velocities). diff_input is the output buffer.
// 'order' is the order of the polynomials.
void polys_ij_transpose_pos(const double *polys_i, const double *polys_j, auto &diff_input, std::uint32_t order)
{
    namespace hy = heyoka;

    assert(diff_input.size() == (order + 1u) * static_cast<decltype(diff_input.size())>(6));

    // Fetch spans over the positional polys.
    const auto pi_span
        = hy::mdspan<const double, hy::extents<std::size_t, 3, std::dynamic_extent>>(polys_i, order + 1u);
    const auto pj_span
        = hy::mdspan<const double, hy::extents<std::size_t, 3, std::dynamic_extent>>(polys_j, order + 1u);

    // Fetch a span over diff_input.
    const auto input_span
        = hy::mdspan<double, hy::extents<std::size_t, std::dynamic_extent, 6>>(diff_input.data(), order + 1u);

    // Transpose into input_span.
    for (auto i = 0u; i < 3u; ++i) {
        for (std::uint32_t j = 0; j <= order; ++j) {
            input_span(j, i) = pi_span(i, j);
            input_span(j, i + 3u) = pj_span(i, j);
        }
    }
}

} // namespace

} // namespace detail

// Narrow-phase conjunction detection.
//
// cd_idx is the index of the current conjunction step, pj the polyjectory, cd_bp_collisions
// the list of aabbs collisions detected in the broad phase, cjd the JIT-compiled data,
// conj_thresh the conjunction threshold, conj_det_interval the conjunction detection
// interval, n_cd_steps the total number of conjunction steps, np_rep the structure for
// logging statistics.
//
// The return value is the list of detect conjunctions.
std::vector<conjunctions::conj>
conjunctions::detect_conjunctions_narrow_phase(std::size_t cd_idx, const polyjectory &pj,
                                               const std::vector<aabb_collision> &cd_bp_collisions,
                                               const detail::conj_jit_data &cjd, double conj_thresh,
                                               double conj_det_interval, std::size_t n_cd_steps, np_report &np_rep)
{
    // Fetch the compiled functions.
    auto *pta6_cfunc = cjd.pta6_cfunc;
    auto *fex_check = cjd.fex_check;
    auto *rtscc = cjd.rtscc;
    auto *pt1 = cjd.pt1;
    auto *cs_enc_func = cjd.cs_enc_func;
    auto *dist2_interp = cjd.dist2_interp_func;

    // Cache the polynomial order.
    const auto order = pj.get_poly_order();

    // Local np_report for this conjunction step.
    np_report cd_np_rep;

    // Cache the square of conj_thresh.
    // NOTE: we checked in the conjunctions constructor that
    // this is safe to compute.
    const auto conj_thresh2 = conj_thresh * conj_thresh;

    // Fetch the begin/end times for the current conjunction step.
    const auto [cd_begin, cd_end] = get_cd_begin_end(pj.get_maxT(), cd_idx, conj_det_interval, n_cd_steps);

    // Initialise the vector to store the results of
    // narrow-phase conjunction detection for this conjunction step.
    //
    // NOTE: we used to use TBB's concurrent_vector here, however:
    //
    // https://github.com/oneapi-src/oneTBB/issues/1531
    //
    // Perhaps we can consider re-enabling it once fixed, but on the
    // other hand the mutex approach seems to perform well enough.
    std::vector<conj> conj_vector;
    std::mutex conj_vector_mutex;

    // We will be using thread-specific data to store temporary results during narrow-phase
    // conjunction detection.
    struct ets_data {
        // Local vector of detected conjunctions.
        std::vector<conj> conj_vec;
        // Polynomial cache for use during real root isolation.
        // NOTE: it is *really* important that this is declared
        // *before* wlist, because wlist will contain references
        // to and interact with r_iso_cache during destruction,
        // and we must be sure that wlist is destroyed *before*
        // r_iso_cache.
        detail::poly_cache r_iso_cache;
        // The working list.
        detail::wlist_t wlist;
        // The list of isolating intervals.
        detail::isol_t isol;
        // Buffer used as temporary storage for the results
        // of operations on polynomials.
        std::vector<double> pbuffer;
        // Buffer to store the input for the cfunc used to compute
        // the distance square polynomial.
        std::vector<double> diff_input;
        // The vector into which detected conjunctions are
        // temporarily written during polynomial root finding.
        // The tuple contains:
        // - the indices of the 2 objects,
        // - the time coordinate of the conjunction (relative
        //   to the time interval in which root finding is performed,
        //   i.e., **NOT** the absolute time in the polyjectory).
        std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> tmp_conj_vec;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([order]() {
        ets_data retval;

        // Prepare pbuffer.
        //
        // NOTE: here we need up to 14 polynomials. In order:
        //
        // - 6 + 6 polys for the translations of the state vectors
        //   of the two objects involved in the conjunction,
        // - 1 poly for the representation of the square of the distance
        //   between the two objects involved in the conjunction,
        // - 1 poly for the representation of the derivative of the square of the distance
        //   between the two objects involved in the conjunction.
        retval.pbuffer.resize(boost::safe_numerics::safe<decltype(retval.pbuffer.size())>(14) * (order + 1u));

        // Prepare diff_input.
        using safe_size_t = boost::safe_numerics::safe<decltype(retval.diff_input.size())>;
        retval.diff_input.resize((order + 1u) * safe_size_t(6));

        return retval;
    });

    // Iterate over the detected aabbs collisions for this conjunction step.
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<decltype(cd_bp_collisions.size())>(0, cd_bp_collisions.size()),
        [&ets, cd_begin, cd_end, &cd_bp_collisions, &pj, pta6_cfunc, fex_check, rtscc, pt1, cs_enc_func, dist2_interp,
         order, conj_thresh2, &conj_vector, &conj_vector_mutex, &cd_np_rep](const auto &bp_range) {
            // Fetch the thread-local data.
            // NOTE: no need to isolate here, as we are not
            // invoking any other TBB primitive from within this
            // scope.
            auto &[local_conj_vec, r_iso_cache, wlist, isol, pbuffer, diff_input, tmp_conj_vec] = ets.local();

            // Fetch the pointers to the temp polys in pbuffer.
            auto *pti_ptr = pbuffer.data();
            auto *ptj_ptr = pti_ptr + static_cast<std::size_t>(6) * (order + 1u);
            auto *ptd2_ptr = ptj_ptr + static_cast<std::size_t>(6) * (order + 1u);
            auto *ptd2p_ptr = ptd2_ptr + (order + 1u);

            // Prepare the local conjunction vector.
            local_conj_vec.clear();

            // Local logging stats.
            unsigned long long n_tot_conj_candidates = 0, n_dist2_check = 0, n_poly_roots = 0, n_fex_check = 0,
                               n_poly_no_roots = 0, n_tot_dist_minima = 0, n_tot_discarded_dist_minima = 0;

            for (auto bp_idx = bp_range.begin(); bp_idx != bp_range.end(); ++bp_idx) {
                const auto [i, j] = cd_bp_collisions[bp_idx];
                assert(i < j);

                // Fetch the trajectory data for i and j.
                const auto [traj_i, time_i, status_i] = pj[i];
                const auto [traj_j, time_j, status_j] = pj[j];

                // Fetch the total number of trajectory steps.
                const auto nsteps_i = traj_i.extent(0);
                const auto nsteps_j = traj_j.extent(0);
                assert(nsteps_i > 0u && nsteps_j > 0u);

                // Overflow checks: we must be sure we can represent the total number
                // of trajectory steps + 1 with std::ptrdiff_t, so that we can safely
                // perform subtractions between pointers to the time data (see code
                // below).
                try {
                    static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(nsteps_i + 1u));
                    static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(nsteps_j + 1u));
                    // LCOV_EXCL_START
                } catch (...) {
                    throw std::overflow_error(
                        "Overflow detected in the trajectory data: the number of steps is too large");
                }
                // LCOV_EXCL_STOP

                // Both trajectories must begin before the end of the conjunction step, and they
                // must end after the begin of the conjunction step.
                assert(time_i[0] < cd_end);
                assert(time_j[0] < cd_end);
                assert(time_i[nsteps_i] > cd_begin);
                assert(time_j[nsteps_j] > cd_begin);

                // Fetch begin/end iterators to the time spans.
                const auto t_begin_i = time_i.data_handle();
                const auto t_end_i = t_begin_i + (nsteps_i + 1u);
                const auto t_begin_j = time_j.data_handle();
                const auto t_end_j = t_begin_j + (nsteps_j + 1u);

                // Determine, for both objects, the range of trajectory steps
                // that temporally overlaps with the current conjunction step.
                // NOTE: same code as in compute_object_aabb().
                const auto ts_begin_i = std::upper_bound(t_begin_i + 1, t_end_i, cd_begin);
                auto ts_end_i = std::lower_bound(ts_begin_i, t_end_i, cd_end);
                ts_end_i += (ts_end_i != t_end_i);

                const auto ts_begin_j = std::upper_bound(t_begin_j + 1, t_end_j, cd_begin);
                auto ts_end_j = std::lower_bound(ts_begin_j, t_end_j, cd_end);
                ts_end_j += (ts_end_j != t_end_j);

#if !defined(NDEBUG)
                bool loop_entered = false;
#endif
                // Iterate until we get to the end of at least one range.
                // NOTE: if either range is empty, this loop is never entered.
                // This should never happen.
                for (auto it_i = ts_begin_i, it_j = ts_begin_j; it_i != ts_end_i && it_j != ts_end_j;) {
#if !defined(NDEBUG)
                    loop_entered = true;
#endif

                    // A small helper to update it_i and it_j for the next iteration of the for-loop.
                    // Normally this is invoked at the end of the loop, however it will be invoked earlier
                    // if the current i and j trajectory steps do not overlap.
                    const auto update_iterators = [&it_i, &it_j]() {
                        // NOTE: explanation on how we compute the iterator increments:
                        //
                        // 1) if *it_i == *it_j, then the trajectory steps end at the same time,
                        //    and we need to move to the next step for both objects (i.e., in the
                        //    current iteration of the for loop we processed the timeline up to
                        //    the same time for both objects). This can happen at the very end
                        //    of a polyjectory, or if both steps end exactly at the same time;
                        //
                        // 2) if *it_i < *it_j, then the trajectory step for object i ends before
                        //    the trajectory step for object j. We must move to the next step for
                        //    object i while remaining on the current step for object j;
                        //
                        // 3) if *it_j < *it_i, we have the specular case of 2.

                        // Compute the increments.
                        const auto inc_i = (*it_i <= *it_j);
                        const auto inc_j = (*it_j <= *it_i);

                        // Apply them.
                        it_i += inc_i;
                        it_j += inc_j;
                    };

                    // Initial time coordinates of the trajectory steps of i and j.
                    assert(it_i != t_begin_i);
                    assert(it_j != t_begin_j);
                    const auto ts_start_i = *(it_i - 1);
                    const auto ts_start_j = *(it_j - 1);

                    // Determine the intersections of the two trajectory steps
                    // with the current conjunction step.
                    // NOTE: min/max is fine here, all involved values are checked
                    // for finiteness.
                    const auto lb_i = std::max(cd_begin, ts_start_i);
                    const auto ub_i = std::min(cd_end, *it_i);
                    const auto lb_j = std::max(cd_begin, ts_start_j);
                    const auto ub_j = std::min(cd_end, *it_j);
                    // NOTE: these must hold because there must be some overlap between
                    // the trajectory steps and the conjunction step, otherwise a candidate
                    // conjunction wouldn't have been flagged.
                    assert(lb_i < ub_i);
                    assert(lb_j < ub_j);

                    // NOTE: if the intersection of [lb_i, ub_i) and [lb_j, ub_j) is empty,
                    // no conjunction is possible because there is no temporal overlap between
                    // the current trajectory steps of i and j. Just move to the next loop
                    // iteration.
                    if (!(ub_i > lb_j && lb_i < ub_j)) {
                        update_iterators();
                        continue;
                    }

                    // Update n_tot_conj_candidates. Do it here, after we have determined
                    // that there is an overlap between the trajectory steps of i and j.
                    ++n_tot_conj_candidates;

                    // Determine the intersection between the two intervals
                    // we just computed. This will be the time range
                    // within which we need to do polynomial root finding.
                    // NOTE: at this stage lb_rf/ub_rf are still absolute time coordinates
                    // within the entire polyjectory time range.
                    // NOTE: min/max is fine here, all quantities are safe.
                    const auto lb_rf = std::max(lb_i, lb_j);
                    const auto ub_rf = std::min(ub_i, ub_j);
                    assert(lb_rf < ub_rf);

                    // The trajectory polynomials for the two objects are time polynomials
                    // in which time is counted from the beginning of the trajectory step. In order to
                    // create the polynomial representing the distance square, we need first to
                    // translate the polynomials of both objects so that they refer to a
                    // common time coordinate, the time elapsed from lb_rf.

                    // Compute the translation amount for the two objects.
                    const auto delta_i = lb_rf - ts_start_i;
                    const auto delta_j = lb_rf - ts_start_j;

                    // Compute the time interval within which we will be performing root finding.
                    const auto rf_int = ub_rf - lb_rf;

                    // Do some checking before moving on.
                    if (!std::isfinite(delta_i) || !std::isfinite(delta_j) || !std::isfinite(rf_int) || delta_i < 0
                        || delta_j < 0 || rf_int < 0) [[unlikely]] {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(
                            fmt::format("During the narrow phase collision detection of objects {} and {}, "
                                        "an invalid time interval for polynomial root finding was generated",
                                        i, j));
                        // LCOV_EXCL_STOP
                    }

                    // Fetch pointers to the trajectory polynomials for the two objects.
                    // NOTE: we verified earlier we can safely compute differences between
                    // pointers to the time data.
                    const auto ts_idx_i = static_cast<std::size_t>(it_i - (t_begin_i + 1));
                    const auto ts_idx_j = static_cast<std::size_t>(it_j - (t_begin_j + 1));
                    const auto *polys_i = &traj_i(ts_idx_i, 0, 0);
                    const auto *polys_j = &traj_j(ts_idx_j, 0, 0);

                    // Perform the translations, if needed.
                    // NOTE: need to re-assign the polys_* pointers if the
                    // translation happens, otherwise we can keep the pointer
                    // to the original polynomials.
                    if (delta_i != 0) {
                        pta6_cfunc(pti_ptr, polys_i, &delta_i, nullptr);
                        polys_i = pti_ptr;
                    }

                    if (delta_j != 0) {
                        pta6_cfunc(ptj_ptr, polys_j, &delta_j, nullptr);
                        polys_j = ptj_ptr;
                    }

                    // Transpose polys_i and polys_j into diff_input.
                    detail::polys_ij_transpose_pos(polys_i, polys_j, diff_input, order);

                    // Compute the dist2 poly.
                    // NOTE: the distance square can assume rather large numerical
                    // values. If this ever becomes a problem for interpolation, we have
                    // the option of rescaling the interpolating values. The rescaling
                    // factor can be chosen as the maximum interpolating value.
                    dist2_interp(ptd2_ptr, diff_input.data(), &rf_int, nullptr);

                    // Compute an enclosure for the distance square.
                    std::array<double, 2> dist2_ieval{};
                    cs_enc_func(dist2_ieval.data(), ptd2_ptr, &rf_int, nullptr);

                    // NOTE: the computation of dist2_ieval will correctly propagate nans, which will then be caught
                    // here if present.
                    if (!std::isfinite(dist2_ieval[0]) || !std::isfinite(dist2_ieval[1])) [[unlikely]] {
                        // LCOV_EXCL_START
                        throw std::invalid_argument(fmt::format(
                            "Non-finite value(s) detected during conjunction tracking for objects {} and {}", i, j));
                        // LCOV_EXCL_STOP
                    }

                    if (dist2_ieval[0] < conj_thresh2) {
                        // The mutual distance between the objects might end up being
                        // less than the conjunction threshold during the current time interval.
                        // This means that a conjunction *may* happen.

                        // Compute the time derivative of the dist2 poly.
                        for (std::uint32_t k = 0; k < order; ++k) {
                            ptd2p_ptr[k] = (k + 1u) * ptd2_ptr[k + 1u];
                        }
                        // NOTE: the highest-order term needs to be set to zero manually.
                        ptd2p_ptr[order] = 0;

                        // Prepare tmp_conj_vec.
                        tmp_conj_vec.clear();

                        // Run polynomial root finding to detect conjunctions.
                        const auto fex_check_res = detail::run_poly_root_finding(
                            ptd2p_ptr, order, rf_int, isol, wlist, fex_check, rtscc, pt1, i, j,
                            // NOTE: positive direction to detect only distance minima.
                            1, tmp_conj_vec, r_iso_cache);

                        // Update n_poly_roots, n_fex_check, n_poly_no_roots and n_tot_dist_minima.
                        ++n_poly_roots;
                        n_fex_check += fex_check_res;
                        n_poly_no_roots += tmp_conj_vec.empty();
                        n_tot_dist_minima += static_cast<unsigned long long>(tmp_conj_vec.size());

                        // Helper to add to local_conj_vec a detected conjunction occurring at time conj_tm with
                        // conjunction distance square of conj_dist2.
                        const auto add_conjunction
                            = [order, &local_conj_vec, lb_rf, polys_i, polys_j, i, j](double conj_tm) {
                                  // Compute the state vector for the two objects.
                                  // NOTE: this is quite ugly, perhaps consider replacing it with a single JITted
                                  // implementation. If not for performance, at least for clarity.
                                  const std::array<double, 3> ri
                                      = {detail::horner_eval(polys_i, order, conj_tm),
                                         detail::horner_eval(polys_i + (order + 1u), order, conj_tm),
                                         detail::horner_eval(polys_i + static_cast<std::size_t>(2) * (order + 1u),
                                                             order, conj_tm)},
                                      vi = {detail::horner_eval(polys_i + static_cast<std::size_t>(3) * (order + 1u),
                                                                order, conj_tm),
                                            detail::horner_eval(polys_i + static_cast<std::size_t>(4) * (order + 1u),
                                                                order, conj_tm),
                                            detail::horner_eval(polys_i + static_cast<std::size_t>(5) * (order + 1u),
                                                                order, conj_tm)};

                                  const std::array<double, 3> rj
                                      = {detail::horner_eval(polys_j, order, conj_tm),
                                         detail::horner_eval(polys_j + (order + 1u), order, conj_tm),
                                         detail::horner_eval(polys_j + static_cast<std::size_t>(2) * (order + 1u),
                                                             order, conj_tm)},
                                      vj = {detail::horner_eval(polys_j + static_cast<std::size_t>(3) * (order + 1u),
                                                                order, conj_tm),
                                            detail::horner_eval(polys_j + static_cast<std::size_t>(4) * (order + 1u),
                                                                order, conj_tm),
                                            detail::horner_eval(polys_j + static_cast<std::size_t>(5) * (order + 1u),
                                                                order, conj_tm)};

                                  // Compute the conjunction distance.
                                  const auto diff_x = ri[0] - rj[0];
                                  const auto diff_y = ri[1] - rj[1];
                                  const auto diff_z = ri[2] - rj[2];
                                  const auto conj_dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

                                  local_conj_vec.emplace_back(i, j,
                                                              // NOTE: we want to store here the absolute
                                                              // time coordinate of the conjunction. conj_tm
                                                              // is a time coordinate relative to the root
                                                              // finding interval, so we need to transform it
                                                              // into an absolute time within the polyjectory.
                                                              lb_rf + conj_tm, conj_dist, ri, vi, rj, vj);
                              };

                        // For each detected conjunction, we need to:
                        // - verify that indeed the conjunction happens below
                        //   the threshold,
                        // - compute the conjunction distance and absolute
                        //   time coordinate.
                        for (const auto &[_1, _2, conj_tm] : tmp_conj_vec) {
                            assert(_1 == i);
                            assert(_2 == j);

                            // Compute the conjunction distance square.
                            const auto conj_dist2 = detail::horner_eval(ptd2_ptr, order, conj_tm);

                            if (!std::isfinite(conj_dist2)) [[unlikely]] {
                                // LCOV_EXCL_START
                                throw std::invalid_argument(
                                    fmt::format("A non-finite conjunction distance square of {} was computed "
                                                "for the objects at indices {} and {}",
                                                conj_dist2, i, j));
                                // LCOV_EXCL_STOP
                            }

                            if (conj_dist2 < conj_thresh2) {
                                // We detected an actual conjunction, add it.
                                add_conjunction(conj_tm);
                            } else {
                                // Update n_tot_discarded_dist_minima.
                                ++n_tot_discarded_dist_minima;
                            }
                        }

                        // NOTE: we now have to handle the special case in which we are at either
                        // the beginning or at the end of the trajectory time range for at least
                        // one of the objects. In this case, we may end up in a situation in which
                        // a conjunction is not detected because there is no minimum in the mutual
                        // distance between the objects - the minimum would occur outside the bounds
                        // of the time range, but we never get to detect it because we have no
                        // trajectory data outside the time bounds.
                        //
                        // An equivalent (and more mathy) way of seeing this is the following. The
                        // minima of the distance square function are given by the zeroes of the derivative
                        // **if** the domain is the entire real line (i.e., infinite time). But if the
                        // domain is restricted to a finite subrange of the real line, then we may
                        // have additional minima in correspondence of the subrange boundaries.

                        // We first handle the case in which we are at the end of trajectory data
                        // for at least one object.
                        //
                        // The it_i + 1 == t_end_i checks that we are at the last trajectory step,
                        // while *it_i == ub_rf checks that the root finding interval ends when the
                        // last trajectory step ends. The second check is needed because being in the
                        // last trajectory step does not necessarily mean that we are considering the
                        // *entire* step for root finding.
                        if ((it_i + 1 == t_end_i && *it_i == ub_rf) || (it_j + 1 == t_end_j && *it_j == ub_rf)) {
                            // We need to evaluate the derivative of the distance function at the
                            // end of the time range. If it is negative, it is a minimum and
                            // we may have another conjunction.

                            // NOTE: the trajectory time ranges are created as half-open intervals,
                            // thus we need to consider for a candidate minimum the time immediately
                            // preceding the end of the time range.
                            const auto min_cand_time = std::nextafter(rf_int, -1.);

                            // Evaluate the derivative of the distance square at min_cand_time.
                            // NOTE: ts_diff_der_ptr was set up previously.
                            const auto min_cand_dval = detail::horner_eval(ptd2p_ptr, order, min_cand_time);
                            if (!std::isfinite(min_cand_dval)) [[unlikely]] {
                                // LCOV_EXCL_START
                                throw std::invalid_argument(fmt::format(
                                    "An invalid value of {} was computed for the "
                                    "derivative of the distance square function at the upper boundary of a trajectory",
                                    min_cand_dval));
                                // LCOV_EXCL_STOP
                            }

                            // NOTE: check for strictly negative derivative. If the derivative had a zero here,
                            // we should have located it during polynomial root finding.
                            if (min_cand_dval < 0) {
                                // The distance square function has negative derivative at the end of the
                                // time range. Evaluate the distance square.
                                const auto min_cand_dist2 = detail::horner_eval(ptd2_ptr, order, min_cand_time);
                                if (!std::isfinite(min_cand_dist2)) [[unlikely]] {
                                    // LCOV_EXCL_START
                                    throw std::invalid_argument(fmt::format(
                                        "An invalid value of {} was computed for a "
                                        "candidate minimum of the distance square function at the upper boundary "
                                        "of a trajectory",
                                        min_cand_dist2));
                                    // LCOV_EXCL_STOP
                                }

                                // Check if the distance square is less than the threshold.
                                // If it is, add the conjunction.
                                if (min_cand_dist2 < conj_thresh2) {
                                    add_conjunction(min_cand_time);
                                }
                            }
                        }

                        // We now handle the case in which we are at the beginning of trajectory data
                        // for at least one object. The logic is similar to the previous case.
                        //
                        // The it_i == t_begin_i + 1 checks that we are at the first trajectory step,
                        // while lb_rf == *t_begin_i checks that the root finding interval begins when the
                        // first trajectory step begins. The second check is needed because being in the
                        // first trajectory step does not necessarily mean that we are considering the
                        // *entire* step in the root finding.
                        if ((it_i == t_begin_i + 1 && lb_rf == *t_begin_i)
                            || (it_j == t_begin_j + 1 && lb_rf == *t_begin_j)) {
                            // Calculate the value of the derivative of the distance square at the beginning
                            // of the root finding interval (which is always zero by definition).
                            const auto min_cand_dval = detail::horner_eval(ptd2p_ptr, order, 0.);
                            if (!std::isfinite(min_cand_dval)) [[unlikely]] {
                                // LCOV_EXCL_START
                                throw std::invalid_argument(fmt::format(
                                    "An invalid value of {} was computed for the "
                                    "derivative of the distance square function at the lower boundary of a trajectory",
                                    min_cand_dval));
                                // LCOV_EXCL_STOP
                            }

                            // The check is now for strictly *positive* derivative.
                            if (min_cand_dval > 0) {
                                const auto min_cand_dist2 = detail::horner_eval(ptd2_ptr, order, 0.);
                                if (!std::isfinite(min_cand_dist2)) [[unlikely]] {
                                    // LCOV_EXCL_START
                                    throw std::invalid_argument(fmt::format(
                                        "An invalid value of {} was computed for a "
                                        "candidate minimum of the distance square function at the lower boundary "
                                        "of a trajectory",
                                        min_cand_dist2));
                                    // LCOV_EXCL_STOP
                                }

                                if (min_cand_dist2 < conj_thresh2) {
                                    add_conjunction(0.);
                                }
                            }
                        }
                    } else {
                        // Update n_dist2_check.
                        ++n_dist2_check;
                    }

                    // Update it_i and it_j.
                    update_iterators();
                }

                assert(loop_entered);
            }

            // Atomically update cd_np_rep.
            cd_np_rep.n_tot_conj_candidates += n_tot_conj_candidates;
            cd_np_rep.n_dist2_check += n_dist2_check;
            cd_np_rep.n_poly_roots += n_poly_roots;
            cd_np_rep.n_fex_check += n_fex_check;
            cd_np_rep.n_poly_no_roots += n_poly_no_roots;
            cd_np_rep.n_tot_dist_minima += n_tot_dist_minima;
            cd_np_rep.n_tot_discarded_dist_minima += n_tot_discarded_dist_minima;

            // Atomically merge local_conj_vec into conj_vector.
            // NOTE: ensure we do this at the end of the scope in order to minimise
            // the locking time.
            std::lock_guard lock(conj_vector_mutex);
            conj_vector.insert(conj_vector.end(), local_conj_vec.begin(), local_conj_vec.end());
        });

    // Sort conj_vector according to tca. If the tcas are identical (this happens
    // often at the boundaries of the polyjectory), order lexicographically: this prevents
    // non-deterministic ordering of conjunctions with identical tca.
    //
    // NOTE: for documentation purposes, it is important to note that we are enforcing
    // lexicographic ordering only within the conjunctions vector produced in a conjunction
    // step. When this is merged into the global conj vector, we cannot guarantee lexicographic
    // ordering any more, and we could have, in principle, two neighbouring conjunctions with
    // equal tca that do not respect the lexicographic ordering. In other words, the only
    // guarantee on the final conjunctions vector is ordering by tca, with undetermined (but still
    // deterministic) ordering if the tcas are equal.
    oneapi::tbb::parallel_sort(conj_vector.begin(), conj_vector.end(), [](const auto &c1, const auto &c2) {
        if (c1.tca == c2.tca) {
            return c1 < c2;
        } else {
            return c1.tca < c2.tca;
        }
    });

    // Atomically update np_rep with the data in cd_np_rep.
    np_rep.n_tot_conj_candidates += cd_np_rep.n_tot_conj_candidates.load();
    np_rep.n_dist2_check += cd_np_rep.n_dist2_check.load();
    np_rep.n_poly_roots += cd_np_rep.n_poly_roots.load();
    np_rep.n_fex_check += cd_np_rep.n_fex_check.load();
    np_rep.n_poly_no_roots += cd_np_rep.n_poly_no_roots.load();
    np_rep.n_tot_dist_minima += cd_np_rep.n_tot_dist_minima.load();
    np_rep.n_tot_discarded_dist_minima += cd_np_rep.n_tot_discarded_dist_minima.load();

    return conj_vector;
}

} // namespace mizuba
