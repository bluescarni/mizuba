// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <boost/align.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>

#include "conjunctions.hpp"
#include "detail/conjunctions_jit.hpp"
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

void conjunctions::narrow_phase(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                                std::size_t n_cd_steps,
                                const std::vector<std::tuple<std::size_t, std::size_t>> &bp_offsets,
                                const std::vector<double> &cd_end_times, const detail::conj_jit_data &cjd,
                                double conj_thresh)
{
    // Cache the polynomial order.
    const auto order = pj.get_poly_order();

    // Fetch a pointer to the bp data.
    boost::iostreams::mapped_file_source file_bp((tmp_dir_path / "bp").string());
    const auto *bp_base_ptr = reinterpret_cast<const aabb_collision *>(file_bp.data());
    assert(boost::alignment::is_aligned(bp_base_ptr, alignof(aabb_collision)));

    // Fetch the compiled functions.
    auto *pta_cfunc = cjd.pta_cfunc;
    auto *pssdiff3_cfunc = cjd.pssdiff3_cfunc;
    auto *fex_check = cjd.fex_check;
    auto *rtscc = cjd.rtscc;
    auto *pt1 = cjd.pt1;

    // Cache the square of conj_thresh.
    // NOTE: we checked in the conjunctions constructor that
    // this is safe to compute.
    const auto conj_thresh2 = conj_thresh * conj_thresh;

    // The global vector of conjunctions.
    oneapi::tbb::concurrent_vector<conj> global_conj_vec_c;

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
        // Buffers used as temporary storage for the results
        // of operations on polynomials.
        // NOTE: if we restructure the code to use JIT more,
        // we should probably re-implement this as a flat
        // 1D buffer rather than a collection of vectors.
        std::array<std::vector<double>, 14> pbuffers;
        // Vector to store the input for the cfunc used to compute
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

        // Prepare pbuffers.
        for (auto &v : retval.pbuffers) {
            v.resize(boost::numeric_cast<decltype(v.size())>(order + 1u));
        }

        // Prepare diff_input.
        using safe_size_t = boost::safe_numerics::safe<decltype(retval.diff_input.size())>;
        retval.diff_input.resize((order + 1u) * safe_size_t(6));

        return retval;
    });

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps), [&pj, order, &ets, &bp_offsets,
                                                                                       bp_base_ptr, &cd_end_times,
                                                                                       pta_cfunc, pssdiff3_cfunc,
                                                                                       conj_thresh2, fex_check, rtscc,
                                                                                       pt1, &global_conj_vec_c](
                                                                                          const auto &cd_range) {
        // Conjunctions vector specific for a conjunction step.
        oneapi::tbb::concurrent_vector<conj> cd_conj_vec;

        for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
            // Reset cd_conj_vec.
            cd_conj_vec.clear();

            // Fetch the broad-phase data for the current conjunction step.
            const auto [bp_offset, bp_size] = bp_offsets[cd_idx];
            const auto *bp_ptr = bp_base_ptr + bp_offset;
            const aabb_collision_span_t bpc{bp_ptr, bp_size};

            // Establish the begin/end times of the conjunction step.
            assert(cd_idx < cd_end_times.size());
            const auto cd_begin = (cd_idx == 0u) ? 0. : cd_end_times[cd_idx - 1u];
            const auto cd_end = cd_end_times[cd_idx];

            // Iterate over all the broad-phase aabb collisions.
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<std::size_t>(0, bpc.extent(0)),
                [&pj, bpc, order, &ets, cd_begin, cd_end, pta_cfunc, pssdiff3_cfunc, conj_thresh2, fex_check, rtscc,
                 pt1, &cd_conj_vec](const auto &bp_range) {
                    // Fetch the thread-local data.
                    // NOTE: no need to isolate here, as we are not
                    // invoking any other TBB primitive from within this
                    // scope.
                    auto &[local_conj_vec, r_iso_cache, wlist, isol, pbuffers, diff_input, tmp_conj_vec] = ets.local();
                    auto &[xi_temp, yi_temp, zi_temp, xj_temp, yj_temp, zj_temp, ts_diff, ts_diff_der, vxi_temp,
                           vyi_temp, vzi_temp, vxj_temp, vyj_temp, vzj_temp]
                        = pbuffers;

                    // Prepare the local conjunction vector.
                    local_conj_vec.clear();

                    for (auto bp_idx = bp_range.begin(); bp_idx != bp_range.end(); ++bp_idx) {
                        const auto [i, j] = bpc(bp_idx);

                        assert(i < j);

                        // Fetch the trajectory data for i and j.
                        const auto [traj_i, time_i, status_i] = pj[i];
                        const auto [traj_j, time_j, status_j] = pj[j];

                        // Overflow checks: we must be sure we can represent the total number
                        // of trajectory steps with std::ptrdiff_t, so that we can safely
                        // perform subtractions between pointers to the time data (see code
                        // below).
                        try {
                            static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(time_i.extent(0)));
                            static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(time_j.extent(0)));
                            // LCOV_EXCL_START
                        } catch (...) {
                            throw std::overflow_error(
                                "Overflow detected in the trajectory data: the number of steps is too large");
                        }
                        // LCOV_EXCL_STOP

                        // Fetch the total number of trajectory steps.
                        const auto nsteps_i = time_i.extent(0);
                        const auto nsteps_j = time_j.extent(0);

                        // Fetch begin/end iterators to the time spans.
                        const auto t_begin_i = time_i.data_handle();
                        const auto t_end_i = t_begin_i + nsteps_i;
                        const auto t_begin_j = time_j.data_handle();
                        const auto t_end_j = t_begin_j + nsteps_j;

                        // Determine, for both objects, the range of trajectory steps
                        // that fully includes the current conjunction step.
                        // NOTE: same code as in compute_object_aabb().
                        const auto ts_begin_i = std::upper_bound(t_begin_i, t_end_i, cd_begin);
                        auto ts_end_i = std::lower_bound(ts_begin_i, t_end_i, cd_end);
                        ts_end_i += (ts_end_i != t_end_i);

                        const auto ts_begin_j = std::upper_bound(t_begin_j, t_end_j, cd_begin);
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

                            // Initial time coordinates of the trajectory steps of i and j.
                            const auto ts_start_i = (it_i == t_begin_i) ? 0. : *(it_i - 1);
                            const auto ts_start_j = (it_j == t_begin_j) ? 0. : *(it_j - 1);

                            // Determine the intersections of the two trajectory steps
                            // with the current conjunction step.
                            // NOTE: min/max is fine here, all involved values are checked
                            // for finiteness.
                            const auto lb_i = std::max(cd_begin, ts_start_i);
                            const auto ub_i = std::min(cd_end, *it_i);
                            const auto lb_j = std::max(cd_begin, ts_start_j);
                            const auto ub_j = std::min(cd_end, *it_j);

                            // Determine the intersection between the two intervals
                            // we just computed. This will be the time range
                            // within which we need to do polynomial root finding.
                            // NOTE: at this stage lb_rf/ub_rf are still absolute time coordinates
                            // within the entire polyjectory time range.
                            // NOTE: min/max fine here, all quantities are safe.
                            const auto lb_rf = std::max(lb_i, lb_j);
                            const auto ub_rf = std::min(ub_i, ub_j);

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
                            if (!std::isfinite(delta_i) || !std::isfinite(delta_j) || !std::isfinite(rf_int)
                                || delta_i < 0 || delta_j < 0 || rf_int < 0) [[unlikely]] {
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
                            const auto ts_idx_i = static_cast<std::size_t>(it_i - t_begin_i);
                            const auto ts_idx_j = static_cast<std::size_t>(it_j - t_begin_j);

                            const auto *poly_xi = &traj_i(ts_idx_i, 0, 0);
                            const auto *poly_yi = &traj_i(ts_idx_i, 1, 0);
                            const auto *poly_zi = &traj_i(ts_idx_i, 2, 0);
                            const auto *poly_vxi = &traj_i(ts_idx_i, 3, 0);
                            const auto *poly_vyi = &traj_i(ts_idx_i, 4, 0);
                            const auto *poly_vzi = &traj_i(ts_idx_i, 5, 0);

                            const auto *poly_xj = &traj_j(ts_idx_j, 0, 0);
                            const auto *poly_yj = &traj_j(ts_idx_j, 1, 0);
                            const auto *poly_zj = &traj_j(ts_idx_j, 2, 0);
                            const auto *poly_vxj = &traj_j(ts_idx_j, 3, 0);
                            const auto *poly_vyj = &traj_j(ts_idx_j, 4, 0);
                            const auto *poly_vzj = &traj_j(ts_idx_j, 5, 0);

                            // Perform the translations, if needed.
                            // NOTE: perhaps we can write a dedicated function
                            // that does the translation for all 3 coordinates/velocities
                            // at once, for better performance?
                            // NOTE: need to re-assign the poly_*i pointers if the
                            // translation happens, otherwise we can keep the pointer
                            // to the original polynomials.
                            if (delta_i != 0) {
                                pta_cfunc(xi_temp.data(), poly_xi, &delta_i, nullptr);
                                poly_xi = xi_temp.data();
                                pta_cfunc(yi_temp.data(), poly_yi, &delta_i, nullptr);
                                poly_yi = yi_temp.data();
                                pta_cfunc(zi_temp.data(), poly_zi, &delta_i, nullptr);
                                poly_zi = zi_temp.data();

                                pta_cfunc(vxi_temp.data(), poly_vxi, &delta_i, nullptr);
                                poly_vxi = vxi_temp.data();
                                pta_cfunc(vyi_temp.data(), poly_vyi, &delta_i, nullptr);
                                poly_vyi = vyi_temp.data();
                                pta_cfunc(vzi_temp.data(), poly_vzi, &delta_i, nullptr);
                                poly_vzi = vzi_temp.data();
                            }

                            if (delta_j != 0) {
                                pta_cfunc(xj_temp.data(), poly_xj, &delta_j, nullptr);
                                poly_xj = xj_temp.data();
                                pta_cfunc(yj_temp.data(), poly_yj, &delta_j, nullptr);
                                poly_yj = yj_temp.data();
                                pta_cfunc(zj_temp.data(), poly_zj, &delta_j, nullptr);
                                poly_zj = zj_temp.data();

                                pta_cfunc(vxj_temp.data(), poly_vxj, &delta_j, nullptr);
                                poly_vxj = vxj_temp.data();
                                pta_cfunc(vyj_temp.data(), poly_vyj, &delta_j, nullptr);
                                poly_vyj = vyj_temp.data();
                                pta_cfunc(vzj_temp.data(), poly_vzj, &delta_j, nullptr);
                                poly_vzj = vzj_temp.data();
                            }

                            // Copy over the data to diff_input.
                            using di_size_t = decltype(diff_input.size());
                            std::copy(poly_xi, poly_xi + (order + 1u), diff_input.data());
                            std::copy(poly_yi, poly_yi + (order + 1u), diff_input.data() + (order + 1u));
                            std::copy(poly_zi, poly_zi + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(2) * (order + 1u));
                            std::copy(poly_xj, poly_xj + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(3) * (order + 1u));
                            std::copy(poly_yj, poly_yj + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(4) * (order + 1u));
                            std::copy(poly_zj, poly_zj + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(5) * (order + 1u));

                            // We can now construct the polynomial for the
                            // square of the distance.
                            auto *ts_diff_ptr = ts_diff.data();
                            pssdiff3_cfunc(ts_diff_ptr, diff_input.data(), nullptr, nullptr);

                            // Evaluate the distance square in the [0, rf_int) interval.
                            const auto dist2_ieval = detail::horner_eval(ts_diff_ptr, order, detail::ival(0, rf_int));

                            if (!std::isfinite(dist2_ieval.lower) || !std::isfinite(dist2_ieval.upper)) [[unlikely]] {
                                // LCOV_EXCL_START
                                throw std::invalid_argument(fmt::format(
                                    "Non-finite value(s) detected during conjunction tracking for objects {} and {}", i,
                                    j));
                                // LCOV_EXCL_STOP
                            }

                            if (dist2_ieval.lower < conj_thresh2) {
                                // The mutual distance between the objects might end up being
                                // less than the conjunction threshold during the current time interval.
                                // This means that a conjunction *may* happen.

                                // Compute the time derivative of the dist2 poly in-place.
                                auto *ts_diff_der_ptr = ts_diff_der.data();
                                for (std::uint32_t k = 0; k < order; ++k) {
                                    ts_diff_der_ptr[k] = (k + 1u) * ts_diff_ptr[k + 1u];
                                }
                                // NOTE: the highest-order term needs to be set to zero manually.
                                ts_diff_der_ptr[order] = 0;

                                // Prepare tmp_conj_vec.
                                tmp_conj_vec.clear();

                                // Run polynomial root finding to detect conjunctions.
                                detail::run_poly_root_finding(
                                    ts_diff_der_ptr, order, rf_int, isol, wlist, fex_check, rtscc, pt1, i, j,
                                    // NOTE: positive direction to detect only distance minima.
                                    1, tmp_conj_vec, r_iso_cache);

                                // For each detected conjunction, we need to:
                                // - verify that indeed the conjunction happens below
                                //   the threshold,
                                // - compute the conjunction distance and absolute
                                //   time coordinate.
                                for (const auto &[_1, _2, conj_tm] : tmp_conj_vec) {
                                    assert(_1 == i);
                                    assert(_2 == j);

                                    // Compute the conjunction distance square.
                                    const auto conj_dist2 = detail::horner_eval(ts_diff_ptr, order, conj_tm);

                                    if (!std::isfinite(conj_dist2)) [[unlikely]] {
                                        // LCOV_EXCL_START
                                        throw std::invalid_argument(
                                            fmt::format("A non-finite conjunction distance square of {} was computed "
                                                        "for the objects at indices {} and {}",
                                                        conj_dist2, i, j));
                                        // LCOV_EXCL_STOP
                                    }

                                    if (conj_dist2 < conj_thresh2) {
                                        // Compute the state vector for the two objects.
                                        const std::array<double, 3> ri = {detail::horner_eval(poly_xi, order, conj_tm),
                                                                          detail::horner_eval(poly_yi, order, conj_tm),
                                                                          detail::horner_eval(poly_zi, order, conj_tm)},
                                                                    vi
                                                                    = {detail::horner_eval(poly_vxi, order, conj_tm),
                                                                       detail::horner_eval(poly_vyi, order, conj_tm),
                                                                       detail::horner_eval(poly_vzi, order, conj_tm)};

                                        const std::array<double, 3> rj = {detail::horner_eval(poly_xj, order, conj_tm),
                                                                          detail::horner_eval(poly_yj, order, conj_tm),
                                                                          detail::horner_eval(poly_zj, order, conj_tm)},
                                                                    vj
                                                                    = {detail::horner_eval(poly_vxj, order, conj_tm),
                                                                       detail::horner_eval(poly_vyj, order, conj_tm),
                                                                       detail::horner_eval(poly_vzj, order, conj_tm)};

                                        local_conj_vec.emplace_back(
                                            // NOTE: we want to store here the absolute
                                            // time coordinate of the conjunction. conj_tm
                                            // is a time coordinate relative to the root
                                            // finding interval, so we need to transform it
                                            // into an absolute time within the polyjectory.
                                            lb_rf + conj_tm,
                                            // NOTE: conj_dist2 is finite but it could still
                                            // be negative due to floating-point rounding
                                            // (e.g., zero-distance conjunctions). Ensure
                                            // we do not produce NaN here.
                                            std::sqrt(std::max(conj_dist2, 0.)), i, j, ri, vi, rj, vj);
                                    }
                                }
                            }

                            // Update it_i and it_j.
                            if (*it_i < *it_j) {
                                // The trajectory step for object i ends
                                // before the trajectory step for object j.
                                ++it_i;
                            } else if (*it_j < *it_i) {
                                // The trajectory step for object j ends
                                // before the trajectory step for object i.
                                ++it_j;
                            } else {
                                // Both trajectory steps end at the same time.
                                // This can happen at the very end of a polyjectory,
                                // or if both steps end exactly at the same time.
                                ++it_i;
                                ++it_j;
                            }
                        }

                        assert(loop_entered);
                    }

                    // Merge the local conjunction vector into cd_conj_vec.
                    cd_conj_vec.grow_by(local_conj_vec.begin(), local_conj_vec.end());
                });

            // Merge cd_conj_vec into global_conj_vec_c.
            global_conj_vec_c.grow_by(cd_conj_vec.begin(), cd_conj_vec.end());
        }
    });

    // Make a copy of global_conj_vec_c for faster sorting and dumping.
    std::vector global_conj_vec(global_conj_vec_c.begin(), global_conj_vec_c.end());

    // Sort the global vector of conjunctions according to TCA.
    oneapi::tbb::parallel_sort(global_conj_vec.begin(), global_conj_vec.end(),
                               [](const auto &c1, const auto &c2) { return c1.tca < c2.tca; });

    // Dump the conjunctions to file.

    // Compute the total required size.
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;
    const auto tot_size = safe_size_t(global_conj_vec.size()) * sizeof(conj);

    // Path to the conjunctions file.
    const auto conj_file_path = tmp_dir_path / "conjunctions";
    // LCOV_EXCL_START
    if (boost::filesystem::exists(conj_file_path)) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("Cannot create the storage file '{}', as it exists already", conj_file_path.string()));
    }
    // LCOV_EXCL_STOP

    // Create the file.
    std::ofstream file(conj_file_path.string(), std::ios::binary | std::ios::out);
    // Make sure we throw on errors.
    file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    // NOTE: if we have no conjunctions, we still want to write something to disk because
    // otherwise we cannot memory-map an empty file.
    if (tot_size > 0u) {
        file.write(reinterpret_cast<const char *>(global_conj_vec.data()), tot_size);
    } else {
        // NOTE: use a single-byte file to signal the lack of conjunctions.
        // Make super extra sure this cannot me mistaken for a single conjunction.
        static_assert(sizeof(conj) > 1u);

        const char empty{};
        file.write(&empty, 1);
    }

    // Close the file and mark it as read-only.
    file.close();
    detail::mark_file_read_only(conj_file_path);
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
