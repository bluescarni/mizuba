// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <tuple>
#include <vector>

#include <boost/align.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

#include "conjunctions.hpp"
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
                                const std::vector<double> &cd_end_times)
{
    using detail::poly_cache;
    using detail::pwrap;

    // Cache the polynomial order.
    const auto order = pj.get_poly_order();

    // Fetch a pointer to the bp data.
    boost::iostreams::mapped_file_source file_bp((tmp_dir_path / "bp").string());
    const auto *bp_base_ptr = reinterpret_cast<const aabb_collision *>(file_bp.data());
    assert(boost::alignment::is_aligned(bp_base_ptr, alignof(aabb_collision)));

    // We will be using thread-specific data to store temporary results during narrow-phase
    // conjunction detection.
    struct ets_data {
        // The working list type used during real root isolation.
        using wlist_t = std::vector<std::tuple<double, double, pwrap>>;
        // The type used to store the list of isolating intervals.
        using isol_t = std::vector<std::tuple<double, double>>;

        // Local vector of detected conjunctions.
        std::vector<conj> conj_vec;
        // Polynomial cache for use during real root isolation.
        // NOTE: it is *really* important that this is declared
        // *before* wlist, because wlist will contain references
        // to and interact with r_iso_cache during destruction,
        // and we must be sure that wlist is destroyed *before*
        // r_iso_cache.
        poly_cache r_iso_cache;
        // The working list.
        wlist_t wlist;
        // The list of isolating intervals.
        isol_t isol;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([]() { return ets_data{}; });

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps),
        [&pj, order, &ets, &bp_offsets, bp_base_ptr, &cd_end_times](const auto &cd_range) {
            for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
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
                    [&pj, bpc, order, &ets, cd_begin, cd_end](const auto &bp_range) {
                        // Fetch the thread-local data.
                        // NOTE: no need to isolate here, as we are not
                        // invoking any other TBB primitive from within this
                        // scope.
                        auto &[local_conj_vec, r_iso_cache, wlist, isol] = ets.local();

                        // Prepare the local conjunction vector.
                        local_conj_vec.clear();

                        // Temporary polynomials used in the bisection loop.
                        pwrap tmp1(r_iso_cache, order), tmp2(r_iso_cache, order), tmp(r_iso_cache, order);

                        for (auto bp_idx = bp_range.begin(); bp_idx != bp_range.end(); ++bp_idx) {
                            const auto [i, j] = bpc(bp_idx);

                            assert(i < j);

                            // Fetch the trajectory data for i and j.
                            const auto [traj_i, time_i, status_i] = pj[i];
                            const auto [traj_j, time_j, status_j] = pj[j];

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

                                // Update it_i and it_j.
                                if (*it_i < *it_j) {
                                    // The trajectory step for particle i ends
                                    // before the trajectory step for particle j.
                                    ++it_i;
                                } else if (*it_j < *it_i) {
                                    // The trajectory step for particle j ends
                                    // before the trajectory step for particle i.
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
                    });
            }
        });
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
