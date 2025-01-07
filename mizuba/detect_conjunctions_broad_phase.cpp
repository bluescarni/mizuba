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

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <span>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>

#include <heyoka/mdspan.hpp>

#include "conjunctions.hpp"

namespace mizuba
{

namespace detail
{

namespace
{

#if !defined(NDEBUG)

// Helper to verify the broad-phase conjunction detection against a naive
// n**2 algorithm.
void verify_broad_phase(auto nobjs, const auto &bp_cv, auto aabbs, const auto &otypes)
{
    // LCOV_EXCL_START
    // Don't run the check if there's too many objects.
    if (nobjs > 10000u) {
        return;
    }
    // LCOV_EXCL_STOP

    // Build a set version of the collision list
    // for fast lookup.
    std::set<conjunctions::aabb_collision> coll_tree;
    for (const auto &c : bp_cv) {
        // Check that, for all collisions (i, j), i is always < j.
        assert(c.i < c.j);
        // Check that the collision pairs are unique.
        assert(coll_tree.insert(c).second);
    }

    // A counter for the N**2 collision detection algorithm below.
    std::atomic<decltype(coll_tree.size())> coll_counter(0);

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<decltype(nobjs)>(0, nobjs), [nobjs, aabbs, &otypes, &coll_tree,
                                                                                      &coll_counter](
                                                                                         const auto &obj_range) {
        for (auto i = obj_range.begin(); i != obj_range.end(); ++i) {
            const auto xi_lb = aabbs(i, 0, 0);
            const auto yi_lb = aabbs(i, 0, 1);
            const auto zi_lb = aabbs(i, 0, 2);
            const auto ri_lb = aabbs(i, 0, 3);

            const auto xi_ub = aabbs(i, 1, 0);
            const auto yi_ub = aabbs(i, 1, 1);
            const auto zi_ub = aabbs(i, 1, 2);
            const auto ri_ub = aabbs(i, 1, 3);

            // Fetch the type of object i.
            const auto otype_i = otypes[i];

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<decltype(nobjs)>(i + 1u, nobjs), [&](const auto &rj) {
                decltype(coll_tree.size()) loc_ncoll = 0;

                for (auto j = rj.begin(); j != rj.end(); ++j) {
                    const auto xj_lb = aabbs(j, 0, 0);
                    const auto yj_lb = aabbs(j, 0, 1);
                    const auto zj_lb = aabbs(j, 0, 2);
                    const auto rj_lb = aabbs(j, 0, 3);

                    const auto xj_ub = aabbs(j, 1, 0);
                    const auto yj_ub = aabbs(j, 1, 1);
                    const auto zj_ub = aabbs(j, 1, 2);
                    const auto rj_ub = aabbs(j, 1, 3);

                    // Fetch the type of object j.
                    const auto otype_j = otypes[j];

                    // Check the overlap condition.
                    const bool overlap = (xi_ub >= xj_lb && xi_lb <= xj_ub) && (yi_ub >= yj_lb && yi_lb <= yj_ub)
                                         && (zi_ub >= zj_lb && zi_lb <= zj_ub) && (ri_ub >= rj_lb && ri_lb <= rj_ub)
                                         && (otype_i + otype_j <= 3);

                    if (overlap) {
                        // Overlap detected in the simple algorithm:
                        // the collision must be present also
                        // in the tree code.
                        assert(coll_tree.find({i, j}) != coll_tree.end());
                    } else {
                        // NOTE: the contrary is not necessarily
                        // true: for multi-object leaves, we
                        // may detect overlaps that do not actually exist.
                    }

                    loc_ncoll += overlap;
                }

                coll_counter.fetch_add(loc_ncoll);
            });
        }
    });

    // NOTE: in case of multi-object leaves, we will have detected
    // non-existing AABBs overlaps. Thus, just require that the number
    // of collisions detected via the tree is at least as large
    // as the number of "true" collisions detected with the N**2 algorithm.
    assert(coll_tree.size() >= coll_counter.load());
}

#endif

} // namespace

} // namespace detail

// Detect aabbs collisions within a conjunction step.
//
// tree is the bvh tree, cd_vidx is the sorted indexing over the objects
// according to their morton codes, otypes contains the object types (in
// the original order), cd_srt_aabbs contains the morton-sorted aabbs,
// cd_aabbs contains the aabbs in the original order.
//
// The return value is the list of detect aabbs collisions.
std::vector<conjunctions::aabb_collision> conjunctions::detect_conjunctions_broad_phase(
    const std::vector<bvh_node> &tree, const std::vector<std::uint32_t> &cd_vidx,
    const std::vector<std::int32_t> &otypes, const std::vector<float> &cd_srt_aabbs,
    // NOTE: this is used only in debug mode.
    [[maybe_unused]] const std::vector<float> &cd_aabbs)
{
    // If the tree is empty, it means that there are no objects
    // with trajectory data for the current conjunction step. Exit early
    // with an empty list of detect aabb collisions.
    if (tree.empty()) {
        return {};
    }

    // Cache the total number of objects.
    const auto tot_nobjs = static_cast<std::size_t>(otypes.size());
    assert(cd_vidx.size() == tot_nobjs);

    // Create a const span into cd_srt_aabbs.
    using const_aabbs_span_t = heyoka::mdspan<const float, heyoka::extents<std::size_t, std::dynamic_extent, 2, 4>>;
    const_aabbs_span_t cd_srt_aabbs_span{cd_srt_aabbs.data(), tot_nobjs + 1u};

#if !defined(NDEBUG)

    // Create a const span into cd_aabbs.
    const_aabbs_span_t cd_aabbs_span{cd_aabbs.data(), tot_nobjs + 1u};

#endif

    // Initialise the vector to store the results of
    // broad-phase conjunction detection for this conjunction step.
    //
    // NOTE: we used to use TBB's concurrent_vector here, however:
    //
    // https://github.com/oneapi-src/oneTBB/issues/1531
    //
    // Perhaps we can consider re-enabling it once fixed, but on the
    // other hand the mutex approach seems to perform well enough.
    std::vector<aabb_collision> bp_coll_vector;
    std::mutex bp_coll_vector_mutex;

    // We will be performing broad-phase conjunction detection only for objects
    // with trajectory data for the current conjunction step. These objects
    // are sorted first into the morton order, and their total number can be
    // determined from the number of objects in the root node of the tree.
    assert(tree.size() > 0u);
    const auto nobjs = tree[0].end - tree[0].begin;
    assert(nobjs <= tot_nobjs);

    // We will be using thread-specific data to store temporary results during broad-phase
    // conjunction detection.
    struct ets_data {
        // Local list of detected AABBs collisions.
        std::vector<aabb_collision> bp;
        // Local stack for the BVH tree traversal.
        std::vector<std::int32_t> stack;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([]() { return ets_data{}; });

    // For each object with trajectory data for this conjunction step,
    // identify collisions between its aabb and the aabbs of the other objects.
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::uint32_t>(0, nobjs),
        [&cd_vidx, &otypes, cd_srt_aabbs_span, &tree, &bp_coll_vector, &bp_coll_vector_mutex,
         &ets](const auto &obj_range) {
            // Fetch the thread-local data.
            // NOTE: no need to isolate here, as we are not
            // invoking any other TBB primitive from within this
            // scope.
            auto &[rng_bp_coll_vec, stack] = ets.local();

            // Clear the local list of aabbs collisions.
            // NOTE: the stack will be cleared at the beginning
            // of the traversal for each object.
            rng_bp_coll_vec.clear();

            // NOTE: the object indices in this for loop refer
            // to the Morton-ordered data.
            for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                // Load the original object index.
                const auto orig_obj_idx = cd_vidx[obj_idx];

                // Fetch the object type.
                const auto obj_type = otypes[orig_obj_idx];

                // Reset the stack, and add the root node to it.
                stack.clear();
                stack.push_back(0);

                // Cache the AABB of the current object.
                const auto x_lb = cd_srt_aabbs_span(obj_idx, 0, 0);
                const auto y_lb = cd_srt_aabbs_span(obj_idx, 0, 1);
                const auto z_lb = cd_srt_aabbs_span(obj_idx, 0, 2);
                const auto r_lb = cd_srt_aabbs_span(obj_idx, 0, 3);

                const auto x_ub = cd_srt_aabbs_span(obj_idx, 1, 0);
                const auto y_ub = cd_srt_aabbs_span(obj_idx, 1, 1);
                const auto z_ub = cd_srt_aabbs_span(obj_idx, 1, 2);
                const auto r_ub = cd_srt_aabbs_span(obj_idx, 1, 3);

                do {
                    // Pop a node.
                    const auto cur_node_idx = stack.back();
                    stack.pop_back();

                    // Fetch the AABB of the node.
                    const auto &cur_node = tree[static_cast<std::uint32_t>(cur_node_idx)];
                    const auto &n_lb = cur_node.lb;
                    const auto &n_ub = cur_node.ub;

                    // Check for overlap with the AABB of the current object.
                    // NOTE: as explained during the computation of the AABBs, the
                    // AABBs we produce via interval arithmetic consist of
                    // closed intervals, and thus we compare with <= and >=.
                    const bool overlap = (x_ub >= n_lb[0] && x_lb <= n_ub[0]) && (y_ub >= n_lb[1] && y_lb <= n_ub[1])
                                         && (z_ub >= n_lb[2] && z_lb <= n_ub[2])
                                         && (r_ub >= n_lb[3] && r_lb <= n_ub[3]);

                    if (overlap) {
                        if (cur_node.left == -1) {
                            // Leaf node: mark the object as a conjunction
                            // candidate with all objects in the node, unless either:
                            //
                            // - obj_idx is having a conjunction with itself (obj_idx == i), or
                            // - obj_idx > i, in order to avoid counting twice
                            //   the conjunctions (obj_idx, i) and (i, obj_idx), or
                            // - obj_idx and i are either two secondaries, or at least
                            //   one of them is masked.
                            //
                            // NOTE: in case of a multi-object leaf,
                            // the node's AABB is the composition of the AABBs
                            // of all objects in the node, and thus, in general,
                            // it is not strictly true that obj_idx will overlap with
                            // *all* objects in the node. In other words, we will
                            // be detecting AABB collisions which are not actually there.
                            // This is ok, as they will be filtered out in the
                            // next stages of conjunction detection.
                            // NOTE: like in the outer loop, the index i here refers
                            // to the morton-ordered data.
                            for (auto i = cur_node.begin; i != cur_node.end; ++i) {
                                // Fetch index i in the original order.
                                const auto orig_i = cd_vidx[i];

                                if (orig_obj_idx >= orig_i) {
                                    continue;
                                }

                                // Fetch the type of the i object.
                                const auto obj_type_i = otypes[orig_i];

                                // NOTE: the type values are constructed so that obj_type + obj_type_i <= 3
                                // for primary-primary and secondary-primary conjunctions only.
                                if (obj_type + obj_type_i <= 3) {
                                    rng_bp_coll_vec.emplace_back(orig_obj_idx, orig_i);
                                }
                            }
                        } else {
                            // Internal node: add both children to the
                            // stack and iterate.
                            stack.push_back(cur_node.left);
                            stack.push_back(cur_node.right);
                        }
                    }
                } while (!stack.empty());
            }

            // Atomically merge rng_bp_coll_vec into bp_coll_vector.
            // NOTE: ensure we do this at the end of the scope in order to minimise
            // the locking time.
            std::lock_guard lock(bp_coll_vector_mutex);
            bp_coll_vector.insert(bp_coll_vector.end(), rng_bp_coll_vec.begin(), rng_bp_coll_vec.end());
        });

    // Sort bp_coll_vector.
    oneapi::tbb::parallel_sort(bp_coll_vector.begin(), bp_coll_vector.end());

#if !defined(NDEBUG)

    // Verify the outcome of collision detection in debug mode.
    detail::verify_broad_phase(nobjs, bp_coll_vector, cd_aabbs_span, otypes);

#endif

    return bp_coll_vector;
}

} // namespace mizuba
