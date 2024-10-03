// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <mutex>
#include <set>
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
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/task_arena.h>

#include "conjunctions.hpp"
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

namespace
{

// Move over the bp data from individual files into a single global file.
// The return value is a vector of offset/size pairs (one for each conjunction
// step) into the global file.
auto consolidate_bp_data(const auto &tmp_dir_path, const auto &bp_coll_sizes)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;
    using aabb_collision = conjunctions::aabb_collision;

    // Do a first pass on bp_coll_sizes to determine the offsets into the global file
    // and the total required size.
    // NOTE: the offsets are measured in number of aabb_collision.
    std::vector<std::tuple<std::size_t, std::size_t>> offsets;
    offsets.reserve(bp_coll_sizes.size());
    safe_size_t cur_offset = 0;
    for (const auto cur_bp_coll_size : bp_coll_sizes) {
        offsets.emplace_back(cur_offset, boost::numeric_cast<std::size_t>(cur_bp_coll_size));
        cur_offset += cur_bp_coll_size;
    }

    // NOTE: there may be no bp data, in which case cur_offset is zero. But it is not allowed to
    // memmap a zero-sized file, hence give it a minimum size of sizeof(aabb_collision).
    cur_offset = std::max(safe_size_t(sizeof(aabb_collision)), cur_offset * sizeof(aabb_collision));

    // Prepare the global file.
    const auto storage_path = tmp_dir_path / "bp";
    create_sized_file(storage_path, cur_offset);

    // Memory-map it.
    boost::iostreams::mapped_file_sink file(storage_path.string());

    // Fetch a pointer to the beginning of the data.
    // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
    // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
    auto *base_ptr = reinterpret_cast<aabb_collision *>(file.data());
    assert(boost::alignment::is_aligned(base_ptr, alignof(aabb_collision)));

    // Copy over the data from the individual files.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<decltype(bp_coll_sizes.size())>(0, bp_coll_sizes.size()),
                              [&tmp_dir_path, base_ptr, &offsets](const auto &rng) {
                                  for (auto idx = rng.begin(); idx != rng.end(); ++idx) {
                                      const auto [bp_coll_offset, bp_coll_size] = offsets[idx];

                                      // Compute the file size.
                                      const auto fsize = sizeof(aabb_collision) * bp_coll_size;

                                      // Build the file path.
                                      const auto fpath = tmp_dir_path / fmt::format("bp_{}", idx);
                                      assert(boost::filesystem::exists(fpath));
                                      assert(boost::filesystem::file_size(fpath) == fsize);

                                      // Open it.
                                      std::ifstream bp_file(fpath.string(), std::ios::binary | std::ios::in);
                                      bp_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

                                      // Copy it into the mapped file.
                                      bp_file.read(reinterpret_cast<char *>(base_ptr + bp_coll_offset),
                                                   boost::numeric_cast<std::streamsize>(fsize));

                                      // Close the file.
                                      bp_file.close();

                                      // Remove it.
                                      boost::filesystem::remove(fpath);
                                  }
                              });

    // Close the memory-mapped file.
    file.close();

    // Mark it as read-only.
    mark_file_read_only(storage_path);

    // Return the offsets.
    return offsets;
}

#if !defined(NDEBUG)

// Helper to verify the broad-phase conjunction detection against a naive
// n**2 algorithm.
void verify_broad_phase(auto cd_idx, auto nobjs, const auto &bp_cv, auto aabbs, const auto &conj_active)
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

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<decltype(nobjs)>(0, nobjs), [nobjs, aabbs, cd_idx,
                                                                                      &conj_active, &coll_tree,
                                                                                      &coll_counter](
                                                                                         const auto &obj_range) {
        for (auto i = obj_range.begin(); i != obj_range.end(); ++i) {
            const auto xi_lb = aabbs(cd_idx, i, 0, 0);
            const auto yi_lb = aabbs(cd_idx, i, 0, 1);
            const auto zi_lb = aabbs(cd_idx, i, 0, 2);
            const auto ri_lb = aabbs(cd_idx, i, 0, 3);

            const auto xi_ub = aabbs(cd_idx, i, 1, 0);
            const auto yi_ub = aabbs(cd_idx, i, 1, 1);
            const auto zi_ub = aabbs(cd_idx, i, 1, 2);
            const auto ri_ub = aabbs(cd_idx, i, 1, 3);

            // Check if i is active for conjunctions.
            const auto conj_active_i = conj_active[i];

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<decltype(nobjs)>(i + 1u, nobjs), [&](const auto &rj) {
                decltype(coll_tree.size()) loc_ncoll = 0;

                for (auto j = rj.begin(); j != rj.end(); ++j) {
                    const auto xj_lb = aabbs(cd_idx, j, 0, 0);
                    const auto yj_lb = aabbs(cd_idx, j, 0, 1);
                    const auto zj_lb = aabbs(cd_idx, j, 0, 2);
                    const auto rj_lb = aabbs(cd_idx, j, 0, 3);

                    const auto xj_ub = aabbs(cd_idx, j, 1, 0);
                    const auto yj_ub = aabbs(cd_idx, j, 1, 1);
                    const auto zj_ub = aabbs(cd_idx, j, 1, 2);
                    const auto rj_ub = aabbs(cd_idx, j, 1, 3);

                    // Check if j is active for conjunctions.
                    const auto conj_active_j = conj_active[j];

                    // Check the overlap condition.
                    const bool overlap = (xi_ub >= xj_lb && xi_lb <= xj_ub) && (yi_ub >= yj_lb && yi_lb <= yj_ub)
                                         && (zi_ub >= zj_lb && zi_lb <= zj_ub) && (ri_ub >= rj_lb && ri_lb <= rj_ub)
                                         && (conj_active_i || conj_active_j);

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

                coll_counter.fetch_add(loc_ncoll, std::memory_order::relaxed);
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

// Broad-phase conjunction detection.
//
// The objective of this phase is to identify collisions (i.e., overlaps) in the aabbs of the
// objects' trajectories.
//
// pj is the polyjectory, tmp_dir_path the temporary dir storing all conjunction data,
// n_cd_steps the number of conjunction steps, tree_offsets the offsets/sizes to access the
// bvh tree data, conj_active a vector of flags signalling which objects are active for
// conjunction detection.
std::vector<std::tuple<std::size_t, std::size_t>>
conjunctions::broad_phase(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path, std::size_t n_cd_steps,
                          const std::vector<std::tuple<std::size_t, std::size_t>> &tree_offsets,
                          const std::vector<bool> &conj_active)
{
    // Cache the total number of objects.
    const auto tot_nobjs = pj.get_nobjs();

    // Fetch read-only spans to the sorted aabbs and indices vectors.
    boost::iostreams::mapped_file_source file_srt_aabbs((tmp_dir_path / "srt_aabbs").string());
    const auto *srt_aabbs_base_ptr = reinterpret_cast<const float *>(file_srt_aabbs.data());
    assert(boost::alignment::is_aligned(srt_aabbs_base_ptr, alignof(float)));
    const aabbs_span_t srt_aabbs{srt_aabbs_base_ptr, n_cd_steps, tot_nobjs + 1u};

    boost::iostreams::mapped_file_source file_srt_idx((tmp_dir_path / "vidx").string());
    const auto *srt_idx_base_ptr = reinterpret_cast<const std::uint32_t *>(file_srt_idx.data());
    assert(boost::alignment::is_aligned(srt_idx_base_ptr, alignof(std::uint32_t)));
    const srt_idx_span_t srt_idx{srt_idx_base_ptr, n_cd_steps, tot_nobjs};

#if !defined(NDEBUG)

    // Fetch a read-only span to the unsorted aabbs.
    // This is used only in debug mode.
    boost::iostreams::mapped_file_source file_aabbs((tmp_dir_path / "aabbs").string());
    const auto *aabbs_base_ptr = reinterpret_cast<const float *>(file_aabbs.data());
    assert(boost::alignment::is_aligned(aabbs_base_ptr, alignof(float)));
    const aabbs_span_t aabbs{aabbs_base_ptr, n_cd_steps, tot_nobjs + 1u};

#endif

    // Fetch a pointer to the tree data.
    boost::iostreams::mapped_file_source file_bvh_trees((tmp_dir_path / "bvh").string());
    const auto *bvh_trees_base_ptr = reinterpret_cast<const bvh_node *>(file_bvh_trees.data());
    assert(boost::alignment::is_aligned(bvh_trees_base_ptr, alignof(bvh_node)));

    // Global vector that will contain the sizes of the aabb collision
    // lists for each conjunction step.
    std::vector<std::vector<aabb_collision>::size_type> bp_coll_sizes;
    bp_coll_sizes.resize(boost::numeric_cast<decltype(bp_coll_sizes.size())>(n_cd_steps));

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

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps), [bvh_trees_base_ptr,
                                                                                       &tree_offsets, &bp_coll_sizes,
                                                                                       tot_nobjs, srt_aabbs, srt_idx,
                                                                                       &ets, &conj_active, &tmp_dir_path
#if !defined(NDEBUG)
                                                                                       ,
                                                                                       aabbs
#endif
    ](const auto &cd_range) {
        for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
            // Fetch the tree for the current conjunction step.
            const auto [tree_offset, tree_size] = tree_offsets[cd_idx];
            const auto *tree_ptr = bvh_trees_base_ptr + tree_offset;
            const tree_span_t tree{tree_ptr, tree_size};

            // Create the vector for storing the list of aabbs collisions
            // for the current conjunction step.
            // NOTE: in principle here we could fetch bp_cv from thread-local
            // storage, rather than creating it each time ex novo. However, the
            // code would become more complex and I am not sure it is worth it
            // performance wise. Something to keep in mind for the future.
            std::vector<aabb_collision> bp_cv;

            // Mutex for concurrently inserting data into bp_cv.
            std::mutex bp_cv_mutex;

            // Fetch the number of objects with trajectory data for the current
            // conjunction step. This can be determined from the number of objects
            // in the root node of the tree.
            assert(tree.extent(0) > 0u);
            const auto nobjs = tree(0).end - tree(0).begin;
            assert(nobjs <= tot_nobjs);

            // For each object with trajectory data for this conjunction step,
            // identify collisions between its aabb and the aabbs of the other objects.
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<std::uint32_t>(0, nobjs),
                [&ets, srt_idx, cd_idx, &conj_active, srt_aabbs, tree, &bp_cv, &bp_cv_mutex](const auto &obj_range) {
                    // Fetch the thread-local data.
                    // NOTE: no need to isolate here, as we are not
                    // invoking any other TBB primitive from within this
                    // scope.
                    auto &[local_bp, stack] = ets.local();

                    // Clear the local list of aabbs collisions.
                    // NOTE: the stack will be cleared at the beginning
                    // of the traversal for each object.
                    local_bp.clear();

                    // NOTE: the object indices in this for loop refer
                    // to the Morton-ordered data.
                    for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                        // Load the original object index.
                        const auto orig_obj_idx = srt_idx(cd_idx, obj_idx);

                        // Check if the object is active for conjunction
                        // detection.
                        const auto obj_is_active = conj_active[orig_obj_idx];

                        // Reset the stack, and add the root node to it.
                        stack.clear();
                        stack.push_back(0);

                        // Cache the AABB of the current object.
                        const auto x_lb = srt_aabbs(cd_idx, obj_idx, 0, 0);
                        const auto y_lb = srt_aabbs(cd_idx, obj_idx, 0, 1);
                        const auto z_lb = srt_aabbs(cd_idx, obj_idx, 0, 2);
                        const auto r_lb = srt_aabbs(cd_idx, obj_idx, 0, 3);

                        const auto x_ub = srt_aabbs(cd_idx, obj_idx, 1, 0);
                        const auto y_ub = srt_aabbs(cd_idx, obj_idx, 1, 1);
                        const auto z_ub = srt_aabbs(cd_idx, obj_idx, 1, 2);
                        const auto r_ub = srt_aabbs(cd_idx, obj_idx, 1, 3);

                        do {
                            // Pop a node.
                            const auto cur_node_idx = stack.back();
                            stack.pop_back();

                            // Fetch the AABB of the node.
                            const auto &cur_node = tree(static_cast<std::uint32_t>(cur_node_idx));
                            const auto &n_lb = cur_node.lb;
                            const auto &n_ub = cur_node.ub;

                            // Check for overlap with the AABB of the current object.
                            // NOTE: as explained during the computation of the AABBs, the
                            // AABBs we produce via interval arithmetic consist of
                            // closed intervals, and thus we compare with <= and >=.
                            const bool overlap
                                = (x_ub >= n_lb[0] && x_lb <= n_ub[0]) && (y_ub >= n_lb[1] && y_lb <= n_ub[1])
                                  && (z_ub >= n_lb[2] && z_lb <= n_ub[2]) && (r_ub >= n_lb[3] && r_lb <= n_ub[3]);

                            if (overlap) {
                                if (cur_node.left == -1) {
                                    // Leaf node: mark the object as a conjunction
                                    // candidate with all objects in the node, unless either:
                                    //
                                    // - obj_idx is having a conjunction with itself (obj_idx == i), or
                                    // - obj_idx > i, in order to avoid counting twice
                                    //   the conjunctions (obj_idx, i) and (i, obj_idx), or
                                    // - obj_idx and i are both inactive.
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
                                        const auto orig_i = srt_idx(cd_idx, i);

                                        if (orig_obj_idx >= orig_i) {
                                            continue;
                                        }

                                        // Check if i is active for conjunctions.
                                        const auto conj_active_i = conj_active[orig_i];

                                        if (obj_is_active || conj_active_i) {
                                            local_bp.emplace_back(orig_obj_idx, orig_i);
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

                    // Append the detected aabbs collisions to bp_cv.
                    std::lock_guard lock(bp_cv_mutex);

                    bp_cv.insert(bp_cv.end(), local_bp.begin(), local_bp.end());
                });

            // Sort the data in bp_cv.
            oneapi::tbb::parallel_sort(bp_cv.begin(), bp_cv.end());

#if !defined(NDEBUG)

            // Verify the outcome of collision detection in debug mode.
            detail::verify_broad_phase(cd_idx, nobjs, bp_cv, aabbs, conj_active);

#endif

            // Record the size of bp_cv in bp_coll_sizes.
            bp_coll_sizes[cd_idx] = bp_cv.size();

            // Write bp_cv to disk.
            const auto bp_file_path = tmp_dir_path / fmt::format("bp_{}", cd_idx);
            // LCOV_EXCL_START
            if (boost::filesystem::exists(bp_file_path)) [[unlikely]] {
                throw std::runtime_error(
                    fmt::format("Cannot create the storage file '{}', as it exists already", bp_file_path.string()));
            }
            // LCOV_EXCL_STOP
            std::ofstream bp_file(bp_file_path.string(), std::ios::binary | std::ios::out);
            bp_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
            bp_file.write(reinterpret_cast<const char *>(bp_cv.data()),
                          boost::safe_numerics::safe<std::streamsize>(bp_cv.size()) * sizeof(aabb_collision));
        }
    });

    // Consolidate all broad-phase data into a single file and return the bp offsets.
    return detail::consolidate_bp_data(tmp_dir_path, bp_coll_sizes);
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
