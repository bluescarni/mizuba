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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <future>
#include <ios>
#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>

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
#include <oneapi/tbb/parallel_scan.h>
#include <oneapi/tbb/task_arena.h>

#include "conjunctions.hpp"
#include "detail/file_utils.hpp"
#include "detail/get_n_effective_objects.hpp"
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

// Initial values for the nodes' bounding boxes.
constexpr auto finf = std::numeric_limits<float>::infinity();
constexpr std::array<float, 4> default_lb = {finf, finf, finf, finf};
constexpr std::array<float, 4> default_ub = {-finf, -finf, -finf, -finf};

// This is auxiliary data attached to each node of a BVH tree,
// used only during construction/verification.
// It will not be present in the final tree, hence it is kept
// separate.
struct aux_node_data {
    // Total number of nodes in the tree level the
    // node belongs to.
    std::uint32_t nn_level;
    // Index of the split bit. This has several meanings and properties:
    //
    // 1) all morton codes in the node will share the same common
    //    initial sequence of split_idx bits (measured from MSB);
    //
    // 2) if the node is internal, split_idx is the bit index in the morton
    //    code (counted from the MSB) used for partitioning the objects
    //    in the node into the left and right children. For instance,
    //    split_idx == 3 means that the objects in the left child will
    //    have morton codes whose fourth bit (from MSB) is 0, while the objects in the
    //    right child will have morton codes whose fourth bit is 1;
    //
    // 3) if the node is a leaf with a single object, then split_idx is the
    //    split_idx of the parent + 1.
    //
    // Property 3) follows immediately from the fact that the split_idx
    // of a node is always initialised as the split_idx of the parent
    // + 1. split_idx is modified after initialisation only if the node
    // contains more than 1 object.
    //
    // From property 1), it follows that in a leaf node with more than
    // 1 object split_idx must be 64, because all morton codes will be
    // identical.
    std::uint32_t split_idx;
    // The parent.
    std::int32_t parent;
};

// Data structure used to temporarily store certain
// properties of a node during the construction of the
// BVH level by level. The data stored in here will eventually
// be transferred to a bvh_node.
struct level_data {
    // Number of children.
    std::uint32_t nc;
    // Number of objects in the left child.
    std::uint32_t nolc;
    // Storage used in the computation of the prefix
    // sum over the number of children for each node
    // in the current level.
    std::uint32_t ps;
};

#if !defined(NDEBUG)

// Helper to verify that the aabb of a node in a bvh tree is correct.
//
// cd_idx is the collisional step the bvh tree belongs to, srt_aabbs the list
// of sorted aabbs for all collisional timesteps, cur_node the node
// we are checking.
void verify_node_aabb(auto cd_idx, auto srt_aabbs, const auto &cur_node)
{
    auto dbg_lb = default_lb, dbg_ub = default_ub;

    for (auto obj_idx = cur_node.begin; obj_idx != cur_node.end; ++obj_idx) {
        for (auto i = 0u; i < 4u; ++i) {
            dbg_lb[i] = std::min(dbg_lb[i], srt_aabbs(cd_idx, obj_idx, 0, i));
            dbg_ub[i] = std::max(dbg_ub[i], srt_aabbs(cd_idx, obj_idx, 1, i));
        }
    }

    assert(cur_node.lb == dbg_lb);
    assert(cur_node.ub == dbg_ub);
}

#if !defined(NDEBUG) && (defined(__GNUC__) || defined(__clang__))

template <typename U>
constexpr auto always_false_v = false;

// Debug helper to compute the index of the first different
// bit between n1 and n2, starting from the MSB.
template <typename T>
int first_diff_bit(T n1, T n2)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);

    const auto res_xor = n1 ^ n2;

    if (res_xor == 0u) {
        return std::numeric_limits<T>::digits;
    } else {
        if constexpr (std::is_same_v<T, unsigned>) {
            return __builtin_clz(res_xor);
        } else if constexpr (std::is_same_v<T, unsigned long>) {
            return __builtin_clzl(res_xor);
        } else if constexpr (std::is_same_v<T, unsigned long long>) {
            return __builtin_clzll(res_xor);
        } else {
            static_assert(always_false_v<T>);
        }
    }
}

#endif

// Helper to run several consistency check on a bvh tree.
//
// cd_idx is the collisional step the bvh tree belongs to, srt_mcodes the list
// of sorted mcodes for all collisional timesteps, bvh_treee the tree
// we are checking, aux_data the per-node auxiliary data, srt_aabbs the list of
// sorted aabbs for all collisional timesteps.
void verify_bvh_tree(auto cd_idx, auto srt_mcodes, const auto &bvh_tree, const auto &aux_data, auto srt_aabbs,
                     auto nobjs)
{
    // Set of object indices. We will use this to verify
    // that all objects appear exactly once in leaf nodes.
    std::set<std::size_t> pset;

    // Traverse the tree.
    for (decltype(bvh_tree.size()) i = 0; i < bvh_tree.size(); ++i) {
        // Load the current node.
        const auto &cur_node = bvh_tree[i];
        const auto &cur_aux_data = aux_data[i];

        // The node must contain 1 or more objects.
        assert(cur_node.end > cur_node.begin);

        // The node must have either 0 or 2 children.
        if (cur_node.left == -1) {
            assert(cur_node.right == -1);
        } else {
            assert(cur_node.left > 0);
            assert(cur_node.right > 0);
        }

        if (cur_node.end - cur_node.begin == 1u) {
            // A leaf with a single object.
            // It must have no children.
            assert(cur_node.left == -1);
            assert(cur_node.right == -1);

            // split_idx must be zero if this is the root node,
            // otherwise it must be the split_idx of the parent + 1.
            if (i == 0u) {
                assert(cur_aux_data.parent == -1);
                assert(cur_aux_data.split_idx == 0u);
            } else {
                assert(cur_aux_data.parent >= 0);
                assert(cur_aux_data.split_idx
                       == aux_data[static_cast<std::uint32_t>(cur_aux_data.parent)].split_idx + 1u);
            }

            // Add the object to the global object set,
            // ensuring the object has not been added to pset yet.
            assert(pset.insert(boost::numeric_cast<std::size_t>(cur_node.begin)).second);
        } else if (cur_node.left == -1) {
            // A leaf with multiple objects:
            // - all objects must have the same Morton code,
            // - split_idx must be 64.
            assert(cur_node.right == -1);

            // Check split_idx.
            assert(cur_aux_data.split_idx == 64u);

            // Fetch the morton code of the first object.
            const auto mc = srt_mcodes(cd_idx, cur_node.begin);

            // Make sure the first object is accounted for in pset.
            assert(pset.insert(boost::numeric_cast<std::size_t>(cur_node.begin)).second);

            // Check the remaining objects.
            for (auto j = cur_node.begin + 1u; j < cur_node.end; ++j) {
                assert(srt_mcodes(cd_idx, j) == mc);
                assert(pset.insert(boost::numeric_cast<std::size_t>(j)).second);
            }
        } else {
            // An internal node.
            assert(cur_node.left > 0);
            assert(cur_node.right > 0);

            const auto uleft = static_cast<std::uint32_t>(cur_node.left);
            const auto uright = static_cast<std::uint32_t>(cur_node.right);

            // The children indices must be greater than the current node's
            // index and within the tree.
            assert(uleft > i && uleft < bvh_tree.size());
            assert(uright > i && uright < bvh_tree.size());

            // Check that the ranges of the children are consistent with
            // the range of the current node.
            assert(bvh_tree[uleft].begin == cur_node.begin);
            assert(bvh_tree[uleft].end < cur_node.end);
            assert(bvh_tree[uright].begin == bvh_tree[uleft].end);
            assert(bvh_tree[uright].end == cur_node.end);

            // The node's split_idx value must be less than 64.
            assert(cur_aux_data.split_idx < 64u);

            // Check that all morton codes in the node share the first
            // split_idx digits (counting from MSB).
            // NOTE: if split_idx is zero, there are no common initial
            // bits to compare.
            if (cur_aux_data.split_idx > 0u) {
                const auto common_init_bits = srt_mcodes(cd_idx, cur_node.begin) >> (64u - cur_aux_data.split_idx);

                for (auto obj_idx = cur_node.begin + 1u; obj_idx != cur_node.end; ++obj_idx) {
                    assert((srt_mcodes(cd_idx, obj_idx) >> (64u - cur_aux_data.split_idx)) == common_init_bits);
                }
            } else {
                // splid_idx == 0 can happen only at the root node.
                assert(i == 0u);
            }

#if defined(__GNUC__) || defined(__clang__)
            // Check this internal node was split correctly (i.e.,
            // split_idx corresponds to the index of the first
            // different bit at the boundary between first and second child).
            const auto split_obj_idx = bvh_tree[uleft].end - 1u;
            assert(static_cast<unsigned>(detail::first_diff_bit(srt_mcodes(cd_idx, split_obj_idx),
                                                                srt_mcodes(cd_idx, split_obj_idx + 1u)))
                   == cur_aux_data.split_idx);
#endif
        }

        // Check the parent info.
        if (i == 0u) {
            assert(cur_aux_data.parent == -1);
        } else {
            assert(cur_aux_data.parent >= 0);

            const auto upar = static_cast<std::uint32_t>(cur_aux_data.parent);

            assert(upar < i);
            assert(cur_node.begin >= bvh_tree[upar].begin);
            assert(cur_node.end <= bvh_tree[upar].end);
            assert(cur_node.begin == bvh_tree[upar].begin || cur_node.end == bvh_tree[upar].end);
        }

        // nn_level must alway be nonzero.
        assert(cur_aux_data.nn_level > 0u);

        // Check that the AABB of the node is correct.
        verify_node_aabb(cd_idx, srt_aabbs, cur_node);
    }

    // Final check on pset.
    assert(pset.size() == nobjs);
    assert(*pset.begin() == 0u);
    auto last_it = pset.end();
    --last_it;
    assert(*last_it == nobjs - 1u);
}

#endif

} // namespace

} // namespace detail

// Construct the bvh trees, one per each conjunction step.
//
// The algorithm is taken from this paper:
//
// https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2009.01377.x
//
// NOTE: here we have implemented the general idea outlined in §3.1.
// The algorithm explained in §3.2 may be faster, and we can consider
// experimenting with it in case we need to improve performance.
//
// pj is the polyjectory, tmp_dir_path the temporary dir storing all conjunction data,
// n_cd_steps the number of conjunction steps.
//
// NOTE: if an object has no trajectory data during a conjunction step, it will not show
// up in the bvh tree.
std::vector<std::tuple<std::size_t, std::size_t>>
conjunctions::construct_bvh_trees(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                                  std::size_t n_cd_steps) const
{
    using safe_u32_t = boost::safe_numerics::safe<std::uint32_t>;

    // Cache the total number of objects.
    const auto tot_nobjs = pj.get_nobjs();

    // Overflow check.
    try {
        // Make sure that std::ptrdiff_t can represent tot_nobjs.
        static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(tot_nobjs));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected during the construction of a BVH tree");
    }
    // LCOV_EXCL_STOP

    // Fetch read-only spans to the sorted aabbs and mcodes.
    boost::iostreams::mapped_file_source file_srt_aabbs((tmp_dir_path / "srt_aabbs").string());
    const auto *srt_aabbs_base_ptr = reinterpret_cast<const float *>(file_srt_aabbs.data());
    assert(boost::alignment::is_aligned(srt_aabbs_base_ptr, alignof(float)));
    const aabbs_span_t srt_aabbs{srt_aabbs_base_ptr, n_cd_steps, tot_nobjs + 1u};

    boost::iostreams::mapped_file_source file_srt_mcodes((tmp_dir_path / "srt_mcodes").string());
    const auto *srt_mcodes_base_ptr = reinterpret_cast<const std::uint64_t *>(file_srt_mcodes.data());
    assert(boost::alignment::is_aligned(srt_mcodes_base_ptr, alignof(std::uint64_t)));
    const mcodes_span_t srt_mcodes{srt_mcodes_base_ptr, n_cd_steps, tot_nobjs};

    // Create the bvh data file.
    const auto bvh_file_path = tmp_dir_path / "bvh";
    if (boost::filesystem::exists(bvh_file_path)) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot create the storage file '{}': the file exists already", bvh_file_path.string()));
        // LCOV_EXCL_STOP
    }
    std::ofstream bvh_file(bvh_file_path.string(), std::ios::binary | std::ios::out);
    bvh_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    // Setup the futures and promises to coordinate between the construction of the trees
    // and the writer thread.
    std::vector<std::promise<std::vector<bvh_node>>> tree_promises;
    tree_promises.resize(boost::numeric_cast<decltype(tree_promises.size())>(n_cd_steps));
    auto tree_fut_view = tree_promises | std::views::transform([](auto &p) { return p.get_future(); });
    std::vector tree_futures(std::ranges::begin(tree_fut_view), std::ranges::end(tree_fut_view));

    // Flag to signal that the writer thread should stop writing.
    std::atomic<bool> stop_writing = false;

    // Vector of offsets and sizes for the tree data
    // stored in the bvh data file.
    //
    // The first element of the pair is the offset at which tree data begins,
    // the second element of the pair is the tree size.
    //
    // NOTE: the offsets are measured in number of bvh_nodes.
    std::vector<std::tuple<std::size_t, std::size_t>> tree_offsets;
    tree_offsets.reserve(n_cd_steps);

    // Launch the writer thread.
    auto writer_future
        = std::async(std::launch::async, [&bvh_file, &tree_futures, &tree_offsets, &stop_writing, n_cd_steps]() {
              using namespace std::chrono_literals;
              using safe_size_t = boost::safe_numerics::safe<std::size_t>;

              // How long should we wait before checking if we should stop writing.
              const auto wait_duration = 250ms;

              // Track the tree offsets to build up tree_offsets.
              safe_size_t cur_offset = 0;

              for (std::size_t i = 0; i < n_cd_steps; ++i) {
                  // Fetch the future.
                  auto &tree_fut = tree_futures[i];

                  // Wait until the future becomes available, or return if a stop is requested.
                  while (tree_fut.wait_for(wait_duration) != std::future_status::ready) {
                      // LCOV_EXCL_START
                      if (stop_writing) [[unlikely]] {
                          return;
                      }
                      // LCOV_EXCL_STOP
                  }

                  // Fetch the tree from the future.
                  auto tree = tree_fut.get();

                  // Fetch the tree size.
                  const auto tree_size = tree.size();

                  // Write the tree.
                  bvh_file.write(reinterpret_cast<const char *>(tree.data()),
                                 boost::safe_numerics::safe<std::streamsize>(tree_size) * sizeof(bvh_node));

                  // Update tree_offsets and cur_offset.
                  tree_offsets.emplace_back(cur_offset, boost::numeric_cast<std::size_t>(tree_size));
                  cur_offset += tree_size;
              }
          });

    // NOTE: at this point, the writer thread has started. From now on, we wrap everything in a try/catch block
    // so that, if any exception is thrown, we can safely stop the writer thread before re-throwing.
    try {
        // We will be using thread-specific data to store the results of intermediate
        // computations in memory, before eventually flushing them to disk.
        struct ets_data {
            // BVH tree.
            std::vector<bvh_node> tree;
            // Auxiliary node data.
            // NOTE: the size of this vector will be kept
            // in sync with the size of tree.
            std::vector<detail::aux_node_data> aux_data;
            // Data used in the level-by-level construction of the treee.
            std::vector<detail::level_data> l_buffer;
        };
        using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                              oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
        ets_t ets([]() { return ets_data{}; });

        // Run the tree construction.
        //
        // NOTE: we process the conjunction steps in chunks because we want the writer thread to make steady progress.
        //
        // If we don't do the chunking, what happens is that TBB starts immediately processing conjunction steps
        // throughout the *entire* range, whereas the writer thread must proceed strictly sequentially. We thus end up
        // in a situation in which a lot of file writing happens at the very end while not overlapping with the
        // computation, and memory usage is high because we have to keep around for long unwritten data.
        //
        // With chunking, the writer thread can fully process chunk N while chunk N+1 is computing.
        //
        // NOTE: there are tradeoffs in selecting the chunk size. If it is too large, we are negating the benefits
        // of chunking wrt computation/transfer overlap and memory usage. If it is too small, we are limiting
        // parallel speedup. The current value is based on preliminary performance evaluation. In general, the "optimal"
        // chunking will depend on several variables such as the number of CPU cores, the available memory, etc.
        //
        // NOTE: I am also not sure whether or not it is possible to achieve the same result more elegantly
        // with some TBB partitioner/range wizardry.
        constexpr auto chunk_size = 128u;

        for (std::size_t start_cd_step_idx = 0; start_cd_step_idx != n_cd_steps;) {
            const auto n_rem_cd_steps = n_cd_steps - start_cd_step_idx;
            const auto end_cd_step_idx
                = start_cd_step_idx + (n_rem_cd_steps < chunk_size ? n_rem_cd_steps : chunk_size);

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<std::size_t>(start_cd_step_idx, end_cd_step_idx),
                [&ets, tot_nobjs, srt_aabbs, srt_mcodes, &tree_promises](const auto &cd_range) {
                    for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
                        // Fetch the number of objects with trajectory data for the current
                        // conjunction step.
                        const auto nobjs = detail::get_n_effective_objects(tot_nobjs, srt_aabbs, cd_idx);

                        // Fetch the thread-local data.
                        auto &[tree, aux_data, l_buffer] = ets.local();

                        // NOTE: isolate to avoid issues with thread-local data. See:
                        // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
                        oneapi::tbb::this_task_arena::isolate([srt_aabbs, cd_idx, srt_mcodes, nobjs, &tree, &aux_data,
                                                               &l_buffer, &tree_promises]() {
                            // Clear out the tree.
                            // NOTE: no need to clear out l_buffer as it is appropriately
                            // resized and written to at every new level during tree construction.
                            tree.clear();
                            aux_data.clear();

                            // Insert the root node.
                            // NOTE: this is inited as a leaf node without a parent.
                            tree.emplace_back(0, nobjs, -1, -1, detail::default_lb, detail::default_ub);
                            // NOTE: nn_level is inited to zero, even if we already know it will be set
                            // to 1 eventually. split_idx is inited to zero (though it may increase later).
                            aux_data.emplace_back(0, 0, -1);

                            // The number of nodes at the current level.
                            std::uint32_t cur_n_nodes = 1;

                            // The total number of levels and nodes in the tree. These are updated
                            // at the end of every iteration of the while-loop.
                            safe_u32_t n_levels = 0, n_nodes = 0;

                            while (cur_n_nodes != 0u) {
                                // Fetch the current tree size.
                                const auto cur_tree_size = tree.size();

                                // The node range for the current level.
                                const auto n_begin = cur_tree_size - cur_n_nodes;
                                const auto n_end = cur_tree_size;

                                // Prepare the level buffer. This will be used to temporarily store
                                // several properties of the nodes at the current level.
                                l_buffer.resize(boost::numeric_cast<decltype(l_buffer.size())>(cur_n_nodes));

                                // Counter for the total number of leaf nodes at the current level.
                                std::atomic<std::uint32_t> n_leaf_nodes_atm = 0;

                                // Step 1: determine, for each node at the current level,
                                // if the node is a leaf or not, and, for an internal node,
                                // the number of objects in the left child.
                                oneapi::tbb::parallel_for(
                                    oneapi::tbb::blocked_range(n_begin, n_end),
                                    [cd_idx, srt_aabbs, srt_mcodes, n_begin, cur_n_nodes, &tree, &aux_data, &l_buffer,
                                     &n_leaf_nodes_atm](const auto &node_range) {
                                        // Local accumulator for the number of leaf nodes
                                        // detected at the current level.
                                        std::uint32_t loc_n_leaf_nodes = 0;

                                        // NOTE: this for loop can *probably* be written in a vectorised
                                        // fashion, using the gather primitives as done in heyoka.
                                        for (auto node_idx = node_range.begin(); node_idx != node_range.end();
                                             ++node_idx) {
                                            assert(node_idx - n_begin < cur_n_nodes);

                                            // Fetch the current node data.
                                            auto &cur_node = tree[node_idx];
                                            auto &adata = aux_data[node_idx];

                                            // Flag to signal that this is a leaf node.
                                            bool is_leaf_node = false;

                                            // Pointer to detect where, in the morton codes range for the current node,
                                            // the bit flip takes place.
                                            const std::uint64_t *split_ptr = nullptr;

                                            // Fetch the morton codes range for the current node.
                                            const auto *mcodes_begin = &srt_mcodes(cd_idx, 0) + cur_node.begin;
                                            const auto *mcodes_end = &srt_mcodes(cd_idx, 0) + cur_node.end;

                                            if (cur_node.end - cur_node.begin > 1u && adata.split_idx < 64u) {
                                                // The node contains more than 1 object,
                                                // and the initial value of split_idx is within
                                                // the bit width of a 64-bit integer.
                                                // Figure out where the bit at index adata.split_idx
                                                // (counted from MSB) flips from 0 to 1
                                                // in the morton codes range.

                                                // Functor that extracts from a morton code the value of the
                                                // bit at index adata.split_idx (counted from MSB).
                                                const auto bit_extractor = [&adata](std::uint64_t mcode) {
                                                    assert(adata.split_idx < 64u);
                                                    const auto mask = static_cast<std::uint64_t>(1)
                                                                      << (63u - adata.split_idx);
                                                    return static_cast<unsigned>((mcode & mask) != 0u);
                                                };

                                                // View that transforms the morton codes in the range [mcodes_begin,
                                                // mcodes_end) into the value (0 or 1) of the bit at index
                                                // adata.split_idx (counted from MSB).
                                                const auto nth_bit_view
                                                    = std::ranges::subrange(mcodes_begin, mcodes_end)
                                                      | std::views::transform(bit_extractor);
                                                assert(std::ranges::is_sorted(nth_bit_view));

                                                // Find out where, in the morton codes range, the bit at index
                                                // adata.split_idx switches from 0 to 1.
                                                split_ptr = std::ranges::lower_bound(nth_bit_view, 1u).base();

                                                while (split_ptr == mcodes_begin || split_ptr == mcodes_end) {
                                                    // There is no bit flip at the current split index.
                                                    // We will try the next split index.
                                                    ++adata.split_idx;

                                                    if (adata.split_idx == 64u) {
                                                        // No more bit indices are available.
                                                        // This will be a leaf node containing more than 1 object.
                                                        is_leaf_node = true;

                                                        break;
                                                    }

                                                    assert(std::ranges::is_sorted(nth_bit_view));
                                                    split_ptr = std::ranges::lower_bound(nth_bit_view, 1u).base();
                                                }
                                            } else {
                                                // Node with either:
                                                //
                                                // - a single object, or
                                                // - split_idx == 64.
                                                //
                                                // The latter means that the node resulted
                                                // from splitting a node whose objects'
                                                // morton codes differred at the least significant
                                                // bit. This also implies that all the objects
                                                // in the node have the same morton code (this is checked
                                                // in the BVH verification function).
                                                // In either case, we cannot split any further
                                                // and the node is a leaf.
                                                assert(cur_node.end - cur_node.begin == 1u || adata.split_idx == 64u);
                                                is_leaf_node = true;
                                            }

                                            // Fill in the properties in l_buffer.
                                            if (is_leaf_node) {
                                                // A leaf node has no children.
                                                l_buffer[node_idx - n_begin].nc = 0;
                                                l_buffer[node_idx - n_begin].nolc = 0;

                                                // Update the leaf nodes counter.
                                                ++loc_n_leaf_nodes;

                                                // NOTE: check that the initial value of the AABB
                                                // was properly set.
                                                assert(cur_node.lb == detail::default_lb);
                                                assert(cur_node.ub == detail::default_ub);

                                                // Compute the AABB for this leaf node.
                                                for (auto obj_idx = cur_node.begin; obj_idx != cur_node.end;
                                                     ++obj_idx) {
                                                    // NOTE: min/max is fine here, we already checked
                                                    // that all AABBs are finite.
                                                    for (auto i = 0u; i < 4u; ++i) {
                                                        cur_node.lb[i] = std::min(cur_node.lb[i],
                                                                                  srt_aabbs(cd_idx, obj_idx, 0, i));
                                                        cur_node.ub[i] = std::max(cur_node.ub[i],
                                                                                  srt_aabbs(cd_idx, obj_idx, 1, i));
                                                    }
                                                }
                                            } else {
                                                assert(split_ptr != nullptr);

                                                // An internal node has 2 children.
                                                l_buffer[node_idx - n_begin].nc = 2;
                                                // NOTE: if we are here, it means that is_leaf_node is false,
                                                // which implies that split_ptr was written to at least once.
                                                // NOTE: the subtraction does not overflow, as we guaranteed
                                                // earlier that std::ptrdiff_t can represent tot_nobjs.
                                                l_buffer[node_idx - n_begin].nolc
                                                    = boost::numeric_cast<std::uint32_t>(split_ptr - mcodes_begin);
                                            }
                                        }

                                        // Update the counter for the total number of leaf nodes in the node range.
                                        n_leaf_nodes_atm.fetch_add(loc_n_leaf_nodes, std::memory_order_relaxed);
                                    });

                                // Load the total number of leaf nodes at the current level
                                // from n_leaf_nodes_atm.
                                const auto n_leaf_nodes = n_leaf_nodes_atm.load();

                                // Number of nodes at the next level. This is, at most, cur_n_nodes, but we have to
                                // subtract n_leaf_nodes * 2 because leaf nodes have no children.
                                auto nn_next_level = static_cast<std::uint32_t>(safe_u32_t(cur_n_nodes) * 2u);
                                assert(n_leaf_nodes * 2u <= nn_next_level);
                                nn_next_level -= n_leaf_nodes * 2u;

                                // Step 2: prepare the tree for the new children nodes in the next level. This will add
                                // new nodes at the end of the tree containing indeterminate values. The properties of
                                // these new nodes will be set up in step 4.
                                tree.resize(boost::safe_numerics::safe<decltype(tree.size())>(cur_tree_size)
                                            + nn_next_level);
                                aux_data.resize(boost::safe_numerics::safe<decltype(aux_data.size())>(cur_tree_size)
                                                + nn_next_level);

                                // Step 3: prefix sum over the number of children for each
                                // node at the current level.
                                // NOTE: the prefix sum cannot overflow because we checked earlier
                                // that we could represent via std::uint32_t the maximum possible
                                // number of nodes in the next level.
                                oneapi::tbb::parallel_scan(
                                    oneapi::tbb::blocked_range<decltype(l_buffer.size())>(0, l_buffer.size()),
                                    static_cast<std::uint32_t>(0),
                                    [&l_buffer](const auto &r, auto sum, bool is_final_scan) {
                                        auto temp = sum;

                                        for (auto i = r.begin(); i < r.end(); ++i) {
                                            temp = temp + l_buffer[i].nc;

                                            if (is_final_scan) {
                                                l_buffer[i].ps = temp;
                                            }
                                        }

                                        return temp;
                                    },
                                    std::plus<>{});

                                // Step 4: finalise the nodes at the current filling in the children pointers,
                                // and perform the initial setup of the children nodes that were
                                // added in step 2.
                                oneapi::tbb::parallel_for(
                                    oneapi::tbb::blocked_range(n_begin, n_end),
                                    [n_begin, cur_n_nodes, &tree, &l_buffer, &aux_data,
                                     cur_tree_size](const auto &node_range) {
                                        for (auto node_idx = node_range.begin(); node_idx != node_range.end();
                                             ++node_idx) {
                                            assert(node_idx - n_begin < cur_n_nodes);

                                            // Fetch the current node data.
                                            auto &cur_node = tree[node_idx];
                                            auto &adata = aux_data[node_idx];

                                            // Fetch the number of children.
                                            const auto nc = l_buffer[node_idx - n_begin].nc;

                                            // Set nn_level. This needs to be done regardless of whether the
                                            // node is internal or a leaf.
                                            adata.nn_level = cur_n_nodes;

                                            if (nc == 0u) {
                                                // NOTE: no need for further finalisation of leaf nodes.
                                                // Ensure that the AABB was correctly set up.
                                                assert(cur_node.lb != detail::default_lb);
                                                assert(cur_node.ub != detail::default_ub);
                                            } else {
                                                // Internal node.

                                                // Fetch the number of objects in the left child.
                                                const auto lsize = l_buffer[node_idx - n_begin].nolc;

                                                // Compute the index in the tree into which the left child will
                                                // be stored.
                                                // NOTE: this computation is safe because we checked earlier
                                                // that cur_tree_size + nn_next_level can be computed safely.
                                                const auto lc_idx
                                                    = cur_tree_size + l_buffer[node_idx - n_begin].ps - 2u;
                                                assert(lc_idx >= cur_tree_size);
                                                assert(lc_idx < tree.size());
                                                assert(lc_idx + 1u > cur_tree_size);
                                                assert(lc_idx + 1u < tree.size());

                                                // Assign the children indices for the current node.
                                                cur_node.left = boost::numeric_cast<std::int32_t>(lc_idx);
                                                cur_node.right = boost::numeric_cast<std::int32_t>(lc_idx + 1u);

                                                // Set up the children's initial properties.
                                                auto &lc = tree[lc_idx];
                                                auto &lc_adata = aux_data[lc_idx];
                                                auto &rc = tree[lc_idx + 1u];
                                                auto &rc_adata = aux_data[lc_idx + 1u];

                                                // NOTE: the children are set up initially in the same
                                                // way as the root node was: as leaf nodes, with nn_level
                                                // initially set to zero and split_idx initially set
                                                // to the split_idx of the parent + 1.
                                                lc.begin = cur_node.begin;
                                                // NOTE: the computation is safe
                                                // because we know we can represent the
                                                // total number of objects as a std::uint32_t.
                                                lc.end = cur_node.begin + lsize;
                                                lc.left = -1;
                                                lc.right = -1;
                                                lc.lb = detail::default_lb;
                                                lc.ub = detail::default_ub;
                                                lc_adata.nn_level = 0;
                                                lc_adata.split_idx = adata.split_idx + 1u;
                                                lc_adata.parent = boost::numeric_cast<std::int32_t>(node_idx);

                                                rc.begin = cur_node.begin + lsize;
                                                rc.end = cur_node.end;
                                                rc.left = -1;
                                                rc.right = -1;
                                                rc.lb = detail::default_lb;
                                                rc.ub = detail::default_ub;
                                                rc_adata.nn_level = 0;
                                                rc_adata.split_idx = adata.split_idx + 1u;
                                                rc_adata.parent = boost::numeric_cast<std::int32_t>(node_idx);
                                            }
                                        }
                                    });

                                // Assign the next value for cur_n_nodes.
                                // If nn_next_level is zero, this means that
                                // all the nodes processed in this iteration
                                // were leaves, and this signals the end of the
                                // construction of the tree.
                                cur_n_nodes = nn_next_level;

                                ++n_levels;
                                n_nodes += cur_n_nodes;
                            }

                            // Perform a backwards pass on the tree to compute the AABBs
                            // of the internal nodes.

                            // Node index range for the last level.
                            auto n_begin = tree.size() - aux_data.back().nn_level;
                            auto n_end = tree.size();

#if !defined(NDEBUG)
                            // Double check that all nodes in the last level are
                            // indeed leaves.
                            assert(std::ranges::all_of(
                                tree.data() + n_begin, tree.data() + n_end,
                                [](const auto &cur_node) { return cur_node.left == -1 && cur_node.right == -1; }));

#endif

                            // NOTE: because the AABBs for the leaf nodes were already computed,
                            // we can skip the AABB computation for the nodes in the last level,
                            // which are all guaranteed to be leaf nodes.
                            // NOTE: if n_begin == 0u, it means the tree consists
                            // only of the root node, which is itself a leaf.
                            if (n_begin == 0u) {
                                assert(n_end == 1u);
                            } else {
                                // Compute the range of the penultimate level.
                                auto new_n_end = n_begin;
                                n_begin -= aux_data[n_begin - 1u].nn_level;
                                n_end = new_n_end;

                                while (true) {
                                    oneapi::tbb::parallel_for(
                                        oneapi::tbb::blocked_range(n_begin, n_end), [&tree](const auto &node_range) {
                                            for (auto node_idx = node_range.begin(); node_idx != node_range.end();
                                                 ++node_idx) {
                                                auto &cur_node = tree[node_idx];

                                                if (cur_node.left == -1) {
                                                    // Leaf node, the bounding box was computed earlier.
                                                    assert(cur_node.right == -1);
                                                } else {
                                                    // Internal node, compute its AABB from the children.
                                                    auto &lc = tree[static_cast<std::uint32_t>(cur_node.left)];
                                                    auto &rc = tree[static_cast<std::uint32_t>(cur_node.right)];

                                                    for (auto j = 0u; j < 4u; ++j) {
                                                        // NOTE: min/max is fine here, we already checked
                                                        // that all AABBs are finite.
                                                        cur_node.lb[j] = std::min(lc.lb[j], rc.lb[j]);
                                                        cur_node.ub[j] = std::max(lc.ub[j], rc.ub[j]);
                                                    }
                                                }
                                            }
                                        });

                                    if (n_begin == 0u) {
                                        // We reached the root node, break out.
                                        assert(n_end == 1u);
                                        break;
                                    } else {
                                        // Compute the range of the previous level.
                                        new_n_end = n_begin;
                                        n_begin -= aux_data[n_begin - 1u].nn_level;
                                        n_end = new_n_end;
                                    }
                                }
                            }

#if !defined(NDEBUG)
                            // Verify the tree.
                            detail::verify_bvh_tree(cd_idx, srt_mcodes, tree, aux_data, srt_aabbs, nobjs);
#endif

                            // Move the tree into the future.
                            tree_promises[cd_idx].set_value(std::move(tree));
                        });
                    }
                });

            start_cd_step_idx = end_cd_step_idx;
        }
        // LCOV_EXCL_START
    } catch (...) {
        // Request a stop on the writer thread.
        stop_writing.store(true);

        // Wait for it to actually stop.
        // NOTE: we use wait() here, because, if the writer thread
        // also threw, get() would throw the exception here. We are
        // not interested in reporting that, as the exception from the numerical
        // integration is likely more interesting.
        // NOTE: in principle wait() could also raise platform-specific exceptions.
        writer_future.wait();

        // Re-throw.
        throw;
    }
    // LCOV_EXCL_STOP

    // Wait for the writer thread to finish.
    // NOTE: get() will throw any exception that might have been
    // raised in the writer thread.
    writer_future.get();

    // Close the bvh file.
    bvh_file.close();

    // Mark it as read-only.
    detail::mark_file_read_only(bvh_file_path);

    // Return the offsets vector.
    return tree_offsets;
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
