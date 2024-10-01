// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include <boost/align.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "conjunctions.hpp"
#include "detail/get_n_effective_objects.hpp"
#include "polyjectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

void conjunctions::broad_phase(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                               std::size_t n_cd_steps,
                               const std::vector<std::tuple<std::size_t, std::size_t>> &tree_offsets)
{
    // Cache the total number of objects.
    const auto tot_nobjs = pj.get_nobjs();

    // Fetch read-only spans to the sorted aabbs and indices vectors.
    boost::iostreams::mapped_file_source file_srt_aabbs((tmp_dir_path / "srt_aabbs").string());
    const auto *srt_aabbs_base_ptr = reinterpret_cast<const float *>(file_srt_aabbs.data());
    assert(boost::alignment::is_aligned(srt_aabbs_base_ptr, alignof(float)));
    const aabbs_span_t srt_aabbs{srt_aabbs_base_ptr, n_cd_steps, tot_nobjs + 1u};

    boost::iostreams::mapped_file_source file_srt_idx((tmp_dir_path / "vidx").string());
    const auto *srt_idx_base_ptr = reinterpret_cast<const std::size_t *>(file_srt_idx.data());
    assert(boost::alignment::is_aligned(srt_idx_base_ptr, alignof(std::size_t)));
    const srt_idx_span_t srt_idx{srt_idx_base_ptr, n_cd_steps, tot_nobjs};

    // Fetch a pointer to the tree data.
    boost::iostreams::mapped_file_source file_bvh_trees((tmp_dir_path / "bvh").string());
    const auto *bvh_trees_base_ptr = reinterpret_cast<const bvh_node *>(file_bvh_trees.data());
    assert(boost::alignment::is_aligned(bvh_trees_base_ptr, alignof(bvh_node)));

    // The global vector of detected aabbs collisions.
    // Each element in this vector will contain the list of aabbs overlaps
    // detected during a conjunction step.
    std::vector<std::vector<std::tuple<std::size_t, std::size_t>>> global_bp_coll;
    global_bp_coll.resize(boost::numeric_cast<decltype(global_bp_coll.size())>(n_cd_steps));

    // We will be using thread-specific data to store temporary results during broad-phase
    // collistion detection.
    struct ets_data {
        // Local list of detected AABBs collisions.
        std::vector<std::tuple<std::size_t, std::size_t>> bp;
        // Local stack for the BVH tree traversal.
        std::vector<std::int32_t> stack;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([]() { return ets_data{}; });

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps),
        [bvh_trees_base_ptr, &tree_offsets, &global_bp_coll, tot_nobjs, srt_aabbs](const auto &cd_range) {
            for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
                // Fetch the tree for the current conjunction step.
                const auto [tree_offset, tree_size] = tree_offsets[cd_idx];
                const auto *tree_ptr = bvh_trees_base_ptr + tree_offset;
                const tree_span_t tree{tree_ptr, tree_size};

                // Fetch the vector to store the list of aabbs overlaps
                // for the current conjunction step.
                auto &bp_cv = global_bp_coll[cd_idx];

                // Fetch the number of objects with trajectory data for the current
                // conjunction step.
                const auto nobjs = detail::get_n_effective_objects(tot_nobjs, srt_aabbs, cd_idx);
            }
        });
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
