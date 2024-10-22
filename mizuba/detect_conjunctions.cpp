// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <ios>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "conjunctions.hpp"
#include "detail/conjunctions_jit.hpp"
#include "detail/file_utils.hpp"
#include "logging.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

std::tuple<std::vector<double>, std::vector<std::tuple<std::size_t, std::size_t>>,
           std::vector<std::tuple<std::size_t, std::size_t>>>
conjunctions::detect_conjunctions(const boost::filesystem::path &tmp_dir_path, const polyjectory &pj,
                                  std::size_t n_cd_steps, double conj_thresh, double conj_det_interval,
                                  const std::vector<bool> &conj_active)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // NOTE: we will be processing conjunction steps in chunks because... TODO
    constexpr std::size_t cd_chunk_size = 128;

    // Cache the polynomial order.
    const auto order = pj.get_poly_order();

    // NOTE: narrow-phase conjunction detection requires JIT compilation
    // of several functions.
    stopwatch sw;
    const auto &cjd = detail::get_conj_jit_data(order);
    log_trace("JIT compilation time: {}s", sw);

    // Cache the total number of objects.
    const auto nobjs = pj.get_nobjs();

    // The total number of aabbs we need to compute and store.
    // NOTE: the +1 is the global AABB to be computed for each conjunction step.
    const auto n_tot_aabbs = (safe_size_t(nobjs) + 1) * n_cd_steps;

    // Prepare the files whose size we know in advance.
    detail::create_sized_file(tmp_dir_path / "aabbs", n_tot_aabbs * sizeof(float) * 8u);
    detail::file_pwrite aabbs_file(tmp_dir_path / "aabbs");

    detail::create_sized_file(tmp_dir_path / "srt_aabbs", n_tot_aabbs * sizeof(float) * 8u);
    detail::file_pwrite srt_aabbs_file(tmp_dir_path / "srt_aabbs");

    detail::create_sized_file(tmp_dir_path / "mcodes", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint64_t));
    detail::file_pwrite mcodes_file(tmp_dir_path / "mcodes");

    detail::create_sized_file(tmp_dir_path / "srt_mcodes", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint64_t));
    detail::file_pwrite srt_mcodes_file(tmp_dir_path / "srt_mcodes");

    // NOTE: we use std::uint32_t to index into the objects, even though in principle a polyjectory could contain more
    // than 2**32-1 objects. std::uint32_t gives us ample room to run large simulations if ever needed, while at the
    // same time reducing memory utilisation wrt 64-bit indices (especially in the representation of bvh trees).
    detail::create_sized_file(tmp_dir_path / "vidx", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint32_t));
    detail::file_pwrite vidx_file(tmp_dir_path / "vidx");

    // Prepare the files whose sizes are not known in advance.
    const auto bvh_file_path = tmp_dir_path / "bvh";
    if (boost::filesystem::exists(bvh_file_path)) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot create the storage file '{}': the file exists already", bvh_file_path.string()));
        // LCOV_EXCL_STOP
    }
    std::ofstream bvh_file(bvh_file_path.string(), std::ios::binary | std::ios::out);
    bvh_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    const auto bp_file_path = tmp_dir_path / "bp";
    if (boost::filesystem::exists(bp_file_path)) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot create the storage file '{}': the file exists already", bp_file_path.string()));
        // LCOV_EXCL_STOP
    }
    std::ofstream bp_file(bp_file_path.string(), std::ios::binary | std::ios::out);
    bp_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    const auto conj_file_path = tmp_dir_path / "conjunctions";
    if (boost::filesystem::exists(conj_file_path)) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("Cannot create the storage file '{}': the file exists already", conj_file_path.string()));
        // LCOV_EXCL_STOP
    }
    std::ofstream conj_file(conj_file_path.string(), std::ios::binary | std::ios::out);
    conj_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    // The global vector of end times for the conjunction steps. This does not need synchronisation
    // because each element will be written to exactly once, at the end of the aabbs computation
    // for each conjunction step.
    std::vector<double> cd_end_times;
    cd_end_times.resize(boost::numeric_cast<decltype(cd_end_times.size())>(n_cd_steps));

    // Vector of offsets and sizes for the tree data stored in the bvh data file.
    //
    // The first element of the pair is the offset at which tree data begins,
    // the second element of the pair is the tree size. The offsets are measured in number
    // of bvh_nodes.
    //
    // This vector is filled-in by the writer thread as it writes bvh data to file.
    std::vector<std::tuple<std::size_t, std::size_t>> tree_offsets;
    tree_offsets.reserve(n_cd_steps);

    // Vector of offsets and sizes for the broad-phase conjunction detection data.
    //
    // The first element of the pair is the offset at which broad-phase data begins,
    // the second element of the pair is the total size of the broad-phase data.
    // The offsets are measured in number of aabb_collision.
    //
    // This vector is filled-in by the writer thread as it writes broad-phase data to file.
    std::vector<std::tuple<std::size_t, std::size_t>> bp_offsets;
    bp_offsets.reserve(n_cd_steps);

    // Counter for the total number of detected conjunctions.
    safe_size_t tot_n_conj = 0;

    // Bag of results which will be outputted each time a conjunction step has finished
    // processing. This data is stored in futures and written sequentially to disk by the writer thread.
    struct f_output {
        // aabbs.
        std::vector<float> aabbs;
        // Morton codes.
        std::vector<std::uint64_t> mcodes;
        // Indices vector.
        std::vector<std::uint32_t> vidx;
        // Sorted aabbs.
        std::vector<float> srt_aabbs;
        // Sorted morton codes.
        std::vector<std::uint64_t> srt_mcodes;
        // BVH tree.
        std::vector<bvh_node> bvh_tree;
        // aabbs collisions.
        std::vector<aabb_collision> bp;
        // Detected conjunctions.
        std::vector<conj> conjunctions;
    };

    // Setup the futures and promises to coordinate between parallel computations and the writer thread.
    std::vector<std::promise<f_output>> promises;
    promises.resize(boost::numeric_cast<decltype(promises.size())>(n_cd_steps));
    auto fut_view = promises | std::views::transform([](auto &p) { return p.get_future(); });
    std::vector futures(std::ranges::begin(fut_view), std::ranges::end(fut_view));

    // Flag to signal that the writer thread should stop writing.
    std::atomic<bool> stop_writing = false;

    // Launch the writer thread.
    auto writer_future = std::async(std::launch::async, [nobjs, n_cd_steps, &futures, &stop_writing, &aabbs_file,
                                                         &mcodes_file, &vidx_file, &srt_aabbs_file, &srt_mcodes_file,
                                                         &bvh_file, &tree_offsets, &bp_file, &bp_offsets, &conj_file,
                                                         &tot_n_conj]() {
        using namespace std::chrono_literals;

        // How long should we wait before checking if we should stop writing.
        const auto wait_duration = 250ms;

        // Track the offsets to build up tree_offsets and bp_offsets.
        safe_size_t cur_tree_offset = 0, cur_bp_offset = 0;

        for (std::size_t i = 0; i < n_cd_steps; ++i) {
            // Fetch the future.
            auto &fut = futures[i];

            // Wait until the future becomes available, or return if a stop is requested.
            while (fut.wait_for(wait_duration) != std::future_status::ready) {
                // LCOV_EXCL_START
                if (stop_writing) [[unlikely]] {
                    return;
                }
                // LCOV_EXCL_STOP
            }

            // Fetch the data from the future.
            auto [aabbs, mcodes, vidx, srt_aabbs, srt_mcodes, bvh_tree, bp, conjs] = fut.get();

            // Write the aabbs.
            aabbs_file.pwrite(aabbs.data(), (nobjs + 1u) * 8u * sizeof(float), i * (nobjs + 1u) * 8u * sizeof(float));

            // Write the mcodes.
            mcodes_file.pwrite(mcodes.data(), nobjs * sizeof(std::uint64_t), i * nobjs * sizeof(std::uint64_t));

            // Write the indices.
            vidx_file.pwrite(vidx.data(), nobjs * sizeof(std::uint32_t), i * nobjs * sizeof(std::uint32_t));

            // Write the sorted aabbs.
            srt_aabbs_file.pwrite(srt_aabbs.data(), (nobjs + 1u) * 8u * sizeof(float),
                                  i * (nobjs + 1u) * 8u * sizeof(float));

            // Write the sorted mcodes.
            srt_mcodes_file.pwrite(srt_mcodes.data(), nobjs * sizeof(std::uint64_t), i * nobjs * sizeof(std::uint64_t));

            // Fetch the tree size.
            const auto tree_size = bvh_tree.size();

            // Write the tree.
            bvh_file.write(reinterpret_cast<const char *>(bvh_tree.data()),
                           boost::safe_numerics::safe<std::streamsize>(tree_size) * sizeof(bvh_node));

            // Update tree_offsets and cur_tree_offset.
            tree_offsets.emplace_back(cur_tree_offset, boost::numeric_cast<std::size_t>(tree_size));
            cur_tree_offset += tree_size;

            // Fetch the broad-phase data size.
            const auto bp_size = bp.size();

            // Write the broad-phase data.
            bp_file.write(reinterpret_cast<const char *>(bp.data()),
                          boost::safe_numerics::safe<std::streamsize>(bp_size) * sizeof(aabb_collision));

            // Update bp_offsets and cur_bp_offset.
            bp_offsets.emplace_back(cur_bp_offset, boost::numeric_cast<std::size_t>(bp_size));
            cur_bp_offset += bp_size;

            // Fetch the detected conjunctions data  size.
            const auto conjs_size = conjs.size();

            // Write the conjunctions data.
            conj_file.write(reinterpret_cast<const char *>(conjs.data()),
                            boost::safe_numerics::safe<std::streamsize>(conjs_size) * sizeof(conj));

            // Update tot_n_conj.
            tot_n_conj += conjs_size;
        }
    });

    try {
        // TODO comment on this.
        struct ets_data {
            // aabbs.
            std::vector<float> aabbs;
            // Morton codes.
            std::vector<std::uint64_t> mcodes;
            // Indices vector.
            std::vector<std::uint32_t> vidx;
            // Sorted aabbs.
            std::vector<float> srt_aabbs;
            // Sorted morton codes.
            std::vector<std::uint64_t> srt_mcodes;
            // BVH tree.
            std::vector<bvh_node> bvh_tree;
            // Auxiliary node data.
            // NOTE: the size of this vector will be kept
            // in sync with the size of tree.
            std::vector<bvh_aux_node_data> bvh_aux_data;
            // Data used in the level-by-level construction of the treee.
            std::vector<bvh_level_data> bvh_l_buffer;
            // Local lists of detected broad-phase AABBs collisions,
            // one for each object.
            std::vector<small_vec<aabb_collision>> bp_collisions;
            // Local stacks for the BVH tree traversal during broad-phase
            // conjunction detection, one for each object.
            std::vector<std::vector<std::int32_t>> bp_stacks;
            // Narrow-phase per-object conjunction detection data.
            std::vector<np_data> npd_vec;
        };
        using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                              oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
        ets_t ets([nobjs, order]() {
            // Setup aabbs.
            std::vector<float> aabbs;
            aabbs.resize(boost::numeric_cast<decltype(aabbs.size())>((nobjs + 1u) * 8u));

            // Setup mcodes.
            std::vector<std::uint64_t> mcodes;
            mcodes.resize(boost::numeric_cast<decltype(mcodes.size())>(nobjs));

            // Setup vidx.
            std::vector<std::uint32_t> vidx;
            vidx.resize(boost::numeric_cast<decltype(vidx.size())>(nobjs));

            // Setup srt_aabbs.
            std::vector<float> srt_aabbs;
            srt_aabbs.resize(boost::numeric_cast<decltype(srt_aabbs.size())>((nobjs + 1u) * 8u));

            // Setup srt_mcodes.
            std::vector<std::uint64_t> srt_mcodes;
            srt_mcodes.resize(boost::numeric_cast<decltype(srt_mcodes.size())>(nobjs));

            // Setup bp_collisions.
            std::vector<small_vec<aabb_collision>> bp_collisions;
            bp_collisions.resize(boost::numeric_cast<decltype(bp_collisions.size())>(nobjs));

            // Setup bp_stacks.
            std::vector<std::vector<std::int32_t>> bp_stacks;
            bp_stacks.resize(boost::numeric_cast<decltype(bp_stacks.size())>(nobjs));

            // Create npd_vec and set it up.
            std::vector<np_data> npd_vec;
            npd_vec.resize(boost::numeric_cast<decltype(npd_vec.size())>(nobjs));

            for (auto &npd : npd_vec) {
                // Prepare pbuffers.
                for (auto &v : npd.pbuffers) {
                    v.resize(boost::numeric_cast<decltype(v.size())>(order + 1u));
                }

                // Prepare diff_input.
                npd.diff_input.resize((order + 1u) * safe_size_t(6));
            }

            return ets_data{.aabbs = std::move(aabbs),
                            .mcodes = std::move(mcodes),
                            .vidx = std::move(vidx),
                            .srt_aabbs = std::move(srt_aabbs),
                            .srt_mcodes = std::move(srt_mcodes),
                            // NOTE: we do not know at this point the sizes
                            // required for the BVH data. These buffers will be
                            // resized approriately as needed.
                            .bvh_tree = {},
                            .bvh_aux_data = {},
                            .bvh_l_buffer = {},
                            .bp_collisions = std::move(bp_collisions),
                            .bp_stacks = std::move(bp_stacks),
                            .npd_vec = std::move(npd_vec)};
        });

        // Iterate in chunks over the conjunction steps.
        for (std::size_t start_cd_step_idx = 0; start_cd_step_idx != n_cd_steps;) {
            const auto n_rem_cd_steps = n_cd_steps - start_cd_step_idx;
            const auto end_cd_step_idx
                = start_cd_step_idx + (n_rem_cd_steps < cd_chunk_size ? n_rem_cd_steps : cd_chunk_size);

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<std::size_t>(start_cd_step_idx, end_cd_step_idx), [&](const auto &cd_range) {
                    // Fetch the thread-local data.
                    auto &[cd_aabbs, cd_mcodes, cd_vidx, cd_srt_aabbs, cd_srt_mcodes, cd_bvh_tree, cd_bvh_aux_data,
                           cd_bvh_l_buffer, cd_bp_collisions, cd_bp_stacks, cd_npd_vec]
                        = ets.local();

                    // NOTE: isolate to avoid issues with thread-local data. See:
                    // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
                    oneapi::tbb::this_task_arena::isolate([&]() {
                        for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
                            // Compute the aabbs for all objects and store them in cd_aabbs.
                            detect_conjunctions_aabbs(cd_idx, cd_aabbs, pj, conj_thresh, conj_det_interval, n_cd_steps,
                                                      cd_end_times);

                            // Compute the morton codes for all objects and sort the aabbs data according to
                            // the morton codes. The morton codes will be written to cd_mcodes, the object
                            // ordering will be written to cd_vidx, the sorted aabbs will be written to cd_srt_aabbs,
                            // the sorted morton codes will be written to cd_srt_mcodes.
                            detect_conjunctions_morton(cd_mcodes, cd_vidx, cd_srt_aabbs, cd_srt_mcodes, cd_aabbs, pj);

                            // Construct the bvh tree, which will be written to cd_bvh_tree.
                            detect_conjunctions_bvh(cd_bvh_tree, cd_bvh_aux_data, cd_bvh_l_buffer, cd_srt_aabbs,
                                                    cd_srt_mcodes);

                            // Detect aabbs collisions.
                            auto bp_coll
                                = detect_conjunctions_broad_phase(cd_bp_collisions, cd_bp_stacks, cd_bvh_tree, cd_vidx,
                                                                  conj_active, cd_srt_aabbs, cd_aabbs);

                            // Detect conjunctions.
                            auto conjs = detect_conjunctions_narrow_phase(cd_npd_vec, cd_idx, pj, cd_bp_collisions, cjd,
                                                                          conj_thresh, conj_det_interval, n_cd_steps);

                            // Prepare the value for the future.
                            f_output fval{.aabbs = cd_aabbs,
                                          .mcodes = cd_mcodes,
                                          .vidx = cd_vidx,
                                          .srt_aabbs = cd_srt_aabbs,
                                          .srt_mcodes = cd_srt_mcodes,
                                          .bvh_tree = cd_bvh_tree,
                                          // NOTE: bp_coll and conjs are created ex-novo
                                          // and destroyed at each conjunction step, thus we move them in.
                                          // The other data persists and is re-used across several conjunction
                                          // steps, thus it is merely copied.
                                          .bp = std::move(bp_coll),
                                          .conjunctions = std::move(conjs)};

                            // Move the data into the future.
                            promises[cd_idx].set_value(std::move(fval));
                        }
                    });
                });

            start_cd_step_idx = end_cd_step_idx;
        }
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

    // Wait for the writer thread to finish.
    // NOTE: get() will throw any exception that might have been
    // raised in the writer thread.
    writer_future.get();

    // If we did not detect any aabb collision, we need to write
    // something into bp_file, otherwise we cannot memory-map it.
    if (bp_offsets.back() == std::make_tuple(0u, 0u)) {
        // Insert a value-inited aabb_collision.
        const aabb_collision empty{};
        bp_file.write(reinterpret_cast<const char *>(&empty), sizeof(aabb_collision));
    }

    // Same goes for conjunctions.
    if (tot_n_conj == 0u) {
        // NOTE: use a single-byte file to signal the lack of conjunctions.
        // Make super extra sure this cannot me mistaken for a single conjunction.
        // This is important because, while for bp data we can deduce the lack
        // of data from bp_offsets, we can deduce the lack of conjunctions only
        // from the file size.
        static_assert(sizeof(conj) > 1u);

        const char empty{};
        conj_file.write(&empty, 1);
    }

    // Close all files.
    aabbs_file.close();
    srt_aabbs_file.close();
    mcodes_file.close();
    srt_mcodes_file.close();
    vidx_file.close();
    bvh_file.close();
    bp_file.close();
    conj_file.close();

    // Mark them as read-only.
    detail::mark_file_read_only(tmp_dir_path / "aabbs");
    detail::mark_file_read_only(tmp_dir_path / "srt_aabbs");
    detail::mark_file_read_only(tmp_dir_path / "mcodes");
    detail::mark_file_read_only(tmp_dir_path / "srt_mcodes");
    detail::mark_file_read_only(tmp_dir_path / "vidx");
    detail::mark_file_read_only(tmp_dir_path / "bvh");
    detail::mark_file_read_only(tmp_dir_path / "bp");
    detail::mark_file_read_only(tmp_dir_path / "conjunctions");

    return std::make_tuple(std::move(cd_end_times), std::move(tree_offsets), std::move(bp_offsets));
}

} // namespace mizuba
