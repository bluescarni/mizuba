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
#include <span>
#include <utility>

#include <boost/align.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/task_arena.h>

#include <heyoka/mdspan.hpp>

#include "conjunctions.hpp"
#include "detail/file_utils.hpp"
#include "polyjectory.hpp"

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#endif

#include "detail/mortonND_LUT.h"

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

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

// Quantise a value x in [min, max) into one of 2**16
// discrete slots, numbered from 0 to 2**16 - 1.
// NOTE: before invoking this function we must ensure that:
// - all args are finite,
// - max > min,
// - max - min gives a finite result.
// We don't check via assertion that x is in [min, max), because
// conceivably in some corner cases FP computations necessary to
// calculate x outside this function could lead to a value slightly outside
// the allowed range. In such case, we will clamp the result.
std::uint64_t disc_single_coord(float x, float min, float max)
{
    assert(std::isfinite(min));
    assert(std::isfinite(max));
    assert(std::isfinite(x));
    assert(max > min);
    assert(std::isfinite(max - min));

    // Determine the interval size.
    const auto isize = max - min;

    // Translate and rescale x so that min becomes zero
    // and max becomes 1.
    auto rx = (x - min) / isize;

    // Ensure that rx is not negative.
    rx = rx >= 0.f ? rx : 0.f;

    // Rescale by 2**16.
    rx *= static_cast<std::uint64_t>(1) << 16;

    // Cast back to integer.
    const auto retval = static_cast<std::uint64_t>(rx);

    // Make sure to clamp it before returning, in case
    // somehow FP arithmetic makes it spill outside
    // the bound.
    // NOTE: std::min is safe with integral types.
    return std::min(retval, static_cast<std::uint64_t>((static_cast<std::uint64_t>(1) << 16) - 1u));
}

// Construct the morton encoder.
constexpr auto morton_enc = mortonnd::MortonNDLutEncoder<4, 16, 8>();

} // namespace

} // namespace detail

void conjunctions::morton_encode_sort_parallel(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                                               std::size_t n_cd_steps) const
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Cache the total number of objects.
    const auto nobjs = pj.get_nobjs();

    // Open the aabbs file and fetch a span to it.
    boost::iostreams::mapped_file_source file_aabbs((tmp_dir_path / "aabbs").string());
    const auto *aabbs_base_ptr = reinterpret_cast<const float *>(file_aabbs.data());
    assert(boost::alignment::is_aligned(aabbs_base_ptr, alignof(float)));
    const aabbs_span_t aabbs{aabbs_base_ptr, n_cd_steps, nobjs + 1u};

    // Construct the sorted aabbs file and fetch a span to it.
    using mut_aabbs_span_t
        = heyoka::mdspan<float, heyoka::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, 2, 4>>;
    detail::create_sized_file(tmp_dir_path / "srt_aabbs", boost::numeric_cast<std::size_t>(file_aabbs.size()));
    boost::iostreams::mapped_file_sink file_srt_aabbs((tmp_dir_path / "srt_aabbs").string());
    auto *srt_aabbs_base_ptr = reinterpret_cast<float *>(file_srt_aabbs.data());
    assert(boost::alignment::is_aligned(srt_aabbs_base_ptr, alignof(float)));
    const mut_aabbs_span_t srt_aabbs{srt_aabbs_base_ptr, n_cd_steps, nobjs + 1u};

    // Construct the morton codes file and fetch a span to it.
    using mut_mcodes_span_t = heyoka::mdspan<std::uint64_t, heyoka::dextents<std::size_t, 2>>;
    detail::create_sized_file(tmp_dir_path / "mcodes", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint64_t));
    boost::iostreams::mapped_file_sink file_mcodes((tmp_dir_path / "mcodes").string());
    auto *mcodes_base_ptr = reinterpret_cast<std::uint64_t *>(file_mcodes.data());
    assert(boost::alignment::is_aligned(mcodes_base_ptr, alignof(std::uint64_t)));
    const mut_mcodes_span_t mcodes{mcodes_base_ptr, n_cd_steps, nobjs};

    // Construct the sorted morton codes file and fetch a span to it.
    detail::create_sized_file(tmp_dir_path / "srt_mcodes", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint64_t));
    boost::iostreams::mapped_file_sink file_srt_mcodes((tmp_dir_path / "srt_mcodes").string());
    auto *srt_mcodes_base_ptr = reinterpret_cast<std::uint64_t *>(file_srt_mcodes.data());
    assert(boost::alignment::is_aligned(srt_mcodes_base_ptr, alignof(std::uint64_t)));
    const mut_mcodes_span_t srt_mcodes{srt_mcodes_base_ptr, n_cd_steps, nobjs};

    // Construct the indices file and fetch a span to it.
    using mut_vidx_span_t = heyoka::mdspan<std::size_t, heyoka::dextents<std::size_t, 2>>;
    detail::create_sized_file(tmp_dir_path / "vidx", safe_size_t(nobjs) * n_cd_steps * sizeof(std::size_t));
    boost::iostreams::mapped_file_sink file_vidx((tmp_dir_path / "vidx").string());
    auto *vidx_base_ptr = reinterpret_cast<std::size_t *>(file_vidx.data());
    assert(boost::alignment::is_aligned(vidx_base_ptr, alignof(std::size_t)));
    const mut_vidx_span_t vidx{vidx_base_ptr, n_cd_steps, nobjs};

    // We will be using thread-specific data to store the results of intermediate
    // computations in memory, before eventually flushing them to disk.
    struct ets_data {
        // Morton codes.
        std::vector<std::uint64_t> mcodes;
        // Indices vector.
        std::vector<std::size_t> vidx;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([nobjs]() {
        // Setup mcodes.
        std::vector<std::uint64_t> mcodes;
        mcodes.resize(boost::numeric_cast<decltype(mcodes.size())>(nobjs));

        // Setup vidx.
        std::vector<std::size_t> vidx;
        vidx.resize(boost::numeric_cast<decltype(vidx.size())>(nobjs));

        return ets_data{.mcodes = std::move(mcodes), .vidx = std::move(vidx)};
    });

    // Run the morton encoding.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps), [&ets, nobjs, aabbs, srt_aabbs,
                                                                                       mcodes, srt_mcodes,
                                                                                       vidx](const auto &cd_range) {
        for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
            // Fetch the global AABB for this conjunction step.
            const auto *glb = &aabbs(cd_idx, nobjs, 0, 0);
            const auto *gub = &aabbs(cd_idx, nobjs, 1, 0);

            // Fetch the thread-local data.
            auto &[local_mcodes, local_vidx] = ets.local();

            // NOTE: isolate to avoid issues with thread-local data. See:
            // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
            oneapi::tbb::this_task_arena::isolate([nobjs, cd_idx, aabbs, glb, gub, &local_mcodes, &local_vidx,
                                                   srt_aabbs, mcodes, srt_mcodes, vidx]() {
                // Computation of the morton codes and initialisation of local_vidx.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                    [cd_idx, aabbs, glb, gub, &local_mcodes, &local_vidx](const auto &obj_range) {
                        // Temporary array to store the coordinates of the centre of the AABB.
                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
                        std::array<float, 4> xyzr_ctr;

                        // NOTE: JIT optimisation opportunity here. Worth it?
                        for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                            if (std::isfinite(aabbs(cd_idx, obj_idx, 0, 0))) {
                                // Compute the centre of the AABB.
                                for (auto i = 0u; i < 4u; ++i) {
                                    // NOTE: the computation here always results in a finite value, as the result
                                    // is a point within the AABB and we know that the AABB bounds are finite, and
                                    // we take care of avoiding overflow by first dividing by 2 and then adding.
                                    xyzr_ctr[i] = aabbs(cd_idx, obj_idx, 0, i) / 2 + aabbs(cd_idx, obj_idx, 1, i) / 2;
                                }

                                // Discretize the coordinates of the centre of the AABB.
                                const auto n0 = detail::disc_single_coord(xyzr_ctr[0], glb[0], gub[0]);
                                const auto n1 = detail::disc_single_coord(xyzr_ctr[1], glb[1], gub[1]);
                                const auto n2 = detail::disc_single_coord(xyzr_ctr[2], glb[2], gub[2]);
                                const auto n3 = detail::disc_single_coord(xyzr_ctr[3], glb[3], gub[3]);

                                // Compute and store the morton code.
                                local_mcodes[obj_idx] = detail::morton_enc.Encode(n0, n1, n2, n3);
                            } else {
                                // The AABB for the current object is not finite. This means
                                // that we do not have trajectory data for the current object
                                // in the current conjunction step. Set the morton code to all ones.
                                local_mcodes[obj_idx] = static_cast<std::uint64_t>(-1);
                            }

                            // Init local_vidx.
                            local_vidx[obj_idx] = obj_idx;
                        }
                    });

                // Indirect sorting of local_vidx according to the morton codes.
                oneapi::tbb::parallel_sort(local_vidx.begin(), local_vidx.end(), [&local_mcodes](auto idx1, auto idx2) {
                    return local_mcodes[idx1] < local_mcodes[idx2];
                });

                // Apply the indirect sorting defined in local_vidx to aabbs and mcodes,
                // writing the sorted data to the the srt_* counterparts on disk.
                // Also, write local_vidx and local_mcodes to disk.
                // NOTE: more parallelism can be extracted here in principle, but performance
                // is bottlenecked by RAM speed anyway. Perhaps revisit on machines
                // with larger core counts during performance tuning.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                    [cd_idx, &local_mcodes, &local_vidx, aabbs, srt_aabbs, mcodes, srt_mcodes,
                     vidx](const auto &obj_range) {
                        for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                            const auto sorted_idx = local_vidx[obj_idx];

                            // Write the sorted aabb.
                            for (auto i = 0u; i < 4u; ++i) {
                                srt_aabbs(cd_idx, obj_idx, 0, i) = aabbs(cd_idx, sorted_idx, 0, i);
                                srt_aabbs(cd_idx, obj_idx, 1, i) = aabbs(cd_idx, sorted_idx, 1, i);
                            }

                            // Write local_mcodes into mcodes and srt_mcodes.
                            mcodes(cd_idx, obj_idx) = local_mcodes[obj_idx];
                            srt_mcodes(cd_idx, obj_idx) = local_mcodes[sorted_idx];

                            // Write local_vidx to disk.
                            vidx(cd_idx, obj_idx) = local_vidx[obj_idx];
                        }
                    });
            });

            // Copy over the global aabb to srt_aabbs.
            for (auto i = 0u; i < 4u; ++i) {
                srt_aabbs(cd_idx, nobjs, 0, i) = glb[i];
                srt_aabbs(cd_idx, nobjs, 1, i) = gub[i];
            }
        }
    });

    // Close all files.
    file_aabbs.close();
    file_srt_aabbs.close();
    file_mcodes.close();
    file_srt_mcodes.close();
    file_vidx.close();

    // Mark as read-only the files we have written to.
    detail::mark_file_read_only(tmp_dir_path / "srt_aabbs");
    detail::mark_file_read_only(tmp_dir_path / "mcodes");
    detail::mark_file_read_only(tmp_dir_path / "srt_mcodes");
    detail::mark_file_read_only(tmp_dir_path / "vidx");
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif