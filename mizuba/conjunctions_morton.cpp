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
#include "detail/fmv_attributes.hpp"
#include "half.hpp"
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

// Compute the morton codes for all objects in each conjunction step, and sort the aabbs
// data according to the morton codes.
//
// pj is the polyjectory, tmp_dir_path the temporary dir storing all conjunction data,
// n_cd_steps the number of conjunction steps.
//
// NOTE: if an object has no trajectory data during a conjunction step, its morton code will
// be set to -1, and in the morton sorting it will be placed at the tail end (i.e., even past any object
// with available trajectory data that may have a morton code of -1).
void conjunctions::morton_encode_sort(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                                      std::size_t n_cd_steps) const
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Cache the total number of objects.
    const auto nobjs = pj.get_nobjs();

    // Open the aabbs file and fetch a span to it.
    boost::iostreams::mapped_file_source aabbs_file((tmp_dir_path / "aabbs").string());
    const auto *aabbs_base_ptr = reinterpret_cast<const float16_t *>(aabbs_file.data());
    assert(boost::alignment::is_aligned(aabbs_base_ptr, alignof(float16_t)));
    const aabbs_span_t aabbs{aabbs_base_ptr, n_cd_steps, nobjs + 1u};

    // Prepare the sorted aabbs file and open it for pwriting.
    detail::create_sized_file(tmp_dir_path / "srt_aabbs", aabbs_file.size());
    detail::file_pwrite srt_aabbs_file(tmp_dir_path / "srt_aabbs");

    // Prepare the morton codes file and open it for pwriting.
    detail::create_sized_file(tmp_dir_path / "mcodes", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint64_t));
    detail::file_pwrite mcodes_file(tmp_dir_path / "mcodes");

    // Prepare the sorted morton codes file and open it for pwriting.
    detail::create_sized_file(tmp_dir_path / "srt_mcodes", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint64_t));
    detail::file_pwrite srt_mcodes_file(tmp_dir_path / "srt_mcodes");

    // Prepare the indices file and open it for pwriting.
    // NOTE: from now on, we use std::uint32_t to index into the objects, even though in principle
    // a polyjectory could contain more than 2**32-1 objects. std::uint32_t gives us ample room to run large
    // simulations if ever needed, while at the same time reducing memory utilisation wrt 64-bit indices
    // (especially in the representation of bvh trees).
    detail::create_sized_file(tmp_dir_path / "vidx", safe_size_t(nobjs) * n_cd_steps * sizeof(std::uint32_t));
    detail::file_pwrite vidx_file(tmp_dir_path / "vidx");

    // We will be using thread-specific data to store the results of intermediate
    // computations in memory, before eventually flushing them to disk.
    struct ets_data {
        // Morton codes.
        std::vector<std::uint64_t> mcodes;
        // Indices vector.
        std::vector<std::uint32_t> vidx;
        // Sorted aabbs.
        std::vector<float16_t> srt_aabbs;
        // Sorted morton codes.
        std::vector<std::uint64_t> srt_mcodes;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([nobjs]() {
        // Setup mcodes.
        std::vector<std::uint64_t> mcodes;
        mcodes.resize(boost::numeric_cast<decltype(mcodes.size())>(nobjs));

        // Setup vidx.
        std::vector<std::uint32_t> vidx;
        vidx.resize(boost::numeric_cast<decltype(vidx.size())>(nobjs));

        // Setup srt_aabbs.
        std::vector<float16_t> srt_aabbs;
        srt_aabbs.resize(boost::numeric_cast<decltype(srt_aabbs.size())>((nobjs + 1u) * 8u));

        // Setup srt_mcodes.
        std::vector<std::uint64_t> srt_mcodes;
        srt_mcodes.resize(boost::numeric_cast<decltype(srt_mcodes.size())>(nobjs));

        return ets_data{.mcodes = std::move(mcodes),
                        .vidx = std::move(vidx),
                        .srt_aabbs = std::move(srt_aabbs),
                        .srt_mcodes = std::move(srt_mcodes)};
    });

    // Run the morton encoding.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_cd_steps), [&ets, nobjs, aabbs,
                                                                                       &srt_aabbs_file, &mcodes_file,
                                                                                       &srt_mcodes_file, &vidx_file](
                                                                                          const auto &cd_range) {
        for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
            // Fetch the global AABB for this conjunction step.
            const auto *glb = &aabbs(cd_idx, nobjs, 0, 0);
            const auto *gub = &aabbs(cd_idx, nobjs, 1, 0);

            // Fetch the thread-local data.
            auto &[local_mcodes, local_vidx, local_srt_aabbs, local_srt_mcodes] = ets.local();

            // NOTE: isolate to avoid issues with thread-local data. See:
            // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
            oneapi::tbb::this_task_arena::isolate([nobjs, cd_idx, aabbs, glb, gub, &local_mcodes, &local_vidx,
                                                   &local_srt_aabbs, &local_srt_mcodes, &srt_aabbs_file, &mcodes_file,
                                                   &srt_mcodes_file, &vidx_file]() MIZUBA_FMV_ATTRIBUTES {
                // Create a mutable span into local_srt_aabbs.
                using mut_aabbs_span_t
                    = heyoka::mdspan<float16_t, heyoka::extents<std::size_t, std::dynamic_extent, 2, 4>>;
                mut_aabbs_span_t local_srt_aabbs_span{local_srt_aabbs.data(), nobjs + 1u};

                // Computation of the morton codes and initialisation of local_vidx.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                    [cd_idx, aabbs, glb, gub, &local_mcodes, &local_vidx](const auto &obj_range) MIZUBA_FMV_ATTRIBUTES {
                        // Temporary array to store the coordinates of the centre of the AABB.
                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
                        std::array<float16_t, 4> xyzr_ctr;

                        // NOTE: JIT optimisation opportunity here. Worth it?
                        for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                            if (detail::isfinite(aabbs(cd_idx, obj_idx, 0, 0))) {
                                // Compute the centre of the AABB.
                                for (auto i = 0u; i < 4u; ++i) {
                                    // NOTE: the computation here always results in a finite value, as the
                                    // result is a point within the AABB and we know that the AABB bounds are
                                    // finite, and we take care of avoiding overflow by first dividing by 2 and
                                    // then adding.
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
                            local_vidx[obj_idx] = boost::numeric_cast<std::uint32_t>(obj_idx);
                        }
                    });

                // Sort the object indices in local_vidx according to the morton codes, also ensuring that
                // objects without trajectory data are placed at the very end.
                oneapi::tbb::parallel_sort(local_vidx.begin(), local_vidx.end(),
                                           [&local_mcodes, cd_idx, aabbs](auto idx1, auto idx2) MIZUBA_FMV_ATTRIBUTES {
                                               if (detail::isinf(aabbs(cd_idx, idx1, 0, 0))) {
                                                   // The first object has no trajectory data, it cannot
                                                   // be less than any other object.
                                                   assert(local_mcodes[idx1] == static_cast<std::uint64_t>(-1));
                                                   return false;
                                               }

                                               if (detail::isinf(aabbs(cd_idx, idx2, 0, 0))) {
                                                   // The first object has trajectory data, while the
                                                   // second one does not.
                                                   assert(local_mcodes[idx2] == static_cast<std::uint64_t>(-1));
                                                   return true;
                                               }

                                               // Both objects have trajectory data, compare the codes.
                                               return local_mcodes[idx1] < local_mcodes[idx2];
                                           });

                // Apply the indirect sorting defined in local_vidx to the local aabbs and mcodes.
                // NOTE: more parallelism can be extracted here in principle, but performance
                // is bottlenecked by RAM speed anyway. Perhaps revisit on machines
                // with larger core counts during performance tuning.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
                    [cd_idx, &local_mcodes, &local_vidx, aabbs, local_srt_aabbs_span,
                     &local_srt_mcodes](const auto &obj_range) MIZUBA_FMV_ATTRIBUTES {
                        for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                            const auto sorted_idx = local_vidx[obj_idx];

                            // Write the sorted aabb.
                            for (auto i = 0u; i < 4u; ++i) {
                                local_srt_aabbs_span(obj_idx, 0, i) = aabbs(cd_idx, sorted_idx, 0, i);
                                local_srt_aabbs_span(obj_idx, 1, i) = aabbs(cd_idx, sorted_idx, 1, i);
                            }

                            // Write the sorted morton codes.
                            local_srt_mcodes[obj_idx] = local_mcodes[sorted_idx];
                        }
                    });

                // Write the global aabb into local_srt_aabbs_span.
                for (auto i = 0u; i < 4u; ++i) {
                    local_srt_aabbs_span(nobjs, 0, i) = glb[i];
                    local_srt_aabbs_span(nobjs, 1, i) = gub[i];
                }

                // Bulk write into the files.
                srt_aabbs_file.pwrite(local_srt_aabbs_span.data_handle(), (nobjs + 1u) * 8u * sizeof(float16_t),
                                      cd_idx * (nobjs + 1u) * 8u * sizeof(float16_t));
                mcodes_file.pwrite(local_mcodes.data(), nobjs * sizeof(std::uint64_t),
                                   cd_idx * nobjs * sizeof(std::uint64_t));
                srt_mcodes_file.pwrite(local_srt_mcodes.data(), nobjs * sizeof(std::uint64_t),
                                       cd_idx * nobjs * sizeof(std::uint64_t));
                vidx_file.pwrite(local_vidx.data(), nobjs * sizeof(std::uint32_t),
                                 cd_idx * nobjs * sizeof(std::uint32_t));
            });
        }
    });

    // Close all files.
    aabbs_file.close();
    srt_aabbs_file.close();
    mcodes_file.close();
    srt_mcodes_file.close();
    vidx_file.close();

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
