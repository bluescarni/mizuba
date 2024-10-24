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
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>

#include <heyoka/mdspan.hpp>

#include "conjunctions.hpp"
#include "polyjectory.hpp"

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#endif

#include "detail/mortonND_LUT.h"

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

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

// Compute the morton codes for a conjunction step and sort the aabbs according to them.
//
// cd_mcodes will contain the morton codes, cd_vidx will contain the sorted indexing over the objects
// according to their morton codes, cd_srt_aabbs will contain the sorted aabbs, cd_srt_mcodes will contain
// the sorted morton codes, cd_aabbs contains the unsorted aabbs, pj is the polyjectory.
void conjunctions::detect_conjunctions_morton(std::vector<std::uint64_t> &cd_mcodes,
                                              std::vector<std::uint32_t> &cd_vidx, std::vector<float> &cd_srt_aabbs,
                                              std::vector<std::uint64_t> &cd_srt_mcodes,
                                              const std::vector<float> &cd_aabbs, const polyjectory &pj)
{
    // Cache the total number of objects.
    const auto nobjs = pj.get_nobjs();
    assert(cd_mcodes.size() == nobjs);
    assert(cd_vidx.size() == nobjs);
    assert(cd_srt_aabbs.size() == (nobjs + 1u) * 8u);
    assert(cd_srt_mcodes.size() == nobjs);
    assert(cd_aabbs.size() == (nobjs + 1u) * 8u);

    // Create a const span into cd_aabbs.
    using const_aabbs_span_t = heyoka::mdspan<const float, heyoka::extents<std::size_t, std::dynamic_extent, 2, 4>>;
    const_aabbs_span_t cd_aabbs_span{cd_aabbs.data(), nobjs + 1u};

    // Create a mutable span into cd_srt_aabbs.
    using mut_aabbs_span_t = heyoka::mdspan<float, heyoka::extents<std::size_t, std::dynamic_extent, 2, 4>>;
    mut_aabbs_span_t cd_srt_aabbs_span{cd_srt_aabbs.data(), nobjs + 1u};

    // Fetch the global AABB for this conjunction step.
    const auto *glb = &cd_aabbs_span(nobjs, 0, 0);
    const auto *gub = &cd_aabbs_span(nobjs, 1, 0);

    // Computation of the morton codes and initialisation of cd_vidx.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, nobjs), [cd_aabbs_span, glb, gub, &cd_mcodes,
                                                                                  &cd_vidx](const auto &obj_range) {
        // Temporary array to store the coordinates of the centre of the AABB.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
        std::array<float, 4> xyzr_ctr;

        // NOTE: JIT optimisation opportunity here. Worth it?
        for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
            if (std::isfinite(cd_aabbs_span(obj_idx, 0, 0))) {
                // Compute the centre of the AABB.
                for (auto i = 0u; i < 4u; ++i) {
                    // NOTE: the computation here always results in a finite value, as the
                    // result is a point within the AABB and we know that the AABB bounds are
                    // finite, and we take care of avoiding overflow by first dividing by 2
                    // and then adding.
                    xyzr_ctr[i] = cd_aabbs_span(obj_idx, 0, i) / 2 + cd_aabbs_span(obj_idx, 1, i) / 2;
                }

                // Discretize the coordinates of the centre of the AABB.
                const auto n0 = detail::disc_single_coord(xyzr_ctr[0], glb[0], gub[0]);
                const auto n1 = detail::disc_single_coord(xyzr_ctr[1], glb[1], gub[1]);
                const auto n2 = detail::disc_single_coord(xyzr_ctr[2], glb[2], gub[2]);
                const auto n3 = detail::disc_single_coord(xyzr_ctr[3], glb[3], gub[3]);

                // Compute and store the morton code.
                cd_mcodes[obj_idx] = detail::morton_enc.Encode(n0, n1, n2, n3);
            } else {
                // The AABB for the current object is not finite. This means
                // that we do not have trajectory data for the current object
                // in the current conjunction step. Set the morton code to all ones.
                cd_mcodes[obj_idx] = static_cast<std::uint64_t>(-1);
            }

            // Init cd_vidx.
            cd_vidx[obj_idx] = boost::numeric_cast<std::uint32_t>(obj_idx);
        }
    });

    // Sort the object indices in cd_vidx according to the morton codes, also ensuring that objects
    // without trajectory data are placed at the very end.
    oneapi::tbb::parallel_sort(cd_vidx.begin(), cd_vidx.end(), [&cd_mcodes, cd_aabbs_span](auto idx1, auto idx2) {
        if (std::isinf(cd_aabbs_span(idx1, 0, 0))) {
            // The first object has no trajectory data, it cannot
            // be less than any other object.
            assert(cd_mcodes[idx1] == static_cast<std::uint64_t>(-1));
            return false;
        }

        if (std::isinf(cd_aabbs_span(idx2, 0, 0))) {
            // The first object has trajectory data, while the
            // second one does not.
            assert(cd_mcodes[idx2] == static_cast<std::uint64_t>(-1));
            return true;
        }

        // Both objects have trajectory data, compare the codes.
        return cd_mcodes[idx1] < cd_mcodes[idx2];
    });

    // Apply the indirect sorting defined in cd_vidx to the aabbs and mcodes.
    // NOTE: more parallelism can be extracted here in principle, but performance
    // is bottlenecked by RAM speed anyway. Perhaps revisit on machines
    // with larger core counts during performance tuning.
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<std::size_t>(0, nobjs),
        [&cd_mcodes, &cd_vidx, cd_aabbs_span, cd_srt_aabbs_span, &cd_srt_mcodes](const auto &obj_range) {
            for (auto obj_idx = obj_range.begin(); obj_idx != obj_range.end(); ++obj_idx) {
                const auto sorted_idx = cd_vidx[obj_idx];

                // Write the sorted aabb.
                for (auto i = 0u; i < 4u; ++i) {
                    cd_srt_aabbs_span(obj_idx, 0, i) = cd_aabbs_span(sorted_idx, 0, i);
                    cd_srt_aabbs_span(obj_idx, 1, i) = cd_aabbs_span(sorted_idx, 1, i);
                }

                // Write the sorted morton codes.
                cd_srt_mcodes[obj_idx] = cd_mcodes[sorted_idx];
            }
        });

    // Write the global aabb into cd_srt_aabbs_span.
    for (auto i = 0u; i < 4u; ++i) {
        cd_srt_aabbs_span(nobjs, 0, i) = glb[i];
        cd_srt_aabbs_span(nobjs, 1, i) = gub[i];
    }
}

} // namespace mizuba
