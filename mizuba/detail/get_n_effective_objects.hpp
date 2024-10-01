// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_DETAIL_GET_N_EFFECTIVE_OBJECTS_HPP
#define MIZUBA_DETAIL_GET_N_EFFECTIVE_OBJECTS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <stdexcept>

#include <boost/numeric/conversion/cast.hpp>

namespace mizuba::detail
{

// Some objects may have no trajectory data during a conjunction step,
// and thus they can (and must) not be included in computations (e.g.,
// when building a bvh tree).
//
// The aabbs of these objects will contain infinities, and they will be placed
// at the tail end of srt_aabbs - we make sure of this when morton-sorting
// the data.
//
// Thus, we can look for the first infinite aabb in srt_aabbs in order to
// determine how many objects with trajectory data we have in the
// conjunction step.
//
// tot_nobjs is the total number of objects (including those with infinite
// aabbs), srt_aabbs is a view on the sorted aabbs for all conjunction steps,
// cd_idx is the conjunction step index.
template <typename SrtAABBs>
auto get_n_effective_objects(std::size_t tot_nobjs, SrtAABBs srt_aabbs, std::size_t cd_idx)
{
    // This is a view that transforms the sorted aabbs in the current conjunction
    // step in something like [false, false, ..., false, true, true, ...], where
    // "true" begins with the first infinite aabb.
    // NOTE: it is important we iota up to tot_nobjs here, even though the aabbs data
    // goes up to tot_nobjs + 1 - the last slot is the global aabb for the current
    // conjunction step.
    const auto isinf_view = std::views::iota(static_cast<std::size_t>(0), tot_nobjs)
                            | std::views::transform(
                                [srt_aabbs, cd_idx](std::size_t n) { return std::isinf(srt_aabbs(cd_idx, n, 0, 0)); });
    assert(std::ranges::is_sorted(isinf_view));
    static_assert(std::ranges::random_access_range<decltype(isinf_view)>);

    // Overflow check.
    try {
        // Make sure the difference type of isinf_view can represent tot_nobjs.
        static_cast<void>(boost::numeric_cast<std::ranges::range_difference_t<decltype(isinf_view)>>(tot_nobjs));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected during the computation of the number of effective objects");
    }
    // LCOV_EXCL_STOP

    // Determine the position of the first infinite aabb.
    const auto it_inf = std::ranges::lower_bound(isinf_view, true);
    // Compute the total number of objects with trajectory data.
    const auto nobjs = boost::numeric_cast<std::uint32_t>(it_inf - std::ranges::begin(isinf_view));
    // NOTE: we cannot have conjunction steps without trajectory data.
    assert(nobjs > 0u);

    return nobjs;
}

} // namespace mizuba::detail

#endif
