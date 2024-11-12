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
#include <stdexcept>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include "conjunctions.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

std::vector<conjunctions::conj> conjunctions::extend(const polyjectory &pj) const
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Check pj.
    if (pj.get_id() == get_polyjectory().get_id()) [[unlikely]] {
        throw std::invalid_argument("Invalid polyjectory extension detected: cannot extend a polyjectory with itself");
    }

    if (pj.get_maxT() > get_polyjectory().get_maxT()) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid polyjectory extension detected: the maximum time coordinate in the extension ({}) is "
                        "greater than the maximum time coordinate in the original polyjectory ({})",
                        pj.get_maxT(), get_polyjectory().get_maxT()));
    }

    if (pj.get_poly_order() != get_polyjectory().get_poly_order()) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid polyjectory extension detected: the polynomial order of the extension ({}) differs "
                        "from the polynomial order of the original polyjectory ({})",
                        pj.get_poly_order(), get_polyjectory().get_poly_order()));
    }

    // Cache the total number of objects in the polyjectory extension.
    const auto nobjs = pj.get_nobjs();

    // We need to determine the subrange of the conjunction steps from the original polyjectory
    // which fully includes the polyjectory extension. We begin by determining which conjunction
    // step ends at or after pj.get_maxT().
    const auto cdet = get_cd_end_times();
    const auto *last_cd_ptr
        = std::ranges::lower_bound(cdet.data_handle(), cdet.data_handle() + cdet.extent(0), pj.get_maxT());
    // NOTE: this must hold because pj.get_maxT() <= get_polyjectory().get_maxT().
    assert(last_cd_ptr != cdet.data_handle() + cdet.extent(0));

    // Overflow check.
    try {
        static_cast<void>(boost::numeric_cast<std::ptrdiff_t>(cdet.extent(0)));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in conjunctions::extend()");
    }
    // LCOV_EXCL_STOP

    // NOTE: the +1 is because last_cd_ptr points to the last conjunction step that contains
    // the polyjectory extension. In order to make a half-open range, we have to bump up by 1.
    const auto end_cd_idx = static_cast<std::size_t>(last_cd_ptr - cdet.data_handle() + 1);

    // NOTE: this is thread-local data specific to a conjunction step. We will use these buffers
    // to temporarily store the results of the various stages of conjunction detection.
    struct ets_data {
        // aabbs.
        std::vector<float> aabbs;
        // BVH tree.
        std::vector<bvh_node> bvh_tree;
    };
    using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([nobjs]() {
        // Setup aabbs.
        std::vector<float> aabbs;
        aabbs.resize((safe_size_t(nobjs) + 1) * 8);

        return ets_data{.aabbs = std::move(aabbs), .bvh_tree = {}};
    });

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, end_cd_idx),
                              [this, &ets, &pj](const auto &cd_range) {
                                  // Fetch the thread-local data.
                                  auto &[cd_aabbs, cd_bvh_tree] = ets.local();

                                  // NOTE: isolate to avoid issues with thread-local data. See:
                                  // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
                                  oneapi::tbb::this_task_arena::isolate([this, &cd_range, &cd_aabbs, &pj]() {
                                      for (auto cd_idx = cd_range.begin(); cd_idx != cd_range.end(); ++cd_idx) {
                                          extend_detect_conjunctions_aabbs(cd_idx, cd_aabbs, pj);
                                      }
                                  });
                              });

    return {};
}

} // namespace mizuba
