// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_POLYJECTORY_HPP
#define MIZUBA_POLYJECTORY_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/mdspan.hpp>

namespace mizuba
{

class polyjectory
{
    struct impl;

    std::shared_ptr<impl> m_impl;

public:
    // NOTE: the three dimensions here are, respectively:
    // - the total number of steps,
    // - the total number of state variables (which is always 7, i.e.,
    //   the Cartesian state vector + radius),
    // - the polynomial order + 1.
    using traj_span_t
        = heyoka::mdspan<const double, heyoka::extents<std::size_t, std::dynamic_extent, 7, std::dynamic_extent>>;
    using time_span_t = heyoka::mdspan<const double, heyoka::dextents<std::size_t, 1>>;

private:
    struct ptag {
    };
    explicit polyjectory(ptag,
                         std::tuple<std::vector<traj_span_t>, std::vector<time_span_t>, std::vector<std::int32_t>>);

    template <typename TrajRng, typename TimeRng, typename StatusRng>
    static auto ctor_impl(TrajRng &&traj_rng, TimeRng &&time_rng, StatusRng &&status_rng)
    {
        std::vector<traj_span_t> traj_spans;
        std::ranges::copy(traj_rng, std::back_inserter(traj_spans));

        std::vector<time_span_t> time_spans;
        std::ranges::copy(time_rng, std::back_inserter(time_spans));

        std::vector<std::int32_t> status;
        std::ranges::copy(status_rng, std::back_inserter(status));

        return std::make_tuple(std::move(traj_spans), std::move(time_spans), std::move(status));
    }

public:
    template <typename TrajRng, typename TimeRng, typename StatusRng>
        requires std::ranges::input_range<TrajRng>
                 && std::same_as<traj_span_t, std::remove_cvref_t<std::ranges::range_reference_t<TrajRng>>>
                 && std::ranges::input_range<TimeRng>
                 && std::same_as<time_span_t, std::remove_cvref_t<std::ranges::range_reference_t<TimeRng>>>
                 && std::ranges::input_range<StatusRng>
                 && std::same_as<std::int32_t, std::remove_cvref_t<std::ranges::range_reference_t<StatusRng>>>
    explicit polyjectory(TrajRng &&traj_rng, TimeRng &&time_rng, StatusRng &&status_rng)
        : polyjectory(ptag{}, ctor_impl(std::forward<TrajRng>(traj_rng), std::forward<TimeRng>(time_rng),
                                        std::forward<StatusRng>(status_rng)))
    {
    }
    polyjectory(const polyjectory &);
    polyjectory(polyjectory &&) noexcept;
    polyjectory &operator=(const polyjectory &);
    polyjectory &operator=(polyjectory &&) noexcept;
    ~polyjectory();

    [[nodiscard]] std::tuple<traj_span_t, time_span_t, std::int32_t> operator[](std::size_t) const;
};

} // namespace mizuba

#endif
