// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_CONJUNCTIONS_HPP
#define MIZUBA_CONJUNCTIONS_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/mdspan.hpp>

#include "polyjectory.hpp"

namespace mizuba
{

class conjunctions
{
    struct impl;

    std::shared_ptr<const impl> m_impl;

    // Private ctor.
    struct ptag {
    };
    explicit conjunctions(ptag, polyjectory, double, double, std::vector<std::size_t>);

    // Helper to convert an input whitelist range into a vector of indices.
    template <typename WRange>
    static auto wrange_to_vec(WRange &&whitelist)
    {
        if constexpr (std::same_as<std::vector<std::size_t>, std::remove_cvref_t<WRange>>) {
            return std::forward<WRange>(whitelist);
        } else {
            std::vector<std::size_t> retval;
            // Prepare the appropriate size if WRange is a sized range
            // with an integral size type.
            if constexpr (requires {
                              requires std::ranges::sized_range<WRange>;
                              requires std::integral<std::remove_cvref_t<std::ranges::range_reference_t<WRange>>>;
                          }) {
                retval.reserve(static_cast<decltype(retval.size())>(std::ranges::size(whitelist)));
            }

            for (auto idx : whitelist) {
                retval.push_back(boost::numeric_cast<std::size_t>(idx));
            }

            return retval;
        }
    }

    [[nodiscard]] std::array<double, 2> get_cd_begin_end(double, std::size_t, double, std::size_t) const;
    std::vector<double> compute_aabbs(const polyjectory &, const boost::filesystem::path &, std::size_t, double,
                                      double) const;

public:
    template <typename WRange = std::vector<std::size_t>>
        requires std::ranges::input_range<WRange>
                 && std::integral<std::remove_cvref_t<std::ranges::range_reference_t<WRange>>>
    explicit conjunctions(polyjectory pj, double conj_thresh, double conj_det_interval, WRange &&whitelist = {})
        : conjunctions(ptag{}, std::move(pj), conj_thresh, conj_det_interval,
                       wrange_to_vec(std::forward<WRange>(whitelist)))
    {
    }
    template <typename T>
        requires std::integral<T>
    explicit conjunctions(polyjectory pj, double conj_thresh, double conj_det_interval,
                          std::initializer_list<T> whitelist)
        : conjunctions(ptag{}, std::move(pj), conj_thresh, conj_det_interval, std::vector<T>(whitelist))
    {
    }

    conjunctions(const conjunctions &);
    conjunctions(conjunctions &&) noexcept;
    conjunctions &operator=(const conjunctions &);
    conjunctions &operator=(conjunctions &&) noexcept;
    ~conjunctions();

    // NOTE: the four dimensions here are, respectively:
    // - the total number of conjunction steps,
    // - the total number of objects + 1 (the +1 is for the global
    //   aabb for the conjunction step),
    // - the lower/upper bounds (always 2),
    // - the number of elements in the bounds (which is
    //   always 4).
    using aabbs_span_t
        = heyoka::mdspan<const float, heyoka::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, 2, 4>>;
    [[nodiscard]] aabbs_span_t get_aabbs() const noexcept;
    using cd_end_times_span_t = heyoka::mdspan<const double, heyoka::dextents<std::size_t, 1>>;
    [[nodiscard]] cd_end_times_span_t get_cd_end_times() const noexcept;
    [[nodiscard]] const polyjectory &get_polyjectory() const noexcept;
};

} // namespace mizuba

#endif