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
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/mdspan.hpp>

#include "polyjectory.hpp"

namespace mizuba
{

// Fwd declaration.
class conjunctions;

namespace detail
{

struct conjunctions_impl;

void close_cj(std::shared_ptr<conjunctions_impl> &) noexcept;
[[nodiscard]] const std::shared_ptr<conjunctions_impl> &fetch_cj_impl(const conjunctions &) noexcept;

} // namespace detail

class conjunctions
{
    std::shared_ptr<detail::conjunctions_impl> m_impl;

    friend const std::shared_ptr<detail::conjunctions_impl> &detail::fetch_cj_impl(const conjunctions &) noexcept;

    // Private ctor.
    struct ptag {
    };
    explicit conjunctions(ptag, polyjectory, double, double, std::vector<std::uint32_t>);

    // Helper to convert an input whitelist range into a vector of indices.
    template <typename WRange>
    static auto wrange_to_vec(WRange &&whitelist)
    {
        if constexpr (std::same_as<std::vector<std::uint32_t>, std::remove_cvref_t<WRange>>) {
            return std::forward<WRange>(whitelist);
        } else {
            std::vector<std::uint32_t> retval;
            // Prepare the appropriate size if WRange is a sized range
            // with an integral size type.
            if constexpr (requires {
                              requires std::ranges::sized_range<WRange>;
                              requires std::integral<std::remove_cvref_t<std::ranges::range_reference_t<WRange>>>;
                          }) {
                retval.reserve(static_cast<decltype(retval.size())>(std::ranges::size(whitelist)));
            }

            for (auto idx : whitelist) {
                retval.push_back(boost::numeric_cast<std::uint32_t>(idx));
            }

            return retval;
        }
    }

    [[nodiscard]] std::array<double, 2> get_cd_begin_end(double, std::size_t, double, std::size_t) const;
    std::vector<double> compute_aabbs(const polyjectory &, const boost::filesystem::path &, std::size_t, double,
                                      double) const;
    void morton_encode_sort_parallel(const polyjectory &, const boost::filesystem::path &, std::size_t) const;
    std::vector<std::tuple<std::size_t, std::size_t>>
    construct_bvh_trees_parallel(const polyjectory &, const boost::filesystem::path &, std::size_t) const;
    std::vector<std::tuple<std::size_t, std::size_t>>
    broad_phase(const polyjectory &, const boost::filesystem::path &, std::size_t,
                const std::vector<std::tuple<std::size_t, std::size_t>> &, const std::vector<bool> &);

public:
    // The BVH node struct.
    struct bvh_node {
        // Object range.
        std::uint32_t begin, end;
        // Pointers to parent and children nodes.
        std::int32_t parent, left, right;
        // AABB.
        std::array<float, 4> lb, ub;
    };

    // Struct to represent collisions between AABBs.
    struct aabb_collision {
        // Indices of the objects whose AABBs collide.
        std::uint32_t i, j;
        auto operator<=>(const aabb_collision &) const = default;
    };

    template <typename WRange = std::vector<std::uint32_t>>
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
    [[nodiscard]] aabbs_span_t get_srt_aabbs() const noexcept;
    using mcodes_span_t = heyoka::mdspan<const std::uint64_t, heyoka::dextents<std::size_t, 2>>;
    [[nodiscard]] mcodes_span_t get_mcodes() const noexcept;
    [[nodiscard]] mcodes_span_t get_srt_mcodes() const noexcept;
    using srt_idx_span_t = heyoka::mdspan<const std::uint32_t, heyoka::dextents<std::size_t, 2>>;
    [[nodiscard]] srt_idx_span_t get_srt_idx() const noexcept;
    using tree_span_t = heyoka::mdspan<const bvh_node, heyoka::dextents<std::size_t, 1>>;
    [[nodiscard]] tree_span_t get_bvh_tree(std::size_t) const;
    using aabb_collision_span_t = heyoka::mdspan<const aabb_collision, heyoka::dextents<std::size_t, 1>>;
    [[nodiscard]] aabb_collision_span_t get_aabb_collisions(std::size_t) const;
};

} // namespace mizuba

#endif
