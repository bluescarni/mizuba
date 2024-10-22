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

#include <boost/container/small_vector.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/mdspan.hpp>

#include "detail/conjunctions_jit.hpp"
#include "detail/poly_utils.hpp"
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

struct conj_jit_data;

} // namespace detail

class conjunctions
{
    std::shared_ptr<detail::conjunctions_impl> m_impl;

    friend const std::shared_ptr<detail::conjunctions_impl> &detail::fetch_cj_impl(const conjunctions &) noexcept;

public:
    // The BVH node struct.
    struct bvh_node {
        // Object range.
        std::uint32_t begin, end;
        // Pointers to the children nodes.
        std::int32_t left, right;
        // AABB.
        std::array<float, 4> lb, ub;
    };

    // Struct to represent collisions between AABBs.
    struct aabb_collision {
        // Indices of the objects whose AABBs collide.
        std::uint32_t i, j;
        auto operator<=>(const aabb_collision &) const = default;
    };

    // Struct to represent a conjunction between two objects.
    struct conj {
        // Time of closest approach.
        double tca;
        // Distance of closest approach.
        double dca;
        // The objects involved in the conjunction.
        std::uint32_t i, j;
        // The state vectors of i and j
        // at TCA.
        std::array<double, 3> ri, vi;
        std::array<double, 3> rj, vj;
    };

private:
    // This is auxiliary data attached to each node of a BVH tree,
    // used only during construction/verification.
    // It will not be present in the final tree, hence it is kept
    // separate.
    struct bvh_aux_node_data {
        // Total number of nodes in the tree level the
        // node belongs to.
        std::uint32_t nn_level;
        // Index of the split bit. This has several meanings and properties:
        //
        // 1) all morton codes in the node will share the same common
        //    initial sequence of split_idx bits (measured from MSB);
        //
        // 2) if the node is internal, split_idx is the bit index in the morton
        //    code (counted from the MSB) used for partitioning the objects
        //    in the node into the left and right children. For instance,
        //    split_idx == 3 means that the objects in the left child will
        //    have morton codes whose fourth bit (from MSB) is 0, while the objects in the
        //    right child will have morton codes whose fourth bit is 1;
        //
        // 3) if the node is a leaf with a single object, then split_idx is the
        //    split_idx of the parent + 1.
        //
        // Property 3) follows immediately from the fact that the split_idx
        // of a node is always initialised as the split_idx of the parent
        // + 1. split_idx is modified after initialisation only if the node
        // contains more than 1 object.
        //
        // From property 1), it follows that in a leaf node with more than
        // 1 object split_idx must be 64, because all morton codes will be
        // identical.
        std::uint32_t split_idx;
        // The parent.
        std::int32_t parent;
    };

    // Data structure used to temporarily store certain
    // properties of a node during the construction of a
    // BVH tree level by level. The data stored in here will eventually
    // be transferred to a bvh_node.
    struct bvh_level_data {
        // Number of children.
        std::uint32_t nc;
        // Number of objects in the left child.
        std::uint32_t nolc;
        // Storage used in the computation of the prefix
        // sum over the number of children for each node
        // in the current level.
        std::uint32_t ps;
    };

    // Handy alias for boost's small vector.
    template <typename T>
    using small_vec = boost::container::small_vector<T, 1>;

    // Bag of per-object data used during narrow-phase conjunction detection.
    struct np_data {
        // Local vector of detected conjunctions.
        small_vec<conj> conj_vec;
        // Polynomial cache for use during real root isolation.
        // NOTE: it is *really* important that this is declared
        // *before* wlist, because wlist will contain references
        // to and interact with r_iso_cache during destruction,
        // and we must be sure that wlist is destroyed *before*
        // r_iso_cache.
        detail::poly_cache r_iso_cache;
        // The working list.
        detail::wlist_t wlist;
        // The list of isolating intervals.
        detail::isol_t isol;
        // Buffers used as temporary storage for the results
        // of operations on polynomials.
        // NOTE: if we restructure the code to use JIT more,
        // we should probably re-implement this as a flat
        // 1D buffer rather than a collection of vectors.
        std::array<std::vector<double>, 14> pbuffers;
        // Vector to store the input for the cfunc used to compute
        // the distance square polynomial.
        std::vector<double> diff_input;
        // The vector into which detected conjunctions are
        // temporarily written during polynomial root finding.
        // The tuple contains:
        // - the indices of the 2 objects,
        // - the time coordinate of the conjunction (relative
        //   to the time interval in which root finding is performed,
        //   i.e., **NOT** the absolute time in the polyjectory).
        std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> tmp_conj_vec;
    };

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

    [[nodiscard]] static std::array<double, 2> get_cd_begin_end(double, std::size_t, double, std::size_t);
    std::vector<double> compute_aabbs(const polyjectory &, const boost::filesystem::path &, std::size_t, double,
                                      double) const;
    void morton_encode_sort(const polyjectory &, const boost::filesystem::path &, std::size_t) const;
    std::vector<std::tuple<std::size_t, std::size_t>>
    construct_bvh_trees(const polyjectory &, const boost::filesystem::path &, std::size_t) const;
    std::vector<std::tuple<std::size_t, std::size_t>>
    broad_phase(const polyjectory &, const boost::filesystem::path &, std::size_t,
                const std::vector<std::tuple<std::size_t, std::size_t>> &, const std::vector<bool> &);
    void narrow_phase(const polyjectory &, const boost::filesystem::path &,
                      const std::vector<std::tuple<std::size_t, std::size_t>> &, const std::vector<double> &,
                      const detail::conj_jit_data &, double);

    static std::tuple<std::vector<double>, std::vector<std::tuple<std::size_t, std::size_t>>,
                      std::vector<std::tuple<std::size_t, std::size_t>>>
    detect_conjunctions(const boost::filesystem::path &, const polyjectory &, std::size_t, double, double,
                        const std::vector<bool> &);
    static void detect_conjunctions_aabbs(std::size_t, std::vector<float> &, const polyjectory &, double, double,
                                          std::size_t, std::vector<double> &);
    static void detect_conjunctions_morton(std::vector<std::uint64_t> &, std::vector<std::uint32_t> &,
                                           std::vector<float> &, std::vector<std::uint64_t> &,
                                           const std::vector<float> &, const polyjectory &);
    static void detect_conjunctions_bvh(std::vector<bvh_node> &, std::vector<bvh_aux_node_data> &,
                                        std::vector<bvh_level_data> &, const std::vector<float> &,
                                        const std::vector<std::uint64_t> &);
    static std::vector<aabb_collision>
    detect_conjunctions_broad_phase(std::vector<small_vec<aabb_collision>> &, std::vector<std::vector<std::int32_t>> &,
                                    const std::vector<bvh_node> &, const std::vector<std::uint32_t> &,
                                    const std::vector<bool> &, const std::vector<float> &, const std::vector<float> &);
    static std::vector<conj> detect_conjunctions_narrow_phase(std::vector<np_data> &, std::size_t, const polyjectory &,
                                                              const std::vector<small_vec<aabb_collision>> &,
                                                              const detail::conj_jit_data &, double, double,
                                                              std::size_t);

public:
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

    [[nodiscard]] std::size_t get_n_cd_steps() const noexcept;

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
    using conj_span_t = heyoka::mdspan<const conj, heyoka::dextents<std::size_t, 1>>;
    [[nodiscard]] conj_span_t get_conjunctions() const noexcept;
    using whitelist_span_t = heyoka::mdspan<const std::uint32_t, heyoka::dextents<std::size_t, 1>>;
    [[nodiscard]] whitelist_span_t get_whitelist() const noexcept;
    [[nodiscard]] double get_conj_thresh() const noexcept;
    [[nodiscard]] double get_conj_det_interval() const noexcept;
};

} // namespace mizuba

#endif
