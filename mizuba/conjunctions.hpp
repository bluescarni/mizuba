// Copyright 2024-2025 Francesco Biscani
//
// This file is part of the mizuba library.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef MIZUBA_CONJUNCTIONS_HPP
#define MIZUBA_CONJUNCTIONS_HPP

#include <array>
#include <atomic>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include <boost/filesystem/path.hpp>

#include <heyoka/mdspan.hpp>

#include "detail/conjunctions_jit.hpp"
#include "mdspan.hpp"
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
        std::uint32_t i, j;
        // Time of closest approach.
        double tca;
        // Distance of closest approach.
        double dca;
        // The objects involved in the conjunction.
        // The state vectors of i and j
        // at TCA.
        std::array<double, 3> ri, vi;
        std::array<double, 3> rj, vj;
        auto operator<=>(const conj &) const = default;
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

    // Struct to collect stats about the narrow-phase conjunction
    // detection step, for logging purposes.
    struct np_report {
        // The total number of conjunction candidates.
        // NOTE: this will in general be larger than the number of
        // aabbs collisions, because for each aabbs collision we typically
        // need to run conjunction detection multiple times due to the trajectory
        // steps not beginning/ending at the same time for different objects.
        std::atomic<unsigned long long> n_tot_conj_candidates = 0;
        // How many conjunction candidates were discarded
        // via the evaluation of the polynomial enclosure of
        // the distance square.
        std::atomic<unsigned long long> n_dist2_check = 0;
        // The total number of polynomial root findings.
        std::atomic<unsigned long long> n_poly_roots = 0;
        // How many polynomial root findings exited early via the
        // fast exclusion check.
        std::atomic<unsigned long long> n_fex_check = 0;
        // How many polynomial root findings resulted in no
        // roots being found.
        std::atomic<unsigned long long> n_poly_no_roots = 0;
        // Total number of distance minima determined via polynomial
        // root finding.
        std::atomic<unsigned long long> n_tot_dist_minima = 0;
        // Number of distance minima that were discarded as conjunctions
        // because the dca is above the threshold.
        std::atomic<unsigned long long> n_tot_discarded_dist_minima = 0;
    };

    [[nodiscard]] static std::array<double, 2> get_cd_begin_end(double, std::size_t, double, std::size_t);

    static std::tuple<std::vector<double>, std::vector<std::tuple<std::size_t, std::size_t>>,
                      std::vector<std::tuple<std::size_t, std::size_t>>>
    detect_conjunctions(const boost::filesystem::path &, const polyjectory &, std::size_t, double, double,
                        const std::vector<std::int32_t> &, bool);
    static void detect_conjunctions_aabbs(std::size_t, std::vector<float> &, const polyjectory &, double, double,
                                          std::size_t, std::vector<double> &, const detail::conj_jit_data &);
    static void detect_conjunctions_morton(std::vector<std::uint64_t> &, std::vector<std::uint32_t> &,
                                           std::vector<float> &, std::vector<std::uint64_t> &,
                                           const std::vector<float> &, const polyjectory &);
    static void detect_conjunctions_bvh(std::vector<bvh_node> &, std::vector<bvh_aux_node_data> &,
                                        std::vector<bvh_level_data> &, const std::vector<float> &,
                                        const std::vector<std::uint64_t> &);
    static std::vector<aabb_collision> detect_conjunctions_broad_phase(const std::vector<bvh_node> &,
                                                                       const std::vector<std::uint32_t> &,
                                                                       const std::vector<std::int32_t> &,
                                                                       const std::vector<float> &,
                                                                       const std::vector<float> &);
    static std::vector<conj> detect_conjunctions_narrow_phase(std::size_t, const polyjectory &,
                                                              const std::vector<aabb_collision> &,
                                                              const detail::conj_jit_data &, double, double,
                                                              std::size_t, np_report &);

public:
    explicit conjunctions(const polyjectory &pj, double conj_thresh, double conj_det_interval,
                          std::optional<std::vector<std::int32_t>>);

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
    [[nodiscard]] dspan_1d<const double> get_cd_end_times() const noexcept;
    [[nodiscard]] aabbs_span_t get_srt_aabbs() const noexcept;
    [[nodiscard]] dspan_2d<const std::uint64_t> get_mcodes() const noexcept;
    [[nodiscard]] dspan_2d<const std::uint64_t> get_srt_mcodes() const noexcept;
    [[nodiscard]] dspan_2d<const std::uint32_t> get_srt_idx() const noexcept;
    [[nodiscard]] dspan_1d<const bvh_node> get_bvh_tree(std::size_t) const;
    [[nodiscard]] dspan_1d<const aabb_collision> get_aabb_collisions(std::size_t) const;
    [[nodiscard]] dspan_1d<const conj> get_conjunctions() const noexcept;
    [[nodiscard]] dspan_1d<const std::int32_t> get_otypes() const noexcept;
    [[nodiscard]] double get_conj_thresh() const noexcept;
    [[nodiscard]] double get_conj_det_interval() const noexcept;
};

} // namespace mizuba

#endif
