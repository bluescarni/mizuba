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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <boost/align.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <oneapi/tbb/parallel_sort.h>

#include "conjunctions.hpp"
#include "detail/file_utils.hpp"
#include "mdspan.hpp"
#include "polyjectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

namespace detail
{

struct conjunctions_impl {
    using bvh_node = conjunctions::bvh_node;
    using aabb_collision = conjunctions::aabb_collision;
    using conj = conjunctions::conj;

    // The path to the temp dir containing all the
    // conjunctions data.
    boost::filesystem::path m_temp_dir_path;
    // The polyjectory.
    polyjectory m_pj;
    // The conjunction threshold.
    double m_conj_thresh = 0;
    // The conjunction detection interval.
    double m_conj_det_interval = 0;
    // The total number of conjunction detection steps.
    std::size_t m_n_cd_steps = 0;
    // End times of the conjunction steps.
    // NOTE: if needed, this one can also be turned into
    // a memory-mapped file.
    std::vector<double> m_cd_end_times;
    // Vector of offsets and sizes for the tree data
    // stored in m_file_bvh_trees.
    //
    // The first element of the pair is the offset at which tree data begins,
    // the second element of the pair is the tree size. The size of
    // this vector is equal to the number of conjunction steps.
    std::vector<std::tuple<std::size_t, std::size_t>> m_tree_offsets;
    // Vector of offsets and sizes for the broad-phase conjunction detection
    // data stored in m_file_bp.
    //
    // The first element of the pair is the offset at which bp data begins,
    // the second element of the pair is the size of the bp data. The size of
    // this vector is equal to the number of conjunction steps.
    std::vector<std::tuple<std::size_t, std::size_t>> m_bp_offsets;
    // Object types.
    std::vector<std::int32_t> m_otypes;
    // The memory-mapped file for the aabbs.
    boost::iostreams::mapped_file_source m_file_aabbs;
    // The memory-mapped file for the sorted aabbs.
    boost::iostreams::mapped_file_source m_file_srt_aabbs;
    // The memory-mapped file for the mcodes.
    boost::iostreams::mapped_file_source m_file_mcodes;
    // The memory-mapped file for the sorted mcodes.
    boost::iostreams::mapped_file_source m_file_srt_mcodes;
    // The memory-mapped file for the sorted indices.
    boost::iostreams::mapped_file_source m_file_srt_idx;
    // The memory-mapped file for the bvh trees.
    boost::iostreams::mapped_file_source m_file_bvh_trees;
    // The memory-mapped file for the bp data.
    boost::iostreams::mapped_file_source m_file_bp;
    // The memory-mapped file for the detected conjunctions.
    boost::iostreams::mapped_file_source m_file_conjs;
    // Pointer to the beginning of m_file_aabbs, cast to float.
    const float *m_aabbs_base_ptr = nullptr;
    // Pointer to the beginning of m_file_srt_aabbs, cast to float.
    const float *m_srt_aabbs_base_ptr = nullptr;
    // Pointer to the beginning of m_file_mcodes, cast to std::uint64_t.
    const std::uint64_t *m_mcodes_base_ptr = nullptr;
    // Pointer to the beginning of m_file_srt_mcodes, cast to std::uint64_t.
    const std::uint64_t *m_srt_mcodes_base_ptr = nullptr;
    // Pointer to the beginning of m_file_srt_idx, cast to std::uint32_t.
    const std::uint32_t *m_srt_idx_base_ptr = nullptr;
    // Pointer to the beginning of m_file_bvh_trees, cast to bvh_node.
    const bvh_node *m_bvh_trees_ptr = nullptr;
    // Pointer to the beginning of m_file_bp, cast to aabb_collision.
    const aabb_collision *m_bp_ptr = nullptr;
    // Pointer to the beginning of m_file_conjs, cast to conj.
    const conj *m_conjs_ptr = nullptr;

    explicit conjunctions_impl(boost::filesystem::path temp_dir_path, polyjectory pj, double conj_thresh,
                               double conj_det_interval, std::size_t n_cd_steps, std::vector<double> cd_end_times,
                               std::vector<std::tuple<std::size_t, std::size_t>> tree_offsets,
                               std::vector<std::tuple<std::size_t, std::size_t>> bp_offsets,
                               std::vector<std::int32_t> otypes)
        : m_temp_dir_path(std::move(temp_dir_path)), m_pj(std::move(pj)), m_conj_thresh(conj_thresh),
          m_conj_det_interval(conj_det_interval), m_n_cd_steps(n_cd_steps), m_cd_end_times(std::move(cd_end_times)),
          m_tree_offsets(std::move(tree_offsets)), m_bp_offsets(std::move(bp_offsets)), m_otypes(std::move(otypes)),
          m_file_aabbs((m_temp_dir_path / "aabbs").string()),
          m_file_srt_aabbs((m_temp_dir_path / "srt_aabbs").string()),
          m_file_mcodes((m_temp_dir_path / "mcodes").string()),
          m_file_srt_mcodes((m_temp_dir_path / "srt_mcodes").string()),
          m_file_srt_idx((m_temp_dir_path / "vidx").string()), m_file_bvh_trees((m_temp_dir_path / "bvh").string()),
          m_file_bp((m_temp_dir_path / "bp").string()), m_file_conjs(m_temp_dir_path / "conjunctions")
    {
        // Sanity checks.
        assert(n_cd_steps > 0u);
        assert(m_cd_end_times.size() == n_cd_steps);
        assert(m_cd_end_times.size() == m_tree_offsets.size());
        assert(m_cd_end_times.size() == m_bp_offsets.size());

        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        m_aabbs_base_ptr = reinterpret_cast<const float *>(m_file_aabbs.data());
        assert(boost::alignment::is_aligned(m_aabbs_base_ptr, alignof(float)));

        m_srt_aabbs_base_ptr = reinterpret_cast<const float *>(m_file_srt_aabbs.data());
        assert(boost::alignment::is_aligned(m_srt_aabbs_base_ptr, alignof(float)));

        m_mcodes_base_ptr = reinterpret_cast<const std::uint64_t *>(m_file_mcodes.data());
        assert(boost::alignment::is_aligned(m_mcodes_base_ptr, alignof(std::uint64_t)));

        m_srt_mcodes_base_ptr = reinterpret_cast<const std::uint64_t *>(m_file_srt_mcodes.data());
        assert(boost::alignment::is_aligned(m_srt_mcodes_base_ptr, alignof(std::uint64_t)));

        m_srt_idx_base_ptr = reinterpret_cast<const std::uint32_t *>(m_file_srt_idx.data());
        assert(boost::alignment::is_aligned(m_srt_idx_base_ptr, alignof(std::uint32_t)));

        m_bvh_trees_ptr = reinterpret_cast<const bvh_node *>(m_file_bvh_trees.data());
        assert(boost::alignment::is_aligned(m_bvh_trees_ptr, alignof(bvh_node)));

        m_bp_ptr = reinterpret_cast<const aabb_collision *>(m_file_bp.data());
        assert(boost::alignment::is_aligned(m_bp_ptr, alignof(aabb_collision)));

        // NOTE: if no conjunctions were detected, only a single byte
        // was written to m_file_conjs. In this case, we leave m_conjs_ptr
        // null in order to signal the lack of conjunctions.
        // NOTE: we do not need to do this for broad-phase conjunction
        // detection data because there we can detect the lack of data from
        // m_bp_offsets.
        if (m_file_conjs.size() > 1u) {
            m_conjs_ptr = reinterpret_cast<const conj *>(m_file_conjs.data());
            assert(boost::alignment::is_aligned(m_conjs_ptr, alignof(conj)));
        }
    }

    [[nodiscard]] bool is_open() noexcept
    {
        return m_file_aabbs.is_open();
    }

    void close() noexcept
    {
        // NOTE: a conjunctions object is not supposed to be closed
        // more than once.
        assert(is_open());

        // Close all memory-mapped files.
        m_file_aabbs.close();
        m_file_srt_aabbs.close();
        m_file_mcodes.close();
        m_file_srt_mcodes.close();
        m_file_srt_idx.close();
        m_file_bvh_trees.close();
        m_file_bp.close();
        m_file_conjs.close();

        // Remove the temp dir and everything within.
        boost::filesystem::remove_all(m_temp_dir_path);
    }

    conjunctions_impl(conjunctions_impl &&) noexcept = delete;
    conjunctions_impl(const conjunctions_impl &) = delete;
    conjunctions_impl &operator=(const conjunctions_impl &) = delete;
    conjunctions_impl &operator=(conjunctions_impl &&) noexcept = delete;
    ~conjunctions_impl()
    {
        if (is_open()) {
            close();
        }
    }
};

// LCOV_EXCL_START

void close_cj(std::shared_ptr<conjunctions_impl> &cj) noexcept
{
    cj->close();
}

// LCOV_EXCL_STOP

const std::shared_ptr<conjunctions_impl> &fetch_cj_impl(const conjunctions &cj) noexcept
{
    return cj.m_impl;
}

} // namespace detail

conjunctions::conjunctions(polyjectory pj, double conj_thresh, double conj_det_interval,
                           std::optional<std::vector<std::int32_t>> otypes)
{
    // Check conj_thresh.
    if (!std::isfinite(conj_thresh) || conj_thresh <= 0) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The conjunction threshold must be finite and positive, but instead a value of {} was provided",
                        conj_thresh));
    }

    // NOTE: we need to square conj_thresh during narrow-phase conjunction detection.
    if (!std::isfinite(conj_thresh * conj_thresh)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("A conjunction threshold of {} is too large and results in an overflow error", conj_thresh));
    }

    // Check conj_det_interval.
    if (!std::isfinite(conj_det_interval) || conj_det_interval <= 0) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The conjunction detection interval must be finite and positive, but instead a value of {} was provided",
            conj_det_interval));
    }

    // Cache the total number of objects in the polyjectory.
    const auto nobjs = pj.get_nobjs();

    // Validation/setup of otypes.
    // NOTE: the skip_cd flag will be set to true if all the objects are either
    // secondaries or masked. In that case, the broad and narrow phase conjunction
    // detection steps are skipped.
    bool skip_cd = true;
    if (otypes) {
        // Check the size of otypes.
        if (otypes->size() != nobjs) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Invalid array of object types passed to the constructor of a conjunctions objects: the "
                            "expected size is {}, but the actual size is {} instead",
                            nobjs, otypes->size()));
        }

        // Check the values in otypes.
        for (const auto val : *otypes) {
            if (val != 1 && val != 2 && val != 4) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "The value of an object type must be one of [1, 2, 4], but a value of {} was detected instead",
                    val));
            }

            // Update the skip_cd flag.
            skip_cd = skip_cd && (val != 1);
        }
    } else {
        // otypes was not provided, mark all objects as primaries.
        otypes.emplace();
        otypes->resize(boost::numeric_cast<decltype(otypes->size())>(nobjs), 1);

        skip_cd = false;
    }

    // Determine the number of conjunction detection steps.
    const auto n_cd_steps = boost::numeric_cast<std::size_t>(std::ceil(pj.get_maxT() / conj_det_interval));

    // Assemble a "unique" dir path into the system temp dir. This will be the root dir
    // for all conjunctions data.
    const auto tmp_dir_path = detail::create_temp_dir("mizuba_conjunctions-%%%%-%%%%-%%%%-%%%%");

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the temp dir in case of exceptions.
    try {
        // Run conjunction detection.
        auto [cd_end_times, tree_offsets, bp_offsets]
            = detect_conjunctions(tmp_dir_path, pj, n_cd_steps, conj_thresh, conj_det_interval, *otypes, skip_cd);

        // Create the impl.
        m_impl = std::make_shared<detail::conjunctions_impl>(
            tmp_dir_path, std::move(pj), conj_thresh, conj_det_interval, n_cd_steps, std::move(cd_end_times),
            std::move(tree_offsets), std::move(bp_offsets), std::move(*otypes));

        // LCOV_EXCL_START
    } catch (...) {
        boost::filesystem::remove_all(tmp_dir_path);
        throw;
    }
    // LCOV_EXCL_STOP
}

conjunctions::conjunctions(const conjunctions &) = default;

conjunctions::conjunctions(conjunctions &&) noexcept = default;

conjunctions &conjunctions::operator=(const conjunctions &) = default;

conjunctions &conjunctions::operator=(conjunctions &&) noexcept = default;

conjunctions::~conjunctions() = default;

// Helper to compute the begin and end time coordinates for a conjunction step.
std::array<double, 2> conjunctions::get_cd_begin_end(double maxT, std::size_t cd_idx, double conj_det_interval,
                                                     std::size_t n_cd_steps)
{
    assert(n_cd_steps > 0u);
    assert(std::isfinite(maxT) && maxT > 0);
    assert(std::isfinite(conj_det_interval) && conj_det_interval > 0);
    assert(cd_idx < n_cd_steps);

    auto cbegin = conj_det_interval * static_cast<double>(cd_idx);
    // NOTE: for the last conjunction step we force the ending at maxT.
    auto cend = (cd_idx == n_cd_steps - 1u) ? maxT : (conj_det_interval * static_cast<double>(cd_idx + 1u));

    // LCOV_EXCL_START
    if (!std::isfinite(cbegin) || !std::isfinite(cend) || !(cend > cbegin) || cbegin < 0 || cend > maxT) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid conjunction step time range: [{}, {})", cbegin, cend));
    }
    // LCOV_EXCL_STOP

    return {cbegin, cend};
}

conjunctions::aabbs_span_t conjunctions::get_aabbs() const noexcept
{
    return aabbs_span_t{m_impl->m_aabbs_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs() + 1u};
}

dspan_1d<const double> conjunctions::get_cd_end_times() const noexcept
{
    return dspan_1d<const double>{m_impl->m_cd_end_times.data(), m_impl->m_n_cd_steps};
}

const polyjectory &conjunctions::get_polyjectory() const noexcept
{
    return m_impl->m_pj;
}

conjunctions::aabbs_span_t conjunctions::get_srt_aabbs() const noexcept
{
    return aabbs_span_t{m_impl->m_srt_aabbs_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs() + 1u};
}

dspan_2d<const std::uint64_t> conjunctions::get_mcodes() const noexcept
{
    return dspan_2d<const std::uint64_t>{m_impl->m_mcodes_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs()};
}

dspan_2d<const std::uint64_t> conjunctions::get_srt_mcodes() const noexcept
{
    return dspan_2d<const std::uint64_t>{m_impl->m_srt_mcodes_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs()};
}

dspan_2d<const std::uint32_t> conjunctions::get_srt_idx() const noexcept
{
    return dspan_2d<const std::uint32_t>{m_impl->m_srt_idx_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs()};
}

dspan_1d<const conjunctions::bvh_node> conjunctions::get_bvh_tree(std::size_t i) const
{
    if (i >= m_impl->m_tree_offsets.size()) [[unlikely]] {
        throw std::out_of_range(fmt::format("Cannot fetch the BVH tree for the conjunction timestep at index {}: the "
                                            "total number of conjunction steps is only {}",
                                            i, m_impl->m_tree_offsets.size()));
    }

    // Fetch the offset and size of the desired tree.
    const auto [tree_offset, tree_size] = m_impl->m_tree_offsets[i];

    // Compute the pointer to the desired tree.
    const auto *tree_ptr = m_impl->m_bvh_trees_ptr + tree_offset;

    // Return the span.
    return dspan_1d<const bvh_node>{tree_ptr, tree_size};
}

dspan_1d<const conjunctions::aabb_collision> conjunctions::get_aabb_collisions(std::size_t i) const
{
    if (i >= m_impl->m_bp_offsets.size()) [[unlikely]] {
        throw std::out_of_range(
            fmt::format("Cannot fetch the list of AABB collisions for the conjunction timestep at index {}: the "
                        "total number of conjunction steps is only {}",
                        i, m_impl->m_bp_offsets.size()));
    }

    // Fetch the offset and size of the desired collision list.
    const auto [bp_offset, bp_size] = m_impl->m_bp_offsets[i];

    // Compute the pointer to the desired collision list.
    const auto *bp_ptr = m_impl->m_bp_ptr + bp_offset;

    // Return the span.
    return dspan_1d<const aabb_collision>{bp_ptr, bp_size};
}

std::size_t conjunctions::get_n_cd_steps() const noexcept
{
    return m_impl->m_n_cd_steps;
}

dspan_1d<const conjunctions::conj> conjunctions::get_conjunctions() const noexcept
{
    if (m_impl->m_conjs_ptr == nullptr) {
        // No conjunctions detected.
        return dspan_1d<const conj>{nullptr, 0};
    } else {
        assert(m_impl->m_file_conjs.size() % sizeof(conj) == 0u);
        // NOTE: the static cast is ok because we made sure that
        // the total size of m_file_conjs can be represented by
        // std::size_t.
        return dspan_1d<const conj>{m_impl->m_conjs_ptr,
                                    static_cast<std::size_t>(m_impl->m_file_conjs.size() / sizeof(conj))};
    }
}

dspan_1d<const std::int32_t> conjunctions::get_otypes() const noexcept
{
    // NOTE: the static cast is ok because we ensured on construction
    // that the size of m_impl->m_otypes is equal to nobjs, which is
    // represented as a std::size_t.
    return dspan_1d<const std::int32_t>{m_impl->m_otypes.data(), static_cast<std::size_t>(m_impl->m_otypes.size())};
}

double conjunctions::get_conj_thresh() const noexcept
{
    return m_impl->m_conj_thresh;
}

double conjunctions::get_conj_det_interval() const noexcept
{
    return m_impl->m_conj_det_interval;
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
