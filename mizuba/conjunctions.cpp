// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
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
#include <boost/unordered/unordered_flat_set.hpp>

#include "conjunctions.hpp"
#include "detail/file_utils.hpp"
#include "polyjectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

struct conjunctions::impl {
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
    // The whitelist.
    boost::unordered_flat_set<std::size_t> m_whitelist;
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
    // Vector of flags to signal which objects are active for conjunction tracking.
    std::vector<bool> m_conj_active;
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
    // Pointer to the beginning of m_file_aabbs, cast to float.
    const float *m_aabbs_base_ptr = nullptr;
    // Pointer to the beginning of m_file_srt_aabbs, cast to float.
    const float *m_srt_aabbs_base_ptr = nullptr;
    // Pointer to the beginning of m_file_mcodes, cast to std::uint64_t.
    const std::uint64_t *m_mcodes_base_ptr = nullptr;
    // Pointer to the beginning of m_file_srt_mcodes, cast to std::uint64_t.
    const std::uint64_t *m_srt_mcodes_base_ptr = nullptr;
    // Pointer to the beginning of m_file_srt_idx, cast to std::size_t.
    const std::size_t *m_srt_idx_base_ptr = nullptr;
    // Pointer to the beginning of m_file_bvh_trees, cast to bvh_node.
    const bvh_node *m_bvh_trees_ptr = nullptr;

    explicit impl(boost::filesystem::path temp_dir_path, polyjectory pj, double conj_thresh, double conj_det_interval,
                  std::size_t n_cd_steps, boost::unordered_flat_set<std::size_t> whitelist,
                  std::vector<double> cd_end_times, std::vector<std::tuple<std::size_t, std::size_t>> tree_offsets)
        : m_temp_dir_path(std::move(temp_dir_path)), m_pj(std::move(pj)), m_conj_thresh(conj_thresh),
          m_conj_det_interval(conj_det_interval), m_n_cd_steps(n_cd_steps), m_whitelist(std::move(whitelist)),
          m_cd_end_times(std::move(cd_end_times)), m_tree_offsets(std::move(tree_offsets)),
          m_file_aabbs((m_temp_dir_path / "aabbs").string()),
          m_file_srt_aabbs((m_temp_dir_path / "srt_aabbs").string()),
          m_file_mcodes((m_temp_dir_path / "mcodes").string()),
          m_file_srt_mcodes((m_temp_dir_path / "srt_mcodes").string()),
          m_file_srt_idx((m_temp_dir_path / "vidx").string()), m_file_bvh_trees((m_temp_dir_path / "bvh").string())
    {
        // Sanity check.
        assert(m_cd_end_times.size() == m_tree_offsets.size());

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

        m_srt_idx_base_ptr = reinterpret_cast<const std::size_t *>(m_file_srt_idx.data());
        assert(boost::alignment::is_aligned(m_srt_idx_base_ptr, alignof(std::size_t)));

        m_bvh_trees_ptr = reinterpret_cast<const bvh_node *>(m_file_bvh_trees.data());
        assert(boost::alignment::is_aligned(m_bvh_trees_ptr, alignof(bvh_node)));

        // Check the whitelist and setup m_conj_active.
        const auto nobjs = m_pj.get_nobjs();
        if (m_whitelist.empty()) {
            m_conj_active.resize(boost::numeric_cast<decltype(m_conj_active.size())>(nobjs), true);
        } else {
            m_conj_active.resize(boost::numeric_cast<decltype(m_conj_active.size())>(nobjs), false);

            for (const auto obj_idx : m_whitelist) {
                if (obj_idx >= nobjs) [[unlikely]] {
                    throw std::invalid_argument(
                        fmt::format("Invalid whitelist detected: the whitelist contains the object index {}, but the "
                                    "total number of objects is only {}",
                                    obj_idx, nobjs));
                }

                m_conj_active[obj_idx] = true;
            }
        }
    }

    ~impl()
    {
        // Close all memory-mapped files.
        m_file_aabbs.close();
        m_file_srt_aabbs.close();
        m_file_mcodes.close();
        m_file_srt_mcodes.close();
        m_file_srt_idx.close();
        m_file_bvh_trees.close();

        // Remove the temp dir and everything within.
        boost::filesystem::remove_all(m_temp_dir_path);
    }
};

conjunctions::conjunctions(ptag, polyjectory pj, double conj_thresh, double conj_det_interval,
                           std::vector<std::size_t> whitelist)
{
    // Check conj_thresh.
    if (!std::isfinite(conj_thresh) || conj_thresh <= 0) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The conjunction threshold must be finite and positive, but instead a value of {} was provided",
                        conj_thresh));
    }

    // Check conj_det_interval.
    if (!std::isfinite(conj_det_interval) || conj_det_interval <= 0) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The conjunction detection interval must be finite and positive, but instead a value of {} was provided",
            conj_det_interval));
    }

    // Determine the number of conjunction detection steps.
    const auto n_cd_steps = boost::numeric_cast<std::size_t>(std::ceil(pj.get_maxT() / conj_det_interval));

    // Assemble a "unique" dir path into the system temp dir. This will be the root dir
    // for all conjunctions data.
    const auto tmp_dir_path = detail::create_temp_dir("mizuba_conjunctions-%%%%-%%%%-%%%%-%%%%");

    // From now on, we have to wrap everything in a try/catch in order to ensure
    // proper cleanup of the temp dir in case of exceptions.
    try {
        // Change the permissions so that only the owner has access.
        boost::filesystem::permissions(tmp_dir_path, boost::filesystem::owner_all);

        // Run the computation of the aabbs.
        auto cd_end_times = compute_aabbs(pj, tmp_dir_path, n_cd_steps, conj_thresh, conj_det_interval);

        // Morton encoding and indirect sorting.
        morton_encode_sort_parallel(pj, tmp_dir_path, n_cd_steps);

        // Construct the bvh trees.
        auto tree_offsets = construct_bvh_trees_parallel(pj, tmp_dir_path, n_cd_steps);

        // Create the impl.
        m_impl
            = std::make_shared<const impl>(tmp_dir_path, std::move(pj), conj_thresh, conj_det_interval, n_cd_steps,
                                           boost::unordered_flat_set<std::size_t>(whitelist.begin(), whitelist.end()),
                                           std::move(cd_end_times), std::move(tree_offsets));

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
                                                     std::size_t n_cd_steps) const
{
    assert(n_cd_steps > 0u);
    assert(std::isfinite(maxT) && maxT > 0);
    assert(std::isfinite(conj_det_interval) && conj_det_interval > 0);

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

conjunctions::cd_end_times_span_t conjunctions::get_cd_end_times() const noexcept
{
    return cd_end_times_span_t{m_impl->m_cd_end_times.data(), m_impl->m_n_cd_steps};
}

const polyjectory &conjunctions::get_polyjectory() const noexcept
{
    return m_impl->m_pj;
}

conjunctions::aabbs_span_t conjunctions::get_srt_aabbs() const noexcept
{
    return aabbs_span_t{m_impl->m_srt_aabbs_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs() + 1u};
}

conjunctions::mcodes_span_t conjunctions::get_mcodes() const noexcept
{
    return mcodes_span_t{m_impl->m_mcodes_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs()};
}

conjunctions::mcodes_span_t conjunctions::get_srt_mcodes() const noexcept
{
    return mcodes_span_t{m_impl->m_srt_mcodes_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs()};
}

conjunctions::srt_idx_span_t conjunctions::get_srt_idx() const noexcept
{
    return srt_idx_span_t{m_impl->m_srt_idx_base_ptr, m_impl->m_n_cd_steps, m_impl->m_pj.get_nobjs()};
}

conjunctions::tree_span_t conjunctions::get_bvh_tree(std::size_t i) const
{
    if (i >= m_impl->m_tree_offsets.size()) [[unlikely]] {
        throw std::out_of_range(fmt::format("Invalid tree index {} specified - the total number of trees is only {}", i,
                                            m_impl->m_tree_offsets.size()));
    }

    // Fetch the offset and size of the desired tree.
    const auto [tree_offset, tree_size] = m_impl->m_tree_offsets[i];

    // Compute the pointer to the desired tree.
    const auto *tree_ptr = m_impl->m_bvh_trees_ptr + tree_offset;

    // Return the span.
    return conjunctions::tree_span_t{tree_ptr, tree_size};
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
