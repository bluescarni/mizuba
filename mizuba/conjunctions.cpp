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
#include <memory>
#include <stdexcept>
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
    // The memory-mapped file for the aabbs.
    boost::iostreams::mapped_file_source m_file_aabbs;
    // Pointer to the beginning of m_file_aabbs, cast to float.
    const float *m_aabbs_base_ptr = nullptr;

    explicit impl(boost::filesystem::path temp_dir_path, polyjectory pj, double conj_thresh, double conj_det_interval,
                  std::size_t n_cd_steps, boost::unordered_flat_set<std::size_t> whitelist,
                  std::vector<double> cd_end_times)
        : m_temp_dir_path(std::move(temp_dir_path)), m_pj(std::move(pj)), m_conj_thresh(conj_thresh),
          m_conj_det_interval(conj_det_interval), m_n_cd_steps(n_cd_steps), m_whitelist(std::move(whitelist)),
          m_cd_end_times(std::move(cd_end_times)), m_file_aabbs((m_temp_dir_path / "aabbs").string())
    {
        // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
        // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
        m_aabbs_base_ptr = reinterpret_cast<const float *>(m_file_aabbs.data());
        assert(boost::alignment::is_aligned(m_aabbs_base_ptr, alignof(float)));
    }

    ~impl()
    {
        // Close all memory-mapped files.
        m_file_aabbs.close();

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
        morton_encode_sort_parallel();

        // Create the impl.
        m_impl = std::make_shared<const impl>(
            tmp_dir_path, std::move(pj), conj_thresh, conj_det_interval, n_cd_steps,
            boost::unordered_flat_set<std::size_t>(whitelist.begin(), whitelist.end()), std::move(cd_end_times));

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

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
