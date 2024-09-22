#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "conjunctions.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

struct conjunctions::impl {
    double m_conj_thresh = 0;
    double m_conj_det_interval = 0;
    std::size_t m_n_cd_steps = 0;
    boost::unordered_flat_set<std::size_t> m_whitelist;
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

    m_impl = std::make_shared<const impl>(conj_thresh, conj_det_interval, n_cd_steps,
                                          boost::unordered_flat_set<std::size_t>(whitelist.begin(), whitelist.end()));
}

conjunctions::conjunctions(const conjunctions &) = default;

conjunctions::conjunctions(conjunctions &&) noexcept = default;

conjunctions &conjunctions::operator=(const conjunctions &) = default;

conjunctions &conjunctions::operator=(conjunctions &&) noexcept = default;

conjunctions::~conjunctions() = default;

} // namespace mizuba
