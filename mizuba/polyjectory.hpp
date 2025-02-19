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

#ifndef MIZUBA_POLYJECTORY_HPP
#define MIZUBA_POLYJECTORY_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/mdspan.hpp>

#include "mdspan.hpp"

namespace mizuba
{

// Fwd declaration.
class polyjectory;

namespace detail
{

struct polyjectory_impl;

void close_pj(std::shared_ptr<polyjectory_impl> &) noexcept;
[[nodiscard]] const std::shared_ptr<polyjectory_impl> &fetch_pj_impl(const polyjectory &) noexcept;

} // namespace detail

class polyjectory
{
    std::shared_ptr<detail::polyjectory_impl> m_impl;

    friend const std::shared_ptr<detail::polyjectory_impl> &detail::fetch_pj_impl(const polyjectory &) noexcept;

public:
    // Span representing a single trajectory. The three dimensions here are, respectively:
    //
    // - the total number of steps,
    // - the polynomial order + 1,
    // - the total number of state variables (which is always 7, i.e.,
    //   the Cartesian state vector + radius).
    using traj_span_t
        = heyoka::mdspan<const double, heyoka::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, 7>>;

    // Span representing time data for a trajectory. If the trajectory has no steps,
    // then the span will be empty. Otherwise, the first value in the span
    // will represent the start time of the trajectory (relative to the polyjectory
    // epoch), while the remaining values will repesent the step end times (relative to
    // the polyjectory epoch).
    using time_span_t = dspan_1d<const double>;

    // Datatype representing:
    //
    // - the offset (in number of floating-point values) at which data for a
    //   specific trajectory begins in a datafile,
    // - the total number of steps in the trajectory.
    //
    // This is used for locating trajectories into a data file.
    struct traj_offset {
        std::size_t offset;
        std::size_t n_steps;
    };

private:
    struct ptag {
    };
    explicit polyjectory(ptag, const std::filesystem::path &);
    explicit polyjectory(ptag,
                         std::tuple<std::vector<traj_span_t>, std::vector<time_span_t>, std::vector<std::int32_t>>,
                         double, double, std::optional<std::filesystem::path>, bool);

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

    void check_attached() const;

public:
    template <typename TrajRng, typename TimeRng, typename StatusRng>
        requires std::ranges::input_range<TrajRng>
                 && std::same_as<traj_span_t, std::remove_cvref_t<std::ranges::range_reference_t<TrajRng>>>
                 && std::ranges::input_range<TimeRng>
                 && std::same_as<time_span_t, std::remove_cvref_t<std::ranges::range_reference_t<TimeRng>>>
                 && std::ranges::input_range<StatusRng>
                 && std::same_as<std::int32_t, std::remove_cvref_t<std::ranges::range_reference_t<StatusRng>>>
    explicit polyjectory(TrajRng &&traj_rng, TimeRng &&time_rng, StatusRng &&status_rng, double epoch, double epoch2,
                         std::optional<std::filesystem::path> data_dir, bool persist)
        : polyjectory(ptag{},
                      ctor_impl(std::forward<TrajRng>(traj_rng), std::forward<TimeRng>(time_rng),
                                std::forward<StatusRng>(status_rng)),
                      epoch, epoch2, std::move(data_dir), persist)
    {
    }
    explicit polyjectory(const std::filesystem::path &, const std::filesystem::path &, std::uint32_t,
                         std::vector<traj_offset>, std::vector<std::int32_t>, double, double,
                         std::optional<std::filesystem::path>, bool);
    polyjectory(const polyjectory &) noexcept;
    polyjectory(polyjectory &&) noexcept;
    polyjectory &operator=(const polyjectory &) noexcept;
    polyjectory &operator=(polyjectory &&) noexcept;
    ~polyjectory();

    [[nodiscard]] static polyjectory mount(const std::filesystem::path &);
    void detach() noexcept;
    [[nodiscard]] bool is_detached() const noexcept;

    [[nodiscard]] std::size_t get_nobjs() const;
    [[nodiscard]] double get_maxT() const;
    [[nodiscard]] std::pair<double, double> get_epoch() const;
    [[nodiscard]] std::uint32_t get_poly_order() const;
    [[nodiscard]] std::filesystem::path get_data_dir() const;

    [[nodiscard]] std::tuple<traj_span_t, time_span_t, std::int32_t> operator[](std::size_t) const;
    [[nodiscard]] dspan_1d<const std::int32_t> get_status() const;

    [[nodiscard]] bool get_persist() const;

    // Span used to store the output of polyjectory evaluation with a *single time* per satellite.
    // The two dimensions are, respectively:
    //
    // - the number of objects,
    // - the 7 state variables.
    using single_eval_span_t = heyoka::mdspan<double, heyoka::extents<std::size_t, std::dynamic_extent, 7>>;

private:
    template <typename Time>
    void state_eval_impl(single_eval_span_t, Time, std::optional<dspan_1d<const std::size_t>>) const;

public:
    void state_eval(single_eval_span_t, double, std::optional<dspan_1d<const std::size_t>>) const;
    void state_eval(single_eval_span_t, dspan_1d<const double>, std::optional<dspan_1d<const std::size_t>>) const;
    // Span used to store the output of polyjectory evaluation with *multiple times* per satellite.
    // The three dimensions are, respectively:
    //
    // - the number of objects,
    // - the number of time evaluations per object,
    // - the 7 state variables.
    using multi_eval_span_t
        = heyoka::mdspan<double, heyoka::extents<std::size_t, std::dynamic_extent, std::dynamic_extent, 7>>;

private:
    template <typename Time>
    void state_meval_impl(multi_eval_span_t, Time, std::optional<dspan_1d<const std::size_t>>) const;

public:
    void state_meval(multi_eval_span_t, dspan_1d<const double>, std::optional<dspan_1d<const std::size_t>>) const;
    void state_meval(multi_eval_span_t, dspan_2d<const double>, std::optional<dspan_1d<const std::size_t>>) const;
};

} // namespace mizuba

#endif
