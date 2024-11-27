// Copyright 2024 Francesco Biscani
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
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/mdspan.hpp>

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
    // NOTE: the three dimensions here are, respectively:
    // - the total number of steps,
    // - the total number of state variables (which is always 7, i.e.,
    //   the Cartesian state vector + radius),
    // - the polynomial order + 1.
    using traj_span_t
        = heyoka::mdspan<const double, heyoka::extents<std::size_t, std::dynamic_extent, 7, std::dynamic_extent>>;
    using time_span_t = heyoka::mdspan<const double, heyoka::dextents<std::size_t, 1>>;

    // Datatype representing:
    // - the offset (in number of floating-point values) at which data for a
    //   specific trajectory begins in a datafile,
    // - the total number of steps in the trajectory.
    // This is used for locating trajectories into a data file.
    struct traj_offset {
        std::size_t offset;
        std::size_t n_steps;
    };

private:
    struct ptag {
    };
    explicit polyjectory(ptag,
                         std::tuple<std::vector<traj_span_t>, std::vector<time_span_t>, std::vector<std::int32_t>>,
                         double);

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
    explicit polyjectory(TrajRng &&traj_rng, TimeRng &&time_rng, StatusRng &&status_rng, double init_epoch)
        : polyjectory(ptag{},
                      ctor_impl(std::forward<TrajRng>(traj_rng), std::forward<TimeRng>(time_rng),
                                std::forward<StatusRng>(status_rng)),
                      init_epoch)
    {
    }
    explicit polyjectory(const std::filesystem::path &, const std::filesystem::path &, std::uint32_t,
                         std::vector<traj_offset>, std::vector<std::int32_t>, double);
    polyjectory(const polyjectory &);
    polyjectory(polyjectory &&) noexcept;
    polyjectory &operator=(const polyjectory &);
    polyjectory &operator=(polyjectory &&) noexcept;
    ~polyjectory();

    [[nodiscard]] std::size_t get_nobjs() const noexcept;
    [[nodiscard]] double get_maxT() const noexcept;
    [[nodiscard]] double get_init_epoch() const noexcept;
    [[nodiscard]] std::uint32_t get_poly_order() const noexcept;

    [[nodiscard]] std::tuple<traj_span_t, time_span_t, std::int32_t> operator[](std::size_t) const;
    using status_span_t = heyoka::mdspan<const std::int32_t, heyoka::extents<std::size_t, std::dynamic_extent>>;
    [[nodiscard]] status_span_t get_status() const noexcept;
};

} // namespace mizuba

#endif
