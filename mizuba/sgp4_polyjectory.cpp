// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/taylor.hpp>

#include "polyjectory.hpp"

namespace mizuba
{

namespace detail
{

namespace
{

// The exit radius (in km).
constexpr auto exit_radius = 8000.;

// The reentry radius, 150km over the average surface.
constexpr auto reentry_radius = 6371 + 150.;

// Helper to check that the initial states of all satellites at jd_begin are well-formed. If there are no
// issues, the initial states will be returned as an mdspan together with the underlying buffer.
auto check_initial_sat_state(
    heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>> sat_data, double jd_begin)
{
    using prop_t = heyoka::model::sgp4_propagator<double>;

    // Construct the propagator for checking.
    prop_t check_prop(sat_data);

    // Init the dates vector.
    std::vector<prop_t::date> dates;
    dates.resize(boost::numeric_cast<decltype(dates.size())>(sat_data.extent(0)), prop_t::date{.jd = jd_begin});

    // Init the dates span.
    prop_t::in_1d<prop_t::date> dates_span(dates.data(), boost::numeric_cast<std::size_t>(dates.size()));

    // Prepare the output buffer.
    std::vector<double> out;
    out.resize(boost::safe_numerics::safe<decltype(out.size())>(check_prop.get_nsats()) * check_prop.get_nouts());

    // Create the output span.
    prop_t::out_2d out_span(out.data(), boost::numeric_cast<std::size_t>(check_prop.get_nouts()),
                            boost::numeric_cast<std::size_t>(check_prop.get_nsats()));

    // Propagate.
    check_prop(out_span, dates_span);

    // Check the otutput.
    // NOTE: this can be easily parallelised if needed.
    for (std::size_t i = 0; i < out_span.extent(1); ++i) {
        // Check finiteness of the state.
        const auto x = out_span(0, i);
        const auto y = out_span(1, i);
        const auto z = out_span(2, i);

        const auto vx = out_span(3, i);
        const auto vy = out_span(4, i);
        const auto vz = out_span(5, i);

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(vx) || !std::isfinite(vy)
            || !std::isfinite(vz)) [[unlikely]] {
            throw std::invalid_argument(fmt::format(
                "The sgp4 propagation of the object at index {} at jd_begin generated a non-finite state vector", i));
        }

        // Check the distance from the Earth.
        const auto dist = std::sqrt(x * x + y * y + z * z);
        if (!std::isfinite(dist) || dist >= exit_radius || dist <= reentry_radius) [[unlikely]] {
            throw std::invalid_argument(fmt::format("The sgp4 propagation of the object at index {} at jd_begin "
                                                    "generated a position vector with invalid radius {}",
                                                    i, dist));
        }

        // Check the sgp4 error code.
        if (out_span(6, i) != 0.) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("The sgp4 propagation of the object at index {} at jd_begin generated the error code {}", i,
                            static_cast<int>(out_span(6, i))));
        }
    }

    // Return the initial states.
    return std::make_pair(out_span, std::move(out));
}

// Construct the ODE system that will be used for the propagation of a single object
// via the sgp4 dynamics.
std::vector<std::pair<heyoka::expression, heyoka::expression>> construct_sgp4_ode()
{
    // Fetch sgp4's formulae.
    auto sgp4_func = heyoka::model::sgp4();

    // The variable representing tsince in the sgp4 formulae.
    const auto tsince = heyoka::expression("tsince");

    // In sgp4_func, replace the TLE data with params, and tsince
    // with tsince + par[7].
    sgp4_func = heyoka::subs(sgp4_func, {{"n0", heyoka::par[0]},
                                         {"e0", heyoka::par[1]},
                                         {"i0", heyoka::par[2]},
                                         {"node0", heyoka::par[3]},
                                         {"omega0", heyoka::par[4]},
                                         {"m0", heyoka::par[5]},
                                         {"bstar", heyoka::par[6]},
                                         {"tsince", tsince + heyoka::par[7]}});

    // Compute the rhs of the sgp4 ODE, substituting tsince with the time placeholder.
    const auto dt = heyoka::diff_tensors(sgp4_func, {tsince});
    auto sgp4_rhs = heyoka::subs(dt.get_jacobian(), {{tsince, heyoka::time}});

    // Create the state variables for the ODE.
    auto [x, y, z, vx, vy, vz, e, r] = heyoka::make_vars("x", "y", "z", "vx", "vy", "vz", "e", "r");

    // Add the differential equation for r.
    sgp4_rhs.push_back(heyoka::sum({x * vx, y * vy, z * vz}) / r);

    // Return the ODE sys.
    using heyoka::prime;
    return {prime(x) = sgp4_rhs[0],  prime(y) = sgp4_rhs[1],  prime(z) = sgp4_rhs[2], prime(vx) = sgp4_rhs[3],
            prime(vy) = sgp4_rhs[4], prime(vz) = sgp4_rhs[5], prime(e) = sgp4_rhs[6], prime(r) = sgp4_rhs[7]};
}

// Construct the template integrator for ensemble propagations.
template <typename ODESys>
auto construct_ode_propagator(const ODESys &sys)
{
    using ta_t = heyoka::taylor_adaptive_batch<double>;

    // Create the reentry and exit events.
    std::vector<ta_t::t_event_t> t_events;
    const auto r = heyoka::make_vars("r");

    // The reentry event.
    t_events.emplace_back(pow(r, 2.) - reentry_radius * reentry_radius,
                          // NOTE: direction is negative in order to detect only crashing into.
                          heyoka::kw::direction = heyoka::event_direction::negative);

    // The exit event.
    t_events.emplace_back(pow(r, 2.) - exit_radius * exit_radius,
                          // NOTE: direction is positive in order to detect only domain exit (not entrance).
                          heyoka::kw::direction = heyoka::event_direction::positive);

    // Fetch the batch size.
    const auto batch_size = heyoka::recommended_simd_size<double>();

    std::vector<double> init_state;
    init_state.resize(boost::safe_numerics::safe<unsigned>(8) * batch_size);

    return ta_t(sys, std::move(init_state), batch_size, heyoka::kw::t_events = std::move(t_events));
}

} // namespace

} // namespace detail

polyjectory
sgp4_polyjectory(heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>> sat_data,
                 double jd_begin, double jd_end)
{
    // Check the date range.
    if (!std::isfinite(jd_begin) || !std::isfinite(jd_end) || !(jd_begin < jd_end)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid Julian date interval [{}, {}) supplied to sgp4_polyjectory(): the begin/end dates "
                        "must be finite and the end date must be strictly after the begin date",
                        jd_begin, jd_end));
    }

    // Fetch the states of the satellites at jd_begin.
    const auto [init_states, init_states_data] = detail::check_initial_sat_state(sat_data, jd_begin);

    // Construct the sgp4 ODE sys.
    const auto sgp4_ode = detail::construct_sgp4_ode();

    // Construct the template ODE integrator.
    const auto ta = detail::construct_ode_propagator(sgp4_ode);
}

} // namespace mizuba
