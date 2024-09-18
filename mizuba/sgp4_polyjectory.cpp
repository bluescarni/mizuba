// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/align.hpp>
#include <boost/filesystem/file_status.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/taylor.hpp>

#include "polyjectory.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace mizuba
{

namespace detail
{

namespace
{

// Helper to compute the initial states of all satellites at jd_begin. If there are no
// issues with the states, they will be returned as an mdspan together with the underlying buffer.
auto sgp4_compute_initial_sat_states(
    heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>> sat_data, double jd_begin,
    double exit_radius, double reentry_radius)
{
    using prop_t = heyoka::model::sgp4_propagator<double>;

    // Construct the propagator for checking.
    prop_t check_prop(sat_data);

    // Init the dates vector.
    std::vector<prop_t::date> dates;
    dates.resize(boost::numeric_cast<decltype(dates.size())>(sat_data.extent(1)), prop_t::date{.jd = jd_begin});

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
        // Check the sgp4 error code first.
        if (out_span(6, i) != 0.) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("The sgp4 propagation of the object at index {} at jd_begin generated the error code {}", i,
                            static_cast<int>(out_span(6, i))));
        }

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
    sgp4_func = heyoka::subs(sgp4_func,
                             std::unordered_map<std::string, heyoka::expression>{{"n0", heyoka::par[0]},
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
    // NOTE: do **not** use vx/vy/vz here. Apparently, in the SGP4 algorithm, if one takes the
    // time derivatives of x/y/z one does not get *exactly* the same values as the vx/vy/vz returned
    // by SGP4. In order for the differential equation for r to be correct, we need the the true time
    // derivatives of x/y/z, and we cannot use what SGP4 says are the velocities.
    sgp4_rhs.push_back(heyoka::sum({x * sgp4_rhs[0], y * sgp4_rhs[1], z * sgp4_rhs[2]}) / r);

    // Return the ODE sys.
    using heyoka::prime;
    return {prime(x) = sgp4_rhs[0],  prime(y) = sgp4_rhs[1],  prime(z) = sgp4_rhs[2], prime(vx) = sgp4_rhs[3],
            prime(vy) = sgp4_rhs[4], prime(vz) = sgp4_rhs[5], prime(e) = sgp4_rhs[6], prime(r) = sgp4_rhs[7]};
}

// Construct the ODE integrator, which includes the terminal events for the
// detection of reentry/exit.
template <typename ODESys>
auto construct_sgp4_ode_integrator(const ODESys &sys, double exit_radius, double reentry_radius)
{
    using ta_t = heyoka::taylor_adaptive_batch<double>;

    // Create the reentry and exit events.
    std::vector<ta_t::t_event_t> t_events;
    const auto r = heyoka::make_vars("r");

    // The reentry event.
    t_events.emplace_back(r - reentry_radius,
                          // NOTE: direction is negative in order to detect only crashing into.
                          heyoka::kw::direction = heyoka::event_direction::negative);

    // The exit event.
    t_events.emplace_back(r - exit_radius,
                          // NOTE: direction is positive in order to detect only domain exit (not entrance).
                          heyoka::kw::direction = heyoka::event_direction::positive);

    // Fetch the batch size.
    const auto batch_size = heyoka::recommended_simd_size<double>();

    // Initial state vector.
    std::vector<double> init_state;
    init_state.resize(boost::safe_numerics::safe<decltype(init_state.size())>(sys.size()) * batch_size);

    return ta_t(sys, std::move(init_state), batch_size, heyoka::kw::t_events = std::move(t_events),
                heyoka::kw::compact_mode = true
#if HEYOKA_VERSION_MAJOR >= 6
                ,
                heyoka::kw::parjit = true
#endif
    );
}

// Run the ODE integration according to the sgp4 dynamics, storing the Taylor coefficients
// and the step end times step-by-step.
template <typename TA, typename Path, typename SatData, typename InitStates>
auto perform_ode_integration(const TA &tmpl_ta, const Path &tmp_dir_path, SatData sat_data, double jd_begin,
                             double jd_end, InitStates init_states)
{
    // Cache the batch size.
    const auto batch_size = tmpl_ta.get_batch_size();

    // Cache the total number of satellites.
    const auto n_sats = sat_data.extent(1);
    assert(n_sats > 0u);

    // Create the sgp4 propagator that we will use to update the sate vector of the ODE
    // integration step by step.
    using prop_t = heyoka::model::sgp4_propagator<double>;
    const auto tmpl_prop = [&]() {
        // NOTE: in order for the construction to be successful, we need to fetch
        // some sensible TLE data. We choose the TLE data from the first satellite
        // in sat_data, and we splat it out in the batch layout.
        std::vector<double> tle_vec;
        tle_vec.reserve(boost::safe_numerics::safe<decltype(tle_vec.size())>(batch_size) * 9);
        for (std::size_t i = 0; i < 9u; ++i) {
            for (std::uint32_t j = 0; j < batch_size; ++j) {
                tle_vec.push_back(sat_data(i, 0));
            }
        }

        return prop_t(heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>>{
            tle_vec.data(), boost::numeric_cast<std::size_t>(batch_size)});
    }();

    // The number of batch-sized blocks into which we divide the satellites.
    // NOTE: round up if batch_size does not divide exactly n_sats.
    const auto n_blocks = n_sats / batch_size + static_cast<unsigned>((n_sats % batch_size) != 0u);

    // Bag of thread-specific data to be used in the batch parallel iterations.
    struct batch_data {
        // The ODE integrator.
        TA ta;
        // The sgp4 propagator.
        prop_t prop;
        // Buffer used to set up new data in prop.
        std::vector<double> new_sat_data;
        // Vector of output files.
        std::vector<std::pair<std::ofstream, std::ofstream>> out_files;
        // Vector of max timestep sizes.
        std::vector<double> max_h;
        // Vector of active batch element flags.
        std::vector<int> active_flags;
        // Vector of Julian dates for use in the sgp4 prop.
        std::vector<prop_t::date> jdates;
        // Buffers used to temporarily store the Taylor coefficients
        // and the time data for each element of a batch. The contents
        // of these buffers will be flushed to file at the end of
        // the integration of a batch.
        std::vector<std::pair<std::vector<double>, std::vector<double>>> tmp_write_buffers;
    };

    // Set up the thread-specific data machinery via TBB.
    using ets_t = oneapi::tbb::enumerable_thread_specific<batch_data, oneapi::tbb::cache_aligned_allocator<batch_data>,
                                                          oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
    ets_t ets([&tmpl_ta, &tmpl_prop, batch_size]() {
        // Make a copy of the template integrator.
        auto ta = tmpl_ta;

        // Make a copy of the template sgp4 propagator.
        auto prop = tmpl_prop;

        // Prepare new_sat_data with the appropriate size.
        std::vector<double> new_sat_data;
        new_sat_data.resize(boost::safe_numerics::safe<decltype(new_sat_data.size())>(batch_size) * 9);

        // Prepare storage for the output files.
        std::vector<std::pair<std::ofstream, std::ofstream>> out_files;
        out_files.reserve(boost::numeric_cast<decltype(out_files.size())>(batch_size));

        // Prepare max_h with the appropriate size.
        std::vector<double> max_h;
        max_h.resize(boost::numeric_cast<decltype(max_h.size())>(batch_size));

        // Prepare active_flags with the appropriate size.
        std::vector<int> active_flags;
        active_flags.resize(boost::numeric_cast<decltype(active_flags.size())>(batch_size));

        // Prepare jdates with the appropriate size.
        std::vector<prop_t::date> jdates;
        jdates.resize(boost::numeric_cast<decltype(jdates.size())>(batch_size));

        // Prepare tmp_write_buffers with the appropriate size.
        std::vector<std::pair<std::vector<double>, std::vector<double>>> tmp_write_buffers;
        tmp_write_buffers.resize(boost::numeric_cast<decltype(tmp_write_buffers.size())>(batch_size));

        return batch_data{.ta = std::move(ta),
                          .prop = std::move(prop),
                          .new_sat_data = std::move(new_sat_data),
                          .out_files = std::move(out_files),
                          .max_h = std::move(max_h),
                          .active_flags = std::move(active_flags),
                          .jdates = std::move(jdates),
                          .tmp_write_buffers = std::move(tmp_write_buffers)};
    });

    // A **global** vector of statuses, one per object.
    // We do not need to protect writes into this, as each status
    // will be written to exactly at most once.
    // NOTE: this is zero-inited, meaning that the default status flag
    // of each object is "no error detected".
    std::vector<int> global_status;
    global_status.resize(boost::numeric_cast<decltype(global_status.size())>(n_sats));

    // Run the numerical integration.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n_blocks), [&ets, batch_size, n_sats, sat_data,
                                                                                     init_states, jd_begin, jd_end,
                                                                                     &tmp_dir_path, &global_status](
                                                                                        const auto &range) {
        using safe_size_t = boost::safe_numerics::safe<std::size_t>;
        using dfloat = heyoka::detail::dfloat<double>;

        // Fetch the thread-local data.
        auto &[ta, prop, new_sat_data, out_files, max_h, active_flags, jdates, tmp_write_buffers] = ets.local();

        // NOTE: isolate to avoid issues with thread-local data. See:
        // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
        oneapi::tbb::this_task_arena::isolate([&]() {
            // View on the state variables.
            const auto state_view = heyoka::mdspan<double, heyoka::dextents<std::size_t, 2>>{
                ta.get_state_data(), boost::numeric_cast<std::size_t>(ta.get_dim()),
                boost::numeric_cast<std::size_t>(batch_size)};

            // View on the pars.
            const auto pars_view = heyoka::mdspan<double, heyoka::dextents<std::size_t, 2>>{
                ta.get_pars_data(), 8, boost::numeric_cast<std::size_t>(batch_size)};

            // View on the Taylor coefficients.
            const auto tc_view = heyoka::mdspan<const double, heyoka::dextents<std::size_t, 3>>{
                ta.get_tc().data(), boost::numeric_cast<std::size_t>(ta.get_dim()),
                boost::numeric_cast<std::size_t>(ta.get_order() + 1u), boost::numeric_cast<std::size_t>(batch_size)};

            for (auto block_idx = range.begin(); block_idx != range.end(); ++block_idx) {
                // Compute begin/end indices for this block.
                const auto b_idx = block_idx * batch_size;
                auto e_idx = static_cast<std::size_t>(safe_size_t(b_idx) + batch_size);

                // Clamp e_idx if we are dealing with the irregular block.
                if (e_idx > n_sats) {
                    e_idx = n_sats;
                }

                // The effective batch size.
                const auto e_batch_size = static_cast<std::uint32_t>(e_idx - b_idx);

                // Prepare out_files and tmp_write_buffers.
                out_files.clear();
                for (std::uint32_t i = 0; i < e_batch_size; ++i) {
                    const auto tc_path = tmp_dir_path / fmt::format("tc_{}", b_idx + i);
                    // LCOV_EXCL_START
                    if (boost::filesystem::exists(tc_path)) [[unlikely]] {
                        throw std::runtime_error(
                            fmt::format("Cannot create the storage file '{}', as it exists already", tc_path.string()));
                    }
                    // LCOV_EXCL_STOP
                    std::ofstream tc_file(tc_path.string(), std::ios::binary | std::ios::out);
                    tc_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

                    const auto time_path = tmp_dir_path / fmt::format("time_{}", b_idx + i);
                    // LCOV_EXCL_START
                    if (boost::filesystem::exists(time_path)) [[unlikely]] {
                        throw std::runtime_error(fmt::format(
                            "Cannot create the storage file '{}', as it exists already", time_path.string()));
                    }
                    // LCOV_EXCL_STOP
                    std::ofstream time_file(time_path.string(), std::ios::binary | std::ios::out);
                    time_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

                    out_files.emplace_back(std::move(tc_file), std::move(time_file));

                    // Clear out the temp buffers.
                    tmp_write_buffers[i].first.clear();
                    tmp_write_buffers[i].second.clear();
                }

                // Set up:
                // - the propagator TLEs,
                // - the initial state in the integrator,
                // - the parameter array in the integrator.
                for (std::uint32_t i = 0; i < e_batch_size; ++i) {
                    // The propagator TLEs.
                    for (decltype(new_sat_data.size()) j = 0u; j < 9u; ++j) {
                        new_sat_data[i + j * batch_size] = sat_data(j, b_idx + i);
                    }

                    // Initial Cartesian state + error code.
                    for (std::size_t j = 0; j < 7u; ++j) {
                        state_view(j, i) = init_states(j, b_idx + i);
                    }
                    // NOTE: the error code should always be zero
                    // since we checked the initial states.
                    assert(state_view(6, i) == 0.);
                    // Set the initial radius.
                    state_view(7, i)
                        = std::sqrt(state_view(0, i) * state_view(0, i) + state_view(1, i) * state_view(1, i)
                                    + state_view(2, i) * state_view(2, i));

                    // Pars. These are 8 in total: 6 TLE elements, the bstar
                    // and the time offset.
                    for (std::size_t j = 0; j < 7u; ++j) {
                        pars_view(j, i) = sat_data(j, b_idx + i);
                    }
                    // Compute and set the time offset.
                    pars_view(7, i)
                        = static_cast<double>(dfloat(jd_begin)
                                              - normalise(dfloat(sat_data(7, b_idx + i), sat_data(8, b_idx + i))))
                          * 1440;
                }

                // Fill the remainder of the irregular block with data
                // from the first satellite.
                for (auto i = e_batch_size; i < batch_size; ++i) {
                    for (decltype(new_sat_data.size()) j = 0u; j < 9u; ++j) {
                        new_sat_data[i + j * batch_size] = sat_data(j, 0);
                    }

                    for (std::size_t j = 0; j < 7u; ++j) {
                        state_view(j, i) = init_states(j, 0);
                    }
                    state_view(7, i)
                        = std::sqrt(state_view(0, i) * state_view(0, i) + state_view(1, i) * state_view(1, i)
                                    + state_view(2, i) * state_view(2, i));

                    for (std::size_t j = 0; j < 7u; ++j) {
                        pars_view(j, i) = sat_data(j, 0);
                    }
                    pars_view(7, i)
                        = static_cast<double>(dfloat(jd_begin) - normalise(dfloat(sat_data(7, 0), sat_data(8, 0))))
                          * 1440;
                }

                // Set the new sat data in the propagator.
                prop.replace_sat_data(
                    heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>>{
                        new_sat_data.data(), boost::numeric_cast<std::size_t>(batch_size)});

                // Initial setup of max_h and active flags.
                const auto final_time = (jd_end - jd_begin) * 1440;
                for (std::uint32_t i = 0; i < e_batch_size; ++i) {
                    max_h[i] = final_time;
                    active_flags[i] = 1;
                }
                // NOTE: for the remainder of the irregular block, set the max timestep
                // to zero as we do not need to perform any integration.
                for (auto i = e_batch_size; i < batch_size; ++i) {
                    max_h[i] = 0;
                    active_flags[i] = 0;
                }

                // Reset all cooldowns.
                ta.reset_cooldowns();

                // Setup the initial time.
                ta.set_time(0);

                // Run the integration step by step.
                while (true) {
                    // First we take the step, capping the timestep size and writing the Taylor coefficients.
                    ta.step(max_h, true);

                    // Next, we want to re-compute the state at the end of the timestep with the sgp4
                    // propagator. This avoids the accumulation of numerical errors
                    // due to ODE integration.

                    // Setup the Julian dates for the propagation.
                    for (std::uint32_t i = 0; i < batch_size; ++i) {
                        const auto jd = dfloat(jd_begin)
                                        + dfloat(ta.get_dtime().first[i] / 1440., ta.get_dtime().second[i] / 1440.);
                        jdates[i] = {jd.hi, jd.lo};
                    }

                    // Propagate writing directly into the state vector of the integrator.
                    const auto prop_out_span
                        = std::experimental::submdspan(state_view, std::pair{0, 7}, std::experimental::full_extent);
                    prop(prop_out_span, heyoka::mdspan<const prop_t::date, heyoka::dextents<std::size_t, 1>>(
                                            jdates.data(), boost::numeric_cast<std::size_t>(batch_size)));

                    // Compute by hand the value of the radius.
                    for (std::uint32_t i = 0; i < batch_size; ++i) {
                        state_view(7, i)
                            = std::sqrt(state_view(0, i) * state_view(0, i) + state_view(1, i) * state_view(1, i)
                                        + state_view(2, i) * state_view(2, i));
                    }

                    // Analyze the integration/propagation outcomes, keeping
                    // track of how many objects are still active at the
                    // end of the step.
                    std::uint32_t n_active = 0;
                    for (std::uint32_t i = 0; i < e_batch_size; ++i) {
                        // No need to do anything for an inactive object.
                        if (active_flags[i] == 0) {
                            continue;
                        }

                        // Fetch the SGP4 error code.
                        // NOTE: this is the error code coming out of the propagator,
                        // not the ODE integrator, as we just replaced by hand the
                        // state vector.
                        const auto ecode = state_view(6, i);

                        if (ecode != 0. && ecode != 6.) {
                            // sgp4 propagation error: set the inactive flag, zero out
                            // max_h, set the status, and continue.
                            active_flags[i] = 0;
                            max_h[i] = 0;
                            global_status[b_idx + i] = 10 + static_cast<int>(ecode);

                            continue;
                        }

                        // Fetch the ODE integration outcome.
                        const auto oc = std::get<0>(ta.get_step_res()[i]);

                        if (oc == heyoka::taylor_outcome::success) {
                            // If the outcome is success, then we are not done
                            // with the current object and we still need to integrate more.
                            ++n_active;

                            // Update the remaining time.
                            const auto rem_time = static_cast<double>(
                                final_time - dfloat(ta.get_dtime().first[i], ta.get_dtime().second[i]));
                            max_h[i] = rem_time;
                        } else if (oc == heyoka::taylor_outcome::time_limit) {
                            // We safely arrived at the time limit, deactivate. No need to
                            // set status as it is by default zero already.
                            active_flags[i] = 0;
                            max_h[i] = 0;
                        } else if (oc < heyoka::taylor_outcome{0}) {
                            // Stopping terminal event. Deactivate and set status.
                            assert(oc == heyoka::taylor_outcome{-1} || oc == heyoka::taylor_outcome{-2});
                            active_flags[i] = 0;
                            max_h[i] = 0;
                            global_status[b_idx + i] = (oc == heyoka::taylor_outcome{-1}) ? 1 : 2;
                        } else {
                            // If we generated a non-finite state during the integration,
                            // we cannot trust the output. Set the inactive flag, zero out
                            // max_h, set the status, and continue.
                            assert(oc == heyoka::taylor_outcome::err_nf_state);
                            active_flags[i] = 0;
                            max_h[i] = 0;
                            global_status[b_idx + i] = 3;

                            continue;
                        }

                        if (ecode == 6.) {
                            // NOTE: if we get here, we are in the following situation:
                            //
                            // - neither SGP4 nor our ODE integration errored out,
                            // - the ODE integration might or might not have reached the time limit or
                            //   a stopping terminal event,
                            // - the SGP4 algorithm detected a decay.
                            //
                            // We treat this equivalently to a decay detected by the ODE
                            // integration.
                            active_flags[i] = 0;
                            max_h[i] = 0;
                            global_status[b_idx + i] = 1;
                        }

                        // NOTE: if we are here, it means that we need to record the Taylor
                        // coefficients and the step end times for the current object.
                        auto &[tc_buf, time_buf] = tmp_write_buffers[i];

                        for (std::size_t j = 0; j < tc_view.extent(0); ++j) {
                            // NOTE: we don't want to store the Taylor coefficients
                            // of the error code (at index 6).
                            if (j == 6u) {
                                continue;
                            }

                            for (std::size_t k = 0; k < tc_view.extent(1); ++k) {
                                tc_buf.push_back(tc_view(j, k, i));
                            }
                        }

                        time_buf.push_back(ta.get_time()[i]);
                    }

                    if (n_active == 0u) {
                        // No more active objects in the batch, exit the endless loop.
                        break;
                    }
                }

                // Flush the temp buffers to file.
                for (std::uint32_t i = 0; i < e_batch_size; ++i) {
                    auto &[tc_file, time_file] = out_files[i];
                    const auto &[tc_buf, time_buf] = tmp_write_buffers[i];

                    tc_file.write(reinterpret_cast<const char *>(tc_buf.data()),
                                  boost::safe_numerics::safe<std::streamsize>(tc_buf.size()) * sizeof(double));

                    time_file.write(reinterpret_cast<const char *>(time_buf.data()),
                                    boost::safe_numerics::safe<std::streamsize>(time_buf.size()) * sizeof(double));
                }
            }
        });
    });

    // Return the status flags.
    return global_status;
}

// Copy all trajectory/time data generated in perform_ode_integration() into
// a single storage file.
//
// NOTE: here we could perhaps improve performance via multi-threading:
//
// - determine the total size required for the single storage file and the offsets
//   for the Taylor coefficients/time data into the single storage file (single-thread),
// - create the single storage file,
// - mmap and write into it from multiple threads (multi-thread).
auto consolidate_data(const boost::filesystem::path &tmp_dir_path, std::size_t n_sats, std::uint32_t order)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Create and open the storage file.
    const auto storage_path = tmp_dir_path / "storage";
    // LCOV_EXCL_START
    if (boost::filesystem::exists(storage_path)) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("Cannot create the storage file '{}', as it exists already", storage_path.string()));
    }
    // LCOV_EXCL_STOP
    std::ofstream storage_file(storage_path.string(), std::ios::binary | std::ios::out);
    // Make sure we throw on errors.
    storage_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    // This is a vector that will contain:
    // - the offset (in number of double-precision values) in the storage file
    //   at which the trajectory data for an object begins,
    // - the number of steps.
    std::vector<std::tuple<std::size_t, std::size_t>> traj_offset;
    traj_offset.reserve(n_sats);

    // Keep track of the offset in the storage file.
    safe_size_t cur_offset = 0;

    // Taylor coefficients.
    for (std::size_t i = 0; i < n_sats; ++i) {
        // Build the file path.
        const auto tc_path = tmp_dir_path / fmt::format("tc_{}", i);

        // Fetch the file size in bytes.
        assert(boost::filesystem::exists(tc_path));
        assert(boost::filesystem::is_regular_file(tc_path));
        const auto tc_size = boost::filesystem::file_size(tc_path);
        assert(tc_size % (safe_size_t(sizeof(double)) * (order + 1u) * 7u) == 0u);

        // Open it.
        std::ifstream tc_file(tc_path.string(), std::ios::binary | std::ios::in);
        tc_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        // Copy into storage_file.
        storage_file << tc_file.rdbuf();

        // Close the file.
        tc_file.close();

        // Remove the file.
        boost::filesystem::remove(tc_path);

        // Update traj_offset.
        const auto n_steps = boost::numeric_cast<std::size_t>(
            tc_size / static_cast<std::size_t>(safe_size_t(sizeof(double)) * (order + 1u) * 7u));
        traj_offset.emplace_back(cur_offset, n_steps);

        // Update cur_offset.
        cur_offset += tc_size / sizeof(double);
    }

    // Offset vector for the time data. It will contain:
    // - the offset (in number of double-precision values) in the storage file
    //   at which the time data for an object begins,
    // - the number of steps.
    std::vector<std::tuple<std::size_t, std::size_t>> time_offset;
    time_offset.reserve(n_sats);

    // Times.
    for (std::size_t i = 0; i < n_sats; ++i) {
        // Build the file path.
        const auto time_path = tmp_dir_path / fmt::format("time_{}", i);

        // Fetch the file size in bytes.
        assert(boost::filesystem::exists(time_path));
        assert(boost::filesystem::is_regular_file(time_path));
        const auto time_size = boost::filesystem::file_size(time_path);
        assert(time_size % sizeof(double) == 0u);

        // Open it.
        std::ifstream time_file(time_path.string(), std::ios::binary | std::ios::in);
        time_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        // Copy into storage_file.
        storage_file << time_file.rdbuf();

        // Close the file.
        time_file.close();

        // Remove the file.
        boost::filesystem::remove(time_path);

        // Update time_offset.
        time_offset.emplace_back(cur_offset, time_size / sizeof(double));

        // Update cur_offset.
        cur_offset += time_size / sizeof(double);
    }

    // Return the offset vectors.
    return std::make_pair(std::move(traj_offset), std::move(time_offset));
}

// Construct the polyjectory resulting from the ODE integration
// of the SGP4 dynamics.
template <typename TrajOffset, typename TimeOffset>
polyjectory build_polyjectory(const boost::filesystem::path &tmp_dir_path, const TrajOffset &traj_offset,
                              const TimeOffset &time_offset, const std::vector<int> &status, std::uint32_t order)
{
    const auto storage_path = tmp_dir_path / "storage";
    boost::iostreams::mapped_file_source file;
    file.open(storage_path.string());

    // LCOV_EXCL_START
    if (boost::numeric_cast<unsigned>(file.alignment()) < alignof(double)) [[unlikely]] {
        throw std::runtime_error(fmt::format("Invalid alignment detected in a memory mapped file: the alignment of "
                                             "the file is {}, but an alignment of {} is required instead",
                                             file.alignment(), alignof(double)));
    }
    // LCOV_EXCL_STOP

    // Fetch a pointer to the beginning of the data.
    // NOTE: this is technically UB. We would use std::start_lifetime_as in C++23:
    // https://en.cppreference.com/w/cpp/memory/start_lifetime_as
    const auto *base_ptr = reinterpret_cast<const double *>(file.data());
    assert(boost::alignment::is_aligned(base_ptr, alignof(double)));

    auto traj_transform = [base_ptr, order](const auto &p) {
        const auto [offset, nsteps] = p;

        return heyoka::mdspan<const double, heyoka::extents<std::size_t, std::dynamic_extent, 7, std::dynamic_extent>>{
            base_ptr + offset, nsteps, order + 1u};
    };

    auto time_transform = [base_ptr](const auto &p) {
        const auto [offset, nsteps] = p;

        return heyoka::mdspan<const double, heyoka::dextents<std::size_t, 1>>{base_ptr + offset, nsteps};
    };

    return polyjectory(traj_offset | std::views::transform(traj_transform),
                       time_offset | std::views::transform(time_transform), status);
}

} // namespace

} // namespace detail

polyjectory
sgp4_polyjectory(heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>> sat_data,
                 double jd_begin, double jd_end, double exit_radius, double reentry_radius)
{
    // Check the date range.
    if (!std::isfinite(jd_begin) || !std::isfinite(jd_end) || !(jd_begin < jd_end)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid Julian date interval [{}, {}) supplied to sgp4_polyjectory(): the begin/end dates "
                        "must be finite and the end date must be strictly after the begin date",
                        jd_begin, jd_end));
    }

    // Check the exit/reentry radiuses.
    if (!std::isfinite(exit_radius) || exit_radius <= 0) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid exit radius {} supplied to sgp4_polyjectory(): the exit radius must be finite and positive",
            exit_radius));
    }
    if (!std::isfinite(reentry_radius) || reentry_radius <= 0 || reentry_radius >= exit_radius) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid reentry radius {} supplied to sgp4_polyjectory(): the reentry "
                                                "radius must be finite, positive and less than the exit radius",
                                                reentry_radius));
    }

    // Fetch the states of the satellites at jd_begin.
    const auto [init_states, init_states_data]
        = detail::sgp4_compute_initial_sat_states(sat_data, jd_begin, exit_radius, reentry_radius);

    // Construct the sgp4 ODE sys.
    const auto sgp4_ode = detail::construct_sgp4_ode();

    // Construct the ODE integrator.
    const auto ta = detail::construct_sgp4_ode_integrator(sgp4_ode, exit_radius, reentry_radius);

    // Assemble a "unique" dir path into the system temp dir.
    const auto tmp_dir_path = boost::filesystem::temp_directory_path()
                              / boost::filesystem::unique_path("mizuba_sgp4_polyjectory-%%%%-%%%%-%%%%-%%%%");

    // Attempt to create it.
    // LCOV_EXCL_START
    if (!boost::filesystem::create_directory(tmp_dir_path)) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("Error while creating a unique temporary directory: the directory '{}' already exists",
                        tmp_dir_path.string()));
    }
    // LCOV_EXCL_STOP

    // NOTE: from now on, we need to ensure that the temp dir is automatically
    // cleaned up, even in case of exceptions. We use this little RAII helper
    // for this purpose.
    struct tmp_cleaner {
        // NOTE: store by reference so that we are sure that constructing
        // a tmp_cleaner cannot possibly throw.
        const boost::filesystem::path &path;
        ~tmp_cleaner()
        {
            boost::filesystem::remove_all(path);
        }
    };
    const tmp_cleaner tmp_clean{tmp_dir_path};

    // Change the permissions so that only the owner has access.
    boost::filesystem::permissions(tmp_dir_path, boost::filesystem::owner_all);

    // Run the numerical integration.
    const auto status = detail::perform_ode_integration(ta, tmp_dir_path, sat_data, jd_begin, jd_end, init_states);

    // Consolidate all the data files into a single file.
    const auto [traj_offset, time_offset] = detail::consolidate_data(tmp_dir_path, sat_data.extent(1), ta.get_order());

    // Build and return the polyjectory.
    return detail::build_polyjectory(tmp_dir_path, traj_offset, time_offset, status, ta.get_order());
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
