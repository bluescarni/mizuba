// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <future>
#include <ios>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/taylor.hpp>

#include "detail/file_utils.hpp"
#include "logging.hpp"
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
            // LCOV_EXCL_START
            throw std::invalid_argument(
                fmt::format("The sgp4 propagation of the object at index {} at jd_begin generated the error code {}", i,
                            static_cast<int>(out_span(6, i))));
            // LCOV_EXCL_STOP
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
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "The sgp4 propagation of the object at index {} at jd_begin generated a non-finite state vector", i));
            // LCOV_EXCL_STOP
        }

        // Check the distance from the Earth.
        const auto dist = std::sqrt(x * x + y * y + z * z);
        if (!std::isfinite(dist) || dist >= exit_radius || dist <= reentry_radius) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format("The sgp4 propagation of the object at index {} at jd_begin "
                                                    "generated a position vector with invalid radius {}",
                                                    i, dist));
            // LCOV_EXCL_STOP
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
                heyoka::kw::compact_mode = true);
}

// Run the ODE integration according to the sgp4 dynamics, storing the Taylor coefficients
// and the step end times step-by-step.
// NOTE: an obvious improvement here would be to split the dynamics of sgp4 into init part and
// time-dependent part. Not sure how much more complicated the implementation becomes though,
// and also the implications are not clear if we put coordinate transformations in the dynamics
// (e.g., for producing the dynamics in ICRF instead of TEME one day).
template <typename TA, typename Path, typename SatData, typename InitStates>
auto perform_ode_integration(const TA &tmpl_ta, const Path &tmp_dir_path, SatData sat_data, double jd_begin,
                             double jd_end, InitStates init_states, std::uint32_t order)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

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

    // A **global** vector of statuses, one per object.
    // We do not need to protect writes into this, as each status
    // will be written to exactly at most once.
    // NOTE: this is zero-inited, meaning that the default status flag
    // of each object is "no error detected".
    std::vector<int> global_status;
    global_status.resize(boost::numeric_cast<decltype(global_status.size())>(n_sats));

    // Create the traj and time data files.
    if (boost::filesystem::exists(tmp_dir_path / "traj")) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Cannot create the storage file '{}': the file exists already",
                                                (tmp_dir_path / "traj").string()));
        // LCOV_EXCL_STOP
    }
    std::ofstream traj_file((tmp_dir_path / "traj").string(), std::ios::binary | std::ios::out);
    traj_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    if (boost::filesystem::exists(tmp_dir_path / "time")) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::invalid_argument(fmt::format("Cannot create the storage file '{}': the file exists already",
                                                (tmp_dir_path / "time").string()));
        // LCOV_EXCL_STOP
    }
    std::ofstream time_file((tmp_dir_path / "time").string(), std::ios::binary | std::ios::out);
    time_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    // Setup the futures and promises to coordinate between the numerical integration and the writer thread.
    std::vector<std::promise<std::vector<double>>> traj_promises, time_promises;

    traj_promises.resize(boost::numeric_cast<decltype(traj_promises.size())>(n_sats));
    time_promises.resize(boost::numeric_cast<decltype(time_promises.size())>(n_sats));

    auto traj_fut_view = traj_promises | std::views::transform([](auto &p) { return p.get_future(); });
    auto time_fut_view = time_promises | std::views::transform([](auto &p) { return p.get_future(); });

    std::vector traj_futures(std::ranges::begin(traj_fut_view), std::ranges::end(traj_fut_view));
    std::vector time_futures(std::ranges::begin(time_fut_view), std::ranges::end(time_fut_view));

    // Flag to signal that the writer thread should stop writing.
    std::atomic<bool> stop_writing = false;

    // Prepare the traj_offsets vector.
    std::vector<polyjectory::traj_offset> traj_offsets;
    traj_offsets.reserve(n_sats);

    // Launch the writer thread.
    auto writer_future = std::async(std::launch::async, [&traj_file, &traj_futures, &traj_offsets, &time_file,
                                                         &time_futures, order, &stop_writing, n_sats]() {
        using namespace std::chrono_literals;

        // How long should we wait before checking if we should stop writing.
        const auto wait_duration = 250ms;

        // Track the trajectory offsets to build up traj_offsets.
        safe_size_t cur_traj_offset = 0;

        for (std::size_t i = 0; i < n_sats; ++i) {
            // Fetch the futures.
            auto &traj_fut = traj_futures[i];
            auto &time_fut = time_futures[i];

            // Wait until the futures become available, or return if a stop is requested.
            while (traj_fut.wait_for(wait_duration) != std::future_status::ready
                   || time_fut.wait_for(wait_duration) != std::future_status::ready) {
                if (stop_writing) [[unlikely]] {
                    return; // LCOV_EXCL_LINE
                }
            }

            // Fetch the data in the futures.
            auto v_traj = traj_fut.get();
            auto v_time = time_fut.get();

            // Write the traj data.
            traj_file.write(reinterpret_cast<const char *>(v_traj.data()),
                            boost::safe_numerics::safe<std::streamsize>(v_traj.size()) * sizeof(double));

            // Compute the number of steps, and update traj_offsets and cur_traj_offset.
            assert(v_traj.size() % ((safe_size_t(order) + 1u) * 7u) == 0u);
            const auto n_steps = v_traj.size() / ((safe_size_t(order) + 1u) * 7u);

            traj_offsets.emplace_back(cur_traj_offset, n_steps);
            cur_traj_offset += v_traj.size();

            // Write the time data.
            time_file.write(reinterpret_cast<const char *>(v_time.data()),
                            boost::safe_numerics::safe<std::streamsize>(v_time.size()) * sizeof(double));
        }
    });

    // NOTE: at this point, the writer thread has started. From now on, we wrap everything in a try/catch block
    // so that, if any exception is thrown, we can safely stop the writer thread before re-throwing.
    try {
        // Bag of thread-specific data to be used in the batch parallel iterations.
        struct batch_data {
            // The ODE integrator.
            TA ta;
            // The sgp4 propagator.
            prop_t prop;
            // Buffer used to set up new data in prop.
            std::vector<double> new_sat_data;
            // Vector of max timestep sizes.
            std::vector<double> max_h;
            // Vector of active batch element flags.
            std::vector<int> active_flags;
            // Vector of Julian dates for use in the sgp4 prop.
            std::vector<prop_t::date> jdates;
            // Buffers used to temporarily store the Taylor coefficients
            // and the time data for each element of a batch. The contents
            // of these buffers will be sent to the writer thread at the end of
            // the integration of a batch.
            std::vector<std::pair<std::vector<double>, std::vector<double>>> tmp_write_buffers;
        };

        // Set up the thread-specific data machinery via TBB.
        using ets_t
            = oneapi::tbb::enumerable_thread_specific<batch_data, oneapi::tbb::cache_aligned_allocator<batch_data>,
                                                      oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
        ets_t ets([&tmpl_ta, &tmpl_prop, batch_size]() {
            // Make a copy of the template integrator.
            auto ta = tmpl_ta;

            // Make a copy of the template sgp4 propagator.
            auto prop = tmpl_prop;

            // Prepare new_sat_data with the appropriate size.
            std::vector<double> new_sat_data;
            new_sat_data.resize(boost::safe_numerics::safe<decltype(new_sat_data.size())>(batch_size) * 9);

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
                              .max_h = std::move(max_h),
                              .active_flags = std::move(active_flags),
                              .jdates = std::move(jdates),
                              .tmp_write_buffers = std::move(tmp_write_buffers)};
        });

        // Run the numerical integration.
        //
        // NOTE: we process the blocks of objects in chunks because we want the writer thread to make steady progress.
        //
        // If we don't do the chunking, what happens is that TBB starts immediately processing blocks throughout
        // the *entire* range, whereas the writer thread must proceed strictly sequentially. We thus end up in a
        // situation in which a lot of file writing happens at the very end while not overlapping with the
        // computation, and memory usage is high because we have to keep around for long unwritten data.
        //
        // With chunking, the writer thread can fully process chunk N while chunk N+1 is computing.
        //
        // NOTE: there are tradeoffs in selecting the chunk size. If it is too large, we are negating the benefits
        // of chunking wrt computation/transfer overlap and memory usage. If it is too small, we are limiting
        // parallel speedup. The current value is based on preliminary performance evaluation with the full LEO
        // catalog, but I am not sure if this can be made more robust/general. In general, the "optimal" chunking
        // will depend on several variables such as the number of CPU cores, the available memory,
        // the integration length, the batch size and so on.
        //
        // NOTE: I am also not sure whether or not it is possible to achieve the same result more elegantly
        // with some TBB partitioner/range wizardry.
        constexpr auto chunk_size = 256u;

        for (std::size_t start_block_idx = 0; start_block_idx != n_blocks;) {
            const auto n_rem_blocks = n_blocks - start_block_idx;
            const auto end_block_idx = start_block_idx + (n_rem_blocks < chunk_size ? n_rem_blocks : chunk_size);

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<std::size_t>(start_block_idx, end_block_idx),
                [&ets, batch_size, n_sats, sat_data, init_states, jd_begin, jd_end, &traj_promises, &time_promises,
                 &global_status](const auto &range) {
                    using dfloat = heyoka::detail::dfloat<double>;

                    // Fetch the thread-local data.
                    // NOTE: no need to isolate here, as we are not
                    // invoking any other TBB primitive from within this
                    // scope. Keep in mind though that if we ever enable parallel
                    // operations in heyoka during ODE integration, we will
                    // have to isolate.
                    auto &[ta, prop, new_sat_data, max_h, active_flags, jdates, tmp_write_buffers] = ets.local();

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
                        boost::numeric_cast<std::size_t>(ta.get_order() + 1u),
                        boost::numeric_cast<std::size_t>(batch_size)};

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

                        // Clear out tmp_write_buffers.
                        for (std::uint32_t i = 0; i < e_batch_size; ++i) {
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
                            pars_view(7, i) = static_cast<double>(
                                                  dfloat(jd_begin)
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
                            pars_view(7, i) = static_cast<double>(dfloat(jd_begin)
                                                                  - normalise(dfloat(sat_data(7, 0), sat_data(8, 0))))
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
                                const auto jd
                                    = dfloat(jd_begin)
                                      + dfloat(ta.get_dtime().first[i] / 1440., ta.get_dtime().second[i] / 1440.);
                                jdates[i] = {jd.hi, jd.lo};
                            }

                            // Propagate writing directly into the state vector of the integrator.
                            const auto prop_out_span = std::experimental::submdspan(state_view, std::pair{0, 7},
                                                                                    std::experimental::full_extent);
                            prop(prop_out_span, heyoka::mdspan<const prop_t::date, heyoka::dextents<std::size_t, 1>>(
                                                    jdates.data(), boost::numeric_cast<std::size_t>(batch_size)));

                            // Compute by hand the value of the radius.
                            for (std::uint32_t i = 0; i < batch_size; ++i) {
                                state_view(7, i) = std::sqrt(state_view(0, i) * state_view(0, i)
                                                             + state_view(1, i) * state_view(1, i)
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

                        // Move the trajectory and time data for this batch
                        // into the futures.
                        for (std::uint32_t i = 0; i < e_batch_size; ++i) {
                            auto &[tc_buf, time_buf] = tmp_write_buffers[i];
                            const auto out_idx = b_idx + i;

                            traj_promises[out_idx].set_value(std::move(tc_buf));
                            time_promises[out_idx].set_value(std::move(time_buf));
                        }
                    }
                });

            start_block_idx = end_block_idx;
        }
        // LCOV_EXCL_START
    } catch (...) {
        // Request a stop on the writer thread.
        stop_writing.store(true);

        // Wait for it to actually stop.
        // NOTE: we use wait() here, because, if the writer thread
        // also threw, get() would throw the exception here. We are
        // not interested in reporting that, as the exception from the numerical
        // integration is likely more interesting.
        // NOTE: in principle wait() could also raise platform-specific exceptions.
        writer_future.wait();

        // Re-throw.
        throw;
    }
    // LCOV_EXCL_STOP

    // Wait for the writer thread to finish.
    // NOTE: get() will throw any exception that might have been
    // raised in the writer thread.
    writer_future.get();

    // Close the data files.
    // NOTE: this is of course unnecessary as the dtors will do the
    // closing themselves, but let us be explicit.
    traj_file.close();
    time_file.close();

    // Return the status flags and the trajectory offsets.
    return std::make_pair(std::move(global_status), std::move(traj_offsets));
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
    stopwatch sw;
    const auto sgp4_ode = detail::construct_sgp4_ode();

    // Construct the ODE integrator.
    const auto ta = detail::construct_sgp4_ode_integrator(sgp4_ode, exit_radius, reentry_radius);
    log_trace("SGP4 ODE integrator construction time: {}s", sw);

    // Assemble a "unique" dir path into the system temp dir.
    const auto tmp_dir_path = detail::create_temp_dir("mizuba_sgp4_polyjectory-%%%%-%%%%-%%%%-%%%%");

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
    sw.reset();
    auto [status, traj_offsets]
        = detail::perform_ode_integration(ta, tmp_dir_path, sat_data, jd_begin, jd_end, init_states, ta.get_order());
    log_trace("SGP4 ODE integration time: {}s", sw);

    // Build and return the polyjectory.
    return polyjectory(std::filesystem::path((tmp_dir_path / "traj").string()),
                       std::filesystem::path((tmp_dir_path / "time").string()), ta.get_order(), std::move(traj_offsets),
                       std::move(status));
}

} // namespace mizuba

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
