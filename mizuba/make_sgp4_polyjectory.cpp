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
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <ios>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/task_group.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/fixed_centres.hpp>
#include <heyoka/model/sgp4.hpp>
#include <heyoka/taylor.hpp>

#include "detail/dfloat_utils.hpp"
#include "detail/file_utils.hpp"
#include "detail/poly_utils.hpp"
#include "detail/sgp4/SGP4.h"
#include "logging.hpp"
#include "make_sgp4_polyjectory.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

namespace detail
{

namespace
{

// NOTE: this is a thin wrapper around oneapi::tbb::task_group that enforces wait() on destruction. We want this
// behaviour in order to automatically wait() in case of exceptions.
class jtask_group
{
    oneapi::tbb::task_group m_tg;

public:
    jtask_group() = default;
    jtask_group(const jtask_group &) = delete;
    jtask_group(jtask_group &&) noexcept = delete;
    jtask_group &operator=(const jtask_group &) = delete;
    jtask_group &operator=(jtask_group &&) noexcept = delete;
    // NOTE: this will terminate if m_tg.wait() throws, but I think it is ok as I don't see a way of recovering from
    // such an occurrence.
    ~jtask_group()
    {
        m_tg.wait();
    }

    template <typename Func>
    void run(Func &&f)
    {
        m_tg.run(std::forward<Func>(f));
    }
};

// Helper to construct a double-double representation of the epoch of a gpe.
auto fetch_gpe_epoch(const gpe &g)
{
    return hilo_to_dfloat(g.epoch_jd1, g.epoch_jd2);
}

// Helper to convert an input double-length UTC Julian date into a double-length TAI Julian date.
// The result is normalised and checked for finiteness.
auto dl_utc_to_tai(auto utc)
{
    const auto [tai1, tai2] = heyoka::model::jd_utc_to_tai(utc.hi, utc.lo);
    return hilo_to_dfloat(tai1, tai2);
}

// Helper to check for finiteness in the gpe data.
void check_gpe_finiteness(const auto &gpes)
{
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, gpes.extent(0)), [&gpes](const auto &range) {
        for (auto i = range.begin(); i != range.end(); ++i) {
#define MIZUBA_GPE_CHECK_FINITE(i, data_member)                                                                        \
    if (!std::isfinite(gpes(i).data_member)) [[unlikely]] {                                                            \
        throw std::invalid_argument(                                                                                   \
            fmt::format("Invalid GPE detected by make_sgp4_polyjectory(): the '{}' of the GPE "                        \
                        "at index {} is not finite",                                                                   \
                        #data_member, i));                                                                             \
    }

            MIZUBA_GPE_CHECK_FINITE(i, epoch_jd1)
            MIZUBA_GPE_CHECK_FINITE(i, epoch_jd2)
            MIZUBA_GPE_CHECK_FINITE(i, n0)
            MIZUBA_GPE_CHECK_FINITE(i, e0)
            MIZUBA_GPE_CHECK_FINITE(i, i0)
            MIZUBA_GPE_CHECK_FINITE(i, node0)
            MIZUBA_GPE_CHECK_FINITE(i, omega0)
            MIZUBA_GPE_CHECK_FINITE(i, m0)
            MIZUBA_GPE_CHECK_FINITE(i, bstar)

#undef MIZUBA_GPE_CHECK_FINITE
        }
    });
}

// Helper to check that the gpe data is sorted properly (that is, first by norad id, and then by epoch).
void check_gpe_order(const auto &gpes)
{
    // The comparator.
    const auto cmp = [](const gpe &g1, const gpe &g2) {
        // Check the norad ids first.
        if (g1.norad_id < g2.norad_id) {
            return true;
        }
        if (g1.norad_id > g2.norad_id) {
            return false;
        }

        // The ids match, check the epochs.
        const auto ep1 = fetch_gpe_epoch(g1);
        const auto ep2 = fetch_gpe_epoch(g2);

        if (ep1 < ep2) {
            return true;
        }
        if (ep1 > ep2) {
            return false;
        }

        // Cannot have multiple identical epochs for the same satellite.
        throw std::invalid_argument(
            fmt::format("Two GPEs with identical NORAD ID {} and epoch ({}, {}) were identified by "
                        "make_sgp4_polyjectory() - this is not allowed",
                        g1.norad_id, g1.epoch_jd1, g1.epoch_jd2));
    };

    if (!std::ranges::is_sorted(gpes.data_handle(), gpes.data_handle() + gpes.extent(0), cmp)) [[unlikely]] {
        throw std::invalid_argument("The set of GPEs passed to make_sgp4_polyjectory() is not sorted correctly: it "
                                    "must be ordered first by NORAD ID and then by epoch");
    }
}

// Helper to build the template integrator, propagator and llvm state which will be used to copy-init
// the corresponding thread locals during parallel operations. The return values are passed in as
// empty optionals.
void build_tplts(auto &ta_kepler_tplt, auto &sgp4_prop_tplt, auto &jit_state_tplt)
{
    namespace hy = heyoka;

    // Build the fixed centre dynamics for the Earth. Positions will be represented in [km],
    // velocities in [km / s], time in [s].
    const auto dyn = hy::model::fixed_centres(hy::kw::Gconst = 6.6743e-11 * 1e-9 * 5.9722e24, hy::kw::masses = {1.},
                                              hy::kw::positions = {0., 0., 0.});

    // Build the integrator.
    // NOTE: set a tolerance higher than the epsilon here, on account that polynomial
    // interpolation is expected to produce better approximations than Taylor series
    // and thus should require a lower polynomial order.
    ta_kepler_tplt.emplace(dyn, hy::kw::tol = 1e-14);

    // Fetch the Taylor order.
    const auto order = ta_kepler_tplt->get_order();

    // Safely compute order + 1.
    const auto op1 = static_cast<std::uint32_t>(boost::safe_numerics::safe<std::uint32_t>(order) + 1);

    // Initialise in parallel the propagator and the jitted functions.
    oneapi::tbb::parallel_invoke(
        [op1, &sgp4_prop_tplt]() {
            // NOTE: init the propagator with zeroed-out data.
            const auto tot_size = static_cast<std::size_t>(boost::safe_numerics::safe<std::size_t>(9) * op1);
            std::vector<double> sgp4_prop_data;
            sgp4_prop_data.resize(boost::numeric_cast<decltype(sgp4_prop_data.size())>(tot_size));
            const auto sgp4_prop_span = hy::mdspan<const double, hy::extents<std::size_t, 9, std::dynamic_extent>>(
                sgp4_prop_data.data(), op1);

            sgp4_prop_tplt.emplace(sgp4_prop_span, hy::kw::batch_size = op1);
        },
        [order, op1, &jit_state_tplt]() {
            // Init the llvm state.
            jit_state_tplt.emplace();

            // Here we need two cfuncs: the function to calculate the r coordinates from the
            // xyz coordinates and the interpolator.

            // The first cfunc.
            auto [x, y, z] = hy::make_vars("x", "y", "z");
            hy::add_cfunc<double>(*jit_state_tplt, "cfunc_r",
                                  {hy::sqrt(hy::sum({hy::pow(x, 2.), hy::pow(y, 2.), hy::pow(z, 2.)}))}, {x, y, z},
                                  hy::kw::batch_size = op1);

            // The second cfunc.
            const auto [interp_inputs, interp_outputs] = vm_interp(order);
            hy::add_cfunc<double>(*jit_state_tplt, "cfunc_interp", interp_outputs, interp_inputs,
                                  hy::kw::batch_size = 7u);

            // Compile.
            jit_state_tplt->compile();
        });
}

// Evaluate the state of a single gpe via the official sgp4 C++ code at multiple time points.
//
// sample_points is the list of time points used for evaluation, and its size is op1.
// satrec is the already-initialised satellite object to be passed to the propagation
// function of the official sgp4 C++ code. interp_buffer is a memory buffer of size
// op1 * 14. The result of the evaluation is supposed to be stored in the second chunk
// of interp_buffer, that is, in the [op1 * 7, op1 * 14) range in row-major format,
// i.e., op1 rows and 7 columns. Each row will contain the evaluation of
// [x, y, z, vx, vy, vz, r] for the corresponding time point in sample_points.
//
// The return value will be 0 if everything went ok, otherwise an sgp4 error code will be returned.
int gpe_eval_vallado(auto &interp_buffer, elsetrec &satrec, const auto &sample_points)
{
    // NOLINTNEXTLINE(misc-unused-alias-decls)
    namespace hy = heyoka;

    // Fetch the interpolation order + 1.
    // NOTE: we checked earlier that this conversion is ok: interp_buffer has a size > op1 and we checked
    // that we can build a std::size_t-sized span on top of it.
    const auto op1 = static_cast<std::size_t>(sample_points.size());
    assert(interp_buffer.size() == op1 * 14u);

    // Evaluate positions and velocities at the sample points, writing the result
    // into the second chunk of interp_buffer.
    const auto ibspan1 = hy::mdspan<double, heyoka::extents<std::size_t, std::dynamic_extent, 7>>{
        interp_buffer.data() + (op1 * 7u), op1};
    for (std::size_t i = 0; i < op1; ++i) {
        // NOTE: we can write directly into ibspan1.
        SGP4Funcs::sgp4(satrec, sample_points[i], &ibspan1[i, 0], &ibspan1[i, 3]);
        // Check for errors.
        if (satrec.error != 0) [[unlikely]] {
            return satrec.error;
        }

        // Compute the radius.
        ibspan1[i, 6] = std::sqrt((ibspan1[i, 0] * ibspan1[i, 0]) + (ibspan1[i, 1] * ibspan1[i, 1])
                                  + (ibspan1[i, 2] * ibspan1[i, 2]));
    }

    return 0;
}

// Evaluate the state of a single gpe via heyoka's sgp4 propagator at multiple time points.
//
// sample_points is the list of time points used for evaluation, and its size is op1. sgp4_prop is the sgp4
// propagator, which has been initialised with op1 copies of the same gpe. cfunc_r is the compiled function to be
// used to compute the radial coordinate from the propagated xyz coordinates. interp_buffer is a memory buffer of
// size op1 * 14. The result of the evaluation is supposed to be stored in the second chunk of interp_buffer, that
// is, in the [op1 * 7, op1 * 14) range in row-major format, i.e., op1 rows and 7 columns. Each row will contain the
// evaluation of [x, y, z, vx, vy, vz, r] for a time point in sample_points.
//
// The return value will be 0 if everything went ok, otherwise an sgp4 error code will be returned.
int gpe_eval_heyoka(auto &interp_buffer, auto &sgp4_prop, const auto &sample_points, auto *cfunc_r)
{
    // NOLINTNEXTLINE(misc-unused-alias-decls)
    namespace hy = heyoka;

    // Fetch the interpolation order + 1.
    // NOTE: we checked earlier that this conversion is ok: interp_buffer has a size > op1 and we checked
    // that we can build a std::size_t-sized span on top of it.
    const auto op1 = static_cast<std::size_t>(sample_points.size());
    assert(sgp4_prop.get_nsats() == op1);
    assert(interp_buffer.size() == op1 * 14u);

    // Step 1: evaluate at the sample points. We will be writing the output at the *beginning* of interp_buffer,
    // i.e., not in the correct final position in the second chunk. We do this because the output of the propagator will
    // have to be transposed, so we use the first chunk of interp_buffer as temporary storage.
    //
    // NOTE: the evaluation produces a 7 x op1 array with the following layout:
    //
    // x0 x1 x2 ...
    // y0 y1 y2 ...
    // ...
    // vz0 vz1 vz2 ...
    // err0 err1 err2 ...
    //
    // Hence, the need to transpose the result of the evaluation into the layout required
    // by interp_buffer.
    const auto ibspan0 = hy::mdspan<double, heyoka::dextents<std::size_t, 2>>{interp_buffer.data(), 7u, op1};
    const auto sample_span = hy::mdspan<const double, heyoka::dextents<std::size_t, 1>>{sample_points.data(), op1};
    sgp4_prop(ibspan0, sample_span);

    // Step 2: we check if any propagation resulted in an error code.
    for (std::size_t i = 0; i < op1; ++i) {
        if (ibspan0[6, i] != 0) [[unlikely]] {
            return static_cast<int>(ibspan0[6, i]);
        }
    }

    // Step 3: we compute the values for the radial coordinate from the xyz coordinates,
    // outputting the result in the last row of ibspan0 (and thus overwriting the error code).
    cfunc_r(&ibspan0[6, 0], &ibspan0[0, 0], nullptr, nullptr);

    // Step 4: we transpose the evaluation output into the second chunk of interp_buffer.
    const auto ibspan1
        = hy::mdspan<double, heyoka::dextents<std::size_t, 2>>{interp_buffer.data() + (op1 * 7u), op1, 7u};
    for (std::size_t i = 0; i < op1; ++i) {
        for (auto j = 0u; j < 7u; ++j) {
            ibspan1[i, j] = ibspan0[j, i];
        }
    }

    return 0;
}

// Estimate the squared interpolation error within an interpolation step.
//
// The estimation is based on the positional error. The interpolating polynomials for xyz are evaluated
// at the half points between the original Cheby nodes that were used to produce the polynomials. The positions
// produced by the interpolating polynomials are compared to the positions produced by the sgp4
// propagators, and the maximum squared positional difference is returned.
//
// cf_ptr contains the interpolating polynomials' coefficients for the entire state vector, stored in column-major
// format. interp_buffer is the evaluation/interpolation buffer, which is assumed to contain in the first chunk the
// Cheby nodes which were used to produce the interpolating polynomials (thus, these are cheby nodes measured in days
// since the beginning of the interpolation step). Similarly, sample_points is assumed to contain the same Cheby nodes
// but measured in minutes since the satellite's epoch (for use in the sgp4 propagators). xyz_eval is a buffer that will
// store the evaluations of the interpolating polynomials. step_begin_sat_epoch represents the step's begin time
// measured in days since the satellite's epoch. is_deep_space signals whether the satellite is a deep-space one or not.
// satrec is the initialised satellite object for computations with the official C++ code. sgp4_prop is the initialised
// heyoka sgp4 propagator. cfunc_r is the jit-compiled function to compute the satellite's radial coordinate from its
// xyz coordinates.
//
// An error code will be returned if either sgp4 propagation produces an error or if non-finite values
// are detected during the computations.
std::variant<double, int> eval_interp_error2(const double *cf_ptr, auto &interp_buffer, auto &sample_points,
                                             auto &xyz_ieval, const double step_begin_sat_epoch, bool is_deep_space,
                                             elsetrec &satrec, auto &sgp4_prop, auto *cfunc_r)
{
    // NOLINTNEXTLINE(misc-unused-alias-decls)
    namespace hy = heyoka;

    // Cache op1.
    // NOTE: we know that by construction sample_points has the size of the poly
    // interpolation order + 1, which is by definition representable as a std::uint32_t.
    const auto op1 = static_cast<std::uint32_t>(sample_points.size());
    assert(xyz_ieval.size() == op1);
    assert(op1 != 0u);

    // Evaluate the interpolating polynomials for xyz halfway between the Cheby nodes and
    // write the results into xyz_eval.
    // NOTE: interp_buffer is assumed to contain the Cheby nodes measured in days since
    // the beginning of the step.
    const auto ipoints_span
        = hy::mdspan<const double, hy::extents<std::size_t, std::dynamic_extent, 7>>(interp_buffer.data(), op1);
    for (std::uint32_t i = 0; i < op1; ++i) {
        const auto prev_tm = (i == 0u) ? 0. : ipoints_span[i - 1u, 0];
        const auto cur_tm = ipoints_span[i, 0];
        const auto eval_tm = (prev_tm / 2) + (cur_tm / 2);

        auto &cur_xyz = xyz_ieval[i];
        for (auto j = 0u; j < 3u; ++j) {
            // NOTE: a stride type of std::uint32_t here is ok, as we know that
            // cf_ptr is coming from a buffer whose size fits std::uint32_t.
            //
            // NOTE: this could be vectorised via a strided cfunc, since we are only
            // interested in the evaluation of the xyz polys. However, the overall cost of Horner
            // evaluation should be quite small compared to the SGP4 propagations
            // we will be performing shortly. Thus, the performance benefit may be
            // quite minimal.
            cur_xyz[j] = horner_eval(cf_ptr + j, op1 - 1u, eval_tm, static_cast<std::uint32_t>(7));
        }
    }

    // Write at the beginning of interp_buffer the midpoints between the Cheby nodes from sample_points.
    // NOTE: here we are using the first chunk of interp_buffer as temporary storage.
    // NOTE: sample_points is assumed to contain the Cheby nodes measured in minutes since
    // the satellite's epoch (this is the time coordinate required by the sgp4 propagators).
    for (std::uint32_t i = 0; i < op1; ++i) {
        const auto prev_tm = (i == 0u) ? (step_begin_sat_epoch * 1440.) : sample_points[i - 1u];
        const auto cur_tm = sample_points[i];
        const auto eval_tm = (prev_tm / 2) + (cur_tm / 2);

        interp_buffer[i] = eval_tm;
    }

    // Copy back to sample_points.
    std::ranges::copy(interp_buffer.data(), interp_buffer.data() + op1, sample_points.begin());

    // Evaluate at sample_points, using one of the two sgp4 propagators.
    const auto res = is_deep_space ? gpe_eval_vallado(interp_buffer, satrec, sample_points)
                                   : gpe_eval_heyoka(interp_buffer, sgp4_prop, sample_points, cfunc_r);
    // Check for errors.
    if (res != 0) [[unlikely]] {
        return res;
    }

    // Fetch a span to the result of the evaluation within interp_buffer.
    const auto epoints_span = hy::mdspan<const double, hy::extents<std::size_t, std::dynamic_extent, 7>>(
        interp_buffer.data() + (op1 * static_cast<std::size_t>(7)), op1);

    // Compute the maximum positional error squared.
    auto max_err2 = 0.;
    for (std::uint32_t i = 0; i < op1; ++i) {
        const auto &cur_xyz = xyz_ieval[i];

        const auto x_err = cur_xyz[0] - epoints_span[i, 0];
        const auto y_err = cur_xyz[1] - epoints_span[i, 1];
        const auto z_err = cur_xyz[2] - epoints_span[i, 2];

        const auto cur_err2 = (x_err * x_err) + (y_err * y_err) + (z_err * z_err);

        // NOTE: if something went awry with propagation or poly evaluation, we will
        // catch it here.
        if (!std::isfinite(cur_err2)) [[unlikely]] {
            return 10;
        }

        max_err2 = std::max(max_err2, cur_err2);
    }

    return max_err2;
}

// Interpolate a gpe within an interpolation step, bisecting the step if necessary in case
// low-precision interpolations are detected.
//
// sgp4 trajectories exhibit occasional discontinuities due to the branchy nature of the sgp4 algorithm.
// These discontinuities (which may show up in the trajectories themselves and/or in their time derivatives)
// can result in a degradation of the interpolation accuracy within a step. Although this degradation
// is unavoidable, we can limit its effect by confining it to a very short interpolation step, so that
// the window of time in which we have an inaccurate interpolation is minimised.
//
// The interpolation step's boundaries are given, in days since the polyjectory's epoch,
// by [init_step_begin, init_step_end), and, in days since the satellite's epoch, by
// [init_step_begin_sat_epoch, init_step_end_sat_epoch). ets is a bag of thread-local data.
// is_deep_space signals whether the satellite is a deep-space one or not. satrec is the
// initialised satellite object for computations with the official C++ code. sgp4_prop is the
// initialised heyoka sgp4 propagator. poly_cf_buf and time_buf are the buffers for storing,
// respectively, the interpolating polynomials' coefficients and the end times of the
// interpolation steps.
int gpe_interpolate_with_bisection(const double init_step_begin, const double init_step_end,
                                   const double init_step_begin_sat_epoch, const double init_step_end_sat_epoch,
                                   auto &ets, bool is_deep_space, elsetrec &satrec, auto &sgp4_prop, auto &poly_cf_buf,
                                   auto &time_buf)
{
    // NOLINTNEXTLINE(misc-unused-alias-decls)
    namespace hy = heyoka;

    // Minimum allowed step size (in days).
    constexpr double min_step_size = 5. / 86400.;

    // Positional interpolation error threshold (in km).
    constexpr double err_thresh = 1e-7;

    // Maximum number of interpolating polynomials for an interpolation step.
    // NOTE: the idea here is that there are pathological cases (e.g., already-decayed
    // satellites) in which the sgp4 algorithm goes crazy without necessarily returning
    // an error code. When this happens, bisection may produce a large number of useless
    // interpolating polynomials while slowing down operations to a crawl. We want to prevent
    // that and return a specific error code instead.
    constexpr unsigned max_n_ipolys = 20;

    // Fetch the Chebyshev nodes buffer.
    const auto &cheby_nodes_unit = ets.cheby_nodes_unit;

    // Cache op1.
    // NOTE: we know that by construction cheby_nodes_unit has the size of the poly
    // interpolation order + 1, which is by definition representable as a std::uint32_t.
    const auto op1 = static_cast<std::uint32_t>(cheby_nodes_unit.size());

    // Fetch the sample points buffer.
    auto &sample_points = ets.sample_points;

    // Fetch the evaluation/interpolation buffer.
    auto &interp_buffer = ets.interp_buffer;

    // Fetch the jitted functions.
    auto *cfunc_r = ets.cfunc_r;
    auto *cfunc_interp = ets.cfunc_interp;

    // Fetch xyz_ieval.
    auto &xyz_ieval = ets.xyz_ieval;

    // Fetch the poly cache.
    auto &interp_poly_pcache = ets.interp_poly_pcache;

    // Clear up the stack and ipolys.
    auto &stack = ets.stack;
    stack.clear();
    auto &ipolys = ets.ipolys;
    ipolys.clear();

    // Seed the starting interpolation interval into the stack.
    stack.push_back({{init_step_begin, init_step_end, init_step_begin_sat_epoch, init_step_end_sat_epoch}});

    while (!stack.empty()) {
        // Check the total number of interpolating polynomials.
        if (ipolys.size() > max_n_ipolys) [[unlikely]] {
            return 14;
        }

        // Pop back an interval from the stack.
        const auto [step_begin, step_end, step_begin_sat_epoch, step_end_sat_epoch] = stack.back();
        stack.pop_back();

        // Compute the durations of the current interval.
        const auto cur_duration = step_end - step_begin;
        const auto cur_duration_sat_epoch = step_end_sat_epoch - step_begin_sat_epoch;
        // NOTE: check for finiteness here, because if these are not finite we may break the logic
        // for sorting ipolys and for accepting the current polynomial. If we have non-finite values being produced
        // during evaluation/interpolation, they may be caught during the evaluation of the interpolation error. In any
        // case, even if we end up producing non-finite poly coefficients or step end times, these will be caught by the
        // polyjectory constructor.
        if (!std::isfinite(cur_duration) || !std::isfinite(cur_duration_sat_epoch)) [[unlikely]] {
            return 10;
        }

        // Setup the interpolation points. Since we will be evaluating the interpolation points
        // with sgp4 propagators, we need to transform the Chebyshev
        // nodes from the [-1, 1] range into the [step_begin, step_end] range, expressed
        // in *minutes elapsed from the satellite epoch* (which is the time coordinate
        // used by the sgp4 propagators). We do this via an affine transformation.
        for (std::uint32_t i = 0; i < op1; ++i) {
            // The affine transformation.
            sample_points[i]
                = (cheby_nodes_unit[i] + 1) / 2 * (cur_duration_sat_epoch * 1440.) + step_begin_sat_epoch * 1440.;
        }

        // Evaluate at the interpolation points, using one of the two sgp4 propagators.
        const auto res = is_deep_space ? gpe_eval_vallado(interp_buffer, satrec, sample_points)
                                       : gpe_eval_heyoka(interp_buffer, sgp4_prop, sample_points, cfunc_r);
        if (res != 0) [[unlikely]] {
            // sgp4 error detected, no point in continuing further.
            return res;
        }

        // Fill in the first chunk of interp_buffer with the interpolation points.
        const auto ipoints_span
            = hy::mdspan<double, hy::extents<std::size_t, std::dynamic_extent, 7>>(interp_buffer.data(), op1);
        for (std::size_t i = 0; i < op1; ++i) {
            // NOTE: for the actual interpolation, we want the interpolation points in the [0, step_end - step_begin]
            // range. We need this because because in the polyjectory the polynomials are to be evaluated with the
            // time counted from the beginning of the step (in days).
            const auto ipoint = (cheby_nodes_unit[i] + 1) / 2 * cur_duration;

            for (auto j = 0u; j < 7u; ++j) {
                ipoints_span[i, j] = ipoint;
            }
        }

        // Fetch a buffer from the poly cache to store the result of polynomial interpolation.
        // NOTE: here we are abusing a bit the pwrap class in the sense that we will be storing
        // multiple polynomials into a single pwrap. This is ok, as pwrap is ultimately nothing but
        // a std::vector with caching.
        pwrap cfs_vec(interp_poly_pcache, boost::safe_numerics::safe<std::uint32_t>(7) * op1);
        auto *cf_ptr = cfs_vec.v.data();

        // Interpolate.
        //
        // NOTE: at this point, the layout of interp_buffer is as follows:
        //
        // c0, c0, c0, ...
        // c1, c1, c1, ...
        // ...
        // x0, y0, z0, ...
        // x1, y1, z1, ...
        //
        // This is, a (2op1 x 7) array divided logically into two (op1 x 7) parts:
        //
        // - the evaluation nodes,
        // - the evaluation values.
        //
        // The evaluation nodes are the Cheby nodes in the [0, step_end - step_begin] range,
        // and they are the same for all coordinates/velocities.
        //
        // The polynomial coefficients will be written into cf_ptr with the following layout:
        //
        // p0x, p0y, p0z, ...
        // p1x, p1y, p1z, ...
        // ...
        //
        // That is, an (op1 x 7) array.
        cfunc_interp(cf_ptr, interp_buffer.data(), nullptr, nullptr);

        // NOTE: unconditionally accept the interpolating polynomials
        // if the step size is below the min threshold.
        if (cur_duration < min_step_size) {
            ipolys.emplace_back(step_begin, step_end, std::move(cfs_vec));
            continue;
        }

        // Estimate the squared interpolation error.
        const auto interp_err2 = eval_interp_error2(cf_ptr, interp_buffer, sample_points, xyz_ieval,
                                                    step_begin_sat_epoch, is_deep_space, satrec, sgp4_prop, cfunc_r);
        if (std::holds_alternative<int>(interp_err2)) [[unlikely]] {
            return std::get<int>(interp_err2);
        }
        const double ierr2 = std::get<double>(interp_err2);

        if (ierr2 < err_thresh * err_thresh) {
            // The interpolation error is below the threshold, accept the polynomial.
            ipolys.emplace_back(step_begin, step_end, std::move(cfs_vec));
        } else {
            // The interpolation error is too high, bisect.
            // NOTE: since step_begin, step_end and cur_duration are all finite,
            // the bisection should not produce any non-finite value.
            const auto mid = step_begin + (cur_duration / 2);
            const auto mid_sat_epoch = step_begin_sat_epoch + (cur_duration_sat_epoch / 2);

            stack.push_back({step_begin, mid, step_begin_sat_epoch, mid_sat_epoch});
            stack.push_back({mid, step_end, mid_sat_epoch, step_end_sat_epoch});
        }
    }

    // Sort the polynomials according to the begin time of the step.
    //
    // NOTE: in principle we would want to provide a swap() primitive to pwrap
    // so that this sorting uses swap() rather than moves. However, in practice,
    // at least on GCC we never end up using swapping anyway for sorting because of this
    // optimisation:
    //
    // https://stackoverflow.com/questions/14212701/stdsort-does-not-always-call-stdswap
    //
    // This happens because the range we are sorting here is typically small. In the future,
    // if needed, we can implement the swap primitive for pwrap.
    std::ranges::sort(ipolys, {}, [](const auto &tup) { return std::get<0>(tup); });

    // NOTE: here in principle we could maybe try to remove sequences of increasingly-short
    // steps in the proximity of a step with discontinuities. In order to achieve this,
    // we would need to:
    //
    // - keep track of the interpolation error for *all* the steps,
    // - identify contiguous sequences of steps with interpolation
    //   error below the threshold,
    // - consolidate these sequences into a single step and re-do the
    //   interpolation over the consolidated step.
    //
    // This should reduce the memory/disk footprint and improve performance during
    // conjunction detection.

    // Write the polynomials and the step end times into poly_cf_buf and time_buf.
    for (const auto &[step_begin, step_end, poly] : ipolys) {
        // Add the polynomial coefficients to poly_cf_buf.
        const auto cf_span
            = hy::mdspan<const double, hy::extents<std::size_t, std::dynamic_extent, 7>>(poly.v.data(), op1);
        for (std::size_t i = 0; i < op1; ++i) {
            for (auto j = 0u; j < 7u; ++j) {
                poly_cf_buf.push_back(cf_span[i, j]);
            }
        }

        // Add the step end time to time_buf.
        if (time_buf.empty()) {
            // NOTE: if time_buf is empty, it means we are at the
            // very beginning of the satellite's trajectory, which must
            // be init_step_begin. Thus, since we sorted ipolys, it means
            // that we must be adding a step beginning at init_step_begin.
            assert(step_begin == init_step_begin);
            time_buf.push_back(step_begin);
        }
        time_buf.push_back(step_end);
    }

    return 0;
}

// Interpolate the gpe g in the [jdate_begin, jdate_end) time range over one or more interpolation steps.
// jdate_begin/end are UTC Julian dates.
//
// The size of the interpolation steps is determined by simulating a numerical integration with Keplerian
// dynamics and using the adaptive integration steps as interpolation steps.
//
// ets is a bag of thread-local data. poly_cf_buf and time_buf are buffers to which the interpolating
// polynomial coefficients and the times of the interpolation steps will be appended. pj_epoch_tai
// is the polyjectory's epoch in the TAI scale of time. reentry_radius and exit_radius are the
// reentry/exit radiuses.
//
// The return value will be 0 if everything went ok, otherwise it will be either an sgp4 error code,
// or error code 10 if non-finite values are detected.
int gpe_interpolate(const gpe &g, const auto &jdate_begin, const auto &jdate_end, auto &ets, auto &poly_cf_buf,
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    auto &time_buf, const auto &pj_epoch_tai, const double reentry_radius, const double exit_radius)
{
    namespace hy = heyoka;

    // Detect if the current gpe is deep-space.
    const auto is_deep_space = hy::model::gpe_is_deep_space(g.n0, g.e0, g.i0);

    // Initialise the satrec for computations with the official sgp4 code.
    elsetrec satrec;
    const auto sat_epoch = fetch_gpe_epoch(g);
    // NOTE: this is the baseline reference epoch used by the C++ SGP4 code,
    // corresponding to "jan 0, 1950. 0 hr".
    constexpr auto jd_sub = 2433281.5;
    SGP4Funcs::sgp4init(wgs72,
                        // NOTE: here we are using the 'a' opsmode in order to closely match propagations
                        // done by celestrak (by extension, I would imagine that this is also the setting
                        // used by spacetrack and associates). Note that the sgp4 Python module by default
                        // uses the 'i' mode instead. In the future, we may allow to make this a user-tunable
                        // parameter (perhaps together with the WGS models?).
                        'a', "",
                        // NOTE: here we are directly subtracting the UTC Julian dates
                        // in order to compute the epoch, which, according to the C++ code,
                        // must be expressed as the "number of days from jan 0, 1950. 0 hr".
                        // As far as I have understood, this epoch is being used by SDP4 to compute the
                        // lunisolar perturbations according to simplified ephemeris included in
                        // the C++ code. But, when using UTC Julian dates to compute the epoch,
                        // we are not accounting for leap seconds and thus the real physical time
                        // elapsed since "jan 0, 1950. 0 hr" is off by a few seconds. The "correct"
                        // thing to do here, I believe, would be to convert the UTC dates to
                        // TAI dates and then subtract to construct the epoch. However, this
                        // is likely not consistent with the the official SGP4 C++ code which seems to
                        // ignore leap seconds altogether (and testing with the Python SGP4 module
                        // indicates that indeed all calculations are being done with UTC Julian dates).
                        // Thus, at least for the time being, we ignore leap seconds for this
                        // calculation.
                        static_cast<double>(sat_epoch - jd_sub), g.bstar,
                        // NOTE: xndot and xnddot are not used during propagation, thus
                        // we just set them to zero.
                        0., 0., g.e0, g.omega0, g.i0, g.m0, g.n0, g.node0, satrec);
    // Check for errors.
    if (satrec.error != 0) [[unlikely]] {
        return satrec.error;
    }

    // Fetch the Keplerian propagator.
    auto &ta_kepler = ets.ta_kepler;

    // Cache the order.
    const auto order = ta_kepler.get_order();
    // NOTE: the order + 1 computation has been proven safe earlier.
    const std::uint32_t op1 = order + 1u;

    // If we are not dealing with a deep-space gpe, we also need
    // to set up heyoka's sgp4 propagator with op1 copies of the gpe g.
    auto &sgp4_prop = ets.sgp4_prop;
    if (!is_deep_space) {
        // Fill in the data buffer containing the gpe data.
        auto &sgp4_sat_data_buffer = ets.sgp4_sat_data_buffer;
        auto *ptr = sgp4_sat_data_buffer.data();
        std::ranges::fill(ptr, ptr + op1, g.n0);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, g.e0);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, g.i0);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, g.node0);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, g.omega0);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, g.m0);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, g.bstar);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, sat_epoch.hi);
        ptr += op1;
        std::ranges::fill(ptr, ptr + op1, sat_epoch.lo);

        // Setup sgp4_prop.
        // NOTE: we checked on construction that sgp4_sat_data_buffer.size()
        // can be represented as a std::size_t.
        sgp4_prop.replace_sat_data(hy::mdspan<const double, hy::extents<std::size_t, 9, std::dynamic_extent>>{
            sgp4_sat_data_buffer.data(), op1});
    }

    // NOTE: we now have to transform several UTC dates/epochs into TAI
    // in order to ensure we are operating within a uniform scale of time.
    // If we do not do that, doing arithmetics on UTC Julian dates would
    // give an incorrect result in case of leap second days.

    // Transform the satellite's epoch to TAI.
    const auto sat_epoch_tai = dl_utc_to_tai(sat_epoch);

    // Transform jdate_begin to TAI.
    const auto jdate_begin_tai = dl_utc_to_tai(jdate_begin);

    // Transform jdate_end to TAI.
    const auto jdate_end_tai = dl_utc_to_tai(jdate_end);

    for (auto cur_jdate_tai = jdate_begin_tai;;) {
        // NOTE: when computing with times and dates, we are careful to use
        // double-length arithmetic as much as possible. We do not however have
        // a double-length multiplication primitive, thus for conversions
        // of time durations we need to cast to single-length. If this ever becomes
        // an issue and we need extra precision, double-length multiplication can be
        // easily implemented on top of std::fma() and the TwoProductFMA() algorithm:
        //
        // https://www-pequan.lip6.fr/~graillat/papers/nolta07.pdf

        // Flag to signal if this is the last iteration.
        bool last_iteration = false;

        // Compute the beginning of the interpolation step in days since the epoch of
        // the polyjectory.
        const auto step_begin = cur_jdate_tai - pj_epoch_tai;

        // Compute it also with respect to the epoch of the satellite. We need this
        // in order to perform computations with the sgp4 propagators.
        const auto step_begin_sat_epoch = cur_jdate_tai - sat_epoch_tai;

        // Evaluate the state of the satellite at the beginning of the interpolation step.
        double sgp4_r[3]{}, sgp4_v[3]{};
        SGP4Funcs::sgp4(satrec,
                        // NOTE: we must pass the propagation time as the number of minutes
                        // since the satellite's epoch.
                        static_cast<double>(step_begin_sat_epoch) * 1440., sgp4_r, sgp4_v);
        // Check for errors.
        if (satrec.error != 0) [[unlikely]] {
            return satrec.error;
        }
        // Check for reentry/exit.
        const auto begin_r2 = (sgp4_r[0] * sgp4_r[0]) + (sgp4_r[1] * sgp4_r[1]) + (sgp4_r[2] * sgp4_r[2]);
        if (begin_r2 < reentry_radius * reentry_radius) [[unlikely]] {
            return 11;
        }
        if (begin_r2 >= exit_radius * exit_radius) [[unlikely]] {
            return 12;
        }

        // Setup ta_kepler.
        ta_kepler.set_time(0.);
        auto *ta_kepler_state_data = ta_kepler.get_state_data();
        std::ranges::copy(sgp4_r, ta_kepler_state_data);
        std::ranges::copy(sgp4_v, ta_kepler_state_data + 3);

        // Take a single step, ensuring that the step does not go past jdate_end_tai.
        // NOTE: ta_kepler measures time in seconds.
        const auto [oc, h] = ta_kepler.step(static_cast<double>(jdate_end_tai - cur_jdate_tai) * 86400.);

        // Check the outcome.
        if (oc == hy::taylor_outcome::time_limit) {
            // If we reached the time limit, this will be the last iteration.
            last_iteration = true;
        } else if (oc == heyoka::taylor_outcome::err_nf_state) [[unlikely]] {
            // Non-finite state detected, error out.
            return 10;
        } else {
            // The only possible outcome at this point is success.
            assert(oc == heyoka::taylor_outcome::success);

            // Check if the step is too short. This is meant to detect pathological trajectories
            // ending up close to the Earth's center.
            if (h < 1.) [[unlikely]] {
                return 13;
            }
        }

        // Compute the end of the current interpolation step (again, in days since the epoch of
        // the polyjectory).
        const auto step_end = step_begin + (h / 86400.);

        // Compute it also with respect to the epoch of the satellite.
        const auto step_end_sat_epoch = step_begin_sat_epoch + (h / 86400.);

        // Interpolate, isolating discontinuities via bisection.
        const auto res = gpe_interpolate_with_bisection(
            static_cast<double>(step_begin), static_cast<double>(step_end), static_cast<double>(step_begin_sat_epoch),
            static_cast<double>(step_end_sat_epoch), ets, is_deep_space, satrec, sgp4_prop, poly_cf_buf, time_buf);

        // Check the return code.
        if (res != 0) [[unlikely]] {
            // Error detected, no point in continuing further.
            return res;
        }

        // Break out if this is the last iteration.
        if (last_iteration) {
            break;
        }

        // Update cur_jdate.
        cur_jdate_tai = cur_jdate_tai + h / 86400.;
    }

    return 0;
}

// Interpolate in parallel all satellites between the UTC Julian dates jd_begin and jd_end.
//
// c_nodes_unit are the Chebyshev interpolation nodes in the [-1, 1] interval. *_tplt are the template
// propagator/integrator/llvm_state objects to be copied and used during interpolation. gpe_groups is the range of GPE
// groups (one per satellite). tmp_dir_path is the path to the output files. epoch_tai is the polyjectory's epoch,
// that is, jd_begin converted to double-length TAI. reentry_radius and exit_radius are the reentry/exit radiuses.
auto interpolate_all(const auto &c_nodes_unit, const auto &ta_kepler_tplt, const auto &sgp4_prop_tplt,
                     // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                     const auto &jit_state_tplt, const auto &gpe_groups, const double jd_begin, const double jd_end,
                     const auto &tmp_dir_path, const auto epoch_tai, const double reentry_radius,
                     const double exit_radius)
{
    // NOLINTNEXTLINE(misc-unused-alias-decls)
    namespace hy = heyoka;
    using dfloat = hy::detail::dfloat<double>;
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Cache the order + 1.
    // NOTE: this cast has been proven safe earlier.
    const auto op1 = static_cast<std::uint32_t>(c_nodes_unit.size());

    // Cache the total number of satellites.
    const auto n_sats = boost::numeric_cast<std::size_t>(gpe_groups.size());
    assert(n_sats > 0u);

    // A **global** vector of statuses, one per satellite.
    // We do not need to protect writes into this, as each status
    // will be written to exactly at most once.
    // NOTE: this is zero-inited, meaning that the default status flag
    // of each satellite is "no error detected".
    std::vector<int> global_status;
    global_status.resize(boost::numeric_cast<decltype(global_status.size())>(n_sats));

    // Create the traj and time data files.
    assert(!boost::filesystem::exists(tmp_dir_path / "traj"));
    std::ofstream traj_file((tmp_dir_path / "traj").string(), std::ios::binary | std::ios::out);
    traj_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);

    assert(!boost::filesystem::exists(tmp_dir_path / "time"));
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

    // Prepare the traj_offsets vector.
    std::vector<polyjectory::traj_offset> traj_offsets;
    traj_offsets.reserve(n_sats);

    // Flag to indicate that the writer thread should stop writing.
    std::atomic<bool> stop_writing = false;

    // NOTE: below we will have several threads waiting on each other. For instance, the writer thread needs to wait for
    // the producer threads to produce the trajectories before writing them to disk. On the other hand, the current
    // thread needs to wait for the writer thread to write enough trajectories to disk before enqueuing another producer
    // task.
    //
    // This creates situations in which an exception in a thread will lead to other threads hanging and waiting forever.
    //
    // We thus introduce a pair of atomic booleans, initially set to false, that will be set to true if the
    // writer/producer threads interrupt due to an exception. Within the wait() calls, we will be periodically checking
    // on these boolean flags and interrupt the wait in case they become true.
    //
    // NOTE: the point of using atomics instead other mechanisms (such as condition variables) is that they provide
    // noexcept behaviour, whereas traditional locking can in principle throw.

    // Flag to indicate that an exception was thrown in the writer thread.
    std::atomic<bool> writer_failure = false;
    // Flag to indicate that an exception was thrown in a producer thread.
    std::atomic<bool> producer_failure = false;

    // Timeout duration for wait operations.
    using namespace std::chrono_literals;
    const auto wait_timeout = 250ms;

    // Data structure to coordinate between the writer thread and the producer threads.
    struct {
        // The total number of satellites whose trajectories have been written to disk.
        std::size_t n_sats_written = 0;
        // The total number of satellites whose trajectories have been produced.
        std::size_t n_sats_processed = 0;
        // Condition variable and mutex to coordinate the access to and notify the modification of n_sats_written and
        // n_sats_processed.
        std::condition_variable cv;
        std::mutex mut;
    } wpc;

    // Launch the writer thread.
    auto writer_future
        = std::async(std::launch::async, [&traj_file, &traj_futures, &traj_offsets, &time_file, &time_futures, op1,
                                          &stop_writing, &writer_failure, n_sats, &wpc, wait_timeout]() {
              // NOTE: we place everything in a try/catch block so that we can set writer_failure to true in case of
              // exceptions before re-throwing.
              try {
                  // Track the trajectory offsets to build up traj_offsets.
                  safe_size_t cur_traj_offset = 0;

                  for (std::size_t i = 0; i < n_sats; ++i) {
                      // Fetch the futures.
                      auto &traj_fut = traj_futures[i];
                      auto &time_fut = time_futures[i];

                      // Wait until the futures become available, or return if a stop is requested.
                      while (traj_fut.wait_for(wait_timeout) != std::future_status::ready
                             || time_fut.wait_for(wait_timeout) != std::future_status::ready) {
                          // LCOV_EXCL_START
                          // NOTE: stop_writing will be set to true if an exception is thrown either by a producer
                          // thread or by the code that enqueues the producer tasks. Thus, here we are also indirectly
                          // checking the status of producer_failure.
                          if (stop_writing) [[unlikely]] {
                              return;
                          }
                          // LCOV_EXCL_STOP
                      }

                      // Fetch the data in the futures.
                      auto v_traj = traj_fut.get();
                      auto v_time = time_fut.get();

                      // Write the traj data.
                      traj_file.write(reinterpret_cast<const char *>(v_traj.data()),
                                      boost::safe_numerics::safe<std::streamsize>(v_traj.size()) * sizeof(double));

                      // Compute the number of steps, and update traj_offsets and cur_traj_offset.
                      assert(v_traj.size() % (safe_size_t(op1) * 7u) == 0u);
                      const auto n_steps = v_traj.size() / (safe_size_t(op1) * 7u);

                      traj_offsets.emplace_back(cur_traj_offset, n_steps);
                      cur_traj_offset += v_traj.size();

                      // Write the time data.
                      time_file.write(reinterpret_cast<const char *>(v_time.data()),
                                      boost::safe_numerics::safe<std::streamsize>(v_time.size()) * sizeof(double));

                      // Update n_sats_written.
                      {
                          const std::lock_guard lock(wpc.mut);
                          ++wpc.n_sats_written;
                      }
                      // Notify waiting threads that n_sats_written has been modified.
                      wpc.cv.notify_all();
                  }
                  // LCOV_EXCL_START
              } catch (...) {
                  writer_failure.store(true);
                  throw;
              }
              // LCOV_EXCL_STOP
          });

    // NOTE: at this point, the writer thread has started. From now on, we wrap everything in a try/catch block
    // so that, if any exception is thrown, we can safely stop the writer thread before re-throwing.
    try {
        // Bag of thread-local data to be used during interpolation.
        struct ets_data {
            using cfunc_t = void (*)(double *, const double *, const double *, const double *) noexcept;

            // The Keplerian integrator.
            hy::taylor_adaptive<double> ta_kepler;
            // The sgp4 propagator (heyoka's implementation).
            hy::model::sgp4_propagator<double> sgp4_prop;
            // llvm state for the auxiliary jitted functions.
            hy::llvm_state jit_state;
            // The auxiliary jitted functions.
            cfunc_t cfunc_r = nullptr;
            cfunc_t cfunc_interp = nullptr;
            // Buffer used to setup the gpe data in sgp4_prop.
            std::vector<double> sgp4_sat_data_buffer;
            // The Chebyshev nodes in the [-1, 1] range.
            // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
            const std::vector<double> cheby_nodes_unit;
            // Sample points evaluation bufffer.
            std::vector<double> sample_points;
            // The evaluation/interpolation buffer.
            std::vector<double> interp_buffer;
            // Stack to be used during bisection of interpolation steps with discontinuities.
            // The 4 values are the begin/end times of an interpolation step, measured both wrt
            // the beginning of the polyjectory and the satellite's epoch.
            std::vector<std::array<double, 4>> stack;
            // Polynomial cache providing storage for the results of polynomial interpolation.
            // NOTE: it is *really* important that this is declared
            // *before* ipolys, because ipolys will contain references
            // to and interact with interp_poly_pcache during destruction,
            // and we must be sure that ipolys is destroyed *before*
            // interp_poly_pcache.
            poly_cache interp_poly_pcache;
            // List of interpolating polynomials, one per bisecting
            // step. Each polynomial is keyed on the begin/end times
            // of the bisection step, measured from the beginning
            // of the polyjectory.
            std::vector<std::tuple<double, double, pwrap>> ipolys;
            // Buffer used to store the xyz positions of a satellite
            // when evaluating the interpolation error.
            std::vector<std::array<double, 3>> xyz_ieval;
        };
        using ets_t = oneapi::tbb::enumerable_thread_specific<ets_data, oneapi::tbb::cache_aligned_allocator<ets_data>,
                                                              oneapi::tbb::ets_key_usage_type::ets_key_per_instance>;
        ets_t ets([op1, &c_nodes_unit, &ta_kepler_tplt, &sgp4_prop_tplt, &jit_state_tplt]() {
            // Init the jit state.
            auto jit_state = *jit_state_tplt;

            // Lookup the jit functions.
            auto *cfunc_r = reinterpret_cast<ets_data::cfunc_t>(jit_state.jit_lookup("cfunc_r"));
            auto *cfunc_interp = reinterpret_cast<ets_data::cfunc_t>(jit_state.jit_lookup("cfunc_interp"));

            // Init sgp4_sat_data_buffer.
            std::vector<double> sgp4_sat_data_buffer;
            sgp4_sat_data_buffer.resize(boost::safe_numerics::safe<decltype(sgp4_sat_data_buffer.size())>(op1) * 9);
            // NOTE: we need to construct a std::size_t-sized span on top of this buffer.
            static_cast<void>(boost::numeric_cast<std::size_t>(sgp4_sat_data_buffer.size()));

            // Init sample_points.
            std::vector<double> sample_points;
            sample_points.resize(c_nodes_unit.size());

            // Init the evaluation/interpolation buffer.
            std::vector<double> interp_buffer;
            interp_buffer.resize(boost::safe_numerics::safe<decltype(interp_buffer.size())>(op1) * 14);
            // NOTE: we need to construct std::size_t-sized spans on top of this buffer.
            static_cast<void>(boost::numeric_cast<std::size_t>(interp_buffer.size()));

            // Init xyz_ieval.
            std::vector<std::array<double, 3>> xyz_ieval;
            xyz_ieval.resize(boost::numeric_cast<decltype(xyz_ieval.size())>(op1));

            return ets_data{.ta_kepler = *ta_kepler_tplt,
                            .sgp4_prop = *sgp4_prop_tplt,
                            .jit_state = std::move(jit_state),
                            .cfunc_r = cfunc_r,
                            .cfunc_interp = cfunc_interp,
                            .sgp4_sat_data_buffer = std::move(sgp4_sat_data_buffer),
                            .cheby_nodes_unit = c_nodes_unit,
                            .sample_points = std::move(sample_points),
                            .interp_buffer = std::move(interp_buffer),
                            .stack{},
                            .interp_poly_pcache{},
                            .ipolys{},
                            .xyz_ieval = std::move(xyz_ieval)};
        });

        // NOTE: it is **very** important that we create the task group *after* ets. This ordering ensures that the task
        // group is destroyed (and thus joined) *before* the destruction of ets. If we created the task group before
        // ets, then ets would be destroyed before the tasks in the task group (which are using ets) are finished,
        // leading to use-after-free, UB, etc.
        detail::jtask_group tg;

        // Run the parallel interpolation loop.
        //
        // NOTE: we process the satellites in chunks because we want the writer thread to make steady progress.
        //
        // If we don't do the chunking, what happens is that TBB starts immediately processing satellites throughout
        // the *entire* range, whereas the writer thread must proceed strictly sequentially. We thus end up in a
        // situation in which a lot of file writing happens at the very end while not overlapping with the
        // computation, and memory usage is high because we have to keep around for long unwritten data.
        //
        // With chunking, the writer thread can fully process chunk N while chunk N+1 is computing.
        //
        // NOTE: there are tradeoffs in selecting the chunk size. If it is too large, we are negating the benefits
        // of chunking wrt computation/transfer overlap and memory usage. If it is too small, we are limiting
        // parallel speedup. The current value is based on preliminary performance evaluation with full gpe catalogs,
        // but I am not sure if this can be made more robust/general. In general, the "optimal" chunking
        // will depend on several variables such as the number of CPU cores, the available memory,
        // the integration length, the batch size and so on.
        //
        // NOTE: I am also not sure whether or not it is possible to achieve the same result more elegantly
        // with some TBB partitioner/range wizardry.
        constexpr auto chunk_size = 256u;

        for (std::size_t start_sat_idx = 0; start_sat_idx != n_sats;) {
            // Flag to signal that an exception was thrown in the writer/producer threads.
            bool wait_failure = false;

            // Flag to signal that we already reported a delay in task processing for this iteration.
            // This is relevant only for debug output.
            bool delay_reported = false;

            // NOTE: if this is the first iteration of the for loop, do not wait on the condition variable and
            // unconditionally proceed to schedule the first task.
            if (start_sat_idx != 0u) {
                // NOTE: before scheduling the next task, we wait until the number of written trajectories is a large
                // fraction of the number of computed trajectories. The goal here is to avoid enqueueing too many tasks
                // if the writer thread is not keeping up - this would lead to high memory consumption because the
                // computed trajectories will be kept in RAM until they are written to disk.

                // Predicate for waiting on wpc.cv.
                const auto pred = [&wpc, &delay_reported] {
                    if (wpc.n_sats_processed == 0u) {
                        // NOTE: if we get here, we know that we already scheduled the first task but it has not made
                        // any progress yet. Wait until some trajectories have been computed.
                        return false;
                    }

                    // Determine the ratio of written/computed trajectories.
                    const auto ratio
                        = static_cast<double>(wpc.n_sats_written) / static_cast<double>(wpc.n_sats_processed);
                    if (!std::isfinite(ratio)) [[unlikely]] {
                        // LCOV_EXCL_START
                        throw std::runtime_error(fmt::format("A non-finite n_sats_written/n_sats_processed ratio of {} "
                                                             "was detected in make_sgp4_polyjectory()",
                                                             ratio));
                        // LCOV_EXCL_STOP
                    }

                    // NOTE: allow to schedule the next task if we have written at least 75% of the computed
                    // trajectories thus far.
                    constexpr auto min_ratio = 0.75;
                    if (!delay_reported && ratio < min_ratio) {
                        log_debug("Delaying task processing in make_sgp4_polyjectory() while waiting for trajectories "
                                  "to be written to disk");
                        delay_reported = true;
                    }
                    return ratio >= min_ratio;
                };

                // Wait on wpc.cv until either the writer thread has made enough progress, or an exception was raised in
                // the writer/producer threads.
                std::unique_lock lock(wpc.mut);
                while (!wpc.cv.wait_for(lock, wait_timeout, pred)) {
                    // NOTE: if we end up here, it means that wait_timeout has elapsed and not enough progress on the
                    // writing has been made yet. Before resuming waiting, we check for failures in the writer and
                    // producer threads. If we detect failures, we set the flag wait_failure to true and break out.

                    // LCOV_EXCL_START
                    // NOTE: we can unlock while we check the failure flags since we are not accessing anything in wpc.
                    lock.unlock();

                    if (writer_failure.load()) [[unlikely]] {
                        log_warning("Exception detected in the writer thread of make_sgp4_polyjectory()");
                        wait_failure = true;
                        break;
                    }
                    if (producer_failure.load()) [[unlikely]] {
                        // NOTE: in case of a producer failure, the writer thread will eventually become stuck waiting
                        // on a trajectory that will never be produced.
                        log_warning("Exception detected in a producer thread of make_sgp4_polyjectory()");
                        wait_failure = true;
                        break;
                    }

                    // Re-lock.
                    lock.lock();
                    // LCOV_EXCL_STOP
                }
            } // LCOV_EXCL_LINE

            if (wait_failure) [[unlikely]] {
                // One of the writer/producer threads threw an exception, break out. The exception will be raised
                // either in the destructor of the task group (which implicitly invokes wait(), which will raise the
                // exception), or when we wait on the writer future.
                break; // LCOV_EXCL_LINE
            }

            // Determine the end index for the next task.
            const auto n_rem_sats = n_sats - start_sat_idx;
            const auto end_sat_idx = start_sat_idx + (n_rem_sats < chunk_size ? n_rem_sats : chunk_size);

            tg.run([&ets, &gpe_groups, dl_jd_begin = dfloat{jd_begin, 0.}, dl_jd_end = dfloat{jd_end, 0.},
                    &global_status, &traj_promises, &time_promises, epoch_tai, reentry_radius, exit_radius,
                    start_sat_idx, end_sat_idx, &wpc, &producer_failure]() {
                // NOTE: we place the entire task in a try/catch block so that we can set producer_failure to true in
                // case of exceptions before re-throwing.
                try {
                    oneapi::tbb::parallel_for(
                        oneapi::tbb::blocked_range<decltype(gpe_groups.size())>(start_sat_idx, end_sat_idx),
                        [&ets, &gpe_groups, dl_jd_begin, dl_jd_end, &global_status, &traj_promises, &time_promises,
                         epoch_tai, reentry_radius, exit_radius, &wpc](const auto &range) {
                            // Fetch the thread-local data.
                            auto &local_ets = ets.local();

                            // NOTE: isolate to avoid issues with thread-local data. See:
                            // https://oneapi-src.github.io/oneTBB/main/tbb_userguide/work_isolation.html
                            // We may be invoking TBB primitives when using heyoka's sgp4 propagator.
                            oneapi::tbb::this_task_arena::isolate(
                                [&local_ets, &range, &gpe_groups, dl_jd_begin, dl_jd_end, &global_status,
                                 &traj_promises, &time_promises, epoch_tai, reentry_radius, exit_radius, &wpc]() {
                                    for (auto sat_idx = range.begin(); sat_idx != range.end(); ++sat_idx) {
                                        // Fetch the gpe group for the current satellite.
                                        const auto gpe_group = *gpe_groups[sat_idx];
                                        assert(std::ranges::size(gpe_group) >= 1);
                                        static_assert(std::ranges::sized_range<decltype(gpe_group)>);

                                        // Cache the end of the group.
                                        const auto group_end = std::ranges::end(gpe_group);

                                        // NOTE: the policy we follow is that for interpolation we want to begin
                                        // with the latest gpe whose epoch is *earlier than or equal to* jd_begin, if
                                        // possible. This may not always result in the *best* (i.e., most precise)
                                        // propagation, because regular gpes are typically most accurate somewhen
                                        // *before* the epoch:
                                        //
                                        // https://celestrak.org/publications/AAS/07-127/AAS-07-127.pdf
                                        //
                                        // This happens because regular gpes are fitted to past observations.
                                        // However, in practice:
                                        //
                                        // - during operational conjunction screening we will typically have no choice
                                        //   other than using the latest available GP data, and
                                        // - SupGP gpes are actually fitted to the future ephemeris provided by
                                        //   the operators (rather than past observations like the regular gpes).
                                        //
                                        // If necessary and if this becomes an issue, we can think of more sophisticated
                                        // policies for selecting the gpes to use for propagation.

                                        // Locate the first gpe in the group whose epoch is
                                        // *greater than* jd_begin. This is the gpe *right after*
                                        // the one we want to target.
                                        auto it_gpe
                                            = std::ranges::upper_bound(gpe_group, dl_jd_begin, {},
                                                                       [](const gpe &g) { return fetch_gpe_epoch(g); });
                                        // If possible, move to the previous GPE.
                                        it_gpe -= (it_gpe != std::ranges::begin(gpe_group));

                                        // Init interpolation_begin.
                                        auto interpolation_begin = dl_jd_begin;

                                        // Prepare the write buffers to store the polynomial coefficients
                                        // and the end times of the interpolation steps.
                                        std::vector<double> poly_cf_buf, time_buf;

                                        // Iterate over the gpes in the group and interpolate.
                                        while (true) {
                                            // Flag to signal that this will be the last iteration.
                                            bool last_iteration = false;

                                            // Compute the iterator to the next gpe.
                                            const auto next_it_gpe = it_gpe + 1;

                                            // Compute the upper time limit for the interpolation with
                                            // the current gpe.
                                            dfloat interpolation_end{};
                                            if (next_it_gpe == group_end) {
                                                // We are at the last gpe of the group, we have no choice
                                                // but to use it until the end.
                                                interpolation_end = dl_jd_end;
                                                last_iteration = true;
                                            } else {
                                                // Compute the epoch of the next gpe.
                                                const auto next_gpe_epoch = fetch_gpe_epoch(*next_it_gpe);

                                                if (next_gpe_epoch >= dl_jd_end) {
                                                    // The next gpe's epoch is at or after the end date.
                                                    // We can keep on using this gpe until the end.
                                                    interpolation_end = dl_jd_end;
                                                    last_iteration = true;
                                                } else {
                                                    // The next gpe's epoch begins before the end date.
                                                    // We need to stop the interpolation with the current gpe
                                                    // at next_gpe_epoch.
                                                    interpolation_end = next_gpe_epoch;
                                                }
                                            }

                                            // Interpolate using the current gpe.
                                            const auto res = gpe_interpolate(
                                                *it_gpe, interpolation_begin, interpolation_end, local_ets, poly_cf_buf,
                                                time_buf, epoch_tai, reentry_radius, exit_radius);

                                            // Check the status. If it is nonzero, set the global status
                                            // flag for the satellite and break out.
                                            if (res != 0) [[unlikely]] {
                                                global_status[sat_idx] = res;
                                                break;
                                            }

                                            // Break out if this is the last iteration.
                                            if (last_iteration) {
                                                break;
                                            }

                                            // Update interpolation_begin.
                                            interpolation_begin = fetch_gpe_epoch(*next_it_gpe);

                                            // Move to the next gpe.
                                            it_gpe = next_it_gpe;
                                        }

                                        // Send the buffers to the futures.
                                        traj_promises[sat_idx].set_value(std::move(poly_cf_buf));
                                        time_promises[sat_idx].set_value(std::move(time_buf));

                                        // Update n_sats_processed.
                                        {
                                            const std::lock_guard lock(wpc.mut);
                                            ++wpc.n_sats_processed;
                                        }
                                        // Notify waiting threads that n_sats_processed has been modified.
                                        wpc.cv.notify_all();
                                    }
                                });
                        });
                    // LCOV_EXCL_START
                } catch (...) {
                    producer_failure.store(true);
                    throw;
                }
                // LCOV_EXCL_STOP
            });

            start_sat_idx = end_sat_idx;
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

polyjectory make_sgp4_polyjectory(heyoka::mdspan<const gpe, heyoka::extents<std::size_t, std::dynamic_extent>> gpes,
                                  const double jd_begin, const double jd_end, const double reentry_radius,
                                  const double exit_radius, std::optional<std::filesystem::path> data_dir, bool persist,
                                  std::optional<std::filesystem::path> tmpdir)
{
    namespace hy = heyoka;
    using dfloat = hy::detail::dfloat<double>;

    // Check the date range.
    if (!std::isfinite(jd_begin) || !std::isfinite(jd_end) || !(jd_begin < jd_end)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "Invalid Julian date interval [{}, {}) supplied to make_sgp4_polyjectory(): the begin/end dates "
            "must be finite and the end date must be strictly after the begin date",
            jd_begin, jd_end));
    }

    // Check the reentry/exit radiuses.
    // NOTE: not sure why, but on the CI clang-tidy seems to think that
    // std::isnan() returns int...
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    if (std::isnan(reentry_radius) || std::isnan(exit_radius)) [[unlikely]] {
        throw std::invalid_argument("The reentry/exit radiuses in make_sgp4_polyjectory() cannot be NaN");
    }
    if (!(reentry_radius < exit_radius)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The reentry radius ({}) must be less than the exit radius ({}) in make_sgp4_polyjectory()",
                        reentry_radius, exit_radius));
    }

    // Cache the total number of GPEs.
    const auto n_gpes = gpes.extent(0);

    // Need at least 1 gpe.
    if (n_gpes == 0u) [[unlikely]] {
        throw std::invalid_argument("make_sgp4_polyjectory() requires a non-empty array of GPEs in input");
    }

    // The view that will be used to group the GPEs by norad_id.
    auto grouped_view = std::ranges::subrange(gpes.data_handle(), gpes.data_handle() + n_gpes)
                        | std::views::chunk_by([](const gpe &g1, const gpe &g2) { return g1.norad_id == g2.norad_id; });
    // NOTE: we will construct a vector of iterators into grouped_view. The reason we do this is that grouped_view is
    // not a random-access ranged, which prevents us from feeding it directly to a TBB parallel for loop.
    std::vector<decltype(grouped_view.begin())> gpe_groups;

    // The Taylor integrator that will be used to infer the interpolation order and step sizes.
    std::optional<hy::taylor_adaptive<double>> ta_kepler_tplt;

    // The sgp4 propagator that will be used for the interpolation of non-deep-space GPEs.
    std::optional<hy::model::sgp4_propagator<double>> sgp4_prop_tplt;

    // The llvm state that will contain auxiliary JITed functions used during interpolation.
    std::optional<hy::llvm_state> jit_state_tplt;

    // The values of the Chebyshev nodes in the [-1, 1] range.
    std::vector<double> c_nodes_unit;

    // The polyjectory epoch. This will be initialised as
    // jd_begin converted to double-length TAI.
    dfloat epoch_tai;

    stopwatch sw;

    // NOTE: several validation and preparation steps can be performed in parallel.
    oneapi::tbb::parallel_invoke(
        // Finiteness check on the GPE data.
        [&gpes]() { detail::check_gpe_finiteness(gpes); },
        // Check that gpes is sorted first by norad id, and then by epoch.
        [&gpes]() { detail::check_gpe_order(gpes); },
        // Fill in gpe_groups with iterators from grouped_view.
        [n_gpes, &grouped_view, &gpe_groups]() {
            gpe_groups.reserve(n_gpes);
            for (auto it = grouped_view.begin(); it != grouped_view.end(); ++it) {
                gpe_groups.push_back(it);
            }
        },
        // Initialise the integrator, the propagator, the jitted functions and c_nodes_unit.
        [&ta_kepler_tplt, &sgp4_prop_tplt, &jit_state_tplt, &c_nodes_unit]() {
            detail::build_tplts(ta_kepler_tplt, sgp4_prop_tplt, jit_state_tplt);

            // Fetch the order + 1.
            // NOTE: no need for overflow checks here as we checked in several
            // places (e.g., vm interpolation, construction of the jit functions, etc.)
            // that this is safe.
            const auto op1 = static_cast<std::uint32_t>(ta_kepler_tplt->get_order() + 1u);

            // Setup the Chebyshev nodes in the [-1, 1] range.
            c_nodes_unit.reserve(op1);
            for (std::uint32_t i = 0; i < op1; ++i) {
                c_nodes_unit.push_back(std::cos((2. * i + 1) / (2. * op1) * boost::math::constants::pi<double>()));
            }

            // Sort them in ascending order.
            //
            // NOTE: for polynomial interpolation with the Bjorck-Pereira algorithm,
            // it seems like sorting the sampling points in ascending order may improve
            // numerical stability:
            //
            // https://link.springer.com/article/10.1007/BF01408579
            std::ranges::reverse(c_nodes_unit);
            assert(std::ranges::is_sorted(c_nodes_unit));
        },
        // Setup epoch_tai.
        [&epoch_tai, jd_begin]() { epoch_tai = detail::dl_utc_to_tai(dfloat{jd_begin, 0.}); });

    log_trace("make_sgp4_polyjectory() validation/preparation time: {}s", sw);

    // Assemble a "unique" dir path into either the system temp dir, or the tmpdir
    // specified by the user.
    const auto tmp_dir_path
        = detail::create_temp_dir("mizuba_sgp4_polyjectory-%%%%-%%%%-%%%%-%%%%",
                                  tmpdir ? boost::filesystem::path(*tmpdir) : std::optional<boost::filesystem::path>{});

    // NOTE: from now on, we need to ensure that the temp dir is automatically
    // cleaned up, even in case of exceptions. We use this little RAII helper
    // for this purpose.
    // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
    struct tmp_cleaner {
        // NOTE: store by reference so that we are sure that constructing
        // a tmp_cleaner cannot possibly throw.
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
        const boost::filesystem::path &path;
        ~tmp_cleaner()
        {
            // NOTE: not sure why the code coverage tool does not pick this up.
            boost::filesystem::remove_all(path); // LCOV_EXCL_LINE
        }
    };
    const tmp_cleaner tmp_clean{tmp_dir_path};

    // Interpolate all satellites.
    sw.reset();
    auto [status, traj_offsets]
        = detail::interpolate_all(c_nodes_unit, ta_kepler_tplt, sgp4_prop_tplt, jit_state_tplt, gpe_groups, jd_begin,
                                  jd_end, tmp_dir_path, epoch_tai, reentry_radius, exit_radius);
    log_trace("make_sgp4_polyjectory() total interpolation time: {}s", sw);

    // Build and return the polyjectory.
    return polyjectory(std::filesystem::path((tmp_dir_path / "traj").string()),
                       std::filesystem::path((tmp_dir_path / "time").string()),
                       // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                       ta_kepler_tplt->get_order(), std::move(traj_offsets), std::move(status), epoch_tai.hi,
                       epoch_tai.lo, std::move(data_dir), persist, std::move(tmpdir));
}

} // namespace mizuba
