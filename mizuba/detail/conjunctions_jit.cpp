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

#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/relational.hpp>
#include <heyoka/math/select.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/mdspan.hpp>

#include "conjunctions_jit.hpp"
#include "poly_utils.hpp"

namespace mizuba::detail
{

namespace
{

// Create an expression for the computation of the translation of a polynomial.
// That is, given a polynomial 'cfs' represented as a list of coefficients c_i, the function
// will compute the coefficients c'_i of the polynomial resulting from substituting
// the original polynomial variable x with x + a. The formula for the translated coefficients is:
//
// c'_i = sum_{k=i}^n (c_k * choose(k, k-i) * a**(k-i))
//
// (where n == order of the polynomial).
//
// NOTE: for my own sanity when re-deriving this formula, remember that
// choose(k, k - i) == choose(k, i).
//
// NOTE: if a is zero, cfs will be returned unchanged.
auto poly_translate(const std::vector<heyoka::expression> &cfs, const heyoka::expression &a)
{
    namespace hy = heyoka;
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;

    assert(!cfs.empty());

    // Fetch the polynomial order.
    const auto order = safe_uint32_t(cfs.size() - 1u);

    // Pre-compute the powers of a up to 'order'.
    std::vector<hy::expression> a_pows;
    a_pows.reserve(order + 1);
    // NOTE: in the corner case of a polynomial of order 0, a_pows will
    // be larger than necessary, but it does not matter.
    a_pows.emplace_back(1.);
    a_pows.push_back(a);
    for (safe_uint32_t i = 2; i <= order; ++i) {
        // NOTE: do it like this, rather than the more
        // straightforward way of multiplying repeatedly
        // by a, in order to improve instruction-level
        // parallelism.
        if (i % 2 == 0) {
            a_pows.push_back(a_pows[i / 2] * a_pows[i / 2]);
        } else {
            a_pows.push_back(a_pows[i / 2 + 1] * a_pows[i / 2]);
        }
    }

    // Compute the translated coefficients, storing them in 'out'.
    std::vector<hy::expression> out;
    out.reserve(order + 1);
    for (safe_uint32_t i = 0; i <= order; ++i) {
        std::vector<hy::expression> tmp;
        tmp.reserve(order + 1 - i);

        for (auto k = i; k <= order; ++k) {
            tmp.push_back(
                cfs[k]
                * boost::math::binomial_coefficient<double>(static_cast<unsigned>(k), static_cast<unsigned>(k - i))
                * a_pows[k - i]);
        }

        out.push_back(hy::sum(std::move(tmp)));
    }

    return out;
}

// Add a compiled function for the computation of the translations of 6 polynomials.
// The translation amount is par[0]. The coefficients of the 6 polynomials are assumed
// to be stored contiguously in row-major format (that is, the coefficients of the first
// poly, followed by the coefficients of the second poly, and so on).
void add_poly_translator_a6(heyoka::llvm_state &s, std::uint32_t order)
{
    namespace hy = heyoka;

    // The original polynomial coefficients are the
    // input variables for the compiled function.
    std::vector<hy::expression> cfs;
    for (auto i = 0u; i < 6u; ++i) {
        for (std::uint32_t j = 0; j <= order; ++j) {
            cfs.emplace_back(fmt::format("c_{}_{}", i, j));
        }
    }

    // The translation amount 'a' is implemented as the
    // first and only parameter of the compiled function.
    const auto a = hy::par[0];

    // Create the expressions for the translation of the coefficients.
    std::vector<hy::expression> out;
    for (decltype(cfs.size()) i = 0u; i < 6u; ++i) {
        auto tmp = poly_translate(std::vector(cfs.data() + i * (order + 1u), cfs.data() + (i + 1u) * (order + 1u)), a);
        out.insert(out.end(), tmp.begin(), tmp.end());
    }

    // Add the compiled function.
    hy::add_cfunc<double>(s, "pta6_cfunc", out, cfs);
}

// Utilities for the implementation of the Cargo-Shisha algorithm.

// min() implementation with nan propagation: (b == b) ? ((b < a) ? b : a) : b.
auto hy_min = [](auto a, auto b) { return select(eq(b, b), select(lt(b, a), b, a), b); };

// max() implementation with nan propagation: (b == b) ? ((a < b) ? b : a) : b.
auto hy_max = [](auto a, auto b) { return select(eq(b, b), select(lt(a, b), b, a), b); };

// Pairwise reduction on a list of values vals using the binary functor f.
auto pairwise_reduce(auto vals, const auto &f)
{
    assert(!vals.empty());

    using safe_size_t = boost::safe_numerics::safe<decltype(vals.size())>;

    while (vals.size() != 1u) {
        decltype(vals) new_vals;

        for (safe_size_t i = 0; i < vals.size(); i += 2) {
            if (i + 1 == vals.size()) {
                new_vals.push_back(vals[i]);
            } else {
                new_vals.push_back(f(vals[i], vals[i + 1]));
            }
        }

        vals.swap(new_vals);
    }

    return vals[0];
}

// Construct the expression corresponding to the Cargo-Shisha algorithm for the
// computation of the enclosure of a polynomial over the interval [lb, ub] for the
// independent variable. See:
//
// https://nvlpubs.nist.gov/nistpubs/jres/70B/jresv70Bn1p79_A1b.pdf
auto cs_enclosure(const std::vector<heyoka::expression> &cfs, const heyoka::expression &lb,
                  const heyoka::expression &ub)
{
    namespace hy = heyoka;
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;

    assert(!cfs.empty());

    // Fetch the polynomial order.
    const auto order = safe_uint32_t(cfs.size() - 1u);

    // Step 1: translate the original polynomial by lb, so that we obtain
    // a new polynomial to be evaluated in the interval [0, delta], where
    // delta = ub - lb.
    const auto delta = ub - lb;
    auto tcfs = poly_translate(cfs, lb);

    // Step 2: rescale the coefficients of tcfs by powers of delta, so that
    // we obtain a polynomial to be evaluated in the interval [0, 1].

    // Pre-compute the powers of delta up to 'order'.
    std::vector<hy::expression> delta_pows;
    delta_pows.reserve(order + 1);
    // NOTE: in the corner case of a polynomial of order 0, delta_pows will
    // be larger than necessary, but it does not matter.
    delta_pows.emplace_back(1.);
    delta_pows.push_back(delta);
    for (safe_uint32_t i = 2; i <= order; ++i) {
        // NOTE: do it like this, rather than the more
        // straightforward way of multiplying repeatedly
        // by a, in order to improve instruction-level
        // parallelism.
        if (i % 2 == 0) {
            delta_pows.push_back(delta_pows[i / 2] * delta_pows[i / 2]);
        } else {
            delta_pows.push_back(delta_pows[i / 2 + 1] * delta_pows[i / 2]);
        }
    }

    // Do the rescaling.
    for (safe_uint32_t i = 0; i <= order; ++i) {
        tcfs[i] *= delta_pows[i];
    }

    // Init the first row of the difference table.
    std::vector<hy::expression> row;
    row.reserve(order + 1);
    for (safe_uint32_t i = 0; i <= order; ++i) {
        row.push_back(
            tcfs[i]
            / boost::math::binomial_coefficient<double>(static_cast<unsigned>(order), static_cast<unsigned>(i)));
    }

    // Init the difference table.
    std::vector<std::vector<hy::expression>> dtable;
    dtable.reserve(order + 1);
    dtable.push_back(std::move(row));

    // Fill in the other rows.
    for (safe_uint32_t i = 1; i <= order; ++i) {
        const auto &prev_row = dtable.back();

        // Init the new row.
        std::vector<hy::expression> new_row;
        new_row.reserve(order + 1 - i);

        // Fill it up.
        for (safe_uint32_t k = 0; k <= order - i; ++k) {
            new_row.push_back(prev_row[k] + prev_row[k + 1]);
        }

        // Add it to the table.
        dtable.push_back(std::move(new_row));
    }

    // Extract the b values.
    std::vector<hy::expression> b_values;
    b_values.reserve(order + 1);
    for (const auto &cur_row : dtable) {
        b_values.push_back(cur_row[0]);
    }

    // Calculate min/max among the b values.
    // NOTE: the hy_min/max primitives ensure the propagation of nans.
    auto min_b = pairwise_reduce(b_values, hy_min);
    auto max_b = pairwise_reduce(b_values, hy_max);

    return std::make_pair(std::move(min_b), std::move(max_b));
}

// Add a compiled function for the computation of the aabb of an object
// via the Cargo-Shisha algorithm.
void add_aabb_cs_func(heyoka::llvm_state &s, std::uint32_t order_)
{
    namespace hy = heyoka;
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;

    const auto order = safe_uint32_t(order_);

    // Create variables representing the poly coefficients for x/y/z,
    // vx/vy/vz and r. We never read anything from vx/vy/vz, but we need
    // them to be present due to the buffer layout requirements for cfuncs.
    std::vector<hy::expression> inputs;
    inputs.reserve((order + 1) * 7);
    for (const auto *name : {"x", "y", "z", "vx", "vy", "vz", "r"}) {
        for (safe_uint32_t i = 0; i <= order; ++i) {
            inputs.emplace_back(fmt::format("{}_{}", name, static_cast<std::uint32_t>(i)));
        }
    }

    // The lower and upper bound for the evaluation are implemented as
    // par[0] and par[1] respectively. The conjunction radius is par[2].
    const auto lb = hy::par[0];
    const auto ub = hy::par[1];
    const auto conj_radius = hy::par[2];

    // Init the outputs.
    std::vector<hy::expression> outputs;
    outputs.reserve(8);
    using osize_t = decltype(outputs.size());
    // Add the outputs for x/y/z.
    for (auto i = 0u; i < 3u; ++i) {
        const auto [cur_min, cur_max]
            = cs_enclosure(std::vector(inputs.data() + static_cast<osize_t>(order + 1) * i,
                                       inputs.data() + static_cast<osize_t>(order + 1) * (i + 1u)),
                           lb, ub);
        outputs.push_back(cur_min - conj_radius);
        outputs.push_back(cur_max + conj_radius);
    }

    // Add the outputs for r.
    const auto [cur_min, cur_max] = cs_enclosure(std::vector(inputs.data() + static_cast<osize_t>(order + 1) * 6,
                                                             inputs.data() + static_cast<osize_t>(order + 1) * 7),
                                                 lb, ub);
    outputs.push_back(cur_min - conj_radius);
    outputs.push_back(cur_max + conj_radius);

    // Add the compiled function.
    heyoka::add_cfunc<double>(s, "aabb_cs", outputs, inputs);
}

// Add a compiled function for the computation of the enclosure of a polynomial
// via the Cargo-Shisha algorithm. The evaluation interval is [0, par[0]].
void add_cs_enc_func(heyoka::llvm_state &s, std::uint32_t order_)
{
    namespace hy = heyoka;
    using namespace hy::literals;
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;

    const auto order = safe_uint32_t(order_);

    // The inputs of the function are the polynomial coefficients.
    std::vector<hy::expression> inputs;
    inputs.reserve(order + 1);
    for (safe_uint32_t i = 0; i <= order; ++i) {
        inputs.emplace_back(fmt::format("cf_{}", static_cast<std::uint32_t>(i)));
    }

    // Init the upper bound of the evaluation interval.
    const auto ub = hy::par[0];

    // NOTE: cs_enclosure() also does an unnecessary translation, because the lower
    // bound here is 0. Luckily, the expression system is able to optimise the
    // translation by zero away.
    const auto [min, max] = cs_enclosure(inputs, 0_dbl, ub);

    heyoka::add_cfunc<double>(s, "cs_enc", {min, max}, inputs);
}

// Add a compiled function for the computation of the interpolating polynomial
// of order 'order' for the distance square between two objects over a time interval.
//
// The evolution in time of the objects' positions is expressed via time-dependent polynomials
// of order 'order'. The interpolation interval is [0, par[0]].
void add_dist2_interp(heyoka::llvm_state &s, std::uint32_t order)
{
    namespace hy = heyoka;
    using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;

    // Safely compute order + 1.
    const auto op1 = static_cast<std::uint32_t>(safe_uint32_t(order) + 1);

    // Create the expressions representing the input polynomial coefficients.
    // NOTE: the polynomials are expected to be stored in column-major format.
    std::vector<hy::expression> cfs;
    for (std::uint32_t i = 0; i < op1; ++i) {
        for (const char *s : {"c_{}_xi", "c_{}_yi", "c_{}_zi", "c_{}_xj", "c_{}_yj", "c_{}_zj"}) {
            cfs.emplace_back(fmt::format(fmt::runtime(s), i));
        }
    }

    // Create the expressions for the evaluation points. These are the Chebyshev nodes
    // rescaled to the [0, par[0]] interval.
    std::vector<hy::expression> eval_points;
    for (std::uint32_t i = 0; i < op1; ++i) {
        eval_points.push_back(
            (std::cos((2. * i + 1.) / (2. * static_cast<std::uint32_t>(op1)) * boost::math::constants::pi<double>())
             + 1.)
            / 2. * hy::par[0]);
    }

    // Compute the distance square at the evaluation points.
    std::vector<hy::expression> dist2;
    for (std::uint32_t i = 0; i < op1; ++i) {
        // Init the array of positions at the current evaluation point.
        std::array<hy::expression, 6> cur_pos;

        // Compute the positions for the current evaluation point.
        for (std::uint32_t j = 0; j < 6u; ++j) {
            cur_pos[j] = horner_eval(cfs.data() + j, order, eval_points[i],
                                     // NOTE: it is important that we cast the stride argument to decltype(cfs.size()),
                                     // this ensures no overflow when computing indices in horner_eval().
                                     static_cast<decltype(cfs.size())>(6));
        }

        // Add the distance square for the current evaluation point.
        const auto &[xi, yi, zi, xj, yj, zj] = cur_pos;
        dist2.push_back(hy::sum({pow(xi - xj, 2.), pow(yi - yj, 2.), pow(zi - zj, 2.)}));
    }

    // Interpolate.
    const auto interp_res = vm_interp(order, std::move(eval_points), std::move(dist2));

    // Add the compiled function.
    heyoka::add_cfunc<double>(s, "dist2_interp", interp_res.second, cfs);
}

} // namespace

// NOTE: consider experimenting with, e.g., slp vectorization here.
conj_jit_data::conj_jit_data(std::uint32_t order)
{
    namespace hy = heyoka;

    // Add the compiled functions.
    auto *fp_t = hy::detail::to_internal_llvm_type<double>(state);
    detail::add_poly_translator_a6(state, order);
    hy::detail::llvm_add_fex_check(state, fp_t, order, 1);
    hy::detail::llvm_add_poly_rtscc(state, fp_t, order, 1);
    detail::add_aabb_cs_func(state, order);
    detail::add_cs_enc_func(state, order);
    detail::add_dist2_interp(state, order);

    // Compile.
    state.compile();

    // Lookup.
    pta6_cfunc = reinterpret_cast<decltype(pta6_cfunc)>(state.jit_lookup("pta6_cfunc"));
    fex_check = reinterpret_cast<decltype(fex_check)>(state.jit_lookup("fex_check"));
    rtscc = reinterpret_cast<decltype(rtscc)>(state.jit_lookup("poly_rtscc"));
    // NOTE: this is implicitly added by llvm_add_poly_rtscc().
    pt1 = reinterpret_cast<decltype(pt1)>(state.jit_lookup("poly_translate_1"));
    aabb_cs_cfunc = reinterpret_cast<decltype(aabb_cs_cfunc)>(state.jit_lookup("aabb_cs"));
    cs_enc_func = reinterpret_cast<decltype(cs_enc_func)>(state.jit_lookup("cs_enc"));
    dist2_interp_func = reinterpret_cast<decltype(dist2_interp_func)>(state.jit_lookup("dist2_interp"));
}

conj_jit_data::~conj_jit_data() = default;

namespace
{

// Mutex for safe access to the global JIT data
// for conjunction detection.
constinit std::mutex conj_jit_data_map_mutex;

} // namespace

const conj_jit_data &get_conj_jit_data(std::uint32_t order)
{
    static std::unordered_map<std::uint32_t, conj_jit_data> conj_jit_data_map;

    std::lock_guard lock{conj_jit_data_map_mutex};

    const auto [it, new_insertion] = conj_jit_data_map.try_emplace(order, order);

    return it->second;
}

} // namespace mizuba::detail
