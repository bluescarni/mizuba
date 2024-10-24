// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/mdspan.hpp>

#include "conjunctions_jit.hpp"

namespace mizuba::detail
{

namespace
{

// Add a compiled function for the computation of the translation of a polynomial.
// That is, given a polynomial represented as a list of coefficients c_i, the function
// will compute the coefficients c'_i of the polynomial resulting from substituting
// the polynomial variable x with x + a, where a = par[0] is a numerical constant.
// The formula for the translated coefficients is:
//
// c'_i = sum_{k=i}^n (c_k * choose(k, k-i) * a**(k-i))
//
// (where n == order of the polynomial).
void add_poly_translator_a(heyoka::llvm_state &s, std::uint32_t order)
{
    namespace hy = heyoka;
    using namespace hy::literals;

    // NOTE: this is guaranteed by the checks in polyjectory.
    assert(order >= 2u); // LCOV_EXCL_LINE

    // The translation amount 'a' is implemented as the
    // first and only parameter of the compiled function.
    auto a = hy::par[0];

    // Pre-compute the powers of a up to 'order'.
    std::vector a_pows = {1_dbl, a};
    for (std::uint32_t i = 2; i <= order; ++i) {
        // NOTE: do it like this, rather than the more
        // straightforward way of multiplying repeatedly
        // by a, in order to improve instruction-level
        // parallelism.
        if (i % 2u == 0u) {
            a_pows.push_back(a_pows[i / 2u] * a_pows[i / 2u]);
        } else {
            a_pows.push_back(a_pows[i / 2u + 1u] * a_pows[i / 2u]);
        }
    }

    // The original polynomial coefficients are the
    // input variables for the compiled function.
    std::vector<hy::expression> cfs;
    for (std::uint32_t i = 0; i <= order; ++i) {
        cfs.emplace_back(fmt::format("c_{}", i));
    }

    // The new coefficients are the function outputs.
    std::vector<hy::expression> out, tmp;
    for (std::uint32_t i = 0; i <= order; ++i) {
        tmp.clear();

        for (std::uint32_t k = i; k <= order; ++k) {
            tmp.push_back(cfs[k]
                          * boost::math::binomial_coefficient<double>(boost::numeric_cast<unsigned>(k),
                                                                      boost::numeric_cast<unsigned>(k - i))
                          * a_pows[k - i]);
        }

        out.push_back(hy::sum(std::move(tmp)));
    }

    // Add the compiled function.
    hy::add_cfunc<double>(s, "pta_cfunc", out, cfs);
}

// Add a compiled function for the computation of the sum of the squares
// of the differences between three polynomials. The computation
// is performed in the truncated power series algebra.
void add_poly_ssdiff3_cfunc(heyoka::llvm_state &s, std::uint32_t order)
{
    namespace stdex = std::experimental;
    namespace hy = heyoka;
    using namespace hy::literals;

    // NOTE: this is guaranteed by the checks in polyjectory.
    assert(order >= 2u); // LCOV_EXCL_LINE

    // The coefficients of are given in row-major
    // format for the polynomials of xi,yi,zi,xj,yj,zj.
    std::vector<hy::expression> vars;
    for (std::uint32_t i = 0; i <= order; ++i) {
        vars.emplace_back(fmt::format("xi_{}", i));
    }
    for (std::uint32_t i = 0; i <= order; ++i) {
        vars.emplace_back(fmt::format("yi_{}", i));
    }
    for (std::uint32_t i = 0; i <= order; ++i) {
        vars.emplace_back(fmt::format("zi_{}", i));
    }
    for (std::uint32_t i = 0; i <= order; ++i) {
        vars.emplace_back(fmt::format("xj_{}", i));
    }
    for (std::uint32_t i = 0; i <= order; ++i) {
        vars.emplace_back(fmt::format("yj_{}", i));
    }
    for (std::uint32_t i = 0; i <= order; ++i) {
        vars.emplace_back(fmt::format("zj_{}", i));
    }

    // Access the polynomials as submdspans into vars.
    stdex::mdspan var_arr(std::as_const(vars).data(), static_cast<decltype(vars.size())>(6), order + 1u);
    auto xi_poly = stdex::submdspan(var_arr, 0u, stdex::full_extent);
    auto yi_poly = stdex::submdspan(var_arr, 1u, stdex::full_extent);
    auto zi_poly = stdex::submdspan(var_arr, 2u, stdex::full_extent);
    auto xj_poly = stdex::submdspan(var_arr, 3u, stdex::full_extent);
    auto yj_poly = stdex::submdspan(var_arr, 4u, stdex::full_extent);
    auto zj_poly = stdex::submdspan(var_arr, 5u, stdex::full_extent);

    // Helper to compute the difference between two polynomials.
    auto pdiff = [order](const auto &p1, const auto &p2) {
        std::vector<hy::expression> ret;
        ret.reserve(order + 1u);

        for (std::uint32_t i = 0; i <= order; ++i) {
            ret.push_back(p1[i] - p2[i]);
        }

        return ret;
    }; // LCOV_EXCL_LINE

    // Compute the differences.
    auto diff_x = pdiff(xi_poly, xj_poly);
    auto diff_y = pdiff(yi_poly, yj_poly);
    auto diff_z = pdiff(zi_poly, zj_poly);

    // Helper to compute the square of a polynomial.
    auto psquare = [order](const auto &p) {
        std::vector<hy::expression> ret, tmp;
        ret.reserve(order + 1u);

        for (std::uint32_t i = 0; i <= order; ++i) {
            tmp.clear();

            if (i % 2u == 0u) {
                if (i == 0u) {
                    ret.push_back(p[0] * p[0]);
                } else {
                    for (std::uint32_t j = 0; j <= i / 2u - 1u; ++j) {
                        tmp.push_back(p[i - j] * p[j]);
                    }

                    ret.push_back(2_dbl * hy::sum(std::move(tmp)) + p[i / 2u] * p[i / 2u]);
                }
            } else {
                for (std::uint32_t j = 0; j <= (i - 1u) / 2u; ++j) {
                    tmp.push_back(p[i - j] * p[j]);
                }

                ret.push_back(2_dbl * hy::sum(std::move(tmp)));
            }
        }

        return ret;
    };

    // Compute the squares.
    auto diff2_x = psquare(diff_x);
    auto diff2_y = psquare(diff_y);
    auto diff2_z = psquare(diff_z);

    // Build the outputs vector as the sum of the squares
    std::vector<hy::expression> out;
    out.reserve(order + 1u);
    for (std::uint32_t i = 0; i <= order; ++i) {
        out.push_back(hy::sum({diff2_x[i], diff2_y[i], diff2_z[i]}));
    }

    // Add the compiled function.
    hy::add_cfunc<double>(s, "ssdiff3_cfunc", out, vars);
}

} // namespace

// NOTE: consider experimenting with, e.g., slp vectorization here.
conj_jit_data::conj_jit_data(std::uint32_t order)
{
    namespace hy = heyoka;

    // Add the compiled functions.
    auto *fp_t = hy::detail::to_internal_llvm_type<double>(state);
    detail::add_poly_translator_a(state, order);
    detail::add_poly_ssdiff3_cfunc(state, order);
    hy::detail::llvm_add_fex_check(state, fp_t, order, 1);
    hy::detail::llvm_add_poly_rtscc(state, fp_t, order, 1);

    // Compile.
    state.compile();

    // Lookup.
    pta_cfunc = reinterpret_cast<decltype(pta_cfunc)>(state.jit_lookup("pta_cfunc"));
    pssdiff3_cfunc = reinterpret_cast<decltype(pssdiff3_cfunc)>(state.jit_lookup("ssdiff3_cfunc"));
    fex_check = reinterpret_cast<decltype(fex_check)>(state.jit_lookup("fex_check"));
    rtscc = reinterpret_cast<decltype(rtscc)>(state.jit_lookup("poly_rtscc"));
    // NOTE: this is implicitly added by llvm_add_poly_rtscc().
    pt1 = reinterpret_cast<decltype(pt1)>(state.jit_lookup("poly_translate_1"));
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
