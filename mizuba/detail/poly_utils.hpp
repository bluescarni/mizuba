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

#ifndef MIZUBA_DETAIL_POLY_UTILS_HPP
#define MIZUBA_DETAIL_POLY_UTILS_HPP

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/expression.hpp>

#include "conjunctions_jit.hpp"

namespace mizuba::detail
{

// Horner evaluation of the univariate polynomial of order 'order'
// with coefficients beginning at iterator 'it'. The
// evaluation value is 'x'. T is the type used to compute
// the evaluation. The polynomial coefficients must be
// convertible to T. 'it' must be a random-access iterator.
template <typename It, typename T>
T horner_eval(It it, std::uint32_t order, const T &x)
{
    auto res = static_cast<T>(it[order]);
    for (std::uint32_t o = 1; o <= order; ++o) {
        res = static_cast<T>(it[order - o]) + res * x;
    }

    return res;
}

// Polynomial cache type. Each entry is a polynomial
// represented as a vector of coefficients, in ascending
// order.
//
// NOTE: a cache must not contain empty coefficient
// vectors and all polynomials in the cache must have
// the same order.
using poly_cache = std::vector<std::vector<double>>;

// A RAII helper to extract polys from a cache and
// return them to the cache upon destruction.
struct pwrap {
    // The cache backing this pwrap.
    poly_cache &pc;

    // A polynomial extracted from pc.
    std::vector<double> v;

    void back_to_cache();
    std::vector<double> get_poly_from_cache(std::uint32_t);

    explicit pwrap(poly_cache &, std::uint32_t);
    pwrap(pwrap &&) noexcept;
    pwrap &operator=(pwrap &&) noexcept;

    // Delete copy semantics.
    pwrap(const pwrap &) = delete;
    pwrap &operator=(const pwrap &) = delete;

    ~pwrap();
};

// The type used to store the list of isolating intervals.
using isol_t = std::vector<std::tuple<double, double>>;

// The working list type used during real root isolation.
using wlist_t = std::vector<std::tuple<double, double, pwrap>>;

bool run_poly_root_finding(const double *, std::uint32_t, double, isol_t &, wlist_t &, conj_jit_data::fex_check_t,
                           conj_jit_data::rtscc_t, conj_jit_data::pt1_t, std::uint32_t, std::uint32_t, int,
                           std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> &, poly_cache &);

// Create the heyoka expression for Vandermonde polynomial interpolation. See:
//
// https://www.ams.org/journals/mcom/1970-24-112/S0025-5718-1970-0290541-1/S0025-5718-1970-0290541-1.pdf
//
// The only input argument is the polynomial interpolation order.
std::pair<std::vector<heyoka::expression>, std::vector<heyoka::expression>> vm_interp(std::uint32_t);

} // namespace mizuba::detail

#endif
