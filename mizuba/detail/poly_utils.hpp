// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_DETAIL_POLY_UTILS_HPP
#define MIZUBA_DETAIL_POLY_UTILS_HPP

#include <vector>

#include <cstdint>

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

} // namespace mizuba::detail

#endif