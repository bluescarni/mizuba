// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_DETAIL_IVAL_HPP
#define MIZUBA_DETAIL_IVAL_HPP

#include <cmath>
#include <limits>

namespace mizuba::detail
{

// Minimal interval class supporting a couple
// of elementary operations.
// NOTE: like in heyoka, the implementation of interval arithmetic
// could be improved by accounting for floating-point truncation to yield results
// which are truly mathematically exact.
struct ival {
    double lower;
    double upper;

    ival() : ival(0) {}
    explicit ival(double val) : ival(val, val) {}
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    explicit ival(double l, double u) : lower(l), upper(u) {}

    // min/max implementations with nan propagation.
    static double min(double a, double b)
    {
        // LCOV_EXCL_START
        if (std::isnan(a) || std::isnan(b)) [[unlikely]] {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // LCOV_EXCL_STOP

        return (a < b) ? a : b;
    }
    static double max(double a, double b)
    {
        // LCOV_EXCL_START
        if (std::isnan(a) || std::isnan(b)) [[unlikely]] {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // LCOV_EXCL_STOP

        return (a > b) ? a : b;
    }
};

// NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
inline ival operator+(ival a, ival b)
{
    return ival(a.lower + b.lower, a.upper + b.upper);
}

inline ival operator+(double a, ival b)
{
    return ival(a) + b;
}

inline ival operator*(ival a, ival b)
{
    const auto tmp1 = a.lower * b.lower;
    const auto tmp2 = a.lower * b.upper;
    const auto tmp3 = a.upper * b.lower;
    const auto tmp4 = a.upper * b.upper;

    const auto l = ival::min(ival::min(tmp1, tmp2), ival::min(tmp3, tmp4));
    const auto u = ival::max(ival::max(tmp1, tmp2), ival::max(tmp3, tmp4));

    return ival(l, u);
}

inline ival operator*(double a, ival b)
{
    return ival(a) * b;
}

} // namespace mizuba::detail

#endif
