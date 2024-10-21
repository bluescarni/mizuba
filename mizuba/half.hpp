// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_HALF_HPP
#define MIZUBA_HALF_HPP

#include <bit>
#include <cmath>
#include <cstdint>

#if __has_include(<stdfloat>) && defined(__STDCPP_FLOAT16_T__)

#include <stdfloat>

#define MIZUBA_HAVE_STD_FLOAT16_T

#elif !defined(MIZUBA_HAVE__FLOAT16)

#include "detail/half.hpp"

#endif

namespace mizuba
{

using float16_t =
#if defined(MIZUBA_HAVE_STD_FLOAT16_T)
    std::float16_t
#elif defined(MIZUBA_HAVE__FLOAT16)
    _Float16
#else
    half_float::half
#endif
    ;

namespace detail
{

// Half-precision math functions used during conjunction detection.
[[nodiscard]] inline bool isfinite(float16_t x) noexcept
{
    return std::isfinite(static_cast<float>(x));
}

[[nodiscard]] inline bool isinf(float16_t x) noexcept
{
    return std::isinf(static_cast<float>(x));
}

[[nodiscard]] inline float16_t nextafter(float16_t from, float16_t to) noexcept
{
#if defined(MIZUBA_HAVE_STD_FLOAT16_T) || defined(MIZUBA_HAVE__FLOAT16)

    // NOTE: this is a mindless porting of the nextafter() function from the half library.

    // A couple of sanity checks.
    static_assert(sizeof(std::uint16_t) == 2u);
    static_assert(sizeof(float16_t) == 2u);

    // As far as I have understood, this returns a quiet NaN if either operand
    // is NaN.
    auto signal
        = [](unsigned x, unsigned y) -> unsigned { return ((x & 0x7FFF) > 0x7C00) ? (x | 0x200) : (y | 0x200); };

    // Convert from/to into 16-bit uints.
    const auto fdata = std::bit_cast<std::uint16_t>(from);
    const auto tdata = std::bit_cast<std::uint16_t>(to);

    const int fabs = fdata & 0x7FFF, tabs = tdata & 0x7FFF;

    if (fabs > 0x7C00 || tabs > 0x7C00) {
        return std::bit_cast<float16_t>(static_cast<std::uint16_t>(signal(fdata, tdata)));
    }

    if (fdata == tdata || !(fabs | tabs)) {
        return to;
    }

    if (!fabs) {
        return std::bit_cast<float16_t>(static_cast<std::uint16_t>((tdata & 0x8000) + 1));
    }

    const unsigned out = fdata
                         + (((fdata >> 15)
                             ^ static_cast<unsigned>((fdata ^ (0x8000 | (0x8000 - (fdata >> 15))))
                                                     < (tdata ^ (0x8000 | (0x8000 - (tdata >> 15))))))
                            << 1)
                         - 1;

    return std::bit_cast<float16_t>(static_cast<std::uint16_t>(out));

#else

    return half_float::nextafter(from, to);

#endif
}

} // namespace detail

} // namespace mizuba

#if defined(MIZUBA_HAVE_STD_FLOAT16_T)

#undef(MIZUBA_HAVE_STD_FLOAT16_T)

#endif

#endif
