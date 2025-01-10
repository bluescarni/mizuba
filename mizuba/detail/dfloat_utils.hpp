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

#ifndef MIZUBA_DETAIL_DFLOAT_UTILS_HPP
#define MIZUBA_DETAIL_DFLOAT_UTILS_HPP

#include <cmath>
#include <stdexcept>

#include <fmt/core.h>

#include <heyoka/detail/dfloat.hpp>

namespace mizuba::detail
{

// Helper to normalise a two-parts floating-point value into a proper
// double-length float (regardless of the magnitudes of the two parts).
// If a non-finite value is detected, an error will be thrown.
inline auto hilo_to_dfloat(double hi, double lo)
{
    // Normalise the two components. By using Knuth's EFT we ensure that the result is correct
    // regardless of the magnitudes of the two components.
    const auto [hi_norm, lo_norm] = heyoka::detail::eft_add_knuth(hi, lo);

    // Check the result.
    if (!std::isfinite(hi_norm) || !std::isfinite(lo_norm)) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "The normalisation of the double-length number with components ({}, {}) produced a non-finite result", hi,
            lo));
    }

    // Return the double-length float.
    return heyoka::detail::dfloat<double>{hi_norm, lo_norm};
}

} // namespace mizuba::detail

#endif
