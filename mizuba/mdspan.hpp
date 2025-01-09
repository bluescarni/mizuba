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

#ifndef MIZUBA_MDSPAN_HPP
#define MIZUBA_MDSPAN_HPP

#include <cstddef>

#include <heyoka/mdspan.hpp>

namespace mizuba
{

// Convenience typedefs for commonly-used spans. All spans
// defined here use std::size_t for sizes.

// 1D span with dynamic size.
template <typename T>
using dspan_1d = heyoka::mdspan<T, heyoka::dextents<std::size_t, 1>>;

// 2D span with dynamic sizes.
template <typename T>
using dspan_2d = heyoka::mdspan<T, heyoka::dextents<std::size_t, 2>>;

} // namespace mizuba

#endif
