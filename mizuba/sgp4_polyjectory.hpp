// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_SGP4_POLYJECTORY_HPP
#define MIZUBA_SGP4_POLYJECTORY_HPP

#include <cstddef>
#include <span>

#include <heyoka/mdspan.hpp>

#include "polyjectory.hpp"

namespace mizuba
{

// Default exit radius.
// This roughly corresponds to the semi-major axis at which
// one should start using the deep space part of the SGP4 algorithm.
inline constexpr double sgp4_exit_radius = 12000;

// Default reentry radius: 150km of altitude over the mean
// Earth radius.
inline constexpr double sgp4_reentry_radius = 6371 + 150.;

polyjectory sgp4_polyjectory(heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>>, double,
                             double, double = sgp4_exit_radius, double = sgp4_reentry_radius);

} // namespace mizuba

#endif
