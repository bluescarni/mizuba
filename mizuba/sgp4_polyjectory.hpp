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

polyjectory sgp4_polyjectory(heyoka::mdspan<const double, heyoka::extents<std::size_t, 9, std::dynamic_extent>>, double,
                             double);

} // namespace mizuba

#endif
