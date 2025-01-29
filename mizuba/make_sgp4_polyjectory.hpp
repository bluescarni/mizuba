// Copyright 2024-2025 Francesco Biscani
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

#ifndef MIZUBA_MAKE_SGP4_POLYJECTORY_HPP
#define MIZUBA_MAKE_SGP4_POLYJECTORY_HPP

#include <cstddef>
#include <cstdint>
#include <span>

#include <heyoka/mdspan.hpp>

#include "polyjectory.hpp"

namespace mizuba
{

// Struct representing a GPE.
struct gpe {
    std::uint64_t norad_id;
    // NOTE: these must be UTC Julian dates.
    double epoch_jd1;
    double epoch_jd2;
    double n0;
    double e0;
    double i0;
    double node0;
    double omega0;
    double m0;
    double bstar;
};

// Construct a polyjectory using the SGP4 propagator.
//
// The set of gpes is passed in as a span, the time interval for the construction of the polyjectory is given by the
// second and third arguments (begin/end as UTC Julian dates). The last two arguments are the reentry and exit
// radiuses.
polyjectory make_sgp4_polyjectory(heyoka::mdspan<const gpe, heyoka::extents<std::size_t, std::dynamic_extent>>, double,
                                  double, double, double);

} // namespace mizuba

#endif
