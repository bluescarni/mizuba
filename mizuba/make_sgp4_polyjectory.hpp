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

#ifndef MIZUBA_MAKE_SGP4_POLYJECTORY_HPP
#define MIZUBA_MAKE_SGP4_POLYJECTORY_HPP

#include <cstddef>
#include <cstdint>
#include <span>

#include <heyoka/mdspan.hpp>

#include "polyjectory.hpp"

namespace mizuba
{

// Struct for storing the gpe data necessary for
// propagation via sgp4.
struct gpe {
    std::uint64_t norad_id;
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

// TODO remember to check for contiguous, aligned array when exposing this in Python.
polyjectory make_sgp4_polyjectory(heyoka::mdspan<const gpe, heyoka::extents<std::size_t, std::dynamic_extent>>, double,
                                  double);

} // namespace mizuba

#endif
