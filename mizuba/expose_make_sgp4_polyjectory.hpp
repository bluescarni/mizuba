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

#ifndef MIZUBA_PY_EXPOSE_MAKE_SGP4_POLYJECTORY_HPP
#define MIZUBA_PY_EXPOSE_MAKE_SGP4_POLYJECTORY_HPP

#include <pybind11/pybind11.h>

namespace mizuba_py
{

void expose_make_sgp4_polyjectory(pybind11::module_ &);

} // namespace mizuba_py

#endif
