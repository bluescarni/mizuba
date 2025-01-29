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

#ifndef MIZUBA_PY_EXPOSE_POLYJECTORY_HPP
#define MIZUBA_PY_EXPOSE_POLYJECTORY_HPP

#include <memory>

#include <pybind11/pybind11.h>

#include "polyjectory.hpp"

namespace mizuba_py
{

void expose_polyjectory(pybind11::module_ &);

void add_pj_weak_ptr(const std::shared_ptr<mizuba::detail::polyjectory_impl> &);
void cleanup_pj_weak_ptrs();

} // namespace mizuba_py

#endif
