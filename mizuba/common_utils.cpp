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

#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "common_utils.hpp"

namespace mizuba_py
{

void check_array_cc_aligned(const py::array &arr, const char *msg)
{
    if (!py::cast<bool>(arr.attr("flags").attr("aligned")) || !py::cast<bool>(arr.attr("flags").attr("c_contiguous")))
        [[unlikely]] {
        throw std::invalid_argument(msg);
    }
}

bool may_share_memory(const py::array &a, const py::array &b)
{
    return py::module_::import("numpy").attr("may_share_memory")(a, b).cast<bool>();
}

} // namespace mizuba_py
