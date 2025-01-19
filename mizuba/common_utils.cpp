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
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "common_utils.hpp"

namespace mizuba_py
{

std::vector<double> sat_list_to_vector(py::list sat_list)
{
    // Prepare the output vector.
    std::vector<double> retval;
    const auto n_sats = boost::safe_numerics::safe<decltype(retval.size())>(py::len(sat_list));
    retval.resize(n_sats * 9);

    // Fill it in.
    for (decltype(retval.size()) i = 0; i < n_sats; ++i) {
        auto sat_obj = sat_list[boost::numeric_cast<py::size_t>(i)];

        retval[i] = sat_obj.attr("no_kozai").template cast<double>();
        retval[i + n_sats] = sat_obj.attr("ecco").template cast<double>();
        retval[i + n_sats * 2] = sat_obj.attr("inclo").template cast<double>();
        retval[i + n_sats * 3] = sat_obj.attr("nodeo").template cast<double>();
        retval[i + n_sats * 4] = sat_obj.attr("argpo").template cast<double>();
        retval[i + n_sats * 5] = sat_obj.attr("mo").template cast<double>();
        retval[i + n_sats * 6] = sat_obj.attr("bstar").template cast<double>();
        retval[i + n_sats * 7] = sat_obj.attr("jdsatepoch").template cast<double>();
        retval[i + n_sats * 8] = sat_obj.attr("jdsatepochF").template cast<double>();
    }

    return retval;
}

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
