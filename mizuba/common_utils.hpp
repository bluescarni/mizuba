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

#ifndef MIZUBA_PY_COMMON_UTILS_HPP
#define MIZUBA_PY_COMMON_UTILS_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <heyoka/mdspan.hpp>

namespace mizuba_py
{

namespace py = pybind11;

void check_array_cc_aligned(const py::array &, const char *);

// Helper to turn an mdspan into a py::array_t. The 'self' object will become
// the 'parent' of the returned array. If T is const, the array will be set
// as read-only.
template <typename T, typename IndexType, std::size_t... Extents>
auto mdspan_to_array(const py::object &self, heyoka::mdspan<T, std::experimental::extents<IndexType, Extents...>> span)
{
    using indices = std::make_index_sequence<sizeof...(Extents)>;

    auto ret = [&span, &self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return py::array_t<std::remove_const_t<T>>(
            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(span.extent(Idx))...}, span.data_handle(), self);
    }(indices{});

    if constexpr (std::is_const_v<T>) {
        ret.attr("flags").attr("writeable") = false;
    }

    return ret;
} // LCOV_EXCL_LINE

// Helper to check if a list of arrays may share any memory with each other.
// Quadratic complexity.
bool may_share_memory(const py::array &, const py::array &);

template <typename... Args>
bool may_share_memory(const py::array &a, const py::array &b, const Args &...args)
{
    const std::array args_arr = {std::cref(a), std::cref(b), std::cref(args)...};
    const auto nargs = args_arr.size();

    for (std::size_t i = 0; i < nargs; ++i) {
        for (std::size_t j = i + 1u; j < nargs; ++j) {
            if (may_share_memory(args_arr[i].get(), args_arr[j].get())) {
                return true;
            }
        }
    }

    return false;
}

} // namespace mizuba_py

#endif
