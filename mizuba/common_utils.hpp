// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_PY_COMMON_UTILS_HPP
#define MIZUBA_PY_COMMON_UTILS_HPP

#include <type_traits>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <heyoka/mdspan.hpp>

namespace mizuba_py
{

namespace py = pybind11;

std::vector<double> sat_list_to_vector(py::list);

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
}

} // namespace mizuba_py

#endif
