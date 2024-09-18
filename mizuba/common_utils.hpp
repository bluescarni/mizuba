// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_PY_COMMON_UTILS_HPP
#define MIZUBA_PY_COMMON_UTILS_HPP

#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace mizuba_py
{

namespace py = pybind11;

void py_throw(PyObject *, const char *);

py::object builtins();

py::object type(const py::handle &);

std::string str(const py::handle &);

std::vector<double> sat_list_to_vector(py::list);

void check_array_cc_aligned(const py::array &, const char *);

} // namespace mizuba_py

#endif
