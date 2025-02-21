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

#include <filesystem>
#include <iostream>
#include <optional>
#include <utility>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "expose_conjunctions.hpp"
#include "expose_make_sgp4_polyjectory.hpp"
#include "expose_polyjectory.hpp"
#include "logging.hpp"
#include "tmpdir.hpp"

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    namespace mz = mizuba;
    namespace mzpy = mizuba_py;
    using namespace py::literals;

    // Disable automatic function signatures in the docs.
    // NOTE: the 'options' object needs to stay alive
    // throughout the whole definition of the module.
    py::options options;
    options.disable_function_signatures();

    m.doc() = "The core mizuba module";

    // Expose the polyjectory class.
    mzpy::expose_polyjectory(m);

    // Expose the conjunctions class.
    mzpy::expose_conjunctions(m);

    // Expose make_sgp4_polyjectory().
    mzpy::expose_make_sgp4_polyjectory(m);

    // Logging utils.
    m.def("set_logger_level_info", &mz::set_logger_level_info);
    m.def("set_logger_level_trace", &mz::set_logger_level_trace);
    m.def("set_logger_level_debug", &mz::set_logger_level_debug);
    m.def("set_logger_level_warning", &mz::set_logger_level_warning);

    // tmpdir getter/setter.
    // NOTE: we cannot directly wrap the C++ functions because Python has no
    // notion of an empty path. Thus, the getter returns None if tmpdir is an
    // empty path, and it sets tmpdir to an empty path if None is passed in input.
    m.def("get_tmpdir", []() -> std::optional<std::filesystem::path> {
        auto ret = mz::get_tmpdir();
        if (ret.empty()) {
            return {};
        } else {
            return ret;
        }
    });
    // NOTE: passing an empty string to this function is equivalent to passing None.
    m.def(
        "set_tmpdir",
        [](std::optional<std::filesystem::path> path) {
            if (path) {
                mz::set_tmpdir(std::move(*path));
            } else {
                mz::set_tmpdir({});
            }
        },
        "path"_a);

    // Register the cleanup machinery.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Running the Python cleanup function" << std::endl;
#endif

        mzpy::cleanup_pj_weak_ptrs();
        mzpy::cleanup_cj_weak_ptrs();
    }));
}
