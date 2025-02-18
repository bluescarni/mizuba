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

#include <cstdlib>
#include <iostream>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "expose_conjunctions.hpp"
#include "expose_make_sgp4_polyjectory.hpp"
#include "expose_polyjectory.hpp"
#include "logging.hpp"

namespace mizuba_py::detail
{

namespace
{

// Wrapper to invoke the cleanup functions at shutdown.
extern "C" void cpp_atexit_wrapper()
{
#if !defined(NDEBUG)
    std::cout << "Running the C++ cleanup function" << std::endl;
#endif

    cleanup_pj_weak_ptrs();
    cleanup_cj_weak_ptrs();
}

} // namespace

} // namespace mizuba_py::detail

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    namespace mz = mizuba;
    namespace mzpy = mizuba_py;

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

    // Register the polyjectory/conjunctions cleanup machinery on the Python side.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Running the Python cleanup function" << std::endl;
#endif

        mzpy::cleanup_pj_weak_ptrs();
        mzpy::cleanup_cj_weak_ptrs();
    }));

    // Register the cleanup machinery also on the C++ side.
    //
    // NOTE: we do this also on the C++ side because we have run into some situations
    // in which, after a Ctrl+C signal, the Python interpreter would not invoke
    // the functions registered with atexit, leaving behind the temporary data on disk.
    //
    // NOTE: because we declared the structures used for cleanup as constinit, they are
    // constructed at compile time and they should be guaranteed to be destroyed *after*
    // the execution of cpp_atexit_wrapper, whose registration happens at runtime.
    //
    // NOTE: perhaps consider keeping only the C++-side cleanup in the future? Also,
    // if this is ever extracted as a separate C++ library, we probably want to keep
    // this cleanup mechanism for those cases in which the destructors of local
    // variables will not be called (e.g., if the user calls std::exit()).
    std::atexit(mzpy::detail::cpp_atexit_wrapper);
}
