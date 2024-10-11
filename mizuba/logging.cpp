// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <string>

#include <pybind11/pybind11.h>

#include "logging.hpp"

namespace mizuba

{

namespace detail
{

void log_info_impl(const std::string &msg)
{
    namespace py = pybind11;

    // NOTE: need to acquire the GIL in case this is invoked
    // from an external thread (e.g., TBB).
    py::gil_scoped_acquire acquire;

    auto logger = py::module_::import("logging").attr("getLogger")("mizuba");
    logger.attr("info")(msg);
}

} // namespace detail

stopwatch::stopwatch() : m_start_tp{clock::now()} {}

std::chrono::duration<double> stopwatch::elapsed() const
{
    return std::chrono::duration<double>(clock::now() - m_start_tp);
}

void stopwatch::reset()
{
    m_start_tp = clock::now();
}

} // namespace mizuba
