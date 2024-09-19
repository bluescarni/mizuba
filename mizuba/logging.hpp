// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_LOGGING_HPP
#define MIZUBA_LOGGING_HPP

#if !defined(NDEBUG)

// NOTE: this means that in release builds all SPDLOG_LOGGER_DEBUG() calls
// will be elided (so that they won't show up in the log even if
// the log level is set to spdlog::level::debug).
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#endif

#include <chrono> // NOTE: needed for the spdlog stopwatch.

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

namespace mizuba
{

namespace detail
{

spdlog::logger *get_logger();

} // namespace detail

void set_logger_level_trace();
void set_logger_level_debug();
void set_logger_level_info();
void set_logger_level_warn();
void set_logger_level_err();
void set_logger_level_critical();

} // namespace mizuba

#endif
