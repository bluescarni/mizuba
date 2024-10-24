// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cstdint>
#include <string>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "logging.hpp"

namespace mizuba

{

namespace detail
{

namespace
{

spdlog::logger *get_logger()
{
    static auto ret = spdlog::stdout_color_mt("mizuba");

    return ret.get();
}

} // namespace

void log_info_impl(const std::string &msg)
{
    get_logger()->info(msg);
}

void log_trace_impl(const std::string &msg)
{
    get_logger()->trace(msg);
}

} // namespace detail

void set_logger_level_info()
{
    detail::get_logger()->set_level(spdlog::level::info);
}

void set_logger_level_trace()
{
    detail::get_logger()->set_level(spdlog::level::trace);
}

stopwatch::stopwatch() : m_start_tp{clock::now()} {}

std::chrono::duration<double> stopwatch::elapsed() const
{
    return std::chrono::duration<double>(clock::now() - m_start_tp);
}

std::chrono::duration<std::int64_t, std::nano> stopwatch::elapsed_ns() const
{
    return std::chrono::duration_cast<std::chrono::duration<std::int64_t, std::nano>>(clock::now() - m_start_tp);
}

void stopwatch::reset()
{
    m_start_tp = clock::now();
}

} // namespace mizuba
