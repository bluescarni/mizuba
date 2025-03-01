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

#include <chrono>
#include <cstdint>
#include <string>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "logging.hpp"

// LCOV_EXCL_START

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

void log_debug_impl(const std::string &msg)
{
    get_logger()->debug(msg);
}

void log_warning_impl(const std::string &msg)
{
    get_logger()->warn(msg);
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

void set_logger_level_debug()
{
    detail::get_logger()->set_level(spdlog::level::debug);
}

void set_logger_level_warning()
{
    detail::get_logger()->set_level(spdlog::level::warn);
}

stopwatch::stopwatch() : m_start_tp{clock::now()} {}

std::chrono::duration<double> stopwatch::elapsed() const
{
    return {clock::now() - m_start_tp};
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

// LCOV_EXCL_STOP
