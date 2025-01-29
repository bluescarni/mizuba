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

#ifndef MIZUBA_LOGGING_HPP
#define MIZUBA_LOGGING_HPP

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#include <fmt/core.h>

namespace mizuba
{

namespace detail
{

void log_info_impl(const std::string &);
void log_trace_impl(const std::string &);

} // namespace detail

// Minimal stopwatch class inspired
// by spdlog.
class stopwatch
{
    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> m_start_tp;

public:
    stopwatch();

    // Elapsed time in seconds, represented as double-precision values.
    std::chrono::duration<double> elapsed() const;
    // Elapsed time in nanoseconds, represented as std::int64_t.
    std::chrono::duration<std::int64_t, std::nano> elapsed_ns() const;
    void reset();
};

// Set the logger level.
void set_logger_level_info();
void set_logger_level_trace();

// Log to info level.
template <typename... T>
void log_info(fmt::format_string<T...> fmt, T &&...args)
{
    detail::log_info_impl(fmt::format(fmt, std::forward<T>(args)...));
}

// Log to trace level.
template <typename... T>
void log_trace(fmt::format_string<T...> fmt, T &&...args)
{
    detail::log_trace_impl(fmt::format(fmt, std::forward<T>(args)...));
}

} // namespace mizuba

namespace fmt
{

// Formatter for the stopwatch class.
template <>
struct formatter<mizuba::stopwatch> : formatter<double> {
    template <typename FormatContext>
    auto format(const mizuba::stopwatch &sw, FormatContext &ctx) const -> decltype(ctx.out())
    {
        return formatter<double>::format(sw.elapsed().count(), ctx);
    }
};

} // namespace fmt

#endif
