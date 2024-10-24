// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
