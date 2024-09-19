// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// NOTE: this needs to go first because of the
// SPDLOG_ACTIVE_LEVEL definition.
#include "logging.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

namespace mizuba
{

namespace detail
{

namespace
{

auto make_logger()
{
    auto ret = spdlog::stdout_color_mt("mizuba");
#if !defined(NDEBUG)
    ret->info("mizuba logger initialised");
#endif

    return ret;
}

} // namespace

spdlog::logger *get_logger()
{
    static auto ret = make_logger();

    return ret.get();
}

} // namespace detail

void set_logger_level_trace()
{
    detail::get_logger()->set_level(spdlog::level::trace);
}

void set_logger_level_debug()
{
    detail::get_logger()->set_level(spdlog::level::debug);
}

void set_logger_level_info()
{
    detail::get_logger()->set_level(spdlog::level::info);
}

void set_logger_level_warn()
{
    detail::get_logger()->set_level(spdlog::level::warn);
}

void set_logger_level_err()
{
    detail::get_logger()->set_level(spdlog::level::err);
}

void set_logger_level_critical()
{
    detail::get_logger()->set_level(spdlog::level::critical);
}

} // namespace mizuba
