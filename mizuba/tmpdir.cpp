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
#include <filesystem>
#include <mutex>
#include <utility>

#include "tmpdir.hpp"

namespace mizuba
{

namespace detail
{

namespace
{

// LCOV_EXCL_START

// Helper to initialise tmpdir as either the content of the MIZUBA_TMPDIR
// env variable, or an empty path.
std::filesystem::path init_tmpdir()
{
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const auto *str = std::getenv("MIZUBA_TMPDIR");

    if (str == nullptr) {
        return {};
    } else {
        return str;
    }
}

// LCOV_EXCL_STOP

// The global tmpdir variable, protected by a mutex.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
constinit std::mutex tmpdir_mutex;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cert-err58-cpp)
std::filesystem::path tmpdir = init_tmpdir();

} // namespace

} // namespace detail

// Getter/setter for tmpdir, with mutex protection.
std::filesystem::path get_tmpdir()
{
    const std::lock_guard lock(detail::tmpdir_mutex);

    return detail::tmpdir;
}

void set_tmpdir(std::filesystem::path path)
{
    const std::lock_guard lock(detail::tmpdir_mutex);

    detail::tmpdir = std::move(path);
}

} // namespace mizuba
