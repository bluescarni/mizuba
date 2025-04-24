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

#ifndef MIZUBA_DETAIL_FILE_UTILS_HPP
#define MIZUBA_DETAIL_FILE_UTILS_HPP

#include <cstddef>
#include <optional>

#if defined(_WIN32)

#include <windows.h>

#endif

#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace mizuba::detail
{

boost::filesystem::path create_temp_dir(const char *, std::optional<boost::filesystem::path> = {});

boost::filesystem::path create_dir_0700(const boost::filesystem::path &);

void create_sized_file(const boost::filesystem::path &, std::size_t);

void mark_file_read_only(const boost::filesystem::path &);

// Wrapper to perform a pwrite()-like operation on an existing file.
class file_pwrite
{
    boost::filesystem::path m_path;

#if defined(_WIN32)
    HANDLE
#else
    int
#endif
    m_fd;

    [[nodiscard]] bool is_closed() const noexcept;

public:
    explicit file_pwrite(boost::filesystem::path);
    file_pwrite(const file_pwrite &) = delete;
    file_pwrite(file_pwrite &&) noexcept = delete;
    file_pwrite &operator=(const file_pwrite &) = delete;
    file_pwrite &operator=(file_pwrite &&) noexcept = delete;
    ~file_pwrite();

    void close() noexcept;

    void pwrite(const void *, std::size_t, std::size_t);
};

void advise_dontneed(boost::iostreams::mapped_file_source &, const boost::filesystem::path &);

} // namespace mizuba::detail

#endif
