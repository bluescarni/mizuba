// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_DETAIL_FILE_UTILS_HPP
#define MIZUBA_DETAIL_FILE_UTILS_HPP

#include <cstddef>
#include <utility>

#if defined(_WIN32)

#include <windows.h>

#endif

#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace mizuba::detail
{

boost::filesystem::path create_temp_dir(const char *);

void create_sized_file(const boost::filesystem::path &, std::size_t);

void mark_file_read_only(const boost::filesystem::path &);

std::pair<const char *, boost::iostreams::mapped_file_source> mmap_at_offset_ro(const boost::filesystem::path &,
                                                                                std::size_t, std::size_t);
std::pair<char *, boost::iostreams::mapped_file_sink> mmap_at_offset_rw(const boost::filesystem::path &, std::size_t,
                                                                        std::size_t);

void madvise_dontneed(const boost::iostreams::mapped_file_source &);

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

    template <typename T>
    void pwrite(const T *ptr, std::size_t size, std::size_t offset)
    {
        pwrite(static_cast<const void *>(ptr), size, offset);
    }
    void pwrite(const void *, std::size_t, std::size_t);
};

} // namespace mizuba::detail

#endif
