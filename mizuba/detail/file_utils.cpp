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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>

// Detect POSIX 2008.
#if __has_include(<unistd.h>)

#include <unistd.h>

#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200809L

#define MIZUBA_HAVE_POSIX_2008

#endif

#endif

// Detect the availability of Unix-style or Windows-style
// low-level IO API.
#if defined(_WIN32)

#include <windows.h>

#elif __has_include(<unistd.h>) && __has_include(<fcntl.h>)

#include <fcntl.h>
#include <unistd.h>

#else

#error Unsupported operating system

#endif

#include <boost/cstdint.hpp>
#include <boost/filesystem/file_status.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include "file_utils.hpp"

namespace mizuba::detail
{

// Helper to create a directory with a "unique" name into the
// system's temporary dir. If the directory to be created exists
// already, an exception will be thrown, otherwise the path
// to the newly-created directory will be returned.
boost::filesystem::path create_temp_dir(const char *tplt)
{
    // Assemble a "unique" dir path into the system's temp dir.
    // NOTE: make sure to canonicalise the temp dir path, so that if we
    // convert the result to string we get an absolute path with resolved symlinks.
    auto tmp_dir_path
        = boost::filesystem::canonical(boost::filesystem::temp_directory_path()) / boost::filesystem::unique_path(tplt);

    // Attempt to create it.
    // LCOV_EXCL_START
    if (!boost::filesystem::create_directory(tmp_dir_path)) [[unlikely]] {
        throw std::runtime_error(
            fmt::format("Error while creating a unique temporary directory: the directory '{}' already exists",
                        tmp_dir_path.string()));
    }
    // LCOV_EXCL_STOP

    return tmp_dir_path;
}

// Create a file at the input path with the given size. If the file exists already, an error will be thrown.
// NOTE: when this function returns, the file will **not** be open.
void create_sized_file(const boost::filesystem::path &path, std::size_t size)
{
    // LCOV_EXCL_START
    if (boost::filesystem::exists(path)) [[unlikely]] {
        throw std::runtime_error(fmt::format("Cannot create the sized file '{}', as it exists already", path.string()));
    }
    // LCOV_EXCL_STOP

#if defined(MIZUBA_HAVE_POSIX_2008)

    // Create the file.
    const auto fd = ::open(path.c_str(), O_RDWR | O_CREAT, S_IRWXU);
    if (fd == -1) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::runtime_error(fmt::format("Unable to create the sized file '{}'", path.string()));
        // LCOV_EXCL_STOP
    }

    // RAII file closer.
    const struct file_closer {
        int fd;
        ~file_closer()
        {
            ::close(fd);
        }
    } fc{.fd = fd};

    // Reserve space.
    if (posix_fallocate(fd, 0, boost::numeric_cast<::off_t>(size)) != 0) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::runtime_error(fmt::format("Unable to reserve space for the file '{}'", path.string()));
        // LCOV_EXCL_STOP
    }

#else

    {
        // NOTE: here we just create the file and close it immediately, so that it will
        // have a size of zero. Then, we will resize it to the necessary size.
        std::ofstream file(path.string(), std::ios::binary | std::ios::out);
        // Make sure we throw on errors.
        file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    }

    // Resize it.
    boost::filesystem::resize_file(path, boost::numeric_cast<boost::uintmax_t>(size));

#endif
}

// Mark the file at the input path as read-only.
void mark_file_read_only(const boost::filesystem::path &path)
{
    assert(boost::filesystem::is_regular_file(path));

    boost::filesystem::permissions(path, boost::filesystem::perms::owner_read);
}

file_pwrite::file_pwrite(boost::filesystem::path path) : m_path(std::move(path))
{
    // Debug checks.
    assert(boost::filesystem::exists(m_path));
    assert(boost::filesystem::is_regular_file(m_path));

#if defined(_WIN32)

    // Attempt to open the file.
    m_fd = CreateFileW(m_path.c_str(), GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                       nullptr);
    if (m_fd == INVALID_HANDLE_VALUE) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::runtime_error(fmt::format("Could not open the file '{}'", m_path.string()));
        // LCOV_EXCL_STOP
    }

#else

    // Attempt to open the file.
    m_fd = ::open(m_path.c_str(), O_WRONLY);
    if (m_fd == -1) [[unlikely]] {
        // LCOV_EXCL_START
        throw std::runtime_error(fmt::format("Could not open the file '{}'", m_path.string()));
        // LCOV_EXCL_STOP
    }

#endif
}

bool file_pwrite::is_closed() const noexcept
{
    return m_fd ==
#if defined(_WIN32)
           INVALID_HANDLE_VALUE
#else
           -1
#endif
        ;
}

file_pwrite::~file_pwrite()
{
    close();
}

void file_pwrite::close() noexcept
{
#if defined(_WIN32)

    if (m_fd != INVALID_HANDLE_VALUE) {
        if (CloseHandle(m_fd) == 0) [[unlikely]] {
            // LCOV_EXCL_START
            std::cerr << "An error was detected while trying to close the file '" << m_path.string() << "'"
                      << std::endl;
            // LCOV_EXCL_STOP
        }

        m_fd = INVALID_HANDLE_VALUE;
    }

#else

    if (m_fd != -1) {
        if (::close(m_fd) == -1) [[unlikely]] {
            // LCOV_EXCL_START
            std::cerr << "An error was detected while trying to close the file '" << m_path.string() << "'"
                      << std::endl;
            // LCOV_EXCL_STOP
        }

        m_fd = -1;
    }

#endif
}

void file_pwrite::pwrite(const void *buffer, std::size_t size, std::size_t offset)
{
    // Check that we are not writing into a closed file.
    assert(!is_closed());

    using safe_size_t = boost::safe_numerics::safe<std::size_t>;
    safe_size_t sz(size);

    // Check that we don't end up writing past the end of the file.
    assert(sz + offset <= boost::filesystem::file_size(m_path));

#if defined(_WIN32)

    // Initial setup the offset in the OVERLAPPED structure.
    auto oset = boost::safe_numerics::safe<std::uint64_t>(offset);
    OVERLAPPED overlapped{};
    overlapped.Offset = oset & 0xFFFFFFFF;
    overlapped.OffsetHigh = (oset >> 32) & 0xFFFFFFFF;

    // NOTE: the WriteFile() function uses DWORD (a 32-bit integer) to
    // represent the number of bytes to write. Thus, we may need to split
    // the write operation in multiple chunks.
    DWORD bytes_written = 0;

    do {
        const auto ret
            = WriteFile(m_fd, buffer, static_cast<DWORD>(std::min(sz, safe_size_t(std::numeric_limits<DWORD>::max()))),
                        &bytes_written, &overlapped);

        if (!ret) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::runtime_error(
                fmt::format("An error was detected while trying to pwrite() into the file '{}'", m_path.string()));
            // LCOV_EXCL_STOP
        }

        // Update size, offset and buffer pointer for
        // the next iteration, if needed.
        sz -= bytes_written;
        oset += bytes_written;
        buffer = static_cast<const void *>(reinterpret_cast<const char *>(buffer) + bytes_written);

        // Re-set the offset in overlapped.
        overlapped.Offset = oset & 0xFFFFFFFF;
        overlapped.OffsetHigh = (oset >> 32) & 0xFFFFFFFF;
    } while (sz != 0);

#else

    using safe_off_t = boost::safe_numerics::safe<::off_t>;
    safe_off_t oset = offset;

    // NOTE: pwrite() may produce partial writes, in which case we are supposed to try again.
    do {
        const auto written_sz = ::pwrite(m_fd, buffer, sz, oset);

        if (written_sz == -1) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::runtime_error(
                fmt::format("An error was detected while trying to pwrite() into the file '{}'", m_path.string()));
            // LCOV_EXCL_STOP
        }

        // Update size, offset and buffer pointer for
        // the next iteration, if needed.
        sz -= written_sz;
        oset += written_sz;
        buffer = static_cast<const void *>(reinterpret_cast<const char *>(buffer) + written_sz);
    } while (sz != 0);

#endif
}

} // namespace mizuba::detail
