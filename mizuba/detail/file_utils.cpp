// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>

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
    {
        // NOTE: here we just create the file and close it immediately, so that it will
        // have a size of zero. Then, we will resize it to the necessary size.
        std::ofstream file(path.string(), std::ios::binary | std::ios::out);
        // Make sure we throw on errors.
        file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    }

    // Resize it.
    boost::filesystem::resize_file(path, boost::numeric_cast<boost::uintmax_t>(size));
}

// Mark the file at the input path as read-only.
void mark_file_read_only(const boost::filesystem::path &path)
{
    assert(boost::filesystem::is_regular_file(path));

    boost::filesystem::permissions(path, boost::filesystem::perms::owner_read);
}

namespace
{

template <typename FileType>
auto mmap_at_offset_impl(const boost::filesystem::path &path, std::size_t size, std::size_t offset)
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // Run a few sanity checks.
    assert(size > 0u);
    assert(boost::filesystem::exists(path));
    assert(boost::filesystem::is_regular_file(path));
    assert(safe_size_t(offset) + size <= boost::filesystem::file_size(path));

    // Fetch the page alignment value.
    const auto palign = safe_size_t(FileType::alignment());

    // offset in general will not be a multiple of the page alignment value.
    // We need to decrease it until we get to a multiple of the page alignment value.
    const auto rem = offset % palign;
    const auto new_offset = offset - rem;

    // Correspondingly, we need to increase the mapping size to account for the
    // decrease in offset.
    const auto new_size = size + rem;

    // We are now ready to memory-map.
    FileType file(path.string(), new_size, new_offset);

    // Fetch the pointer to the beginning of the mapped region.
    auto *ptr = file.data();

    // Increase it by rem to get to the desired offset.
    ptr += static_cast<std::size_t>(rem);

    return std::make_pair(ptr, std::move(file));
}

} // namespace

std::pair<const char *, boost::iostreams::mapped_file_source> mmap_at_offset_ro(const boost::filesystem::path &path,
                                                                                std::size_t size, std::size_t offset)
{
    return mmap_at_offset_impl<boost::iostreams::mapped_file_source>(path, size, offset);
}

std::pair<char *, boost::iostreams::mapped_file_sink> mmap_at_offset_rw(const boost::filesystem::path &path,
                                                                        std::size_t size, std::size_t offset)
{
    return mmap_at_offset_impl<boost::iostreams::mapped_file_sink>(path, size, offset);
}

} // namespace mizuba::detail
