// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>

#include <boost/cstdint.hpp>
#include <boost/filesystem/file_status.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/numeric/conversion/cast.hpp>

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

// Create a file at the input path with the given size.
// The file will **not** be opened.
// If the file exists already, an error will be thrown.
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

} // namespace mizuba::detail
