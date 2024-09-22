// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <fmt/core.h>

#include "file_utils.hpp"

namespace mizuba::detail
{

// Helper to create a directory with a unique name into the
// system's temporary dir. If the directory to be created exists
// already, an exception will be thrown.
boost::filesystem::path create_temp_dir(const char *tplt)
{
    // Assemble a "unique" dir path into the system temp dir.
    // NOTE: make sure to canonicalise the result, so that if we
    // convert it to string we get an absolute path with resolved symlinks.
    auto tmp_dir_path
        = boost::filesystem::canonical(boost::filesystem::temp_directory_path() / boost::filesystem::unique_path(tplt));

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

} // namespace mizuba::detail
