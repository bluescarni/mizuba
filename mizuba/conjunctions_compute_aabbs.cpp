// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>

#include <boost/filesystem/path.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include "conjunctions.hpp"
#include "polyjectory.hpp"

namespace mizuba
{

#if 0

void conjunctions::compute_aabbs(const polyjectory &pj, const boost::filesystem::path &tmp_dir_path,
                                 std::size_t n_cd_steps) const
{
    using safe_size_t = boost::safe_numerics::safe<std::size_t>;

    // The total number of aabbs we need to compute and store.
    const auto n_tot_aabbs = safe_size_t(pj.get_nobjs()) * n_cd_steps;

    // The total required size in bytes.
    const auto tot_size = static_cast<std::size_t>(n_tot_aabbs * sizeof(float) * 4u);
}

#endif

} // namespace mizuba
