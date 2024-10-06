// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include "poly_utils.hpp"

namespace mizuba::detail
{

void pwrap::back_to_cache()
{
    // NOTE: v will be empty when this has been
    // moved-from. In that case, we do not want
    // to send v back to the cache.
    if (!v.empty()) {
        assert(pc.empty() || pc[0].size() == v.size());

        // Move v into the cache.
        pc.push_back(std::move(v));
    }
}

std::vector<double> pwrap::get_poly_from_cache(std::uint32_t n)
{
    if (pc.empty()) {
        // No polynomials are available, create a new one.
        return std::vector<double>(boost::numeric_cast<std::vector<double>::size_type>(n + 1u));
    } else {
        // Extract an existing polynomial from the cache.
        auto retval = std::move(pc.back());
        pc.pop_back();

        assert(retval.size() == n + 1u);

        return retval;
    }
}

pwrap::pwrap(poly_cache &cache, std::uint32_t n) : pc(cache), v(get_poly_from_cache(n)) {}

pwrap::pwrap(pwrap &&other) noexcept : pc(other.pc), v(std::move(other.v))
{
    // Make sure we moved from a valid pwrap.
    assert(!v.empty()); // LCOV_EXCL_LINE

    // NOTE: we must ensure that other.v is cleared out, because
    // otherwise, when other is destructed, we could end up
    // returning to the cache a polynomial of the wrong size.
    //
    // In basically every existing implementation, moving a std::vector
    // will leave the original object empty, but technically this
    // does not seem to be guaranteed by the standard, so, better
    // safe than sorry. Quick checks on godbolt indicate that compilers
    // are anyway able to elide this clearing out of the vector.
    other.v.clear();
}

// NOTE: this does not support self-move, and requires that
// the cache of other is the same as the cache of this.
pwrap &pwrap::operator=(pwrap &&other) noexcept
{
    // Disallow self move.
    assert(this != &other); // LCOV_EXCL_LINE

    // Make sure the polyomial caches match.
    assert(&pc == &other.pc); // LCOV_EXCL_LINE

    // Make sure we are not moving from a
    // moved-from pwrap.
    assert(!other.v.empty()); // LCOV_EXCL_LINE

    // Put the current v in the cache.
    back_to_cache();

    // Do the move-assignment.
    v = std::move(other.v);

    // NOTE: we must ensure that other.v is cleared out, because
    // otherwise, when other is destructed, we could end up
    // returning to the cache a polynomial of the wrong size.
    //
    // In basically every existing implementation, moving a std::vector
    // will leave the original object empty, but technically this
    // does not seem to be guaranteed by the standard, so, better
    // safe than sorry. Quick checks on godbolt indicate that compilers
    // are anyway able to elide this clearing out of the vector.
    other.v.clear();

    return *this;
}

pwrap::~pwrap()
{
    // Put the current v in the cache.
    back_to_cache();
}

} // namespace mizuba::detail
