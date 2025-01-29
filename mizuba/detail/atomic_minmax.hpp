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

#ifndef MIZUBA_DETAIL_ATOMIC_MINMAX_HPP
#define MIZUBA_DETAIL_ATOMIC_MINMAX_HPP

#include <algorithm>
#include <atomic>

namespace mizuba::detail
{

// Helper to atomically set out to std::min(out, val).
// NOTE: this assumes no NaNs are involved in the comparison.
template <typename T>
void atomic_min(std::atomic<T> &out, T val)
{
    // Load the current value from the atomic.
    auto orig_val = out.load();
    T new_val;

    do {
        // Compute the new value.
        new_val = std::min(val, orig_val);
    } while (!out.compare_exchange_weak(orig_val, new_val));
}

// Helper to atomically set out to std::max(out, val).
// NOTE: this assumes no NaNs are involved in the comparison.
template <typename T>
void atomic_max(std::atomic<T> &out, T val)
{
    // Load the current value from the atomic.
    auto orig_val = out.load();
    T new_val;

    do {
        // Compute the new value.
        new_val = std::max(val, orig_val);
    } while (!out.compare_exchange_weak(orig_val, new_val));
}

} // namespace mizuba::detail

#endif
