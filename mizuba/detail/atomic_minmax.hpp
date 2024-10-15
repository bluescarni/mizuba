// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
    auto orig_val = out.load(std::memory_order_relaxed);
    T new_val;

    do {
        // Compute the new value.
        new_val = std::min(val, orig_val);
    } while (!out.compare_exchange_weak(orig_val, new_val, std::memory_order_relaxed, std::memory_order_relaxed));
}

// Helper to atomically set out to std::max(out, val).
// NOTE: this assumes no NaNs are involved in the comparison.
template <typename T>
void atomic_max(std::atomic<T> &out, T val)
{
    // Load the current value from the atomic.
    auto orig_val = out.load(std::memory_order_relaxed);
    T new_val;

    do {
        // Compute the new value.
        new_val = std::max(val, orig_val);
    } while (!out.compare_exchange_weak(orig_val, new_val, std::memory_order_relaxed, std::memory_order_relaxed));
}

} // namespace mizuba::detail

#endif
