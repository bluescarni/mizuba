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

#ifndef MIZUBA_PY_COMMON_UTILS_HPP
#define MIZUBA_PY_COMMON_UTILS_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>
#include <unordered_set>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <heyoka/mdspan.hpp>

namespace mizuba_py
{

void check_array_cc_aligned(const pybind11::array &, const char *);

// Helper to turn an mdspan into a pybind11::array_t. The 'self' object will become
// the 'parent' of the returned array. If T is const, the array will be set
// as read-only.
template <typename T, typename IndexType, std::size_t... Extents>
auto mdspan_to_array(const pybind11::object &self,
                     heyoka::mdspan<T, std::experimental::extents<IndexType, Extents...>> span)
{
    using indices = std::make_index_sequence<sizeof...(Extents)>;

    auto ret = [&span, &self]<std::size_t... Idx>(std::index_sequence<Idx...>) {
        return pybind11::array_t<std::remove_const_t<T>>(
            pybind11::array::ShapeContainer{boost::numeric_cast<pybind11::ssize_t>(span.extent(Idx))...},
            span.data_handle(), self);
    }(indices{});

    if constexpr (std::is_const_v<T>) {
        ret.attr("flags").attr("writeable") = false;
    }

    return ret;
} // LCOV_EXCL_LINE

// Helper to check if a list of arrays may share any memory with each other.
// Quadratic complexity.
bool may_share_memory(const pybind11::array &, const pybind11::array &);

template <typename... Args>
bool may_share_memory(const pybind11::array &a, const pybind11::array &b, const Args &...args)
{
    const std::array args_arr = {std::cref(a), std::cref(b), std::cref(args)...};
    const auto nargs = args_arr.size();

    for (std::size_t i = 0; i < nargs; ++i) {
        for (std::size_t j = i + 1u; j < nargs; ++j) {
            if (may_share_memory(args_arr[i].get(), args_arr[j].get())) {
                return true;
            }
        }
    }

    return false;
}

// TLDR: machinery to clean up objects at Python shutdown.
//
// Python does not guarantee that all objects are garbage-collected at
// shutdown. This means that we may find ourselves in a situation where
// the temporary memory-mapped files used internally by the polyjectory,
// conjunction, etc. classes are not deleted when the program terminates
// because the C++ destructors are never called.
//
// In order to avoid this, we adopt the following approach:
//
// - every time a new polyjectory/conjunction/etc. is constructed, we grab a weak pointer
//   to its implementation and store it in a global vector;
// - we register a cleanup function that, at shutdown, goes through
//   the unexpired weak pointers and manually closes the polyjectories/conjunctions/etc.,
//   thus ensuring that the temporary files are removed.
//
// All newly-created polyjectories/conjunctions/etc. which end up exposed as a py::object
// are affected by this issue. This means that the weak pointer registration
// should be done every time a new polyjectory/conjunction/etc. is created in C++ before it is
// wrapped and returned as a py::object.
//
// For instance, both the polyjectory __init__() and the make_sgp4_polyjectory() factory need
// to register a weak pointer for the new polyjectory they create.
//
// This all sounds unfortunately complicated, let us hope it does not get too messy :/
//
// NOTE: we will have to re-examine this approach if/when we implement des11n, as that
// results in a creation of a new Python-wrapped polyjectory/conjunction/etc. without any weak pointer
// registration. Probably we will need to enforce the registration at unpickling time?
// Note also that des11n of, e.g., a conjunctions object will also create a new internal
// polyjectory object, which will also need to be registered.
//
// NOTE: in jupyterlab the cleanup functions registered to run at exit sometimes
// do not run to completion, thus leaving behind temporary files after shutdown.
// I think we are seeing this issue:
//
// https://github.com/jupyterlab/jupyterlab/issues/16276

// Add a shared pointer to T to the vector of weak pointers weak_ptr_vector.
//
// The random number engine weak_ptr_rng is used internally to avoid weak_ptr_vector
// growing to unbounded size. The mutex mut is intended to protect concurrent access to
// weak_ptr_vector and weak_ptr_rng.
template <typename T, typename Rng>
void add_weak_ptr_cleanup(std::mutex &mut, std::vector<std::weak_ptr<T>> &weak_ptr_vector, Rng &weak_ptr_rng,
                          const std::shared_ptr<T> &ptr)
{
    // Lock down.
    std::lock_guard lock(mut);

    // NOTE: if weak_ptr_vector is not empty, we want to randomly poke into
    // it and see if we find an expired pointer. If we do, we remove it by swapping it
    // with the last pointer and then popping back. Like this, we avoid weak_ptr_vector
    // growing unbounded in size.
    if (!weak_ptr_vector.empty()) {
        // Pick randomly an index into weak_ptr_vector.
        std::uniform_int_distribution<decltype(weak_ptr_vector.size())> dist(0, weak_ptr_vector.size() - 1u);
        const auto idx = dist(weak_ptr_rng);

        if (weak_ptr_vector[idx].expired()) {
            // We picked an expired pointer: swap it to the end of the vector
            // and pop back to erase it.
            // NOTE: the self swap corner case should be ok here.
            weak_ptr_vector[idx].swap(weak_ptr_vector.back());
            weak_ptr_vector.pop_back();
        }
    }

    // Add ptr.
    weak_ptr_vector.emplace_back(ptr);
}

// Helper to invoke a closing function on the non-expired weak pointers in weak_ptr_vector.
//
// The mutex mut is intended to protect concurrent access to weak_ptr_vector. class_name is
// used for debug output only. close_func is the function to be used for closing the pointers.
template <typename T, typename F>
void cleanup_weak_ptrs(std::mutex &mut, std::vector<std::weak_ptr<T>> &weak_ptr_vector,
                       [[maybe_unused]] const char *class_name, F *close_func)
{
    // Lock down.
    std::lock_guard lock(mut);

#if !defined(NDEBUG)
    std::cout << "Running the " << class_name << " cleanup function" << std::endl;

    // NOTE: we want to make sure that all non-expired weak pointers
    // are unique, because otherwise we will be closing the same object
    // twice, which is not allowed. Uniqueness of the weak pointers should be
    // guaranteed by the fact that all items added to weak_ptr_vector are
    // constructed ex-novo.
    std::unordered_set<std::shared_ptr<T>> ptr_set;
    for (const auto &wptr : weak_ptr_vector) {
        if (auto sptr = wptr.lock()) {
            assert(ptr_set.insert(sptr).second); // LCOV_EXCL_LINE
        }
    }
    ptr_set.clear();

#endif

    // Close the non-expired weak pointers.
    for (auto &wptr : weak_ptr_vector) {
        if (auto sptr = wptr.lock()) {
            // LCOV_EXCL_START
#if !defined(NDEBUG)
            std::cout << "Cleaning up a " << class_name << " still alive at shutdown" << std::endl;
#endif
            close_func(sptr);
            // LCOV_EXCL_STOP
        }
    }
}

} // namespace mizuba_py

#endif
