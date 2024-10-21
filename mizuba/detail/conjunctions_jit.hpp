// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_DETAIL_CONJUNCTIONS_JIT_HPP
#define MIZUBA_DETAIL_CONJUNCTIONS_JIT_HPP

#include <cstdint>

#include <heyoka/llvm_state.hpp>

namespace mizuba::detail
{

struct conj_jit_data {
    using pta_cfunc_t = void (*)(double *, const double *, const double *, const double *) noexcept;
    using pssdiff3_cfunc_t = void (*)(double *, const double *, const double *, const double *) noexcept;
    using fex_check_t = void (*)(const double *, const double *, const std::uint32_t *, std::uint32_t *) noexcept;
    using rtscc_t = void (*)(double *, double *, std::uint32_t *, const double *) noexcept;
    using pt1_t = void (*)(double *, const double *) noexcept;

    explicit conj_jit_data(std::uint32_t);
    conj_jit_data(const conj_jit_data &) = delete;
    conj_jit_data(conj_jit_data &&) noexcept = delete;
    conj_jit_data &operator=(const conj_jit_data &) = delete;
    conj_jit_data &operator=(conj_jit_data &&) noexcept = delete;
    ~conj_jit_data();

    heyoka::llvm_state state;
    pta_cfunc_t pta_cfunc = nullptr;
    pssdiff3_cfunc_t pssdiff3_cfunc = nullptr;
    fex_check_t fex_check = nullptr;
    rtscc_t rtscc = nullptr;
    pt1_t pt1 = nullptr;
};

const conj_jit_data &get_conj_jit_data(std::uint32_t);

} // namespace mizuba::detail

#endif
