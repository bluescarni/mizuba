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

#ifndef MIZUBA_DETAIL_CONJUNCTIONS_JIT_HPP
#define MIZUBA_DETAIL_CONJUNCTIONS_JIT_HPP

#include <cstdint>

#include <heyoka/llvm_state.hpp>

namespace mizuba::detail
{

// Struct holding several JIT-compiled functions used throughout
// the conjunction-detection code. The functions are compiled
// when an instance of this struct is constructed.
struct conj_jit_data {
    // NOTE: some of the jitted functions are created as cfuncs, and thus they share the
    // same prototype. Others are custom written and need their own prototype.
    using cfunc_func_t = void (*)(double *, const double *, const double *, const double *) noexcept;
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
    cfunc_func_t pta7_cfunc = nullptr;
    fex_check_t fex_check = nullptr;
    rtscc_t rtscc = nullptr;
    pt1_t pt1 = nullptr;
    cfunc_func_t aabb_cs_cfunc = nullptr;
    cfunc_func_t cs_enc_func = nullptr;
    cfunc_func_t batched_cheby_eval6_func = nullptr;
    cfunc_func_t pinterp_func = nullptr;
};

// Helper to access a cached instance of conj_jit_data.
const conj_jit_data &get_conj_jit_data(std::uint32_t);

} // namespace mizuba::detail

#endif
