// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef MIZUBA_DETAIL_FMV_ATTRIBUTES_HPP
#define MIZUBA_DETAIL_FMV_ATTRIBUTES_HPP

// NOTE: on some platforms we want to compile certain functions for multiple
// target architectures in order to take advantage of extended instruction sets.
// This most notably applies to the conversion of 16-bit floating-point numbers
// to/from float, which is much faster if done in hardware with the f16c instruction set:
//
// https://en.wikipedia.org/wiki/F16C
//
// Compilation for multiple targets is achieved via the GCC-style target_clones function attribute:
//
// https://lwn.net/Articles/691932/
//
// Note that the target_clones attribute does not extend to nested function calls within
// the original function. We thus also add the "flatten" function attribute, which should
// maximise the chance that all code in the original function (including inlineable nested
// function calls) is compiled for multiple architectures.
//
// NOTE: at this time we need this only for x86_64. It seems like 64-bit arm always provides
// fast float16 conversion functions, while ppc64 at this time does not seem to have
// specific instructions for float16 conversion.
#if defined(MIZUBA_HAVE_TARGET_CLONES_ATTRIBUTE) && (defined(__x86_64__) || defined(_M_AMD64))

#define MIZUBA_FMV_ATTRIBUTES                                                                                          \
    __attribute__((target_clones("default", "arch=x86-64-v2", "arch=x86-64-v3", "arch=x86-64-v4"), flatten))

#else

#define MIZUBA_FMV_ATTRIBUTES

#endif

#endif
