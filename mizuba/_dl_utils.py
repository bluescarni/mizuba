# Copyright 2024-2025 Francesco Biscani
#
# This file is part of the mizuba library.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Implementation-detail utilities for double-length arithmetic.

from typing import Any, Tuple


def _eft_add_knuth(a: Any, b: Any) -> Tuple[Any, Any]:
    # Error-free transformation of the sum of two floating point numbers.
    # This is Knuth's algorithm. See algorithm 2.1 here:
    # https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)

    return x, y


def _eft_add_dekker(a: Any, b: Any) -> Tuple[Any, Any]:
    # Error-free transformation of the sum of two floating point numbers.
    # This is Dekker's algorithm, which requires abs(a) >= abs(b). See algorithm 2.2 here:
    # https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
    x = a + b
    y = (a - x) + b

    return x, y


def _dl_add(a_hi: Any, a_lo: Any, b_hi: Any, b_lo: Any) -> Tuple[Any, Any]:
    # Double-length addition using error-free transformations.
    x_hi, y_hi = _eft_add_knuth(a_hi, b_hi)
    x_lo, y_lo = _eft_add_knuth(a_lo, b_lo)

    u, v = _eft_add_dekker(x_hi, y_hi + x_lo)
    u, v = _eft_add_dekker(u, v + y_lo)

    return u, v


del Any, Tuple
