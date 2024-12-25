# Copyright 2024 Francesco Biscani
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

import polars as pl
from typing import Any, Tuple


def _common_validate_gpes(gpes: pl.DataFrame, unique_norad_id: bool = True) -> None:
    # Common logic for the validation of GPEs downloaded from space-track.org or celestrak.org.
    import polars as pl

    # We need all GPEs to have a non-null norad id.
    if not gpes["NORAD_CAT_ID"].is_not_null().all():
        raise ValueError("One or more NULL NORAD IDs detected in GPEs")

    # If unique_norad_id is True, check that the norad ids
    # are unique.
    if unique_norad_id and not gpes["NORAD_CAT_ID"].cast(pl.UInt64).is_unique().all():
        raise ValueError("Non-unique NORAD IDs detected in GPEs")

    # Check that all the data used for orbital propagation is present.
    for col in [
        "EPOCH",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "ARG_OF_PERICENTER",
        "RA_OF_ASC_NODE",
        "MEAN_ANOMALY",
        "BSTAR",
    ]:
        if not gpes[col].is_not_null().all():
            raise ValueError(
                f"One or more NULL values detected in the GPE column '{col}'"
            )


def _common_deduplicate_gpes(gpes: pl.DataFrame) -> pl.DataFrame:
    # GPE sets may contain duplicates (e.g., ISS servicing vehicles often
    # have their own GPE identical to the ISS one). We want to eliminate such
    # duplicates as they will just clutter up the results of conjunction detection.
    #
    # In order to detect duplicates, we consider all the values in the GPEs
    # which affect orbital propagation. We assume that these are contained
    # in the [3, 12) column range of the dataframe.
    #
    # NOTE: unique() will scramble the ordering of the rows.
    return gpes.unique(subset=gpes.columns[3:12], keep="first")


def _eft_add_knuth(a: Any, b: Any) -> Tuple[Any, Any]:
    # Error-free transformation of the sum of two floating point numbers.
    # This is Knuth's algorithm. See algorithm 2.1 here:
    # https://www.researchgate.net/publication/228568591_Error-free_transformations_in_real_and_complex_floating_point_arithmetic
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)

    return x, y


del pl
del Any
del Tuple
