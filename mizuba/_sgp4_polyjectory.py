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

from .core import polyjectory, gpe_dtype
import polars as pl
from typing import Union
import numpy as np

# The fields expected to be in a dataframe
# containing GPEs.
_gpe_fields = {
    "norad_id": pl.UInt64,
    "epoch_jd1": float,
    "epoch_jd2": float,
    "n0": float,
    "e0": float,
    "i0": float,
    "node0": float,
    "omega0": float,
    "m0": float,
    "bstar": float,
}


def _make_satrec_from_dict(d):
    # Helper to construct a Satrec object from
    # GPE values contained in a dictionary.
    from sgp4.api import Satrec, WGS72
    from ._dl_utils import _dl_add, _eft_add_knuth

    # NOTE: this is the baseline reference epoch
    # used by the C++ SGP4 code.
    jd_sub = 2433281.5

    # Normalise the components of the Julian dates.
    jd1, jd2 = _eft_add_knuth(d["epoch_jd1"], d["epoch_jd2"])

    # Compute the date required by sgp4init().
    jd = _dl_add(
        jd1,
        jd2,
        -jd_sub,
        0.0,
    )[0]

    # Construct and init the Satrec.
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        d["norad_id"],
        jd,
        d["bstar"],
        # NOTE: ignore ndot and nddot, as they
        # are not used during propagation.
        0.0,
        0.0,
        d["e0"],
        d["omega0"],
        d["i0"],
        d["m0"],
        d["n0"],
        d["node0"],
    )

    return sat


def make_sgp4_polyjectory(
    gpes: Union[pl.DataFrame, np.ndarray[gpe_dtype]], jd_begin: float, jd_end: float
) -> polyjectory:
    from .core import _make_sgp4_polyjectory
    import polars as pl
    import numpy as np

    if isinstance(gpes, pl.DataFrame):
        # Make sure the expected fields in gpes have the correct type.
        gpes = gpes.with_columns(
            [pl.col(name).cast(_gpe_fields[name]) for name in _gpe_fields]
        )

        # Extract the fields in a separate dataframe.
        gpes = gpes.select([pl.col(name) for name in _gpe_fields])

        # Convert to numpy.
        # NOTE: since we enforced types and ordering of fields,
        # it should not be necessary to cast manually to the gpe
        # dtype.
        gpes_arr = np.ascontiguousarray(gpes.to_numpy(structured=True))
    else:
        gpes_arr = gpes

    # Invoke the C++ function.
    return _make_sgp4_polyjectory(gpes_arr, jd_begin, jd_end)


del polyjectory, pl, gpe_dtype, Union, np
