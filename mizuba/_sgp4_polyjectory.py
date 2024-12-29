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

from .core import polyjectory
import polars as pl


def make_sgp4_polyjectory(
    gpes: pl.DataFrame, jd_begin: float, jd_end: float
) -> polyjectory:
    from .core import _make_sgp4_polyjectory, _gpe_dtype
    import polars as pl
    import numpy as np

    # Fields (and their types) to extract from gpes.
    fields = {
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

    # Make sure the expected fields in gpes have the correct type.
    gpes = gpes.with_columns([pl.col(name).cast(fields[name]) for name in fields])

    # Extract the fields in a separate dataframe.
    gpes = gpes.select([pl.col(name) for name in fields])

    # Convert to numpy.
    # NOTE: since we enforced types and ordering of fields,
    # it should not be necessary to cast manually to the gpe
    # dtype.
    gpes_arr = np.ascontiguousarray(gpes.to_numpy(structured=True))

    # Invoke the C++ function.
    return _make_sgp4_polyjectory(gpes_arr, jd_begin, jd_end)


del polyjectory, pl
