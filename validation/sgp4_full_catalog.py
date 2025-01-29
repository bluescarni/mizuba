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

# NOTE: this validation case is meant to test the mechanism that bisects
# low-precision interpolation steps in make_sgp4_trajectory().

from pathlib import Path
import polars as pl
from astropy.time import Time
import mizuba as mz
from mizuba._sgp4_polyjectory import _make_satrec_from_dict as make_satrec
from sgp4.api import SatrecArray
import numpy as np

# Load the data.
data_path = Path(".") / "data" / "full_catalog.parquet"
gpes = pl.read_parquet(data_path)

# Retain only the GPEs with TLE data.
# NOTE: we do this because the sgp4 propagator does not like
# satellite numbers with too many digits.
gpes = gpes.filter(~pl.col("tle_line1").is_null())

# Setup the polyjectory epoch.
tm = Time("2025-01-12T12:00:00Z", format="isot", scale="utc")
jd_begin = tm.jd1

# Build the sat list.
sat_list = [make_satrec(_) for _ in gpes.iter_rows(named=True)]
sat_arr = SatrecArray(sat_list)

# Build the polyjectory.
pj = mz.make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 5.1)[0]

# Setup the sample times.
N_times = 1000
tm_range = np.linspace(0, 5, N_times)

# Evaluate with the sgp4 propagator.
e, r, v = sat_arr.sgp4(np.full((N_times,), jd_begin), tm_range)

# Evaluate with the polyjectory.
pj_state = pj.state_meval(tm_range)

# Compute the positional difference.
diff = np.linalg.norm(pj_state[:, :, :3] - r, axis=2).reshape((-1,))

# Filter out trajectories which errored out and compute the max err.
max_err = np.max(diff[~np.isnan(diff)])

if max_err >= 3e-6:
    raise ValueError(
        f"A max error below 3e-6 was expected, but an error of {max_err} was computed instead"
    )
