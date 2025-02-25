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

# NOTE: this is the same test as in celestrak_on_orbit_20241101.py,
# only that the purpose here is to make sure that different conjunction
# detection intervals return the same conjunctions.

import polars as pl
from astropy.time import Time
import numpy as np
import pathlib
import mizuba as mz

# Fetch the current directory.
cur_dir = pathlib.Path(__file__).parent.resolve()

# Load the original Socrates data.
on_orbit = pl.read_csv(
    cur_dir / "data" / "celestrak_on_orbit_20241101.csv",
    schema_overrides={"MEAN_MOTION_DDOT": pl.Float64},
)

# Rename the columns.
rename_map = {
    "NORAD_CAT_ID": "norad_id",
    "MEAN_MOTION": "n0",
    "ECCENTRICITY": "e0",
    "INCLINATION": "i0",
    "RA_OF_ASC_NODE": "node0",
    "ARG_OF_PERICENTER": "omega0",
    "MEAN_ANOMALY": "m0",
    "BSTAR": "bstar",
}
on_orbit = on_orbit.rename(rename_map)

# Setup the epoch_jd columns.
utc_dates = Time(on_orbit["EPOCH"], format="isot", scale="utc")
on_orbit = on_orbit.with_columns(
    pl.Series(name="epoch_jd1", values=utc_dates.jd1),
    pl.Series(name="epoch_jd2", values=utc_dates.jd2),
)

# Change units of measurement.
deg2rad = 2.0 * np.pi / 360.0
on_orbit = on_orbit.with_columns(
    n0=pl.col("n0") * (2.0 * np.pi / 1440.0),
    i0=pl.col("i0") * deg2rad,
    node0=pl.col("node0") * deg2rad,
    omega0=pl.col("omega0") * deg2rad,
    m0=pl.col("m0") * deg2rad,
)

# Setup the propagation period.
date_begin = Time("2024-11-01 00:00:00", format="iso", scale="utc")
date_end = Time("2024-11-08 00:00:00", format="iso", scale="utc")

# Build the polyjectory.
mz.set_logger_level_trace()
pj, norad_ids = mz.make_sgp4_polyjectory(on_orbit, date_begin.jd, date_end.jd)

# Detect the conjunctions using different conjunction intervals.
cj1 = mz.conjunctions(pj, 5.0, 1.0 / 1440.0)
cj2 = mz.conjunctions(pj, 5.0, 2.0 / 1440.0)
cj3 = mz.conjunctions(pj, 5.0, 3.0 / 1440.0)

if len(cj1.conjunctions) != len(cj2.conjunctions):
    raise ValueError(
        f"cj1 detected {len(cj1.conjunctions)} conjunctions, cj2 detected {len(cj2.conjunctions)}"
    )

if len(cj1.conjunctions) != len(cj3.conjunctions):
    raise ValueError(
        f"cj1 detected {len(cj1.conjunctions)} conjunctions, cj3 detected {len(cj3.conjunctions)}"
    )

if np.any(cj1.conjunctions["i"] != cj2.conjunctions["i"]):
    raise ValueError("Differences detected in the 'i' objects between cj1 and cj2")

if np.any(cj1.conjunctions["i"] != cj3.conjunctions["i"]):
    raise ValueError("Differences detected in the 'i' objects between cj1 and cj3")

if np.any(cj1.conjunctions["j"] != cj2.conjunctions["j"]):
    raise ValueError("Differences detected in the 'j' objects between cj1 and cj2")

if np.any(cj1.conjunctions["j"] != cj3.conjunctions["j"]):
    raise ValueError("Differences detected in the 'j' objects between cj1 and cj3")

# Check TCAs.
nonzero = np.where(cj1.conjunctions["tca"] != 0)

tca_diff = np.abs(cj1.conjunctions["tca"][nonzero] - cj2.conjunctions["tca"][nonzero])
if np.any(tca_diff > 5e-8):
    raise ValueError("Differences detected in the tcas between cj1 and cj2")
print(f"Largest TCA difference between cj1 and cj2: {np.max(tca_diff)}")


tca_diff = np.abs(cj1.conjunctions["tca"][nonzero] - cj3.conjunctions["tca"][nonzero])
if np.any(tca_diff > 5e-8):
    raise ValueError("Differences detected in the tcas between cj1 and cj3")
print(f"Largest TCA difference between cj1 and cj3: {np.max(tca_diff)}")
