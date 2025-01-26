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

# NOTE: this is the same test as in celestrak_on_orbit_20241101.py,
# only that the purpose here is to make sure that different conjunction
# detection intervals return the same conjunctions.

import numpy as np
from pathlib import Path
from sgp4.api import Satrec
from skyfield.api import EarthSatellite, load
from sgp4 import omm
import mizuba as mz
from astropy import time

# Load the TLE data.
ts = load.timescale()
sats = []
with open(Path(".") / "data" / "celestrak_on_orbit_20241101.csv") as datafile:
    for fields in omm.parse_csv(datafile):
        sat = Satrec()
        omm.initialize(sat, fields)
        sats.append(EarthSatellite.from_satrec(sat, ts))
sats_arr = np.array(sats)

# Define begin/end times for the conjunction screening.
date_begin = time.Time("2024-11-01 00:00:00", format="iso", scale="utc", precision=9)
date_end = time.Time("2024-11-08 00:00:00", format="iso", scale="utc", precision=9)

# Build the polyjectory.
pt, df, mask = mz.sgp4_polyjectory(
    sats, date_begin.jd, date_end.jd, exit_radius=60000.0
)

# Detect the conjunctions using different conjunction intervals.
cj1 = mz.conjunctions(pt, 5.0, 1.0)
cj2 = mz.conjunctions(pt, 5.0, 2.0)
cj3 = mz.conjunctions(pt, 5.0, 3.0)

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

tca_diff = np.abs(
    (cj1.conjunctions["tca"][nonzero] - cj2.conjunctions["tca"][nonzero])
    / cj1.conjunctions["tca"][nonzero]
)
if np.any(tca_diff > 1e-10):
    raise ValueError("Differences detected in the tcas between cj1 and cj2")

tca_diff = np.abs(
    (cj1.conjunctions["tca"][nonzero] - cj3.conjunctions["tca"][nonzero])
    / cj1.conjunctions["tca"][nonzero]
)
if np.any(tca_diff > 1e-10):
    raise ValueError("Differences detected in the tcas between cj1 and cj3")
