# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# NOTE: this test case uses data downloaded from the celestrak/socrates
# system on November 1st, 2024. It checks that all collisions identified
# by socrates are identified by mizuba as well.

import numpy as np
from pathlib import Path
from sgp4.api import Satrec
from skyfield.api import EarthSatellite, load
from sgp4 import omm
import mizuba as mz
from astropy import time
import pandas as pd

# Load the TLE data.
ts = load.timescale()
sats = []
with open(Path(".") / "data" / "celestrak_on_orbit_20241101.csv") as datafile:
    for fields in omm.parse_csv(datafile):
        sat = Satrec()
        omm.initialize(sat, fields)
        sats.append(EarthSatellite.from_satrec(sat, ts))
sats_arr = np.array(sats)

# Fetch the socrates results.
soc_df = pd.read_csv(Path(".") / "data" / "socrates_minrange_20241101.csv")
soc_df["TCA"] = pd.to_datetime(soc_df["TCA"])

# Define begin/end times for the conjunction screening.
date_begin = time.Time("2024-11-01 00:00:00", format="iso", scale="utc", precision=9)
date_end = time.Time("2024-11-08 00:00:00", format="iso", scale="utc", precision=9)

# Build the polyjectory.
pt, df, mask = mz.sgp4_polyjectory(
    sats, date_begin.jd, date_end.jd, exit_radius=60000.0
)

# Detect the conjunctions.
cj = mz.conjunctions(pt, 5.0, 1.0)

# Create the conjunctions dataframe.
cdf = mz.make_sgp4_conjunctions_df(cj, df, date_begin.jd)
cdfs = cdf.sort_values("dca")


def locate_conj(idx):
    # Helper to locate the socrates conjunction at index
    # idx into the mizuba results.
    soc_conj = soc_df.iloc[idx]
    i, j = soc_conj[["NORAD_CAT_ID_1", "NORAD_CAT_ID_2"]]

    # NOTE: mizuba reports conjunctions always with i < j.
    if i > j:
        i, j = j, i
    mz_conjs = cdfs.loc[(cdfs["i_satnum"] == i) & (cdfs["j_satnum"] == j)]

    if len(mz_conjs) == 0:
        # If we do not find mizuba conjunctions, we may be dealing with deep-space
        # or duplicate objects.
        if (
            df.loc[i, "init_code"] == mz.sgp4_pj_status.DEEP_SPACE
            or df.loc[j, "init_code"] == mz.sgp4_pj_status.DEEP_SPACE
        ):
            return 0
        elif (
            df.loc[i, "init_code"] == mz.sgp4_pj_status.DUPLICATE
            or df.loc[j, "init_code"] == mz.sgp4_pj_status.DUPLICATE
        ):
            return 1
        else:
            raise ValueError(
                f"The conjunction at index {idx} between objects {i} and {j} at time {soc_conj['TCA']} was not located"
            )

    # NOTE: this is code for computing the tca/dca differences between mizuba and socrates.
    # Not really needed, but let us keep it for reference.
    mz_closest = mz_conjs.sort_values(
        by="tca (UTC)", key=lambda tca: abs(tca - soc_conj["TCA"])
    ).iloc[0]

    diff_tca = abs(mz_closest["tca (UTC)"] - soc_conj["TCA"]).total_seconds()
    diff_dca = abs(mz_closest["dca"] - soc_conj["TCA_RANGE"])

    return diff_tca, diff_dca


# Check that all conjunctions detected by socrates are also
# detected by mizuba. The tca/dca differences will end up
# stored in res.
res = []
N_ds = 0
N_dup = 0
for i in range(len(soc_df)):
    tmp = locate_conj(i)
    if tmp == 0:
        N_ds = N_ds + 1
        res.append((float("nan"), float("nan")))
    elif tmp == 1:
        N_dup = N_dup + 1
        res.append((float("nan"), float("nan")))
    else:
        res.append(tmp)

if N_ds != 320:
    raise ValueError(
        f"320 conjunctions involving deep-space objects were expected, but {N_ds} were found instead"
    )

if N_dup != 6:
    raise ValueError(
        f"6 conjunctions involving duplicate objects were expected, but {N_dup} were found instead"
    )
