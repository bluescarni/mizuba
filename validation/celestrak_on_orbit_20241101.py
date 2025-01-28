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

# NOTE: this test case uses data downloaded from the celestrak/socrates
# system on November 1st, 2024. It checks that all conjunctions identified
# by socrates are identified by mizuba as well.

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

# Run conjunction detection.
cj = mz.conjunctions(pj, 5.0, 2.0 / 1440.0)

# Load the conjunction data from socrates.
soc_df = pl.read_csv(cur_dir / "data" / "socrates_minrange_20241101.csv")

# Ensure correct ordering of norad IDs.
soc_df = soc_df.with_columns(
    norad_id_i=pl.when(pl.col("NORAD_CAT_ID_1") < pl.col("NORAD_CAT_ID_2"))
    .then(pl.col("NORAD_CAT_ID_1"))
    .otherwise(pl.col("NORAD_CAT_ID_2")),
    norad_id_j=pl.when(pl.col("NORAD_CAT_ID_1") < pl.col("NORAD_CAT_ID_2"))
    .then(pl.col("NORAD_CAT_ID_2"))
    .otherwise(pl.col("NORAD_CAT_ID_1")),
)

# Rename.
soc_df = soc_df.with_columns(
    pl.col("TCA").alias("tca"),
    pl.col("TCA_RANGE").alias("dca"),
    pl.col("TCA_RELATIVE_SPEED").alias("relative_speed"),
)

# Transform tca column into datetime.
soc_df = soc_df.with_columns(
    pl.col("tca").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.3f").alias("tca")
)

# Clean up.
soc_df = soc_df.drop(
    "NORAD_CAT_ID_1",
    "NORAD_CAT_ID_2",
    "OBJECT_NAME_1",
    "OBJECT_NAME_2",
    "DSE_1",
    "DSE_2",
    "MAX_PROB",
    "DILUTION",
    "TCA",
    "TCA_RANGE",
    "TCA_RELATIVE_SPEED",
)
soc_df = soc_df.cast({"norad_id_i": pl.UInt64, "norad_id_j": pl.UInt64})

# Start constructing the conjunctions dataframe.
conj = cj.conjunctions

# Fetch the norad ids.
norad_id_i = norad_ids[conj["i"]]
norad_id_j = norad_ids[conj["j"]]

# Build the tca column, representing it as a ISO UTC
# string with millisecond resolution.
pj_epoch1, pj_epoch2 = cj.polyjectory.epoch
tca = Time(
    val=pj_epoch1, val2=pj_epoch2 + conj["tca"], format="jd", scale="tai", precision=3
).utc.iso

# Build the dca column.
dca = conj["dca"]

# Build the relative speed column.
rel_speed = np.linalg.norm(conj["vi"] - conj["vj"], axis=1)

# Assemble the dataframe.
sgp4_cdf = pl.DataFrame(
    {
        "norad_id_i": norad_id_i,
        "norad_id_j": norad_id_j,
        "tca": tca,
        "dca": dca,
        "relative_speed": rel_speed,
    }
)

# Transform the tca column into datetime.
sgp4_cdf = sgp4_cdf.with_columns(
    pl.col("tca").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.3f").alias("tca")
)

# NOTE: this is a join operation that, for each row in soc_df, will:
#
# - perform a left join with sgp4_cdf on the norad_id_i/j columns, thus selecting the subset
#   of rows from the sgp4_cdf dataframe containing the same norad_id_i/j values;
# - within this subset, it will pick the row with the tca nearest to the row in
#   soc_df being joined.
#
# Essentially, for each conjunction in soc_df we are locating in sgp4_cdf the closest conjunction
# (in terms of tca) that involves the same satellites.
#
# NOTE: in order for join_asof() to work, both dataframes need to be sorted wrt tca.
jdf = soc_df.sort("tca").join_asof(
    sgp4_cdf.sort("tca"),
    by=["norad_id_i", "norad_id_j"],
    on="tca",
    strategy="nearest",
    coalesce=False,
    check_sortedness=True,
)

# Now we can run the actual checks.

# Every i/j pair in soc_df must be in sgp4_df.
if jdf["tca_right"].is_null().any():
    raise ValueError("Not all conjunctions detected by socrates were found")

# The greatest tca difference must be under 1 second.
if np.max(np.abs((jdf["tca"] - jdf["tca_right"]).to_numpy())) >= 1000:
    raise ValueError("A TCA difference >= 1 second was detected")

# The greates difference in relative speed must be under 1 m/s.
if (
    np.max(np.abs((jdf["relative_speed"] - jdf["relative_speed_right"]).to_numpy()))
    >= 0.001
):
    raise ValueError("A relative speed difference >= 1 m/s was detected")
