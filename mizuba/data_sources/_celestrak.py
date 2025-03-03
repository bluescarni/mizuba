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

import polars as pl
from typing import Optional


def _reformat_supgp_celestrak(gpes: pl.DataFrame) -> pl.DataFrame:
    # Reformat the supgp data downloaded from celestrak.org.
    #
    # Here we will:
    #
    # - change the name of some columns,
    # - drop some other columns,
    # - change the units of measurement in several columns.
    import polars as pl
    from astropy.time import Time
    import numpy as np
    from .._dl_utils import _eft_add_knuth

    # Convert the epochs to astropy Time objects.
    apy_tm = Time(gpes["EPOCH"].cast(str), format="isot", scale="utc", precision=9)

    # Normalise the hi/lo parts of the Julian dates.
    # NOTE: we do this in order to make absolutely sure that
    # lexicographic ordering first by jd1 and then by jd2 produces
    # chronological order.
    jd1, jd2 = _eft_add_knuth(apy_tm.jd1, apy_tm.jd2)

    # Degree to radians conversion factor.
    deg2rad = 2.0 * np.pi / 360.0

    # Assemble the reformatted dataframe.
    # NOTE: use explicit casting in order to make sure
    # we are constructing a dataframe with the correct types.
    ret = pl.DataFrame(
        {
            "norad_id": gpes["NORAD_CAT_ID"].cast(pl.UInt64),
            "cospar_id": gpes["OBJECT_ID"].cast(str),
            "name": gpes["OBJECT_NAME"].cast(str),
            "epoch": apy_tm.iso,
            "epoch_jd1": jd1,
            "epoch_jd2": jd2,
            "n0": gpes["MEAN_MOTION"].cast(float) * (2.0 * np.pi / 1440.0),
            "e0": gpes["ECCENTRICITY"].cast(float),
            "i0": gpes["INCLINATION"].cast(float) * deg2rad,
            "node0": gpes["RA_OF_ASC_NODE"].cast(float) * deg2rad,
            "omega0": gpes["ARG_OF_PERICENTER"].cast(float) * deg2rad,
            "m0": gpes["MEAN_ANOMALY"].cast(float) * deg2rad,
            "bstar": gpes["BSTAR"].cast(float),
            # NOTE: the RMS is stored as a string, which may be empty
            # in some cases, which results in failures when attempting
            # the cast to float. We handle this manually below.
            "rms": gpes["RMS"].cast(str),
        }
    )

    # First, we replace empty strings in the rms
    # column with null values.
    ret = ret.with_columns(
        pl.when(pl.col("rms") == "").then(None).otherwise(pl.col("rms")).alias("rms")
    )

    # Then, we can cast.
    ret = ret.with_columns(pl.col("rms").cast(float))

    return ret


def _fetch_supgp_celestrak(group_name: str) -> Optional[pl.DataFrame]:
    # Fetch the supgp data for the group group_name from celestrak.
    import requests as rq
    from io import StringIO
    import polars as pl
    from ._common import _common_validate_gpes, _common_deduplicate_gpes
    import logging

    logger = logging.getLogger("mizuba")
    logger.debug(
        f"Attempting to fetch the supgp data for the group '{group_name}' from celestrak"
    )

    download_url = rf"https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={group_name}&FORMAT=json"
    download_response = rq.get(download_url)

    if not download_response.ok:
        raise RuntimeError(
            f"Unable to download GPEs from celestrak.org for the group '{group_name}':"
            f" {download_response.reason}"
        )

    # Parse the gpes into a polars dataframe.
    try:
        gpes = pl.read_json(StringIO(download_response.text))
    except Exception as e:
        # NOTE: in case of an invalid group name, we will get an exception here.
        # Downgrade the exception to a warning and return None instead. The idea
        # here is that every now and then there may be additions/removals of supgp
        # groups on celestrak, and if that happens we do not want to produce
        # a "hard" error.
        logger.warning(
            f'Error parsing the data for the celestrak supgp group "{group_name}"',
            exc_info=True,
            stack_info=True,
        )
        return None

    logger.debug(f"supgp data for the group '{group_name}' downloaded and parsed")

    # Validate.
    # NOTE: supgp data may have duplicate norad ids.
    _common_validate_gpes(gpes, False)

    # Reformat.
    gpes = _reformat_supgp_celestrak(gpes)

    # Deduplicate.
    # NOTE: I am not 100% sure this is required for supgp data,
    # but it sohuld not hurt.
    gpes = _common_deduplicate_gpes(gpes)

    # Sort by norad id first, then by epoch. Then return.
    return gpes.sort(["norad_id", "epoch_jd1", "epoch_jd2"])


def _validate_satcat_celestrak(satcat: pl.DataFrame) -> None:
    # Helper to run minimal validation on the satcat downloaded
    # from celestrak.org.
    import polars as pl

    # We need all satellites to have a non-null and unique norad id.
    if not satcat["NORAD_CAT_ID"].is_not_null().all():
        raise ValueError(
            "One or more NULL NORAD IDs detected in the satcat downloaded from"
            " celestrak.org"
        )
    if not satcat["NORAD_CAT_ID"].cast(pl.UInt64).is_unique().all():
        raise ValueError(
            "Non-unique NORAD IDs detected in the satcat downloaded from celestrak.org"
        )


def _fetch_satcat_celestrak() -> pl.DataFrame:
    # Download the satcat from celestrak.org.
    import requests as rq
    from io import StringIO
    import polars as pl
    import logging

    logger = logging.getLogger("mizuba")
    logger.debug("Attempting to fetch the satcat from celestrak")

    download_url = "https://celestrak.org/pub/satcat.csv"
    download_response = rq.get(download_url)

    if not download_response.ok:
        raise RuntimeError(
            "Unable to download the satcat from celestrak.org:"
            f" {download_response.reason}"
        )

    # Parse the satcat into a polars dataframes.
    # NOTE: use truncate_ragged_lines=True since occasionally we have seen malformed
    # CSV files from celestrak.
    satcat = pl.read_csv(StringIO(download_response.text), truncate_ragged_lines=True)

    logger.debug("celestrak satcat downloaded and parsed")

    # Validate.
    _validate_satcat_celestrak(satcat)

    # Return.
    return satcat


# List of group names for supgp GPEs datasets.
_supgp_group_names = [
    "starlink",
    "oneweb",
    "planet",
    "iridium",
    "gps",
    "glonass",
    "intelsat",
    "ses",
    "telesat",
    # NOTE: orbcomm satellites are not showing
    # up any more in the supgp data since mid/late
    # December.
    # "orbcomm",
    "iss",
    "css",
    "cpf",
    "kuiper",
    "ast",
]


def _supgp_pick_lowest_rms(gpes: pl.DataFrame) -> pl.DataFrame:
    # A helper to identify GPEs which have identical norad_id
    # and epoch, and, amongst these, pick the GPE with the
    # lowest RMS.
    # NOTE: we are assuming that the input gpes are supgps
    # datasets from celestrak.org
    import polars as pl

    # Let's break this down.
    gpes = (
        # First, we sort all GPEs by RMS, ensuring null values
        # are sorted at the bottom.
        gpes.sort("rms", nulls_last=True)
        # Then, we split the dataframe in groups with equal
        # norad ids and epochs.
        # NOTE: importantly, within each group, the data
        # is still sorted by RMS.
        .group_by(["norad_id", "epoch_jd1", "epoch_jd2"])
        # Then, we run an aggregation operation in which,
        # from each group, we pick the first GPE (that is, the
        # one with lowest RMS). pl.all() specifies to retain
        # all columns.
        .agg(pl.all().first())
        # Then, we restore the original column order.
        .select(gpes.columns)
        # Finally, we apply the usual sorting by
        # norad id and epoch.
        .sort(["norad_id", "epoch_jd1", "epoch_jd2"])
    )

    return gpes


del pl, Optional
