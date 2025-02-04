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
import requests as rq
from typing import Optional


def _spacetrack_login(identity: Optional[str], password: Optional[str]) -> rq.Session:
    # Attempt to log into space-track.org. If successful, an http session will be returned.
    import requests as rq
    import os
    import logging

    id_pass_err_msg = (
        "In order to access the data on space-track.org, you must provide an"
        " identity and a password, either passing them as arguments or via the"
        " environment variables MIZUBA_SPACETRACK_IDENTITY and"
        " MIZUBA_SPACETRACK_PASSWORD"
    )

    # Either both identity and password must be provided, or neither of them.
    if (identity is None) != (password is None):
        raise ValueError(id_pass_err_msg)

    if identity is None:
        # The user did not pass identity and password,
        # try to fetch them from the environment.
        identity = os.getenv("MIZUBA_SPACETRACK_IDENTITY")
        password = os.getenv("MIZUBA_SPACETRACK_PASSWORD")
        if identity is None or password is None:
            raise ValueError(id_pass_err_msg)
    else:
        # The user provided identity and password,
        # check their types.
        if not isinstance(identity, str):
            raise TypeError(
                f"The spacetrack identity must be a string, but an object of type '{type(identity)}' was provided instead"
            )
        if not isinstance(password, str):
            raise TypeError(
                f"The spacetrack password must be a string, but an object of type '{type(password)}' was provided instead"
            )

    # Open an http session.
    session = rq.Session()

    logger = logging.getLogger("mizuba")
    logger.debug("Attempting to log into space-track.org")

    # Try to log in.
    login_url = r"https://www.space-track.org/ajaxauth/login"
    login_response = session.post(
        login_url,
        {
            "identity": identity,
            "password": password,
        },
    )
    if not login_response.ok:
        raise RuntimeError(
            f"Unable to log into space-track.org: {login_response.reason}"
        )

    logger.debug("space-track.org login successful")

    return session


def _validate_gpes_spacetrack(gpes: pl.DataFrame) -> None:
    # Validate the GPEs downloaded from space-track.org.
    from ._common import _common_validate_gpes

    _common_validate_gpes(gpes)

    # Check the time system, as we are assuming UTC.
    if not (gpes["TIME_SYSTEM"].cast(str) == "UTC").all():
        raise ValueError(
            "One or more non-UTC time systems detected in the GPEs downloaded from"
            " space-track.org"
        )

    # Check the reference frame.
    if not (gpes["REF_FRAME"].cast(str) == "TEME").all():
        raise ValueError(
            "One or more non-TEME reference frames detected in the GPEs downloaded from"
            " space-track.org"
        )

    # Check the center.
    if not (gpes["CENTER_NAME"].cast(str) == "EARTH").all():
        raise ValueError(
            "One or more non-Earth centers detected in the GPEs downloaded from"
            " space-track.org"
        )


def _reformat_gpes_spacetrack(gpes: pl.DataFrame) -> pl.DataFrame:
    # Reformat the GPEs downloaded from space-track.org.
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
            # NOTE: these two are kept for debugging.
            "tle_line1": gpes["TLE_LINE1"].cast(str),
            "tle_line2": gpes["TLE_LINE2"].cast(str),
        }
    )

    # Replace "UNKNOWN" cospar ids and "TBA - TO BE ASSIGNED" names
    # with null values.
    ret = ret.with_columns(
        pl.when(pl.col("cospar_id") == "UNKNOWN")
        .then(None)
        .otherwise(pl.col("cospar_id"))
        .alias("cospar_id"),
        pl.when(pl.col("name") == "TBA - TO BE ASSIGNED")
        .then(None)
        .otherwise(pl.col("name"))
        .alias("name"),
    )

    return ret


def _fetch_gpes_spacetrack(session: rq.Session) -> pl.DataFrame:
    # Fetch the most recent GPEs from space-track.org.
    from io import StringIO
    import polars as pl
    from ._common import _common_deduplicate_gpes
    import logging

    logger = logging.getLogger("mizuba")
    logger.debug("Attempting to download gpes from space-track.org")

    # Try to fetch the gpes.
    #
    # NOTE: this is the recommended URL for retrieving the newest propagable
    # element set for all on-orbit objects. See:
    #
    # https://www.space-track.org/documentation#/api
    #
    # From what I understand, this query fetches the GPEs of all objects that:
    #
    # - have not decayed yet,
    # - received a GPE update in the last 30 days.
    #
    # The results are ordered by NORAD cat id and formatted as JSON.
    download_url = r"https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-30/orderby/norad_cat_id/format/json"
    download_response = session.get(download_url)
    if not download_response.ok:
        raise RuntimeError(
            f"Unable to download GPEs from space-track.org: {download_response.reason}"
        )

    # Parse the gpes into a polars dataframes.
    gpes = pl.read_json(StringIO(download_response.text))

    logger.debug("gpes from space-track.org downloaded and parsed")

    # Validate.
    _validate_gpes_spacetrack(gpes)

    # Reformat.
    gpes = _reformat_gpes_spacetrack(gpes)

    # Deduplicate.
    gpes = _common_deduplicate_gpes(gpes)

    # Sort by norad id and return.
    return gpes.sort("norad_id")


del pl, rq, Optional
