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
import requests as rq


def _spacetrack_login() -> rq.Session:
    # Attempt to log into space-track.org. If successful,
    # an http session will be returned.
    import requests as rq
    import os

    # Open an http session.
    session = rq.Session()

    # Try to log in.
    login_url = r"https://www.space-track.org/ajaxauth/login"
    login_response = session.post(
        login_url,
        {
            "identity": os.getenv("MIZUBA_SPACETRACK_IDENTITY"),
            "password": os.getenv("MIZUBA_SPACETRACK_PASSWORD"),
        },
    )
    if not login_response.ok:
        raise RuntimeError(
            f"Unable to log into space-track.org: {login_response.reason}"
        )

    return session


def _validate_gpes_spacetrack(gpes: pl.DataFrame) -> None:
    # Validate the GPEs downloaded from space-track.org.
    if not gpes["NORAD_CAT_ID"].is_unique().all():
        raise ValueError(
            "Non-unique NORAD IDs detected in the GPEs downloaded from space-track.org"
        )

    if not gpes["NORAD_CAT_ID"].is_not_null().all():
        raise ValueError(
            "One or more NULL NORAD IDs detected in the GPEs downloaded from space-track.org"
        )

    if not (gpes["TIME_SYSTEM"] == "UTC").all():
        raise ValueError(
            "One or more non-UTC time systems detected in the GPEs downloaded from space-track.org"
        )

    if not (gpes["REF_FRAME"] == "TEME").all():
        raise ValueError(
            "One or more non-TEME reference frames detected in the GPEs downloaded from space-track.org"
        )

    if not (gpes["CENTER_NAME"] == "EARTH").all():
        raise ValueError(
            "One or more non-Earth centers detected in the GPEs downloaded from space-track.org"
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

    # Convert the epochs to astropy Time objects.
    apy_tm = Time(gpes["EPOCH"], format="isot", scale="utc", precision=9)

    return pl.DataFrame(
        {
            "norad_id": gpes["NORAD_CAT_ID"].cast(int),
            "cospar_id": gpes["OBJECT_ID"],
            "name": gpes["OBJECT_NAME"],
            "epoch_jd1": apy_tm.jd1,
            "epoch_jd2": apy_tm.jd2,
            "n0": gpes["MEAN_MOTION"].cast(float) * (2.0 * np.pi / 1440.0),
            "ecc0": gpes["ECCENTRICITY"].cast(float),
            "incl0": gpes["INCLINATION"].cast(float) * (2.0 * np.pi / 360.0),
            "argp0": gpes["ARG_OF_PERICENTER"].cast(float) * (2.0 * np.pi / 360.0),
            "node0": gpes["RA_OF_ASC_NODE"].cast(float) * (2.0 * np.pi / 360.0),
            "m0": gpes["MEAN_ANOMALY"].cast(float) * (2.0 * np.pi / 360.0),
            "bstar": gpes["BSTAR"].cast(float),
            "rcs_size": gpes["RCS_SIZE"],
            # NOTE: these two are kept for debugging.
            "tle_line1": gpes["TLE_LINE1"],
            "tle_line2": gpes["TLE_LINE2"],
        }
    )


def _deduplicate_gpes_spacetrack(gpes: pl.DataFrame) -> pl.DataFrame:
    # GPEs from spacetrack often contain duplicates (e.g., ISS servicing
    # vehicles often have their own GPE identical to the ISS one). We want
    # to eliminate such duplicates as they will just clutter up the results
    # of conjunction detection.
    #
    # In order to detect duplicates, we consider all the values in the GPEs
    # which affect orbital propagation. These are contained in the [3, 12)
    # column range of the dataframe.
    return gpes.unique(subset=gpes.columns[3:12], keep="first").sort("norad_id")


def _fetch_gpes_spacetrack(session: rq.Session) -> pl.DataFrame:
    # Fetch the most recent GPEs from space-track.org.
    from io import StringIO
    import polars as pl

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

    # Parse the gpes as polars dataframes.
    gpes = pl.read_json(StringIO(download_response.text))

    # Validate.
    _validate_gpes_spacetrack(gpes)

    # Reformat.    
    gpes = _reformat_gpes_spacetrack(gpes)

    # Deduplicate and return.
    return _deduplicate_gpes_spacetrack(gpes)


def download_gpes_spacetrack() -> pl.DataFrame:
    session = _spacetrack_login()
    return _fetch_gpes_spacetrack(session)


del pl, rq
