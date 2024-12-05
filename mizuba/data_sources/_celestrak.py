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
    from ._common import _eft_knuth

    # Convert the epochs to astropy Time objects.
    apy_tm = Time(gpes["EPOCH"], format="isot", scale="utc", precision=9)

    # Normalise the hi/lo parts of the Julian dates.
    # NOTE: we do this in order to make absolutely sure that
    # lexicographic ordering first by jd1 and then by jd2 produces
    # chronological order.
    jd1, jd2 = _eft_knuth(apy_tm.jd1, apy_tm.jd2)

    # Degree to radians conversion factor.
    deg2rad = 2.0 * np.pi / 360.0

    return pl.DataFrame(
        {
            "norad_id": gpes["NORAD_CAT_ID"],
            "cospar_id": gpes["OBJECT_ID"],
            "name": gpes["OBJECT_NAME"],
            "epoch_jd1": jd1,
            "epoch_jd2": jd2,
            "n0": gpes["MEAN_MOTION"] * (2.0 * np.pi / 1440.0),
            "ecc0": gpes["ECCENTRICITY"],
            "incl0": gpes["INCLINATION"] * deg2rad,
            "argp0": gpes["ARG_OF_PERICENTER"] * deg2rad,
            "node0": gpes["RA_OF_ASC_NODE"] * deg2rad,
            "m0": gpes["MEAN_ANOMALY"] * deg2rad,
            "bstar": gpes["BSTAR"],
            "rms": gpes["RMS"],
        }
    )


def _fetch_supgp_celestrak(group_name: str) -> pl.DataFrame:
    # Fetch the supgp data for the group group_name from celestrak.
    import requests as rq
    from io import StringIO
    import polars as pl
    from ._common import _common_validate_gpes, _common_deduplicate_gpes

    download_url = rf"https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE={group_name}&FORMAT=json"
    download_response = rq.get(download_url)

    if not download_response.ok:
        raise RuntimeError(
            f"Unable to download GPEs from celestrak.org for the group '{group_name}': {download_response.reason}"
        )

    # Parse the gpes as polars dataframes.
    gpes = pl.read_json(StringIO(download_response.text))

    # Validate.
    # NOTE: supgp data may have duplicate norad ids.
    _common_validate_gpes(gpes, False)

    # Reformat.
    gpes = _reformat_supgp_celestrak(gpes)

    # Deduplicate.
    gpes = _common_deduplicate_gpes(gpes)

    # Sort by norad id first, then by epoch. Then return.
    return gpes.sort(["norad_id", "epoch_jd1", "epoch_jd2"])


del pl
