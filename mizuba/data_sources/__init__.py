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


def download_gpes_spacetrack() -> pl.DataFrame:
    from ._spacetrack import _spacetrack_login, _fetch_gpes_spacetrack

    with _spacetrack_login() as session:
        return _fetch_gpes_spacetrack(session)


def download_gpes_celestrak() -> pl.DataFrame:
    import polars as pl
    from concurrent.futures import ThreadPoolExecutor
    from ._celestrak import (
        _fetch_supgp_celestrak,
        _supgp_group_names,
        _supgp_pick_lowest_rms,
    )

    # Download in parallel the GPEs for all supgp groups.
    with ThreadPoolExecutor() as executor:
        ret = executor.map(_fetch_supgp_celestrak, _supgp_group_names)

    # Concatenate the datasets into a single dataframe.
    gpes = pl.concat(ret)

    # Pick the GPEs with the lowest rms.
    gpes = _supgp_pick_lowest_rms(gpes)

    return gpes


def download_satcat_celestrak() -> pl.DataFrame:
    from ._celestrak import _fetch_satcat_celestrak

    return _fetch_satcat_celestrak()


# The schema for the dataframe returned
# by download_all_gpes().
gpes_schema = pl.Schema(
    {
        "norad_id": pl.Int64,
        "cospar_id": pl.String,
        "name": pl.String,
        "epoch_jd1": pl.Float64,
        "epoch_jd2": pl.Float64,
        "n0": pl.Float64,
        "ecc0": pl.Float64,
        "incl0": pl.Float64,
        "argp0": pl.Float64,
        "node0": pl.Float64,
        "m0": pl.Float64,
        "bstar": pl.Float64,
        "tle_line1": pl.String,
        "tle_line2": pl.String,
        "rms": pl.Float64,
        "rcs": pl.Float64,
        "ops_code": pl.String,
    }
)


def download_all_gpes(with_supgp: bool = True) -> pl.DataFrame:
    import polars as pl
    from concurrent.futures import ThreadPoolExecutor
    from ._celestrak import (
        _fetch_supgp_celestrak,
        _supgp_group_names,
        _supgp_pick_lowest_rms,
    )

    with ThreadPoolExecutor() as executor:
        # Fetch all data asynchronously.
        gpe_st = executor.submit(download_gpes_spacetrack)
        satcat = executor.submit(download_satcat_celestrak)

        if with_supgp:
            gpe_ct_list = [
                executor.submit(_fetch_supgp_celestrak, gname)
                for gname in _supgp_group_names
            ]

            # Concatenate the datasets into a single dataframe.
            gpe_ct = pl.concat(fut.result() for fut in gpe_ct_list)

            # Pick the GPEs with the lowest rms.
            gpe_ct = _supgp_pick_lowest_rms(gpe_ct)

        # Fetch the futures.
        gpe_st = gpe_st.result()
        satcat = satcat.result()

    # Merge the supgp data into the spacetrack data, if requested.
    if with_supgp:
        # Run a full join, coalescing norad_id. The columns from
        # gpe_ct which are also in gpe_st will have a special suffix
        # added to their names.
        #
        # NOTE: here we have the following possibilities:
        #
        # - a norad_id shows up exactly once in both dataframes: the joined
        #   dataframe will contain a single row for that norad_id;
        # - a norad_id shows up once in spacetrack and never in celestrak:
        #   the joined dataframe will contain a single row for that norad_id,
        #   the columns from celestrak will contain nulls;
        # - a norad_id shows up N>0 times in celestrak and never in spacetrack:
        #   the joined dataframe will contain N rows for that norad_id, the columns
        #   from spacetrack will contain nulls;
        # - a norad_id shows up N>0 times in celestrak and once in spacetrack:
        #   the joined dataframe will contain N rows for that norad_id, the columns
        #   from spacetrack will contain the same data from spacetrack repeated
        #   N times.
        #
        # Note that we cannot have a situation with multiple norad ids from spacetrack,
        # as we are always only getting a single GPE from spacetrack.
        supgp_suffix = ":supgp"
        supgp_suffix_len = len(supgp_suffix)
        gpes = gpe_st.join(
            gpe_ct, on="norad_id", how="full", coalesce=True, suffix=supgp_suffix
        )

        # The list of columns which will be coalesced. These are the columns
        # with the special suffix (i.e., those common between spacetrack and celestrak,
        # with the exception of 'norad_id' which has already been coalesced).
        c_cols = list(
            filter(lambda col_name: col_name.endswith(supgp_suffix), gpes.columns)
        )

        # Now we are going to coalesce the common columns. Because we pass first
        # the supgp column and then the spacetrack column to coalesce(), priority
        # will be given to the values from supgp. If the supgp values are null,
        # then we will retain the values from spacetrack.
        gpes = gpes.with_columns(
            [
                pl.coalesce(
                    pl.col(c_col_name), pl.col(c_col_name[:-supgp_suffix_len])
                ).alias(c_col_name[:-supgp_suffix_len])
                for c_col_name in c_cols
            ]
        ).drop(c_cols)

        # Finally, we will re-sort the data in the canonical order.
        gpes = gpes.sort(["norad_id", "epoch_jd1", "epoch_jd2"])
    else:
        # No supgp data requested. Add an rms column filled with nulls in order
        # to match the layout of the dataframe from the other branch.
        gpes = gpe_st.with_columns(pl.lit(None).cast(float).alias("rms"))

    # We are now going to include data from the satcat into gpes.
    # For every satellite in satcat that appears in gpes, we fetch
    # the radar cross section and the ops status code.
    gpes = (
        gpes.join(
            satcat.select(["NORAD_CAT_ID", "RCS", "OPS_STATUS_CODE"]),
            left_on="norad_id",
            right_on="NORAD_CAT_ID",
            how="left",
        )
        .with_columns(
            pl.col("RCS").cast(float).alias("rcs"),
            pl.col("OPS_STATUS_CODE").cast(str).alias("ops_code"),
        )
        .drop(["RCS", "OPS_STATUS_CODE"])
    )

    return gpes


del pl
