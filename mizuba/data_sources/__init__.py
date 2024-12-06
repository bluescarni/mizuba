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


del pl
