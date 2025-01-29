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

import unittest as _ut


class data_sources_test_case(_ut.TestCase):
    def test_download_all_gpes(self):
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from ..data_sources import download_all_gpes, gpes_schema
        from .._sgp4_polyjectory import _make_satrec_from_dict as make_satrec
        from sgp4.api import Satrec
        import numpy as np
        import polars as pl

        # Download the full GPE datasets with and without supgp data.
        gpes = download_all_gpes()
        gpes_no_supgp = download_all_gpes(with_supgp=False)

        # Verify that both are sorted.
        self.assertTrue(gpes.equals(gpes.sort(["norad_id", "epoch_jd1", "epoch_jd2"])))
        self.assertTrue(
            gpes_no_supgp.equals(
                gpes_no_supgp.sort(["norad_id", "epoch_jd1", "epoch_jd2"])
            )
        )

        # Verify the schemas.
        self.assertEqual(gpes.schema, gpes_schema)
        self.assertEqual(gpes_no_supgp.schema, gpes_schema)

        # The no_supgp data must have unique norad ids.
        self.assertTrue(gpes_no_supgp["norad_id"].is_unique().all())

        # The no_supgp data must have all null rms.
        self.assertTrue(gpes_no_supgp["rms"].is_null().all())

        # For the no_supgp data, we want to randomly pick a couple
        # of satellites and verify the consistency between the original
        # TLEs and the columns in the dataframe. We do this to check
        # that we performed proper unit and time conversions.
        sat_attrs = [
            "bstar",
            "inclo",
            "nodeo",
            "ecco",
            "argpo",
            "mo",
            "no_kozai",
            "jdsatepoch",
            "jdsatepochF",
        ]
        for row in gpes_no_supgp[0, 100, 1000, 10000].iter_rows(named=True):
            sat1 = make_satrec(row)
            sat2 = Satrec.twoline2rv(row["tle_line1"], row["tle_line2"])

            for attr in sat_attrs:
                self.assertAlmostEqual(
                    getattr(sat1, attr), getattr(sat2, attr), places=15
                )

        # For the supgp data, we want to make sure that
        # we reset to null the TLE fields.
        # NOTE: test is not great, as it relies on the assumption
        # of segmented ISS data in the supgp dataset.
        self.assertTrue(
            gpes.filter(pl.col("name").str.contains(r"ISS.*Segment"))["tle_line1"]
            .is_null()
            .all()
        )
        self.assertTrue(
            gpes.filter(pl.col("name").str.contains(r"ISS.*Segment"))["tle_line2"]
            .is_null()
            .all()
        )
