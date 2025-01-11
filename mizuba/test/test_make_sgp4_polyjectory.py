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

import unittest as _ut


class make_sgp4_polyjectory_test_case(_ut.TestCase):
    def _compare_sgp4(
        self, jd_begin, i, sat, rng, cfs, end_times, pos_atol=1e-8, vel_atol=1e-11
    ):
        # Helper to compare the i-th step of a trajectory
        # with the output of the Python sgp4 propagator.
        # jd_begin is the start time of the polyjectory,
        # sat the Satrec object, rng the random engine,
        # cfs and end_times the poly coefficients and the
        # end times for the entire trajectory.
        import numpy as np

        # Pick 5 random times.
        step_begin = end_times[i - 1]
        step_end = end_times[i]
        random_times = rng.uniform(0, step_end - step_begin, (5,))

        xvals = np.polyval(cfs[i - 1, 0, ::-1], random_times)
        yvals = np.polyval(cfs[i - 1, 1, ::-1], random_times)
        zvals = np.polyval(cfs[i - 1, 2, ::-1], random_times)
        vxvals = np.polyval(cfs[i - 1, 3, ::-1], random_times)
        vyvals = np.polyval(cfs[i - 1, 4, ::-1], random_times)
        vzvals = np.polyval(cfs[i - 1, 5, ::-1], random_times)
        rvals = np.polyval(cfs[i - 1, 6, ::-1], random_times)

        e, r, v = sat.sgp4_array(np.array([jd_begin] * 5), step_begin + random_times)

        self.assertTrue(np.all(e == 0))

        self.assertTrue(np.allclose(r[:, 0], xvals, rtol=0.0, atol=pos_atol))
        self.assertTrue(np.allclose(r[:, 1], yvals, rtol=0.0, atol=pos_atol))
        self.assertTrue(np.allclose(r[:, 2], zvals, rtol=0.0, atol=pos_atol))
        self.assertTrue(
            np.allclose(np.linalg.norm(r, axis=1), rvals, rtol=0.0, atol=pos_atol)
        )
        self.assertTrue(np.allclose(v[:, 0], vxvals, rtol=0.0, atol=vel_atol))
        self.assertTrue(np.allclose(v[:, 1], vyvals, rtol=0.0, atol=vel_atol))
        self.assertTrue(np.allclose(v[:, 2], vzvals, rtol=0.0, atol=vel_atol))

    def test_single_gpe(self):
        # Simple test with a single gpe.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        from sgp4.api import Satrec
        import polars as pl
        import numpy as np

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe.parquet")

        # Build the polyjectory.
        jd_begin = 2460669.0
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)

        # Build the satrec.
        s = gpes["tle_line1"][0]
        t = gpes["tle_line2"][0]
        sat = Satrec.twoline2rv(s, t)

        # Iterate over the trajectory steps, sampling randomly,
        # evaluating the polynomials and comparing with the
        # sgp4 python module.
        cfs, end_times, _ = pj[0]
        for i in range(1, len(end_times)):
            self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times)

    def test_single_gpe_ds(self):
        # Simple test with a single gpe.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        from sgp4.api import Satrec
        import polars as pl
        import numpy as np

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe_ds.parquet")

        # Build the polyjectory.
        jd_begin = 2460669.0
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)

        # Build the satrec.
        s = gpes["tle_line1"][0]
        t = gpes["tle_line2"][0]
        sat = Satrec.twoline2rv(s, t)

        # Iterate over the trajectory steps, sampling randomly,
        # evaluating the polynomials and comparing with the
        # sgp4 python module.
        cfs, end_times, _ = pj[0]
        for i in range(1, len(end_times)):
            self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times, 1e-7, 1e-10)

    def test_multi_gpes(self):
        # Simple test with multiple satellites,
        # one GPE per satellite.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        from sgp4.api import Satrec
        import polars as pl
        import numpy as np

        # Deterministic seeding.
        rng = np.random.default_rng(123)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "multi_gpes.parquet")

        # Build the polyjectory.
        jd_begin = 2460669.0
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)

        for sat_idx in range(len(gpes)):
            # Build the satrec.
            s = gpes["tle_line1"][sat_idx]
            t = gpes["tle_line2"][sat_idx]
            sat = Satrec.twoline2rv(s, t)

            cfs, end_times, _ = pj[sat_idx]
            for i in range(1, len(end_times)):
                self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times)

    def test_iss_gpes(self):
        # Test for a single satellite (ISS) with multiple GPEs.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        # NOTE: this test requires at least Python 3.10
        # in order to use the 'key' argument to the bisect functions.
        import sys

        if sys.version_info < (3, 10):
            return

        from .. import make_sgp4_polyjectory
        from ..data_sources._common import _eft_add_knuth
        import pathlib
        from sgp4.api import Satrec, WGS72
        import polars as pl
        import numpy as np
        import bisect

        # Double-length addition using error-free
        # transformations.
        def _dl_add(a_hi, a_lo, b_hi, b_lo):
            def _eft_add_dekker(a, b):
                x = a + b
                y = (a - x) + b

                return x, y

            x_hi, y_hi = _eft_add_knuth(a_hi, b_hi)
            x_lo, y_lo = _eft_add_knuth(a_lo, b_lo)

            u, v = _eft_add_dekker(x_hi, y_hi + x_lo)
            u, v = _eft_add_dekker(u, v + y_lo)

            return u, v

        # Small helper to construct a Satrec from a row
        # in the gpe dataset.
        def make_satrec(row):
            # NOTE: this is the baseline reference epoch
            # used by the C++ SGP4 code.
            jd_sub = 2433281.5

            sat = Satrec()
            sat.sgp4init(
                WGS72,
                "i",
                row["norad_id"],
                _dl_add(
                    # NOTE: we are assuming here that the two
                    # jd components are already normalised.
                    row["epoch_jd1"],
                    row["epoch_jd2"],
                    -jd_sub,
                    0.0,
                )[0],
                row["bstar"],
                0.0,
                0.0,
                row["e0"],
                row["omega0"],
                row["i0"],
                row["m0"],
                row["n0"],
                row["node0"],
            )

            return sat

        # Deterministic seeding.
        rng = np.random.default_rng(123)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "iss_gpes.parquet")

        # Build the polyjectory.
        # NOTE: we pick as jd_begin an epoch slightly earlier
        # than the epoch of the first GPE. Like this, we check
        # also a time range before the first GPE epoch.
        jd_begin = 2460667.0
        # Similarly, this is slightly after the epoch of the last GPE.
        jd_end = 2460684.0
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_end)

        # Create the satellites list.
        # NOTE: we manually attach the epochs from the dataframe
        # to each satellite because we cannot trust the sgp4 python module
        # to reconstruct the double-length epoch to full accuracy.
        sats = list(
            (row["epoch_jd1"], row["epoch_jd2"], make_satrec(row))
            for row in gpes.iter_rows(named=True)
        )

        # Pick several time points randomly between jd_begin and jd_end.
        # Time is measured in days since jd_begin.
        random_times = rng.uniform(0, jd_end - jd_begin, (100,))

        # The bisection key, this will extract the delta
        # (in days) between a satellite epoch and the jd_begin
        # of the polyjectory.
        def bisect_key(tup):
            # NOTE: we know that the double-length epochs from the
            # GPEs dataframe are already normalised.
            return _dl_add(tup[0], tup[1], -jd_begin, 0)[0]

        for tm in random_times:
            # Locate the first gpe in sats whose epoch
            # (measured relative to jd_begin) is *greater than* tm.
            idx = bisect.bisect_right(sats, tm, key=bisect_key)
            # Move backwards, if possible, to pick the previous gpe.
            idx -= int(idx != 0)

            # Accurately calculate the tsince by computing
            # "jd_begin + tm - gpe_epoch" in double-length.
            tsince = _dl_add(
                *_dl_add(jd_begin, 0.0, tm, 0.0),
                -sats[idx][0],
                -sats[idx][1],
            )[0]

            # Compute the state according to the sgp4 Python module.
            e, r, v = sats[idx][2].sgp4_tsince(tsince * 1440)
            self.assertEqual(e, 0)

            # Compute the state according to the polyjectory.
            mz_state = pj(tm)

            # Compare.
            self.assertTrue(np.allclose(r, mz_state[0, :3], rtol=0, atol=1e-8))
            self.assertTrue(np.allclose(v, mz_state[0, 3:6], rtol=0, atol=1e-11))

    def test_leap_seconds(self):
        # Test creation of a polyjectory over
        # a timespan including a leap second day.
        from .. import _have_sgp4_deps, _have_heyoka_deps

        if not _have_sgp4_deps() or not _have_heyoka_deps():
            return

        from sgp4.api import Satrec
        import polars as pl
        from .. import make_sgp4_polyjectory
        from heyoka.model import sgp4_propagator
        import numpy as np
        from astropy.time import Time

        s = "1 00045U 60007A   05363.79166667  .00000504  00000-0  14841-3 0  9992"
        t = "2 00045  66.6943  81.3521 0257384 317.3173  40.8180 14.34783636277898"

        sat = Satrec.twoline2rv(s, t)

        # Fetch the data from sat.
        n0 = [sat.no_kozai]
        e0 = [sat.ecco]
        i0 = [sat.inclo]
        node0 = [sat.nodeo]
        omega0 = [sat.argpo]
        m0 = [sat.mo]
        bstar = [sat.bstar]
        epoch_jd1 = [sat.jdsatepoch]
        epoch_jd2 = [sat.jdsatepochF]
        norad_id = [1234]

        gpes = pl.DataFrame(
            {
                "n0": n0,
                "e0": e0,
                "i0": i0,
                "node0": node0,
                "omega0": omega0,
                "m0": m0,
                "bstar": bstar,
                "epoch_jd1": epoch_jd1,
                "epoch_jd2": epoch_jd2,
                "norad_id": norad_id,
            }
        )

        # Build the polyjectory up to 10 days in the future,
        # well beyond year's end.
        jd_begin = sat.jdsatepoch + sat.jdsatepochF
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 10)

        # Build the heyoka propagator.
        prop = sgp4_propagator([sat])

        # Fetch the poly coefficients and the end times.
        cfs, end_times, _ = pj[0]

        # Deterministic seeding.
        rng = np.random.default_rng(420)

        for i in range(1, len(end_times)):
            # Pick 5 random times.
            step_begin = end_times[i - 1]
            step_end = end_times[i]
            random_times = rng.uniform(0, step_end - step_begin, (5,))

            xvals = np.polyval(cfs[i - 1, 0, ::-1], random_times)
            yvals = np.polyval(cfs[i - 1, 1, ::-1], random_times)
            zvals = np.polyval(cfs[i - 1, 2, ::-1], random_times)
            vxvals = np.polyval(cfs[i - 1, 3, ::-1], random_times)
            vyvals = np.polyval(cfs[i - 1, 4, ::-1], random_times)
            vzvals = np.polyval(cfs[i - 1, 5, ::-1], random_times)
            rvals = np.polyval(cfs[i - 1, 6, ::-1], random_times)

            # Convert the times to UTC Julian dates.
            utc_jds = Time(
                val=[pj.epoch[0]] * 5,
                val2=step_begin + random_times + [pj.epoch[1]] * 5,
                format="jd",
                scale="tai",
            ).utc

            # Evaluate with the heyoka propagator.
            dates = np.zeros((5, 1), dtype=prop.jdtype)
            dates[:, 0]["jd"] = utc_jds.jd1
            dates[:, 0]["frac"] = utc_jds.jd2

            res = prop(dates)

            self.assertTrue(np.allclose(res[:, 0, 0], xvals, rtol=0.0, atol=1e-8))
            self.assertTrue(np.allclose(res[:, 1, 0], yvals, rtol=0.0, atol=1e-8))
            self.assertTrue(np.allclose(res[:, 2, 0], zvals, rtol=0.0, atol=1e-8))

            self.assertTrue(np.allclose(res[:, 3, 0], vxvals, rtol=0.0, atol=1e-11))
            self.assertTrue(np.allclose(res[:, 4, 0], vyvals, rtol=0.0, atol=1e-11))
            self.assertTrue(np.allclose(res[:, 5, 0], vzvals, rtol=0.0, atol=1e-11))

            self.assertTrue(
                np.allclose(
                    np.linalg.norm(res[:, :3, 0], axis=1), rvals, rtol=0.0, atol=1e-8
                )
            )
