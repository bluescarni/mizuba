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
    def _compare_sgp4(self, jd_begin, i, sat, rng, cfs, end_times):
        # Helper to compare the i-th step of a trajectory
        # with the output of the Python sgp4 propagator.
        # jd_begin is the start time of the polyjectory,
        # sat the Satrec object, rng the random engine,
        # cfs and end_times the poly coefficients and the
        # end times for the entire trajectory.
        import numpy as np

        # Pick 5 random times.
        step_begin = 0.0 if i == 0 else end_times[i - 1]
        step_end = end_times[i]
        random_times = rng.uniform(0, step_end - step_begin, (5,))

        xvals = np.polyval(cfs[i, 0, ::-1], random_times)
        yvals = np.polyval(cfs[i, 1, ::-1], random_times)
        zvals = np.polyval(cfs[i, 2, ::-1], random_times)
        vxvals = np.polyval(cfs[i, 3, ::-1], random_times)
        vyvals = np.polyval(cfs[i, 4, ::-1], random_times)
        vzvals = np.polyval(cfs[i, 5, ::-1], random_times)
        rvals = np.polyval(cfs[i, 6, ::-1], random_times)

        e, r, v = sat.sgp4_array(np.array([jd_begin] * 5), step_begin + random_times)

        self.assertTrue(np.all(e == 0))

        self.assertTrue(np.allclose(r[:, 0], xvals, rtol=0.0, atol=1e-8))
        self.assertTrue(np.allclose(r[:, 1], yvals, rtol=0.0, atol=1e-8))
        self.assertTrue(np.allclose(r[:, 2], zvals, rtol=0.0, atol=1e-8))
        self.assertTrue(
            np.allclose(np.linalg.norm(r, axis=1), rvals, rtol=0.0, atol=1e-8)
        )
        self.assertTrue(np.allclose(v[:, 0], vxvals, rtol=0.0, atol=1e-11))
        self.assertTrue(np.allclose(v[:, 1], vyvals, rtol=0.0, atol=1e-11))
        self.assertTrue(np.allclose(v[:, 2], vzvals, rtol=0.0, atol=1e-11))

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
        for i in range(len(end_times)):
            self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times)

    def test_multi_gpes(self):
        # Simple test with multiple GPEs.
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
            for i in range(len(end_times)):
                self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times)
