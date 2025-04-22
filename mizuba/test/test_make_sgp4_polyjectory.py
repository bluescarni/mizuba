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

# TLEs of an object whose orbital radius goes
# over 8000km.
_s_8000 = "1 00011U 59001A   24187.51496924  .00001069  00000-0  55482-3 0  9992"
_t_8000 = "2 00011  32.8711 255.0638 1455653 332.1888  20.7734 11.88503118450690"

# TLEs of an object which eventually decays.
_s_dec = "1 04206U 69082BV  24187.08533867  .00584698  00000-0  52886-2 0  9990"
_t_dec = "2 04206  69.8949  69.3024 0029370 203.3165 156.6698 15.65658911882875"


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

        xvals = np.polyval(cfs[i - 1, ::-1, 0], random_times)
        yvals = np.polyval(cfs[i - 1, ::-1, 1], random_times)
        zvals = np.polyval(cfs[i - 1, ::-1, 2], random_times)
        vxvals = np.polyval(cfs[i - 1, ::-1, 3], random_times)
        vyvals = np.polyval(cfs[i - 1, ::-1, 4], random_times)
        vzvals = np.polyval(cfs[i - 1, ::-1, 5], random_times)
        rvals = np.polyval(cfs[i - 1, ::-1, 6], random_times)

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
        from .._dl_utils import _eft_add_knuth
        import pathlib
        import polars as pl
        import numpy as np
        from astropy.time import Time
        from .._sgp4_polyjectory import _make_satrec_from_dict

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe.parquet")

        # Build the polyjectory.
        jd_begin = 2460669.0
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)[0]

        # Check that the initial time of the trajectory is exactly zero.
        self.assertEqual(pj[0][1][0], 0.0)

        # Build the satrec.
        sat = _make_satrec_from_dict(gpes.row(0, named=True))

        # Iterate over the trajectory steps, sampling randomly,
        # evaluating the polynomials and comparing with the
        # sgp4 python module.
        cfs, end_times, _ = pj[0]
        for i in range(1, len(end_times)):
            self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times)

        # Check also the conversion of the polyjectory epoch to TAI.
        tm = Time(val=jd_begin, format="jd", scale="utc").tai
        jd1, jd2 = _eft_add_knuth(tm.jd1, tm.jd2)
        self.assertEqual(jd1, pj.epoch[0])
        self.assertEqual(jd2, pj.epoch[1])

    def test_single_gpe_ds(self):
        # Simple test with a single deep-space gpe.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        import polars as pl
        from .._sgp4_polyjectory import _make_satrec_from_dict
        import numpy as np

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe_ds.parquet")

        # Build the polyjectory.
        jd_begin = 2460669.0
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)[0]

        # Check that the initial time of the trajectory is exactly zero.
        self.assertEqual(pj[0][1][0], 0.0)

        # Build the satrec.
        sat = _make_satrec_from_dict(gpes.row(0, named=True))

        # Iterate over the trajectory steps, sampling randomly,
        # evaluating the polynomials and comparing with the
        # sgp4 python module.
        cfs, end_times, _ = pj[0]
        for i in range(1, len(end_times)):
            self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times, 1e-7, 1e-10)

    def test_single_gpe_syncom(self):
        # This is a test with a deep-space satellite that, for some
        # interpolation steps, exhibits a degraded interpolation accuracy
        # with respect to what we normally expect. This is
        # due to the sgp4 algorithm exhibiting occasional spikes in the
        # values of the time derivatives at orders 2 and higher which throw
        # off the interpolation algorithm.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        import polars as pl
        from .._sgp4_polyjectory import _make_satrec_from_dict
        import numpy as np

        # Deterministic seeding.
        rng = np.random.default_rng(24)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "syncom_gpe.parquet")

        # Build the polyjectory.
        jd_begin = 2460666.5
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 10)[0]

        # Check that the initial time of the trajectory is exactly zero.
        self.assertEqual(pj[0][1][0], 0.0)

        # Build the satrec.
        sat = _make_satrec_from_dict(gpes.row(0, named=True))

        # Iterate over the trajectory steps, sampling randomly,
        # evaluating the polynomials and comparing with the
        # sgp4 python module.
        cfs, end_times, _ = pj[0]
        for i in range(1, len(end_times)):
            self._compare_sgp4(jd_begin, i, sat, rng, cfs, end_times, 1e-6, 1e-10)

    def test_multi_gpes(self):
        # Simple test with multiple satellites,
        # one GPE per satellite.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        from .._sgp4_polyjectory import _make_satrec_from_dict
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
        pj, norad_ids = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)

        self.assertEqual(pj.n_objs, len(norad_ids))
        self.assertTrue((gpes["norad_id"].to_numpy() == norad_ids).all())

        # Check that the initial times of the trajectories are exactly zero.
        for _, tm, _ in pj:
            self.assertEqual(tm[0], 0.0)

        for sat_idx in range(len(gpes)):
            # Build the satrec.
            sat = _make_satrec_from_dict(gpes.row(sat_idx, named=True))

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
        from .._dl_utils import _dl_add
        from .._sgp4_polyjectory import _make_satrec_from_dict as make_satrec
        import pathlib
        import polars as pl
        import numpy as np
        import bisect

        # Deterministic seeding.
        rng = np.random.default_rng(123)

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "iss_gpes.parquet")

        # NOTE: we pick two sets of jd_begin/end dates,
        # so that we test both the case in which the date
        # range is a superset of the GPE epochs, and vice-versa.
        jd_ranges = [(2460667.0, 2460684.0), (2460672.0, 2460679.0)]

        for jd_begin, jd_end in jd_ranges:
            # Build the polyjectory.
            pj, norad_ids = make_sgp4_polyjectory(gpes, jd_begin, jd_end)

            self.assertEqual(len(norad_ids), 1)
            self.assertEqual(norad_ids[0], 25544)

            # Check that the initial time of the trajectory is exactly zero.
            self.assertEqual(pj[0][1][0], 0.0)

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
                mz_state = pj.state_eval(tm)

                # Compare.
                self.assertTrue(np.allclose(r, mz_state[0, :3], rtol=0, atol=1e-8))
                self.assertTrue(np.allclose(v, mz_state[0, 3:6], rtol=0, atol=1e-11))

        pj.hint_release()

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
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 10)[0]

        # Check that the initial time of the trajectory is exactly zero.
        self.assertEqual(pj[0][1][0], 0.0)

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

            xvals = np.polyval(cfs[i - 1, ::-1, 0], random_times)
            yvals = np.polyval(cfs[i - 1, ::-1, 1], random_times)
            zvals = np.polyval(cfs[i - 1, ::-1, 2], random_times)
            vxvals = np.polyval(cfs[i - 1, ::-1, 3], random_times)
            vyvals = np.polyval(cfs[i - 1, ::-1, 4], random_times)
            vzvals = np.polyval(cfs[i - 1, ::-1, 5], random_times)
            rvals = np.polyval(cfs[i - 1, ::-1, 6], random_times)

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

    def test_malformed_data(self):
        from .. import make_sgp4_polyjectory, gpe_dtype, _have_sgp4_deps
        import numpy as np

        # Wrong dimensionality for the array of gpes.
        arr = np.zeros((1, 1), dtype=gpe_dtype)
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0)[0]
        self.assertTrue(
            "The array of gpes passed to make_sgp4_polyjectory() must have 1 dimension, but the number of dimensions is 2 instead"
            in str(cm.exception)
        )

        arr = np.zeros((), dtype=gpe_dtype)
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0)[0]
        self.assertTrue(
            "The array of gpes passed to make_sgp4_polyjectory() must have 1 dimension, but the number of dimensions is 0 instead"
            in str(cm.exception)
        )

        # Non-contiguous array.
        arr = np.zeros((10,), dtype=gpe_dtype)
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr[::2], 0.0, 1.0)[0]
        self.assertTrue(
            "The array of gpes passed to make_sgp4_polyjectory() must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        # Empty array.
        arr = np.zeros((0,), dtype=gpe_dtype)
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0)[0]
        self.assertTrue(
            "make_sgp4_polyjectory() requires a non-empty array of GPEs in input"
            in str(cm.exception)
        )

        # Identical NORAD ids and epochs.
        arr = np.zeros((2,), dtype=gpe_dtype)
        arr["norad_id"][:] = 1
        arr["epoch_jd1"][:] = 123
        arr["epoch_jd2"][:] = 1
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0)[0]
        self.assertTrue(
            "Two GPEs with identical NORAD ID 1 and epoch (123, 1) were identified by make_sgp4_polyjectory() - this is not allowed"
            in str(cm.exception)
        )

        # Wrong ordering.
        arr = np.zeros((2,), dtype=gpe_dtype)
        arr["norad_id"][0] = 1
        arr["epoch_jd1"][0] = 123
        arr["epoch_jd2"][0] = 1
        arr["norad_id"][1] = 0
        arr["epoch_jd1"][1] = 123
        arr["epoch_jd2"][1] = 1
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0)[0]
        self.assertTrue(
            "The set of GPEs passed to make_sgp4_polyjectory() is not sorted correctly: it must be ordered first by NORAD ID and then by epoch"
            in str(cm.exception)
        )

        arr = np.zeros((2,), dtype=gpe_dtype)
        arr["norad_id"][0] = 1
        arr["epoch_jd1"][0] = 123
        arr["epoch_jd2"][0] = 1
        arr["norad_id"][1] = 1
        arr["epoch_jd1"][1] = 122
        arr["epoch_jd2"][1] = 1
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0)[0]
        self.assertTrue(
            "The set of GPEs passed to make_sgp4_polyjectory() is not sorted correctly: it must be ordered first by NORAD ID and then by epoch"
            in str(cm.exception)
        )

        # Invalid polyjectory dates.
        arr = np.zeros((2,), dtype=gpe_dtype)
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, float("inf"), 1.0)[0]
        self.assertTrue("Invalid Julian date interval " in str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 1.0, float("nan"))[0]
        self.assertTrue("Invalid Julian date interval " in str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 1.0, 1.0)[0]
        self.assertTrue("Invalid Julian date interval " in str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 2.0, 1.0)[0]
        self.assertTrue("Invalid Julian date interval " in str(cm.exception))

        # Invalid reentry/exit radiuses.
        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0, reentry_radius=float("nan"))[0]
        self.assertTrue("The reentry/exit radiuses cannot be NaN" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0, exit_radius=float("nan"))[0]
        self.assertTrue("The reentry/exit radiuses cannot be NaN" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0, reentry_radius=2, exit_radius=2)[0]
        self.assertTrue(
            "The reentry radius (2) must be less than the exit radius (2)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            make_sgp4_polyjectory(arr, 0.0, 1.0, reentry_radius=3, exit_radius=2)[0]
        self.assertTrue(
            "The reentry radius (3) must be less than the exit radius (2)"
            in str(cm.exception)
        )

        # Wrong type for the floating-point arguments.
        with self.assertRaises(TypeError) as cm:
            make_sgp4_polyjectory(arr, [], 1.0, reentry_radius=3, exit_radius=2)[0]
        self.assertTrue(
            "The jd_begin, jd_end, reentry_radius and exit_radius arguments to make_sgp4_polyjectory() must all be floats or ints"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            make_sgp4_polyjectory(arr, 0, 1.0, reentry_radius=3, exit_radius=[])[0]
        self.assertTrue(
            "The jd_begin, jd_end, reentry_radius and exit_radius arguments to make_sgp4_polyjectory() must all be floats or ints"
            in str(cm.exception)
        )

        # Wrong type of the gpes argument.
        with self.assertRaises(TypeError) as cm:
            make_sgp4_polyjectory(1.0, 0, 1.0)[0]
        self.assertTrue(
            "The 'gpes' argument to make_sgp4_polyjectory() must be either a polars dataframe, a NumPy array or a list of Satrec objects, but it is of type"
            in str(cm.exception)
        )

        # Wrong dtype for the gpes argument,
        with self.assertRaises(TypeError) as cm:
            make_sgp4_polyjectory(np.zeros((1,)), 0, 1.0)[0]
        self.assertTrue(
            "When passing the 'gpes' argument to make_sgp4_polyjectory() as a NumPy array, the dtype must be 'gpe_dtype', but it is '"
            in str(cm.exception)
        )

        if not _have_sgp4_deps():
            return

        # Wrong gpes list argument.
        with self.assertRaises(TypeError) as cm:
            make_sgp4_polyjectory([1, 2, 3], 0, 1.0)[0]
        self.assertTrue(
            "When passing the 'gpes' argument to make_sgp4_polyjectory() as a list, all elements of the list must be Satrec instances"
            in str(cm.exception)
        )

    def test_disc_gpe_01(self):
        # Test a GPE with known discontinuities.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        from .._sgp4_polyjectory import _make_satrec_from_dict as make_satrec
        import pathlib
        import polars as pl
        import numpy as np
        from astropy.time import Time

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "disc_gpe_01.parquet")

        # Create the satrec object.
        sat = make_satrec(gpes.row(0, named=True))

        # Setup the time.
        tm = Time("2025-01-12T12:00:00Z", format="isot", scale="utc")
        jd_begin = tm.jd1

        # Create the polyjectory.
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 5.0)[0]

        # Fetch the poly coefficients and the end times.
        cfs, end_times, _ = pj[0]
        for idx in range(1, len(end_times)):
            # Establish begin/end of the interpolation step.
            step_begin = end_times[idx - 1]
            step_end = end_times[idx]

            # Sample uniformly within.
            times = np.linspace(step_begin, np.nextafter(step_end, -1.0), 1000)

            # Evaluate the state with the sgp4 propagator.
            e, r, v = sat.sgp4_array(np.full((1000,), jd_begin), times)
            self.assertTrue(np.all(e == 0))

            # Evaluate the state using the polyjectory.
            pjs = pj.state_meval(time=times, obj_idx=0)

            # Evaluate the max positional error (in km).
            max_err = np.max(np.linalg.norm(r - pjs[0, :, :3], axis=1))

            # Compute the step duration (in seconds).
            duration = (step_end - step_begin) * 86400.0

            # NOTE: we want to check that high-error interpolation
            # has been isolated in a short step.
            if max_err > 1e-6:
                self.assertLess(duration, 5.0)

    def test_nz_status(self):
        # Test nonzero statuses at the beginning of the polyjectory.
        import pathlib
        import polars as pl
        from astropy.time import Time
        from .. import make_sgp4_polyjectory

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        try:
            gpes = pl.read_parquet(cur_dir / "full_catalog.parquet")
        except Exception:
            return

        # Setup the time.
        tm = Time("2025-01-12T12:00:00Z", format="isot", scale="utc")
        jd_begin = tm.jd1

        # Build the polyjectory, propagating only for 10 seconds.
        pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 10 / 86400.0)[0]

        # Check that all trajectories without data have a nonzero status.
        for cfs, times, status in pj:
            if times.shape[0] == 0:
                self.assertNotEqual(status, 0)

    def test_satrec_construction(self):
        # Test construction from a list of satrec objects.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        from .._sgp4_polyjectory import _make_satrec_from_dict
        import polars as pl
        import numpy as np

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe.parquet")

        # Build the first polyjectory.
        jd_begin = 2460669.0
        pj1 = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1)[0]

        # Build a satrec from the satellite.
        sat = _make_satrec_from_dict(gpes.row(0, named=True))

        # Build the second polyjectory.
        pj2 = make_sgp4_polyjectory([sat], jd_begin, jd_begin + 1)[0]

        # Fetch the content of the polyjectories.
        cfs1, times1, status1 = pj1[0]
        cfs2, times2, status2 = pj2[0]

        # Compare.
        self.assertEqual(status1, status2)
        self.assertTrue(np.all(cfs1 == cfs2))
        self.assertTrue(np.all(times1 == times2))

    def test_exit_decay(self):
        # Test exiting and decaying satellites.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        from sgp4.api import Satrec
        import numpy as np

        sat = Satrec.twoline2rv(_s_8000, _t_8000)
        sat_dec = Satrec.twoline2rv(_s_dec, _t_dec)
        pj = make_sgp4_polyjectory(
            [sat, sat_dec], 2460496.5 + 1.0 / 32, 2460496.5 + 7, exit_radius=8000.0
        )[0]
        self.assertTrue(np.all(pj.status == [12, 0]))

        sat = Satrec.twoline2rv(_s_dec, _t_dec)
        pj = make_sgp4_polyjectory([sat], 2460496.5 + 30.0, 2460496.5 + 30.0 + 7)[0]
        self.assertTrue(np.all(pj.status == [6]))

    def test_strack(self):
        # Tests with datasets from space-track.org.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        from sgp4.api import SatrecArray
        from .._sgp4_polyjectory import _make_satrec_from_dict

        # from sgp4.api import Satrec
        import polars as pl
        import numpy as np

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Total propagation time (4 hours).
        prop_time = 1 / 6.0

        # NOTE: these datasets contains GPEs of decayed
        # satellites that trigger the bisection limit.
        for fname, begin_jd in zip(
            ["strack_20240705.parquet", "strack_20240917.parquet"],
            [2460496.5, 2460569.5],
        ):
            # Load the test data.
            try:
                gpes = pl.read_parquet(cur_dir / fname)
            except Exception:
                return

            # Build the polyjectory.
            pj = make_sgp4_polyjectory(gpes, begin_jd, begin_jd + prop_time)[0]

            # Check the presence of the bisection limit error code.
            self.assertTrue(np.any(pj.status == 14))

            # Create the evaluation timespan.
            N_times = 10
            tspan = np.linspace(0.0, np.nextafter(prop_time, -1.0), N_times)

            # Create the satellite objects.
            sat_list = [_make_satrec_from_dict(_) for _ in gpes.iter_rows(named=True)]

            # Create the satrec array.
            sat_arr = SatrecArray(sat_list)

            # Evaluate with the sgp4 propagator.
            e, r, v = sat_arr.sgp4(np.full((N_times,), begin_jd), tspan)

            # Evaluate with the polyjectory.
            pj_state = pj.state_meval(tspan)

            # Compute the positional difference.
            diff = np.linalg.norm(pj_state[:, :, :3] - r, axis=2).reshape((-1,))

            # Filter out trajectories which errored out and compute the max err.
            max_err = np.max(diff[~np.isnan(diff)])

            self.assertLess(max_err, 1e-6)

    def test_custom_data_dir(self):
        # Simple test custom data dir.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        import tempfile
        import polars as pl

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe.parquet")

        # Build the polyjectory.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = (pathlib.Path(tmpdirname) / "data_dir").resolve()
            jd_begin = 2460669.0
            pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1, data_dir=data_dir)[
                0
            ]
            self.assertEqual(data_dir, pj.data_dir)

            # NOTE: ensure we remove the datafiles before the temp dir,
            # otherwise the test will fail on Windows.
            pj.detach()

    def test_persist(self):
        # Simple test with persistence.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        import tempfile
        import polars as pl

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe.parquet")

        # Build the polyjectory.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = (pathlib.Path(tmpdirname) / "data_dir").resolve()
            jd_begin = 2460669.0
            pj = make_sgp4_polyjectory(
                gpes, jd_begin, jd_begin + 1, data_dir=data_dir, persist=True
            )[0]
            self.assertEqual(data_dir, pj.data_dir)

            # NOTE: detach will *not* remove the data dirs since persist is active.
            pj.detach()

            # Check the data dir still exists.
            self.assertTrue(data_dir.is_dir())
            self.assertTrue(data_dir.exists())

    def test_tmpdir(self):
        # A test checking custom setting for tmpdir.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from .. import make_sgp4_polyjectory
        import pathlib
        import tempfile
        import polars as pl

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        gpes = pl.read_parquet(cur_dir / "single_gpe.parquet")

        # Build the polyjectory.
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = pathlib.Path(tmpdirname)

            # NOTE: this checks that the dir is empty.
            self.assertTrue(not any(tmpdir.iterdir()))

            jd_begin = 2460669.0
            pj = make_sgp4_polyjectory(gpes, jd_begin, jd_begin + 1, tmpdir=tmpdir)[0]

            self.assertTrue(any(tmpdir.iterdir()))

            pj.detach()

    def test_opsmode(self):
        # This is a test checking that we correctly set the
        # opsmode to 'a' when propagating with sgp4. The test case
        # comes from a SOCRATES conjunction that we missed due
        # to the mismatch in opsmode.
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        from astropy.time import Time
        from sgp4.api import Satrec
        from .. import make_sgp4_polyjectory
        import numpy as np

        # The tles of the two satellites.
        line1i = "1 25541U          25053.26376711  .00000000  00000-0  48065-2 0    03"
        line2i = "2 25541   7.2593   0.4196 7073454 171.4819 214.8548  2.39999221    04"

        line1j = "1 46172U          25055.07880679  .00000000  00000-0  17505-3 0    06"
        line2j = "2 46172  53.0525 196.2605 0001187  60.4679 299.6428 15.06411185    02"

        # The TCA reported by SOCRATES.
        tca_time = Time(
            "2025-02-26 14:23:41.901", format="iso", scale="utc", precision=9
        )

        # Init the satrec objects.
        # NOTE: the sgp4 python module by default uses the 'i' opsmode, but it
        # does not matter here as we do not use these satrecs for propagation.
        sat_i = Satrec.twoline2rv(line1i, line2i)
        sat_j = Satrec.twoline2rv(line1j, line2j)

        # Build a polyjectory.
        pj = make_sgp4_polyjectory([sat_i, sat_j], tca_time.jd - 1, tca_time.jd + 1)[0]

        # Evaluate the state at TCA.
        teval = (
            tca_time.tai
            - Time(val=pj.epoch[0], val2=pj.epoch[1], format="jd", scale="tai")
        ).to_value("d")
        st = pj.state_eval(teval)

        # Compute the distance.
        dist = np.linalg.norm(st[0, :3] - st[1, :3])

        self.assertAlmostEqual(dist, 4.8980344574062675, places=8)
