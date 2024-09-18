# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut

# NOTE: all the TLEs used for testing were donwloaded from space-track.org
# in July 2024:
# https://www.space-track.org/

# TLEs of an object whose orbital radius goes
# over 8000km.
_s_8000 = "1 00011U 59001A   24187.51496924  .00001069  00000-0  55482-3 0  9992"
_t_8000 = "2 00011  32.8711 255.0638 1455653 332.1888  20.7734 11.88503118450690"

# TLEs of an object which eventually decays.
_s_dec = "1 04206U 69082BV  24187.08533867  .00584698  00000-0  52886-2 0  9990"
_t_dec = "2 04206  69.8949  69.3024 0029370 203.3165 156.6698 15.65658911882875"


class sgp4_polyjectory_test_case(_ut.TestCase):
    def test_basics(self):
        try:
            from sgp4.api import Satrec
        except ImportError:
            return

        from .. import sgp4_polyjectory

        sat = Satrec.twoline2rv(_s_dec, _t_dec)

        # Input sanity checking.
        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([], float("inf"), 1.0)
        self.assertTrue(
            "The sgp4_polyjectory() function requires a non-empty list of satellites in input"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            sgp4_polyjectory([int], float("inf"), 1.0)
        self.assertTrue(
            "The sgp4_polyjectory() function requires in input a list of Satrec objects from the 'sgp4' module or EarthSatellite objects from the 'skyfield' module"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5, 2460496.5)
        self.assertTrue("Invalid Julian date interval" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5 + 0.1, 2460496.5)
        self.assertTrue("Invalid Julian date interval" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory(
                [sat], 2460496.5, 2460496.5 + 0.1, exit_radius=float("inf")
            )
        self.assertTrue("Invalid exit radius" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5, 2460496.5 + 0.1, exit_radius=-0.1)
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5, 2460496.5 + 0.1, exit_radius=0.0)
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory(
                [sat], 2460496.5, 2460496.5 + 0.1, reentry_radius=float("inf")
            )
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5, 2460496.5 + 0.1, reentry_radius=-0.1)
        self.assertTrue("Invalid reentry radius" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5, 2460496.5 + 0.1, reentry_radius=0.0)
        self.assertTrue("Invalid reentry radius" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory(
                [sat],
                2460496.5,
                2460496.5 + 0.1,
                reentry_radius=100.0,
                exit_radius=100.0,
            )
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory(
                [sat],
                2460496.5,
                2460496.5 + 0.1,
                reentry_radius=100.1,
                exit_radius=100.0,
            )
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

    def test_invalid_initial_states(self):
        try:
            from sgp4.api import Satrec
        except ImportError:
            return

        from .. import sgp4_polyjectory
        import numpy as np

        sat = Satrec.twoline2rv(_s_8000, _t_8000)
        sat_dec = Satrec.twoline2rv(_s_dec, _t_dec)
        pt, mask = sgp4_polyjectory(
            [sat, sat_dec], 2460496.5 + 1.0 / 32, 2460496.5 + 7, exit_radius=8000.0
        )
        self.assertTrue(np.all(mask == [False, True]))

        sat = Satrec.twoline2rv(_s_dec, _t_dec)
        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5 + 40.0, 2460496.5 + 40.0 + 7)
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

        sat = Satrec.twoline2rv(_s_dec, _t_dec)
        with self.assertRaises(ValueError) as cm:
            sgp4_polyjectory([sat], 2460496.5 + 30.0, 2460496.5 + 30.0 + 7)
        self.assertTrue(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
            in str(cm.exception)
        )

    def test_taylor_cfs(self):
        try:
            from skyfield.api import load
            from skyfield.iokit import parse_tle_file
        except ImportError:
            return

        from .. import sgp4_polyjectory
        import numpy as np
        from ._sgp4_test_data_202407 import sgp4_test_tle as sgp4_test_tle_202407
        from ._sgp4_test_data_202409 import sgp4_test_tle as sgp4_test_tle_202409

        def run_test(sgp4_test_tle, begin_jd):
            # Load the test TLEs.
            ts = load.timescale()
            sat_list = list(
                parse_tle_file(
                    (bytes(_, "ascii") for _ in sgp4_test_tle.split("\n")), ts
                )
            )

            # Use only some of the satellites.
            sat_list = sat_list[::20]

            # Build the polyjectory.
            pt, mask = sgp4_polyjectory(sat_list, begin_jd, begin_jd + 1)

            # Filter out the masked satellites from sat_list.
            sat_list = list(np.array(sat_list)[mask])

            # Check consistency between pt and sat_list.
            self.assertEqual(pt.nobjs, len(sat_list))

            # For each satellite, we compute the state
            # towards the end of the trajectory via Taylor
            # series evaluation, and we then compare it to the
            # state computed by the sgp4 propagator.
            for tdata, sat in zip(pt, sat_list):
                traj, tm, status = tdata
                mod = sat.model

                # Take the midpoint of the last step.
                last_mp = (tm[-2] + tm[-1]) / 2
                jd, fr = begin_jd, last_mp / 1440

                # Compute the state at last_mp via the sgp4 propagator.
                e, r, v = mod.sgp4(jd, fr)

                # The error code here should be 0 (all good)
                # or 6 (decay). The other error codes should not
                # show up here, because no trajectory data is recorded
                # in the polyjectory when they occur.
                self.assertTrue((e == 0 or e == 6))

                # We do not want to check the state vector
                # in case of reentry.
                if e == 6:
                    continue

                # Taylor series evaluation needs the time elapsed
                # since the beginning of the last timestep.
                dt = (tm[-1] - tm[-2]) / 2

                # Fetch the state vector and compare.
                # NOTE: need ::-1 because numpy wants polynomials from highest
                # power to lowest, but we store them in the opposite order.
                ts_x = traj[-1, 0, ::-1]
                self.assertAlmostEqual(np.polyval(ts_x, dt), r[0], delta=1e-8)
                ts_y = traj[-1, 1, ::-1]
                self.assertAlmostEqual(np.polyval(ts_y, dt), r[1], delta=1e-8)
                ts_z = traj[-1, 2, ::-1]
                self.assertAlmostEqual(np.polyval(ts_z, dt), r[2], delta=1e-8)

                ts_vx = traj[-1, 3, ::-1]
                self.assertAlmostEqual(np.polyval(ts_vx, dt), v[0], delta=1e-8)
                ts_vy = traj[-1, 4, ::-1]
                self.assertAlmostEqual(np.polyval(ts_vy, dt), v[1], delta=1e-8)
                ts_vz = traj[-1, 5, ::-1]
                self.assertAlmostEqual(np.polyval(ts_vz, dt), v[2], delta=1e-8)

        for sgp4_test_tle, begin_jd in zip(
            [sgp4_test_tle_202407, sgp4_test_tle_202409], [2460496.5, 2460569.5]
        ):
            run_test(sgp4_test_tle, begin_jd)
