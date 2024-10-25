# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class boundary_conjunctions_test_case(_ut.TestCase):
    # This is a test to check that conjunctions at the time
    # boundaries of trajectory data are correctly detected.
    def test_incoming(self):
        # In this test, we have 2 objects initially
        # placed on the x axis at +-1. The two objects
        # move with uniform unitary speed towards the origin.
        # The conjunction threshold is set to 0.25,
        # but the simulation is stopped before the distance
        # minimum (which would be zero) is reached. The code
        # must report the conjunction even if a distance
        # minimum was not reached.
        from .. import polyjectory, conjunctions
        import numpy as np
        from copy import copy

        # The conjunction threshold.
        cthresh = 0.25

        # Time data for both trajectories.
        tm_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Construct the trajectory data for both objects.
        traj_data = [[], []]
        for tm in tm_data:
            tdata_0 = np.zeros((7, 4))
            tdata_0[0, 0] = 1.0 - (tm - 0.1)
            tdata_0[0, 1] = -1.0
            tdata_0[3, 0] = -1.0

            tdata_1 = -copy(tdata_0)

            traj_data[0].append(tdata_0)
            traj_data[1].append(tdata_1)

        # Construct the polyjectory.
        pj = polyjectory(traj_data, [tm_data, tm_data], [0, 0])

        # Run conjunction detection.
        cj = conjunctions(pj=pj, conj_thresh=cthresh, conj_det_interval=0.03)

        self.assertEqual(len(cj.conjunctions), 1)
        self.assertAlmostEqual(cj.conjunctions["tca"][0], 0.9, places=15)
        self.assertAlmostEqual(cj.conjunctions["dca"][0], 0.2, places=15)
        self.assertEqual(cj.conjunctions["i"][0], 0)
        self.assertEqual(cj.conjunctions["j"][0], 1)
        self.assertTrue(
            np.allclose(cj.conjunctions["ri"][0], [0.1, 0, 0], rtol=1e-15, atol=0.0)
        )
        self.assertTrue(
            np.allclose(cj.conjunctions["rj"][0], [-0.1, 0, 0], rtol=1e-15, atol=0.0)
        )
        self.assertTrue(
            np.allclose(cj.conjunctions["vi"][0], [-1, 0, 0], rtol=1e-15, atol=0.0)
        )
        self.assertTrue(
            np.allclose(cj.conjunctions["vj"][0], [1, 0, 0], rtol=1e-15, atol=0.0)
        )

    def test_outgoing(self):
        # Similar to the previous test, but this
        # time the objects start close and then
        # drift apart.
        from .. import polyjectory, conjunctions
        import numpy as np
        from copy import copy

        # The conjunction threshold.
        cthresh = 0.25

        # Time data for both trajectories.
        tm_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Construct the trajectory data for both objects.
        traj_data = [[], []]
        for tm in tm_data:
            tdata_0 = np.zeros((7, 4))
            tdata_0[0, 0] = 0.1 + (tm - 0.1)
            tdata_0[0, 1] = 1.0
            tdata_0[3, 0] = 1.0

            tdata_1 = -copy(tdata_0)

            traj_data[0].append(tdata_0)
            traj_data[1].append(tdata_1)

        # Construct the polyjectory.
        pj = polyjectory(traj_data, [tm_data, tm_data], [0, 0])

        # Run conjunction detection.
        cj = conjunctions(pj=pj, conj_thresh=cthresh, conj_det_interval=0.03)

        self.assertEqual(len(cj.conjunctions), 1)
        self.assertAlmostEqual(cj.conjunctions["tca"][0], 0.0, places=15)
        self.assertAlmostEqual(cj.conjunctions["dca"][0], 0.2, places=15)
        self.assertEqual(cj.conjunctions["i"][0], 0)
        self.assertEqual(cj.conjunctions["j"][0], 1)
        self.assertTrue(
            np.allclose(cj.conjunctions["ri"][0], [0.1, 0, 0], rtol=1e-15, atol=0.0)
        )
        self.assertTrue(
            np.allclose(cj.conjunctions["rj"][0], [-0.1, 0, 0], rtol=1e-15, atol=0.0)
        )
        self.assertTrue(
            np.allclose(cj.conjunctions["vi"][0], [1, 0, 0], rtol=1e-15, atol=0.0)
        )
        self.assertTrue(
            np.allclose(cj.conjunctions["vj"][0], [-1, 0, 0], rtol=1e-15, atol=0.0)
        )
