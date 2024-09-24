# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class conjunctions_test_case(_ut.TestCase):
    # Helper to verify that in a conjunctions instance
    # the computed aabbs are consistent with the positions
    # of the objects.
    def _verify_conj_aabbs(self, c, rng):
        import numpy as np

        pj = c.polyjectory

        # For every conjunction step, pick random times within,
        # evaluate the polyjectory at the corresponding times and
        # assert that the positions are within the aabbs.
        for cd_idx, end_time in enumerate(c.cd_end_times):
            begin_time = 0.0 if cd_idx == 0 else c.cd_end_times[cd_idx - 1]

            # Pick 5 random times.
            random_times = rng.uniform(begin_time, end_time, (5,))

            # Iterate over all objects.
            for obj_idx in range(pj.nobjs):
                # Fetch the polyjectory data for the current object.
                traj, traj_times, status = pj[obj_idx]

                # Fetch the AABB of the object.
                aabb = c.aabbs[cd_idx, obj_idx]

                # Iterate over the random times.
                for time in random_times:
                    # Look for the trajectory step which ends at
                    # or after the end time.
                    step_idx = np.searchsorted(traj_times, time)

                    # Skip the current time if there's no corresponding
                    # trajectory data.
                    if step_idx == len(traj_times):
                        continue

                    # Fetch the polynomials for all state variables
                    # in the trajectory step.
                    traj_polys = traj[step_idx]

                    # Compute the poly evaluation interval.
                    h = time - (0.0 if step_idx == 0 else traj_times[step_idx - 1])

                    # Evaluate the polynomials and check that
                    # the results fit in the aabb.
                    for coord_idx, aabb_idx in zip([0, 1, 2, 6], range(4)):
                        pval = np.polyval(traj_polys[coord_idx, ::-1], h)
                        self.assertGreater(pval, aabb[0][aabb_idx])
                        self.assertLess(pval, aabb[1][aabb_idx])

    def test_basics(self):
        from .. import conjunctions as conj, polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        # Test error handling on construction.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        with self.assertRaises(ValueError) as cm:
            conj(pj, conj_thresh=0.0, conj_det_interval=0.0)
        self.assertTrue(
            "The conjunction threshold must be finite and positive, but instead a value of"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            conj(pj, conj_thresh=float("inf"), conj_det_interval=0.0)
        self.assertTrue(
            "The conjunction threshold must be finite and positive, but instead a value of"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            conj(pj, conj_thresh=1.0, conj_det_interval=0.0)
        self.assertTrue(
            "The conjunction detection interval must be finite and positive,"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            conj(pj, conj_thresh=1.0, conj_det_interval=float("nan"))
        self.assertTrue(
            "The conjunction detection interval must be finite and positive,"
            in str(cm.exception)
        )

    def test_aabbs(self):
        import numpy as np
        from .. import conjunctions as conj, polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Single planar circular orbit case.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        # Run the test for several conjunction detection intervals.
        for conj_det_interval in [0.01, 0.1, 0.5, 2.0, 5.0, 7.0]:
            c = conj(pj, conj_thresh=0.1, conj_det_interval=conj_det_interval)

            # Check shapes.
            self.assertEqual(c.aabbs.shape[0], c.cd_end_times.shape[0])

            # The global aabbs must all coincide
            # exactly with the only object's aabbs.
            self.assertTrue(np.all(c.aabbs[:, 0] == c.aabbs[:, 1]))

            # In the z and r coordinates, all aabbs
            # should be of size circa 0.1 accounting for the
            # conjunction threshold.
            self.assertTrue(np.all(c.aabbs[:, 0, 0, 2] >= -0.05001))
            self.assertTrue(np.all(c.aabbs[:, 0, 1, 2] <= 0.05001))

            self.assertTrue(np.all(c.aabbs[:, 0, 0, 3] >= 1 - 0.05001))
            self.assertTrue(np.all(c.aabbs[:, 0, 1, 3] <= 1 + 0.05001))

            # Verify the aabbs.
            self._verify_conj_aabbs(c, rng)
