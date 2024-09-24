# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class polyjectory_test_case(_ut.TestCase):
    def test_basics(self):
        import numpy as np
        import sys
        from .. import polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        with self.assertRaises(ValueError) as cm:
            polyjectory([[]], [], [])
        self.assertTrue(
            "A trajectory array must have 3 dimensions, but instead 1 dimension(s) were detected"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[]]], [], [])
        self.assertTrue(
            "A trajectory array must have 3 dimensions, but instead 2 dimension(s) were detected"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[[]]]], [], [])
        self.assertTrue(
            "A trajectory array must have a size of 7 in the second dimension, but instead a size of 1 was detected"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[[]] * 7]], [1.0], [0])
        self.assertTrue(
            "A time array must have 1 dimension, but instead 0 dimension(s) were detected"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[[]] * 7]], [[1.0]], [[0]])
        self.assertTrue(
            "A status array must have 1 dimension, but instead 2 dimension(s) were detected"
            in str(cm.exception)
        )

        # Check with non-contiguous arrays.
        traj_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        state_data = np.array([[traj_data] * 7])[:,:,::2]

        with self.assertRaises(ValueError) as cm:
            polyjectory([state_data], [[1.0]], [0])
        print(cm.exception)
        self.assertTrue(
            "All trajectory arrays must be C contiguous and properly aligned"
            in str(cm.exception)
        )
