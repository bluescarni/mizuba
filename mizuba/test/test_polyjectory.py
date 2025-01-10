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


class polyjectory_test_case(_ut.TestCase):
    def test_basics(self):
        import numpy as np
        import sys
        from .. import polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        with self.assertRaises(ValueError) as cm:
            polyjectory([[]], [], [])
        self.assertTrue(
            "A trajectory array must have 3 dimensions, but instead 1 dimension(s) were"
            " detected" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[]]], [], [])
        self.assertTrue(
            "A trajectory array must have 3 dimensions, but instead 2 dimension(s) were"
            " detected" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[[]]]], [], [])
        self.assertTrue(
            "A trajectory array must have a size of 7 in the second dimension, but"
            " instead a size of 1 was detected" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[[]] * 7]], [1.0], [0])
        self.assertTrue(
            "A time array must have 1 dimension, but instead 0 dimension(s) were"
            " detected" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([[[[]] * 7]], [[1.0]], [[0]])
        self.assertTrue(
            "A status array must have 1 dimension, but instead 2 dimension(s) were"
            " detected" in str(cm.exception)
        )

        # Check with non-contiguous arrays.
        traj_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        state_data = np.array([[traj_data] * 7])[:, :, ::2]

        with self.assertRaises(ValueError) as cm:
            polyjectory([state_data], [[1.0]], [0])
        self.assertTrue(
            "All trajectory arrays must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        state_data = np.array([[traj_data] * 7])
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                [state_data, state_data], [np.array([1.0, 2.0, 3.0, 4.0])[::2]], [0]
            )
        self.assertTrue(
            "All time arrays must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                [state_data, state_data],
                [np.array([1.0, 2.0])],
                np.array([0, 0, 0, 0], dtype=np.int32)[::2],
            )
        self.assertTrue(
            "The status array must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        # Checks from C++.
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[],
                times=[np.array([1.0])],
                status=np.array([0, 1], dtype=np.int32),
            )
        self.assertTrue(
            "Cannot initialise a polyjectory object from an empty list of trajectories"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([1.0])],
                status=np.array([0, 1], dtype=np.int32),
            )
        self.assertTrue(
            "In the construction of a polyjectory, the number of objects deduced from"
            " the list of trajectories (2) is inconsistent with the number of objects"
            " deduced from the list of times (1)" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([1.0]), np.array([3.0])],
                status=np.array([0], dtype=np.int32),
            )
        self.assertTrue(
            "In the construction of a polyjectory, the number of objects deduced from"
            " the list of trajectories (2) is inconsistent with the number of objects"
            " deduced from the status list (1)" in str(cm.exception)
        )

        # Check trajectories without steps.
        pj = polyjectory(
            trajs=[state_data, state_data[1:]],
            times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
            status=np.array([0, 0], dtype=np.int32),
        )
        self.assertEqual(pj.maxT, 1.0)
        self.assertTrue(np.all(pj[0][0] == state_data))
        self.assertTrue(np.all(pj[0][1] == [0.0, 1.0]))
        self.assertTrue(np.all(pj[1][0] == np.zeros((0, 7, 8), dtype=float)))
        self.assertTrue(np.all(pj[1][1] == np.zeros((0,), dtype=float)))

        pj = polyjectory(
            trajs=[state_data[1:], state_data],
            times=[np.array([], dtype=float), np.array([0.0, 1.0])],
            status=np.array([0, 0], dtype=np.int32),
        )
        self.assertEqual(pj.maxT, 1.0)
        self.assertTrue(np.all(pj[0][0] == np.zeros((0, 7, 8), dtype=float)))
        self.assertTrue(np.all(pj[0][1] == np.zeros((0,), dtype=float)))
        self.assertTrue(np.all(pj[1][0] == state_data))
        self.assertTrue(np.all(pj[1][1] == [0.0, 1.0]))

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data[1:], state_data[1:]],
                times=[np.array([], dtype=float), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "All the trajectories in a polyjectory have a number of steps equal to"
            " zero: this is not allowed" in str(cm.exception)
        )

        short_state_data = np.array([[traj_data[:2]] * 7])
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[short_state_data, short_state_data],
                times=[np.array([1.0]), np.array([3.0])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "The trajectory polynomial order for the first object is less than 2 - this"
            " is not allowed" in str(cm.exception)
        )

        short_state_data = np.array([[traj_data[:5]] * 7])
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, short_state_data],
                times=[np.array([1.0]), np.array([3.0])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "The trajectory polynomial order for the object at index 1 is inconsistent"
            " with the polynomial order deduced from the first object (7)"
            in str(cm.exception)
        )

        # Inconsistencies between traj and time data sizes.
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([1.0]), np.array([1.0, 3.0])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "The number of steps for the trajectory of the object at index 0 is 1, but"
            " the number of times is 1 (the number of times must be equal to the number of steps + 1)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([1.0, 2.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "The trajectory of the object at index 1 has a nonzero number of steps but no associated time data"
            in str(cm.exception)
        )

        inf_state_data = np.array([[traj_data] * 7])
        inf_state_data[0, 0, 0] = float("inf")
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, inf_state_data],
                times=[np.array([0.0, 1.0]), np.array([0.0, 1.0])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "A non-finite value was found in the trajectory at index 1"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([0.0, 1.0]), np.array([1.0, float("nan")])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "A non-finite time coordinate was found for the object at index 1"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([-1.0, 0.0]), np.array([0.0, 1.0])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "A negative time coordinate was found for the object at index 0"
            in str(cm.exception)
        )

        two_state_data = np.array([[traj_data] * 7, [traj_data] * 7])
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[two_state_data, two_state_data],
                times=[np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0])],
                status=np.array([0, 0], dtype=np.int32),
            )
        self.assertTrue(
            "The sequence of times for the object at index 1 is not monotonically"
            " increasing" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                status=np.array([0, 0], dtype=np.int32),
                times=[np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.5])],
                trajs=[two_state_data, two_state_data],
            )
        self.assertTrue(
            "The sequence of times for the object at index 1 is not monotonically"
            " increasing" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([0.0, 1.0]), np.array([0.0, 3.0])],
                status=np.array([0, 1], dtype=np.int32),
                epoch=float("inf"),
            )
        self.assertTrue(
            "The initial epoch of a polyjectory must be finite, but instead a value of"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([0.0, 1.0]), np.array([0.0, 3.0])],
                status=np.array([0, 1], dtype=np.int32),
                epoch2=float("inf"),
            )
        self.assertTrue(
            "The second component of the initial epoch of a polyjectory must be finite, "
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data],
                times=[np.array([0.0, 1.0]), np.array([0.0, 3.0])],
                status=np.array([0, 1], dtype=np.int32),
                epoch=float("nan"),
            )
        self.assertTrue(
            "The initial epoch of a polyjectory must be finite, but instead a value of"
            in str(cm.exception)
        )

        # Test properties.
        pj = polyjectory(
            trajs=[state_data, state_data],
            times=[np.array([0.0, 1.0]), np.array([0.0, 3.0])],
            status=np.array([0, 1], dtype=np.int32),
            epoch=42.0,
            epoch2=1.0,
        )

        self.assertEqual(pj.nobjs, 2)
        self.assertEqual(pj.maxT, 3)
        self.assertEqual(pj.epoch, (43.0, 0.0))
        self.assertEqual(pj.poly_order, 7)

        rc = sys.getrefcount(pj)

        traj, time, status = pj[0]
        self.assertEqual(sys.getrefcount(pj), rc + 2)
        self.assertTrue(np.all(traj == traj_data))
        self.assertTrue(np.all(time == np.array([0.0, 1.0])))
        self.assertEqual(status, 0)

        with self.assertRaises(ValueError) as cm:
            traj[:] = 0

        with self.assertRaises(ValueError) as cm:
            time[:] = 0

        self.assertTrue(np.all(traj == traj_data))
        self.assertTrue(np.all(time == np.array([0.0, 1.0])))

        traj, time, status = pj[1]
        self.assertTrue(np.all(traj == traj_data))
        self.assertTrue(np.all(time == np.array([0.0, 3.0])))
        self.assertEqual(status, 1)

        with self.assertRaises(IndexError) as cm:
            pj[2]

    def test_bug_traj_init(self):
        # This is a test about a bug in the implementation
        # of the polyjectory constructor where we would
        # read from dangling memory due to returning pointers
        # to temporary arrays,
        import numpy as np
        from .. import polyjectory

        tdata0 = np.zeros((7, 6))
        tdata0[0, 0] = 1.0
        tdata1 = np.zeros((7, 6))
        tdata1[0, 0] = -1.0

        tdata2 = np.zeros((7, 6))
        tdata2[1, 0] = 1.0
        tdata3 = np.zeros((7, 6))
        tdata3[1, 0] = -1.0

        tdata4 = np.zeros((7, 6))
        tdata4[2, 0] = 1.0
        tdata5 = np.zeros((7, 6))
        tdata5[2, 0] = -1.0

        tdata6 = np.zeros((7, 6))

        tdata7 = np.zeros((7, 6))
        tdata7[:, 0] = 1

        # NOTE: the first 10 objects will have traj
        # data only for the first step, not the second.
        pj = polyjectory(
            [
                [tdata0] * 2,
                [tdata1] * 2,
                [tdata2] * 2,
                [tdata3] * 2,
                [tdata4] * 2,
                [tdata5] * 2,
                [tdata6] * 2,
                [tdata7] * 2,
            ],
            [[0.0, 1.0, 2.0]] * 8,
            [0] * 8,
        )

        self.assertEqual(pj.epoch[0], 0.0)

        self.assertTrue(np.all(pj[0][0][0] == tdata0))
        self.assertTrue(np.all(pj[0][0][1] == tdata0))

        self.assertTrue(np.all(pj[1][0][0] == tdata1))
        self.assertTrue(np.all(pj[1][0][1] == tdata1))

        self.assertTrue(np.all(pj[2][0][0] == tdata2))
        self.assertTrue(np.all(pj[2][0][1] == tdata2))

        self.assertTrue(np.all(pj[3][0][0] == tdata3))
        self.assertTrue(np.all(pj[3][0][1] == tdata3))

        self.assertTrue(np.all(pj[4][0][0] == tdata4))
        self.assertTrue(np.all(pj[4][0][1] == tdata4))

        self.assertTrue(np.all(pj[5][0][0] == tdata5))
        self.assertTrue(np.all(pj[5][0][1] == tdata5))

        self.assertTrue(np.all(pj[6][0][0] == tdata6))
        self.assertTrue(np.all(pj[6][0][1] == tdata6))

        self.assertTrue(np.all(pj[7][0][0] == tdata7))
        self.assertTrue(np.all(pj[7][0][1] == tdata7))

    def test_eval(self):
        import numpy as np
        from .. import polyjectory

        # NOTE: we use the rectilinear trajectories from
        # the conjunctions test for testing evaluations
        # (see test_cd_begin_end()).
        tm_data = np.array(
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
            ]
        )

        tm_data_0 = tm_data[:10]
        tm_data_1 = tm_data[11:]

        traj_data_0 = []
        for tm in tm_data_0[1:]:
            tdata = np.zeros((7, 4))
            tdata[0, 0] = 1.0 - (tm - 0.1)
            tdata[0, 1] = -1.0
            tdata[3, 0] = -1.0

            traj_data_0.append(tdata)

        traj_data_1 = []
        for tm in tm_data_1[1:]:
            tdata = np.zeros((7, 4))
            tdata[0, 0] = -(1.0 - (tm - 0.1))
            tdata[0, 1] = 1.0
            tdata[3, 0] = 1.0

            traj_data_1.append(tdata)

        pj = polyjectory([traj_data_0, traj_data_1], [tm_data_0, tm_data_1], [0, 0])

        # Error checking first.
        with self.assertRaises(ValueError) as cm:
            pj(np.zeros((5,))[::2])
        self.assertTrue(
            "The time array passed to the call operator of a polyjectory must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj(np.zeros((5, 5)))
        self.assertTrue(
            "The time array passed to the call operator of a polyjectory must have 1 dimension, but the number of dimensions is 2 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj(np.zeros((5,)))
        self.assertTrue(
            "Invalid input array passed to the call operator of a polyjectory: the number of objects is 2 but the size of the array is 5 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj(np.full((2,), float("inf")))
        self.assertTrue("An non-finite evaluation time of " in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            pj(float("nan"))
        self.assertTrue("An non-finite evaluation time of " in str(cm.exception))

        # NOTE: the first trajectory ends at 0.9, the second trajectory
        # begins at 1.1.
        tm = np.array([0.1, 0.1])
        res = pj(time=tm)
        self.assertTrue(
            np.allclose(
                res[0], [0.9, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))

        tm = np.array([1.2, 1.2])
        res = pj(time=tm)
        self.assertTrue(np.all(np.isnan(res[0])))
        self.assertTrue(
            np.allclose(
                res[1], [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )

        tm = np.array([0.1, 1.2])
        res = pj(time=tm)
        self.assertTrue(
            np.allclose(
                res[0], [0.9, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(
            np.allclose(
                res[1], [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )

        # Scalar overloads too.
        res = pj(time=0.1)
        self.assertTrue(
            np.allclose(
                res[0], [0.9, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))

        res = pj(time=1.2)
        self.assertTrue(np.all(np.isnan(res[0])))
        self.assertTrue(
            np.allclose(
                res[1], [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
