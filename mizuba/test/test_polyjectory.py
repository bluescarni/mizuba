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


class polyjectory_test_case(_ut.TestCase):
    def test_basics(self):
        import numpy as np
        import sys
        from .. import polyjectory
        from pathlib import Path

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
            "A trajectory array must have a size of 7 in the third dimension, but"
            " instead a size of 0 was detected" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            polyjectory([np.zeros((1, 1, 7))], [1.0], [0])
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
        state_data = np.zeros((1, 8, 7))[:, ::2, :]

        with self.assertRaises(ValueError) as cm:
            polyjectory([state_data], [[1.0]], [0])
        self.assertTrue(
            "All trajectory arrays must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        state_data = np.zeros((1, 8, 7))
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
        self.assertTrue(np.all(pj[1][0] == np.zeros((0, 8, 7), dtype=float)))
        self.assertTrue(np.all(pj[1][1] == np.zeros((0,), dtype=float)))
        pj.hint_release()

        pj = polyjectory(
            trajs=[state_data[1:], state_data],
            times=[np.array([], dtype=float), np.array([0.0, 1.0])],
            status=np.array([0, 0], dtype=np.int32),
        )
        self.assertEqual(pj.maxT, 1.0)
        self.assertTrue(np.all(pj[0][0] == np.zeros((0, 8, 7), dtype=float)))
        self.assertTrue(np.all(pj[0][1] == np.zeros((0,), dtype=float)))
        self.assertTrue(np.all(pj[1][0] == state_data))
        self.assertTrue(np.all(pj[1][1] == [0.0, 1.0]))
        pj.hint_release()

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

        short_state_data = np.zeros((1, 2, 7))
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

        short_state_data = np.zeros((1, 4, 7))
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

        inf_state_data = np.zeros((1, 8, 7))
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

        two_state_data = np.zeros((2, 8, 7))
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

        self.assertEqual(pj.n_objs, 2)
        self.assertEqual(pj.maxT, 3)
        self.assertEqual(pj.epoch, (43.0, 0.0))
        self.assertEqual(pj.poly_order, 7)
        self.assertTrue(isinstance(pj.data_dir, Path))
        self.assertFalse(pj.persist)

        rc = sys.getrefcount(pj)

        traj, time, status = pj[0]
        self.assertEqual(sys.getrefcount(pj), rc + 2)
        self.assertTrue(np.all(traj == state_data))
        self.assertTrue(np.all(time == np.array([0.0, 1.0])))
        self.assertEqual(status, 0)

        with self.assertRaises(ValueError) as cm:
            traj[:] = 0

        with self.assertRaises(ValueError) as cm:
            time[:] = 0

        self.assertTrue(np.all(traj == state_data))
        self.assertTrue(np.all(time == np.array([0.0, 1.0])))

        traj, time, status = pj[1]
        self.assertTrue(np.all(traj == state_data))
        self.assertTrue(np.all(time == np.array([0.0, 3.0])))
        self.assertEqual(status, 1)

        with self.assertRaises(IndexError) as cm:
            pj[2]

    def test_from_datafiles(self):
        import tempfile
        from .. import polyjectory
        import numpy as np
        from pathlib import Path

        # NOTE: several error handling tests which do not
        # involve moving files.
        with (
            tempfile.NamedTemporaryFile() as traj_file,
            tempfile.NamedTemporaryFile() as time_file,
        ):
            for order in [0, 1]:
                with self.assertRaises(ValueError) as cm:
                    polyjectory.from_datafiles(
                        traj_file=traj_file.name,
                        time_file=time_file.name,
                        order=order,
                        traj_offsets=[],
                        status=[],
                    )
                self.assertTrue(
                    f"Invalid polynomial order {order} specified during the construction of a polyjectory"
                    in str(cm.exception)
                )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=[],
                    status=[],
                )
            self.assertTrue(
                "Invalid trajectory offsets vector passed to the constructor of a polyjectory: the vector cannot be empty"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.zeros((1, 1), dtype=polyjectory.traj_offset),
                    status=[],
                )
            self.assertTrue(
                "The array of trajectory offsets passed to 'from_datafiles()' must have 1 dimension, but the number of dimensions is 2 instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.zeros((1,), dtype=polyjectory.traj_offset),
                    status=[],
                )
            self.assertTrue(
                "Invalid status vector passed to the constructor of a polyjectory: the expected size is 1, but the actual size is 0"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.zeros((1,), dtype=polyjectory.traj_offset),
                    status=[1],
                    epoch=float("inf"),
                )
            self.assertTrue(
                "The initial epoch of a polyjectory must be finite, but instead a value of"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.zeros((1,), dtype=polyjectory.traj_offset),
                    status=[1],
                    epoch=1.0,
                    epoch2=float("inf"),
                )
            self.assertTrue(
                "The second component of the initial epoch of a polyjectory must be finite, but instead a value of"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=traj_file.name,
                    order=2,
                    traj_offsets=np.zeros((1,), dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "Invalid data file(s) passed to the constructor of a polyjectory: the trajectory data file and the time data file are the same file"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(1, 1)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "Invalid trajectory offsets vector passed to the constructor of a polyjectory: the initial offset value must be zero but it is 1 instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array(
                        [(0, 1), (21, 1), (0, 2)], dtype=polyjectory.traj_offset
                    ),
                    status=[1, 3, 4],
                )
            self.assertTrue(
                "Invalid trajectory offsets vector passed to the constructor of a polyjectory: the offset of the object at index 2 is less than the offset of the previous object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array(
                        [(0, 1), (2, 1)], dtype=polyjectory.traj_offset
                    ),
                    status=[1, 3],
                )
            self.assertTrue(
                "Inconsistent data detected in the trajectory offsets vector passed to the constructor of a polyjectory: the offset of the object at index 1 is inconsistent with the offset and number of steps of the previous object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.zeros((1,), dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "All the trajectories in a polyjectory have a number of steps equal to zero: this is not allowed"
                in str(cm.exception)
            )

        # The following error checking tests do involve
        # moving files, so we have a bit more complicated setup.
        tmpdir1 = tempfile.TemporaryDirectory()
        tmpdir2 = tempfile.TemporaryDirectory()
        time_file = open(Path(tmpdir1.name) / "time", "wb")
        time_file.close()
        with self.assertRaises(ValueError) as cm:
            polyjectory.from_datafiles(
                traj_file=tmpdir2.name,
                time_file=Path(tmpdir1.name) / "time",
                order=2,
                traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                status=[1],
            )
        self.assertTrue(
            "Invalid trajectory data file passed to the constructor of a polyjectory: the file is not a regular file"
            in str(cm.exception)
        )
        tmpdir1.cleanup()
        tmpdir2.cleanup()

        tmpdir1 = tempfile.TemporaryDirectory()
        tmpdir2 = tempfile.TemporaryDirectory()
        traj_file = open(Path(tmpdir1.name) / "traj", "wb")
        traj_file.close()
        with self.assertRaises(ValueError) as cm:
            polyjectory.from_datafiles(
                traj_file=Path(tmpdir1.name) / "traj",
                time_file=tmpdir2.name,
                order=2,
                traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                status=[1],
            )
        self.assertTrue(
            "Invalid time data file passed to the constructor of a polyjectory: the file is not a regular file"
            in str(cm.exception)
        )
        tmpdir1.cleanup()
        tmpdir2.cleanup()

        with tempfile.TemporaryDirectory() as tmpdirname:
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.zeros((20,), dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            np.zeros((2,), dtype=float).tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "Invalid trajectory data file passed to the constructor of a polyjectory: the expected size in bytes is"
                in str(cm.exception)
            )

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.zeros((21,), dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            np.zeros((1,), dtype=float).tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "Invalid time data file passed to the constructor of a polyjectory: the expected size in bytes is"
                in str(cm.exception)
            )

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((21,), float("inf"), dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            np.zeros((2,), dtype=float).tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "A non-finite value was found in the trajectory at index 0"
                in str(cm.exception)
            )

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((21,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            np.full((2,), float("nan"), dtype=float).tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "A non-finite time coordinate was found for the object at index 0"
                in str(cm.exception)
            )

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((21,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            np.full((2,), -1.0, dtype=float).tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 1)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "A negative time coordinate was found for the object at index 0"
                in str(cm.exception)
            )

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.0])
            time_data.tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "The sequence of times for the object at index 0 is not monotonically increasing"
                in str(cm.exception)
            )

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 0.9])
            time_data.tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                    status=[1],
                )
            self.assertTrue(
                "The sequence of times for the object at index 0 is not monotonically increasing"
                in str(cm.exception)
            )

            # A simple check with successful construction.
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            pj = polyjectory.from_datafiles(
                traj_file=traj_file.name,
                time_file=time_file.name,
                order=2,
                traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                status=[1],
            )

            self.assertEqual(pj.maxT, 1.1)

    def test_bug_traj_init(self):
        # This is a test about a bug in the implementation
        # of the polyjectory constructor where we would
        # read from dangling memory due to returning pointers
        # to temporary arrays,
        import numpy as np
        from .. import polyjectory

        tdata0 = np.zeros((6, 7))
        tdata0[0, 0] = 1.0
        tdata1 = np.zeros((6, 7))
        tdata1[0, 0] = -1.0

        tdata2 = np.zeros((6, 7))
        tdata2[0, 1] = 1.0
        tdata3 = np.zeros((6, 7))
        tdata3[0, 1] = -1.0

        tdata4 = np.zeros((6, 7))
        tdata4[0, 2] = 1.0
        tdata5 = np.zeros((6, 7))
        tdata5[0, 2] = -1.0

        tdata6 = np.zeros((6, 7))

        tdata7 = np.zeros((6, 7))
        tdata7[0, :] = 1

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
        traj_data_0 = np.ascontiguousarray(np.array(traj_data_0).transpose((0, 2, 1)))

        traj_data_1 = []
        for tm in tm_data_1[1:]:
            tdata = np.zeros((7, 4))
            tdata[0, 0] = -(1.0 - (tm - 0.1))
            tdata[0, 1] = 1.0
            tdata[3, 0] = 1.0

            traj_data_1.append(tdata)
        traj_data_1 = np.ascontiguousarray(np.array(traj_data_1).transpose((0, 2, 1)))

        pj = polyjectory([traj_data_0, traj_data_1], [tm_data_0, tm_data_1], [0, 0])

        # Error checking first.
        with self.assertRaises(ValueError) as cm:
            pj.state_eval(np.zeros((5,))[::2])
        self.assertTrue(
            "The time array passed to state_eval() must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(np.zeros((5, 5)))
        self.assertTrue(
            "The time array passed to state_eval() must have 1 dimension, but the number of dimensions is 2 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=np.zeros((5,)))
        self.assertTrue(
            "Invalid time array passed to state_eval(): the number of objects is 2 but the size of the array is 5 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=np.full((2,), float("inf")))
        self.assertTrue("An non-finite evaluation time of " in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(float("nan"))
        self.assertTrue("An non-finite evaluation time of " in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=np.array([0.11, 0.11]), obj_idx=1)
        self.assertTrue(
            "Invalid time array passed to state_eval(): the number of selected objects is 1 but the size of the array is 2 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(IndexError) as cm:
            pj.state_eval(time=0.11, obj_idx=2)
        self.assertTrue(
            "Invalid object index 2 specified - the total number of objects in the polyjectory is only 2"
            in str(cm.exception)
        )

        with self.assertRaises(IndexError) as cm:
            pj.state_eval(time=0.11, obj_idx=[45])
        self.assertTrue(
            "Invalid object index 45 specified - the total number of objects in the polyjectory is only 2"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(
                time=0.11, obj_idx=np.array([0, 0, 0, 0], dtype=np.uintp)[::2]
            )
        self.assertTrue(
            "The selector array passed to state_eval() must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=0.11, obj_idx=np.zeros((2, 2)))
        self.assertTrue(
            "The selector array passed to state_eval() must have 1 dimension, but the number of dimensions is 2 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=0.11, out=np.array([1.0, 2.0, 3.0])[::2])
        self.assertTrue(
            "The output array passed to state_eval() must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=0.11, out=np.full((1,), 1.0))
        self.assertTrue(
            "The output array passed to state_eval() must have 2 dimensions, but the number of dimensions is 1 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=0.11, out=np.full((1, 6), 1.0))
        self.assertTrue(
            "The output array passed to state_eval() must have a size of 7 in the second dimension, but the size in the second dimension is 6 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=0.11, out=np.full((3, 7), 1.0))
        self.assertTrue(
            "Invalid output array passed to state_eval(): the number of objects is 2 but the size of the first dimension of the array is 3 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=0.11, out=np.full((3, 7), 1.0), obj_idx=[1])
        self.assertTrue(
            "Invalid output array passed to state_eval(): the number of objects selected for evaluation is 1 but the size of the first dimension of the array is 3 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=[0.11, 0.11], out=np.full((3, 7), 1.0))
        self.assertTrue(
            "Invalid output array passed to state_eval(): the number of objects is 2 but the size of the first dimension of the array is 3 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=[0.11, 0.11], out=np.full((3, 7), 1.0), obj_idx=[1])
        self.assertTrue(
            "Invalid output array passed to state_eval(): the number of objects selected for evaluation is 1 but the size of the first dimension of the array is 3 (the two numbers must be equal)"
            in str(cm.exception)
        )

        out = np.zeros((2, 7))
        tm = out[:2]
        with self.assertRaises(ValueError) as cm:
            pj.state_eval(time=tm, out=out)
        self.assertTrue(
            "Potential memory overlap detected between the output array passed to state_eval() and the time array"
            in str(cm.exception)
        )

        if np.dtype(np.uintp).itemsize == np.dtype(np.float64).itemsize:
            with self.assertRaises(ValueError) as cm:
                pj.state_eval(
                    time=[1.1, 1.1], out=out, obj_idx=out[0, :2].view(dtype=np.uintp)
                )
            self.assertTrue(
                "Potential memory overlap detected between the output array passed to state_eval() and the array of object indices"
                in str(cm.exception)
            )

        # For state_meval() too.
        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=np.array([0.11, 0.11, 0.11, 0.11])[::2])
        self.assertTrue(
            "The time array passed to state_meval() must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=np.zeros((3, 3, 3)))
        self.assertTrue(
            "The time array passed to state_meval() must have either 1 or 2 dimensions, but the number of dimensions is 3 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[0.11, 0.11], out=np.array([1.0, 2.0, 0.3, 0.4])[::2])
        self.assertTrue(
            "The output array passed to state_meval() must be C contiguous and properly aligned"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[0.11, 0.11], out=np.array([1.0, 2.0, 0.3, 0.4]))
        self.assertTrue(
            "The output array passed to state_meval() must have 3 dimensions, but the number of dimensions is 1 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[0.11, 0.11], out=np.zeros((3, 4, 5)))
        self.assertTrue(
            "The output array passed to state_meval() must have a size of 7 in the third dimension, but the size in the third dimension is 5 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[0.11, 0.11], out=np.zeros((3, 4, 7)))
        self.assertTrue(
            "Invalid output array passed to state_meval(): the number of time evaluations per object is 2 but the size of the second dimension of the array is 4 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[0.11, 0.11], out=np.zeros((2, 2, 7)), obj_idx=[1])
        self.assertTrue(
            "Invalid output array passed to state_meval(): the number of selected objects is 1 but the size of the first dimension of the array is 2 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[[0.11, 0.11]], out=np.zeros((2, 2, 7)), obj_idx=[1])
        self.assertTrue(
            "Invalid output array passed to state_meval(): the number of selected objects is 1 but the size of the first dimension of the array is 2 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(
                time=[[0.11, 0.11], [0.11, 0.11]], out=np.zeros((1, 2, 7)), obj_idx=[1]
            )
        self.assertTrue(
            "Invalid time array passed to state_meval(): the number of selected objects is 1 but the size of the first dimension of the array is 2 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[0.11, 0.11], out=np.zeros((3, 2, 7)))
        self.assertTrue(
            "Invalid output array passed to state_meval(): the number of objects is 2 but the size of the first dimension of the array is 3 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[[0.11, 0.11]], out=np.zeros((3, 2, 7)))
        self.assertTrue(
            "Invalid output array passed to state_meval(): the number of objects is 2 but the size of the first dimension of the array is 3 (the two numbers must be equal)"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=[[0.11, 0.11]], out=np.zeros((2, 2, 7)))
        self.assertTrue(
            "Invalid time array passed to state_meval(): the number of objects is 2 but the size of the first dimension of the array is 1 (the two numbers must be equal)"
            in str(cm.exception)
        )

        out = np.zeros((2, 2, 7))
        tm = out[0, 0, :2]
        with self.assertRaises(ValueError) as cm:
            pj.state_meval(time=tm, out=out)
        self.assertTrue(
            "Potential memory overlap detected between the output array passed to state_meval() and the time array"
            in str(cm.exception)
        )

        if np.dtype(np.uintp).itemsize == np.dtype(np.float64).itemsize:
            with self.assertRaises(ValueError) as cm:
                pj.state_meval(
                    time=[1.1, 1.1], out=out, obj_idx=out[0, 0, :2].view(dtype=np.uintp)
                )
            self.assertTrue(
                "Potential memory overlap detected between the output array passed to state_meval() and the array of object indices"
                in str(cm.exception)
            )

        # NOTE: the first trajectory ends at 0.9, the second trajectory
        # begins at 1.1.
        tm = np.array([0.11, 0.11])
        res = pj.state_eval(time=tm)
        self.assertTrue(
            np.allclose(
                res[0], [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))

        tm = np.array([0.11, 0.11])
        out = np.zeros((2, 7))
        res = pj.state_eval(time=tm, out=out)
        self.assertTrue(
            np.allclose(
                res[0], [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))
        self.assertEqual(id(out), id(res))

        tm = np.array([0.11, 0.12])
        res = pj.state_meval(time=tm)
        self.assertTrue(
            np.allclose(
                res[0],
                [
                    [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.88, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                ],
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))

        tm = np.array([0.11, 0.12])
        out = np.zeros((2, 2, 7))
        res = pj.state_meval(out=out, time=tm)
        self.assertTrue(
            np.allclose(
                res[0],
                [
                    [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.88, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                ],
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))
        self.assertEqual(id(out), id(res))

        tm = np.array([[0.11], [0.12]])
        res = pj.state_meval(time=tm)
        self.assertTrue(
            np.allclose(
                res[0],
                [[0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]],
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))

        tm = np.array([0.11, 0.12] * 1000)
        res = pj.state_eval(time=tm, obj_idx=[0, 1] * 1000)
        self.assertTrue(
            np.allclose(
                res[::2],
                [[0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]] * 1000,
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1::2])))

        tm = np.array([0.11, 0.12] * 1000)
        res = pj.state_meval(time=tm, obj_idx=[0, 1] * 1000)
        self.assertTrue(
            np.allclose(
                res[::2],
                [
                    [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.88, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                ]
                * 1000,
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1::2])))

        tm = np.array([[0.11, 0.12] * 1000, [1.2, 1.21] * 1000])
        res = pj.state_meval(time=tm, obj_idx=[0, 1])
        self.assertTrue(
            np.allclose(
                res[0],
                [
                    [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.88, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                ]
                * 1000,
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(
            np.allclose(
                res[1],
                [
                    [0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.21, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
                * 1000,
                rtol=0.0,
                atol=1e-15,
            ),
        )

        tm = np.array([[0.11, 0.12] * 1000, [1.2, 1.21] * 1000] * 1000)
        res = pj.state_meval(time=tm, obj_idx=[0, 1] * 1000)
        self.assertTrue(
            np.allclose(
                res[::2],
                [
                    [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [0.88, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                ]
                * 1000,
                rtol=0.0,
                atol=1e-15,
            ),
        )
        self.assertTrue(
            np.allclose(
                res[1::2],
                [
                    [0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.21, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
                * 1000,
                rtol=0.0,
                atol=1e-15,
            ),
        )

        tm = np.array([1.2, 1.21])
        res = pj.state_eval(time=tm)
        self.assertTrue(np.all(np.isnan(res[0])))
        self.assertTrue(
            np.allclose(
                res[1], [0.21, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )

        tm = np.array([0.11, 1.21])
        res = pj.state_eval(time=tm)
        self.assertTrue(
            np.allclose(
                res[0], [0.89, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(
            np.allclose(
                res[1], [0.21, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )

        # Scalar overloads too.
        res = pj.state_eval(time=0.1)
        self.assertTrue(
            np.allclose(
                res[0], [0.9, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )
        self.assertTrue(np.all(np.isnan(res[1])))

        res = pj.state_eval(time=1.2)
        self.assertTrue(np.all(np.isnan(res[0])))
        self.assertTrue(
            np.allclose(
                res[1], [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-15
            ),
        )

        # Test single-object evaluations.
        res = pj.state_eval(time=0.1, obj_idx=0)
        self.assertEqual(res.shape, (1, 7))
        self.assertTrue(
            np.allclose(
                res, [[0.9, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]], rtol=0.0, atol=1e-15
            ),
        )

        res = pj.state_eval(time=1.2, obj_idx=1)
        self.assertTrue(
            np.allclose(
                res, [[0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rtol=0.0, atol=1e-15
            ),
        )

        # Also test with a trajectory without data.
        pj = polyjectory([traj_data_0, np.empty((0, 4, 7))], [tm_data_0, []], [0, 0])
        tm = np.array([0.1, 0.1])
        res = pj.state_eval(time=tm)
        self.assertTrue(np.all(np.isnan(res[1])))

    def test_custom_data_dir(self):
        from .. import polyjectory
        import os
        from pathlib import Path
        import tempfile
        import numpy as np

        state_data = np.zeros((1, 8, 7))

        # Test failure with already-existing dir.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = Path(tmpdirname) / "data_dir"
            os.mkdir(data_dir)

            with self.assertRaises(RuntimeError) as cm:
                polyjectory(
                    trajs=[state_data, state_data[1:]],
                    times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                    status=np.array([0, 0], dtype=np.int32),
                    data_dir=data_dir,
                )
            self.assertTrue("Error while creating the directory" in str(cm.exception))

        # Check proper creation and usage of custom data dir.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = (Path(tmpdirname) / "data_dir").resolve()

            pj = polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
                data_dir=data_dir,
            )
            self.assertEqual(data_dir, pj.data_dir)

            # NOTE: ensure we remove the datafiles before the temp dir,
            # otherwise the test will fail on Windows.
            pj.detach()
            self.assertFalse(data_dir.exists())

        # Check the ctor from datafiles too.
        with tempfile.TemporaryDirectory() as tmpdirname:
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            data_dir = (Path(tmpdirname) / "data_dir").resolve()

            pj = polyjectory.from_datafiles(
                traj_file=traj_file.name,
                time_file=time_file.name,
                order=2,
                traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                status=[1],
                data_dir=data_dir,
            )
            self.assertEqual(data_dir, pj.data_dir)

            # NOTE: ensure we remove the datafiles before the temp dir,
            # otherwise the test will fail on Windows.
            pj.detach()
            self.assertFalse(data_dir.exists())

        # Check that an empty data dir path is treated as if not provided.
        # NOTE: here we only check the lack of throwing.
        pj = polyjectory(
            trajs=[state_data, state_data[1:]],
            times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
            status=np.array([0, 0], dtype=np.int32),
            data_dir="",
        )

        # Check the ctor from datafiles too.
        with tempfile.TemporaryDirectory() as tmpdirname:
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            pj = polyjectory.from_datafiles(
                traj_file=traj_file.name,
                time_file=time_file.name,
                order=2,
                traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                status=[1],
                data_dir="",
            )

    def test_persist(self):
        from .. import polyjectory
        import numpy as np
        from pathlib import Path
        import tempfile

        state_data = np.zeros((1, 8, 7))

        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = (Path(tmpdirname) / "data_dir").resolve()

            pj = polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
                persist=True,
                data_dir=data_dir,
            )
            self.assertTrue(data_dir.is_dir())
            self.assertTrue(data_dir.exists())
            self.assertTrue(pj.persist)

            # NOTE: detach will *not* remove the data dirs since persist is active.
            pj.detach()

            # Check the data dir still exists.
            self.assertTrue(data_dir.is_dir())
            self.assertTrue(data_dir.exists())

        # Check the ctor from datafiles too.
        with tempfile.TemporaryDirectory() as tmpdirname:
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            data_dir = (Path(tmpdirname) / "data_dir").resolve()

            pj = polyjectory.from_datafiles(
                traj_file=traj_file.name,
                time_file=time_file.name,
                order=2,
                traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                status=[1],
                data_dir=data_dir,
                persist=True,
            )
            self.assertEqual(data_dir, pj.data_dir)

            # NOTE: detach will *not* remove the data dirs since persist is active.
            pj.detach()

            # Check the data dir still exists.
            self.assertTrue(data_dir.is_dir())
            self.assertTrue(data_dir.exists())

    def test_detached_throw(self):
        # A test to check throwing behaviour after a polyjectory
        # has been detached.
        from .. import polyjectory
        import numpy as np

        state_data = np.zeros((1, 8, 7))

        pj = polyjectory(
            trajs=[state_data, state_data[1:]],
            times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
            status=np.array([0, 0], dtype=np.int32),
        )

        pj.detach()

        # detach() and is_detached() still work.
        pj.detach()
        self.assertTrue(pj.is_detached)

        # All the other properties/methods must throw.
        def check_raise(attr, *args, **kwargs):
            with self.assertRaises(ValueError) as cm:
                # NOTE: this line will already throw if we are dealing
                # with a property. Otherwise, this will fetch the method.
                attr = getattr(pj, attr)
                attr(*args, **kwargs)

            self.assertTrue(
                "Cannot operate on a detached polyjectory" in str(cm.exception)
            )

        check_raise("n_objs")
        check_raise("maxT")
        check_raise("epoch")
        check_raise("poly_order")
        check_raise("data_dir")
        check_raise("__getitem__", 0)
        check_raise("status")
        check_raise("persist")
        check_raise("state_eval", 0.0)
        check_raise("state_meval", [0.0])

    def test_mount(self):
        from .. import polyjectory
        import numpy as np
        import tempfile
        from pathlib import Path

        state_data = np.zeros((1, 8, 7))

        pj = polyjectory(
            trajs=[state_data, state_data[1:]],
            times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
            status=np.array([0, 0], dtype=np.int32),
        )

        # Check mounting a non-persistent data dir.
        with self.assertRaises(ValueError) as cm:
            polyjectory.mount(pj.data_dir)
        self.assertTrue(": the data is not persistent" in str(cm.exception))

        # Check mounting a non-existing data dir.
        with tempfile.TemporaryDirectory() as tmpdirname:
            with self.assertRaises(ValueError) as cm:
                polyjectory.mount(Path(tmpdirname) / "foobar")
            self.assertTrue(
                "could not be canonicalised (does it exist?)" in str(cm.exception)
            )

        # Check mounting a file.
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(Path(tmpdirname) / "foobar", "wb"):
                pass

            with self.assertRaises(ValueError) as cm:
                polyjectory.mount(Path(tmpdirname) / "foobar")
            self.assertTrue("the path is not a directory" in str(cm.exception))

        # Check a working case.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = Path(tmpdirname) / "foo"

            pj = polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
                data_dir=data_dir,
                persist=True,
            )

            pj.detach()

            self.assertTrue(data_dir.exists())
            self.assertTrue(data_dir.is_dir())

            pj = polyjectory.mount(data_dir)

            self.assertTrue(np.all(pj[0][0] == state_data))
            self.assertTrue(np.all(pj[0][1] == np.array([0.0, 1.0])))
            self.assertTrue(np.all(pj[1][0] == state_data[1:]))
            self.assertTrue(np.all(pj[1][1] == np.array([], dtype=float)))

            pj.detach()

            self.assertTrue(data_dir.exists())
            self.assertTrue(data_dir.is_dir())

    def test_dir_removal(self):
        # A test checking behaviour when polyjectory data
        # is deleted before the polyjectory is garbage-collected.
        from .. import polyjectory
        import numpy as np
        import os
        import shutil

        state_data = np.full((1, 8, 7), 42.0)

        pj = polyjectory(
            trajs=[state_data, state_data[1:]],
            times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
            status=np.array([0, 0], dtype=np.int32),
        )

        if os.name == "nt":
            # On Windows, we won't be able to delete the data
            # because it is "owned" by the polyjectory object.
            with self.assertRaises(Exception):
                shutil.rmtree(pj.data_dir)

            self.assertTrue(pj.data_dir.exists())

        elif os.name == "posix":
            # On posix, we assume we can remove the data.
            # The memory-mapped regions should remain valid
            # because we have not closed the file descriptors.
            shutil.rmtree(pj.data_dir)

            self.assertFalse(pj.data_dir.exists())

            self.assertTrue(np.all(pj[0][0] == state_data))
            self.assertTrue(np.all(pj[0][1] == np.array([0.0, 1.0])))
            self.assertTrue(np.all(pj[1][0] == state_data[1:]))
            self.assertTrue(np.all(pj[1][1] == np.array([], dtype=float)))

    def test_tmpdir(self):
        # A test checking custom setting for tmpdir in the constructors.
        from .. import polyjectory, set_tmpdir, get_tmpdir
        import numpy as np
        import tempfile
        from pathlib import Path
        import os

        state_data = np.full((1, 8, 7), 42.0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            # NOTE: this checks that the dir is empty.
            self.assertTrue(not any(tmpdir.iterdir()))

            pj = polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
                tmpdir=tmpdir,
            )

            self.assertTrue(any(tmpdir.iterdir()))

            pj.detach()

        # Check the ctor from datafiles too.
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.makedirs(Path(tmpdirname) / "foo")
            tmpdir = Path(tmpdirname) / "foo"

            # NOTE: this checks that the dir is empty.
            self.assertTrue(not any(tmpdir.iterdir()))

            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            pj = polyjectory.from_datafiles(
                traj_file=traj_file.name,
                time_file=time_file.name,
                order=2,
                traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                status=[1],
                tmpdir=tmpdir,
            )

            self.assertTrue(any(tmpdir.iterdir()))

            pj.detach()

        # A test to check that a custom tmpdir overrides
        # the global tmpdir.
        orig_global_tmpdir = get_tmpdir()
        set_tmpdir(__file__)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            # NOTE: this checks that the dir is empty.
            self.assertTrue(not any(tmpdir.iterdir()))

            pj = polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
                tmpdir=tmpdir,
            )

            self.assertTrue(any(tmpdir.iterdir()))

            pj.detach()

        # Restore the original global temp dir.
        set_tmpdir(orig_global_tmpdir)

        # Check that we cannot specify both tmpdir and data_dir at the same time.
        with self.assertRaises(ValueError) as cm:
            polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
                tmpdir="",
                data_dir="",
            )
        self.assertTrue(
            "The 'data_dir' and 'tmpdir' construction arguments cannot be provided at the same time"
            in str(cm.exception)
        )

        # Datafile ctor too.
        with tempfile.TemporaryDirectory() as tmpdirname:
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            with self.assertRaises(ValueError) as cm:
                polyjectory.from_datafiles(
                    traj_file=traj_file.name,
                    time_file=time_file.name,
                    order=2,
                    traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                    status=[1],
                    tmpdir="",
                    data_dir="",
                )
            self.assertTrue(
                "The 'data_dir' and 'tmpdir' construction arguments cannot be provided at the same time"
                in str(cm.exception)
            )

        # Check that an empty tmpdir is interpreted as if not provided.
        # NOTE: here we only check the lack of throwing.
        pj = polyjectory(
            trajs=[state_data, state_data[1:]],
            times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
            status=np.array([0, 0], dtype=np.int32),
            tmpdir="",
        )

        # Check the ctor from datafiles too.
        with tempfile.TemporaryDirectory() as tmpdirname:
            traj_file = open(Path(tmpdirname) / "traj", "wb")
            np.full((42,), 0.0, dtype=float).tofile(traj_file)
            traj_file.close()

            time_file = open(Path(tmpdirname) / "time", "wb")
            time_data = np.array([0.0, 1.0, 1.1])
            time_data.tofile(time_file)
            time_file.close()

            pj = polyjectory.from_datafiles(
                traj_file=traj_file.name,
                time_file=time_file.name,
                order=2,
                traj_offsets=np.array([(0, 2)], dtype=polyjectory.traj_offset),
                status=[1],
                tmpdir="",
            )
