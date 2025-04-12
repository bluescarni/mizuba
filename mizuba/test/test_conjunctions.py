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


class conjunctions_test_case(_ut.TestCase):
    @classmethod
    def setUpClass(cls):
        from .. import _have_sgp4_deps

        if not _have_sgp4_deps():
            return

        import pathlib
        import polars as pl
        from bisect import bisect_left
        from .._sgp4_polyjectory import _make_satrec_from_dict

        # Fetch the current directory.
        cur_dir = pathlib.Path(__file__).parent.resolve()

        # Load the test data.
        try:
            gpes = pl.read_parquet(cur_dir / "strack_20240705.parquet")
        except Exception:
            return

        # Create the satellite objects.
        sat_list = [_make_satrec_from_dict(_) for _ in gpes.iter_rows(named=True)]

        # Create a sparse list of satellites.
        # NOTE: we manually include an object for which the
        # trajectory data terminates early if the exit_radius
        # is set to 12000.
        cls.sparse_sat_list = sorted(
            sat_list[::2000] + [sat_list[220]], key=lambda sat: sat.satnum
        )

        # Identify the new index of the added satellite in the sorted list.
        norad_id_list = [_.satnum for _ in cls.sparse_sat_list]
        idx = bisect_left(norad_id_list, sat_list[220].satnum)
        cls.exiting_idx = idx

        # List of 9000 satellites.
        cls.half_sat_list = sat_list[:9000]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "sparse_sat_list"):
            del cls.sparse_sat_list
            del cls.half_sat_list

    # Helper to verify that the aabbs are consistent
    # with the positions of the objects computed via
    # polynomial evaluation.
    def _verify_conj_aabbs(self, c, pj, rng):
        import numpy as np

        # For every conjunction step, pick random times within,
        # evaluate the polyjectory at the corresponding times and
        # assert that the positions are within the aabbs.
        for cd_idx, end_time in enumerate(c.cd_end_times):
            begin_time = 0.0 if cd_idx == 0 else c.cd_end_times[cd_idx - 1]

            # Pick 5 random times.
            random_times = rng.uniform(begin_time, end_time, (5,))

            # Fetch the global aabb for this conjunction step.
            global_lb = c.aabbs[cd_idx, pj.n_objs, 0]
            global_ub = c.aabbs[cd_idx, pj.n_objs, 1]

            if not np.isfinite(global_lb[0]):
                # Non-finite value detected in the global AABB.

                # All global AABB values must be infinity.
                self.assertTrue(np.all(np.isinf(global_lb)))
                self.assertTrue(np.all(np.isinf(global_ub)))

                # The AABBs of all objects must be infinities.
                self.assertTrue(
                    all(
                        np.all(np.isinf(c.aabbs[cd_idx, obj_idx]))
                        for obj_idx in range(pj.n_objs)
                    )
                )

                # Continue to the next conjunction step.
                continue

            # Iterate over all objects.
            for obj_idx in range(pj.n_objs):
                # Fetch the polyjectory data for the current object.
                traj, traj_times, status = pj[obj_idx]

                # Fetch the AABB of the object.
                aabb = c.aabbs[cd_idx, obj_idx]

                # If there is no trajectory data for the current
                # object, just check that its aabb is infinite.
                if traj.shape[0] == 0:
                    self.assertTrue(np.all(np.isinf(aabb)))
                    continue

                if begin_time >= traj_times[-1]:
                    # The trajectory data for the current object
                    # ends before the beginning of the current conjunction
                    # step. Skip the current object and assert that its
                    # aabb is infinite.
                    self.assertTrue(np.all(np.isinf(aabb)))
                    continue
                elif traj_times[0] >= end_time:
                    # The trajectory data for the current object
                    # begins at or after the end time of the conjunction
                    # step. Skip the current object and assert that its
                    # aabb is infinite.
                    self.assertTrue(np.all(np.isinf(aabb)))
                    continue
                else:
                    # The time data for the current object overlaps
                    # with the conjunction step. The aabb must be finite.
                    self.assertTrue(np.all(np.isfinite(aabb)))

                # The aabb must be included in the global one.
                self.assertTrue(np.all(aabb[0] >= global_lb))
                self.assertTrue(np.all(aabb[1] <= global_ub))

                # Iterate over the random times.
                for time in random_times:
                    # Look for the first trajectory time data point *after* 'time'.
                    step_idx = np.searchsorted(traj_times, time, side="right")

                    # Skip the current 'time' if it is past the end of
                    # trajectory data.
                    if step_idx == len(traj_times):
                        continue

                    # Skip the current 'time' if it is before the beginning
                    # of trajectory data.
                    if step_idx == 0:
                        continue

                    # Fetch the polynomials for all state variables
                    # in the trajectory step.
                    # NOTE: step_idx - 1 because we need the
                    # trajectory step that ends at traj_times[step_idx].
                    traj_polys = traj[step_idx - 1]

                    # Compute the poly evaluation interval.
                    # This is the time elapsed since the beginning
                    # of the trajectory step.
                    h = time - traj_times[step_idx - 1]

                    # Evaluate the polynomials and check that
                    # the results fit in the aabb.
                    for coord_idx, aabb_idx in zip([0, 1, 2, 6], range(4)):
                        pval = np.polyval(traj_polys[::-1, coord_idx], h)
                        self.assertGreater(pval, aabb[0][aabb_idx])
                        self.assertLess(pval, aabb[1][aabb_idx])

    def test_basics(self):
        import sys
        from .. import conjunctions as conj, polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times
        import numpy as np

        # Test error handling on construction.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        with self.assertRaises(ValueError) as cm:
            conj(conj_det_interval=0.0, pj=pj, conj_thresh=0.0)
        self.assertTrue(
            "The conjunction threshold must be finite and positive, but instead a"
            " value of" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            conj(conj_det_interval=0.0, pj=pj, conj_thresh=float(np.finfo(float).max))
        self.assertTrue(
            "is too large and results in an overflow error" in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            conj(pj, conj_thresh=float("inf"), conj_det_interval=0.0)
        self.assertTrue(
            "The conjunction threshold must be finite and positive, but instead a"
            " value of" in str(cm.exception)
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

        # Test accessors.
        c = conj(pj, conj_thresh=1.0, conj_det_interval=0.1)

        self.assertEqual(c.n_cd_steps, len(c.cd_end_times))
        self.assertTrue(isinstance(c.bvh_node, np.dtype))
        self.assertTrue(isinstance(c.aabb_collision, np.dtype))
        self.assertTrue(isinstance(c.conj, np.dtype))
        self.assertEqual(c.conj_thresh, 1.0)
        self.assertEqual(c.conj_det_interval, 0.1)

        # aabbs.
        rc = sys.getrefcount(c)
        aabbs = c.aabbs
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(ValueError) as cm:
            aabbs[:] = aabbs
        with self.assertRaises(AttributeError) as cm:
            c.aabbs = aabbs

        # cd_end_times.
        rc = sys.getrefcount(c)
        cd_end_times = c.cd_end_times
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(ValueError) as cm:
            cd_end_times[:] = cd_end_times
        with self.assertRaises(AttributeError) as cm:
            c.cd_end_times = cd_end_times

        # srt_aabbs.
        rc = sys.getrefcount(c)
        srt_aabbs = c.srt_aabbs
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(ValueError) as cm:
            srt_aabbs[:] = srt_aabbs
        with self.assertRaises(AttributeError) as cm:
            c.srt_aabbs = srt_aabbs

        # mcodes.
        rc = sys.getrefcount(c)
        mcodes = c.mcodes
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(ValueError) as cm:
            mcodes[:] = mcodes
        with self.assertRaises(AttributeError) as cm:
            c.mcodes = mcodes

        # srt_mcodes.
        rc = sys.getrefcount(c)
        srt_mcodes = c.srt_mcodes
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(ValueError) as cm:
            srt_mcodes[:] = srt_mcodes
        with self.assertRaises(AttributeError) as cm:
            c.srt_mcodes = srt_mcodes

        # srt_idx.
        rc = sys.getrefcount(c)
        srt_idx = c.srt_idx
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(ValueError) as cm:
            srt_idx[:] = srt_idx
        with self.assertRaises(AttributeError) as cm:
            c.srt_idx = srt_idx

        c.hint_release()

    def test_main(self):
        import numpy as np
        import sys
        from .. import (
            conjunctions as conj,
            polyjectory,
            otype,
        )
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Single planar circular orbit case.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        # Run the test for several conjunction detection intervals.
        for conj_det_interval in [0.01, 0.1, 0.5, 2.0, 5.0, 7.0]:
            c = conj(pj, conj_thresh=0.1, conj_det_interval=conj_det_interval)

            # Shape checks.
            self.assertEqual(c.aabbs.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_aabbs.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_aabbs.shape, c.aabbs.shape)
            self.assertEqual(c.mcodes.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_mcodes.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_idx.shape[0], c.cd_end_times.shape[0])

            # The conjunction detection end time must coincide
            # with the trajectory end time.
            self.assertEqual(c.cd_end_times[-1], pj[0][1][-1])

            # The global aabbs must all coincide
            # exactly with the only object's aabbs.
            self.assertTrue(np.all(c.aabbs[:, 0] == c.aabbs[:, 1]))
            # With only one object, aabbs and srt_aabbs must be identical.
            self.assertTrue(np.all(c.aabbs == c.srt_aabbs))

            # In the z and r coordinates, all aabbs
            # should be of size circa 0.1 accounting for the
            # conjunction threshold.
            self.assertTrue(np.all(c.aabbs[:, 0, 0, 2] >= -0.05001))
            self.assertTrue(np.all(c.aabbs[:, 0, 1, 2] <= 0.05001))

            self.assertTrue(np.all(c.aabbs[:, 0, 0, 3] >= 1 - 0.05001))
            self.assertTrue(np.all(c.aabbs[:, 0, 1, 3] <= 1 + 0.05001))

            # Verify the aabbs.
            self._verify_conj_aabbs(c, pj, rng)

            # No aabb collisions or conjunctions expected.
            for i in range(c.n_cd_steps):
                self.assertEqual(len(c.get_aabb_collisions(i)), 0)
            self.assertEqual(len(c.conjunctions), 0)

            # Check the object types.
            self.assertTrue(np.all(c.otypes == [1] * pj.n_objs))
            self.assertTrue(np.all(c.otypes == [otype.PRIMARY] * pj.n_objs))

            # Test otypes initialisation.
            c = conj(
                pj,
                conj_thresh=0.1,
                conj_det_interval=conj_det_interval,
                otypes=[1] * pj.n_objs,
            )

            # Check the otypes property.
            rc = sys.getrefcount(c)
            otypes = c.otypes
            self.assertEqual(sys.getrefcount(c), rc + 1)
            with self.assertRaises(ValueError) as cm:
                otypes[:] = otypes
            with self.assertRaises(AttributeError) as cm:
                c.otypes = otypes
            self.assertEqual(len(otypes), pj.n_objs)

            c.hint_release()

            # Error handling.
            with self.assertRaises(ValueError) as cm:
                conj(
                    pj,
                    conj_thresh=0.1,
                    conj_det_interval=conj_det_interval,
                    otypes=[],
                )
            self.assertTrue(
                "Invalid array of object types passed to the constructor of a"
                f" conjunctions objects: the expected size is {pj.n_objs}, but the"
                " actual size is 0 instead" in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                conj(
                    pj,
                    conj_thresh=0.1,
                    conj_det_interval=conj_det_interval,
                    otypes=[-5],
                )
            self.assertTrue(
                "The value of an object type must be one of [1, 2, 4], but a value of"
                " -5 was detected instead" in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                conj(
                    pj,
                    conj_thresh=0.1,
                    conj_det_interval=conj_det_interval,
                    otypes=[5],
                )
            self.assertTrue(
                "The value of an object type must be one of [1, 2, 4], but a value of 5"
                " was detected instead" in str(cm.exception)
            )

        # Test that if we specify a conjunction detection interval
        # larger than maxT, the time data in the conjunctions object
        # is correctly clamped.
        c = conj(pj, conj_thresh=0.1, conj_det_interval=42.0)
        self.assertEqual(c.n_cd_steps, 1)
        self.assertEqual(c.cd_end_times[0], pj[0][1][-1])

        # Run the sgp4 tests, if possible.
        if not hasattr(type(self), "sparse_sat_list"):
            return

        from .. import make_sgp4_polyjectory

        # Use the sparse satellite list.
        sat_list = self.sparse_sat_list

        begin_jd = 2460496.5

        # Build the polyjectory.
        pt = make_sgp4_polyjectory(
            sat_list, begin_jd, begin_jd + 0.25, exit_radius=12000.0
        )[0]
        tot_n_objs = pt.n_objs

        # Build the conjunctions object. Keep a small threshold not to interfere
        # with aabb checking.
        c = conj(pt, conj_thresh=1e-8, conj_det_interval=1.0 / 1440.0)

        # Verify the aabbs.
        self._verify_conj_aabbs(c, pt, rng)

        # Shape checks.
        self.assertEqual(c.aabbs.shape, c.srt_aabbs.shape)
        self.assertEqual(c.mcodes.shape, c.srt_mcodes.shape)
        self.assertEqual(c.srt_idx.shape, (c.n_cd_steps, pt.n_objs))

        # The global aabbs must be the same in srt_aabbs.
        self.assertTrue(
            np.all(c.aabbs[:, pt.n_objs, :, :] == c.srt_aabbs[:, pt.n_objs, :, :])
        )

        # The individual aabbs for the objects will differ.
        self.assertFalse(
            np.all(c.aabbs[:, : pt.n_objs, :, :] == c.srt_aabbs[:, : pt.n_objs, :, :])
        )

        # The morton codes won't be sorted.
        self.assertFalse(np.all(np.diff(c.mcodes.astype(object)) >= 0))

        # The sorted morton codes must be sorted.
        self.assertTrue(np.all(np.diff(c.srt_mcodes.astype(object)) >= 0))

        # srt_idx is not sorted.
        self.assertFalse(np.all(np.diff(c.srt_idx.astype(object)) >= 0))

        # Indexing into aabbs and mcodes via srt_idx must produce
        # srt_abbs and srt_mcodes.
        for cd_idx in range(c.n_cd_steps):
            self.assertEqual(sorted(c.srt_idx[cd_idx]), list(range(pt.n_objs)))

            self.assertTrue(
                np.all(
                    c.aabbs[cd_idx, c.srt_idx[cd_idx], :, :]
                    == c.srt_aabbs[cd_idx, : pt.n_objs, :, :]
                )
            )

            self.assertTrue(
                np.all(c.mcodes[cd_idx, c.srt_idx[cd_idx]] == c.srt_mcodes[cd_idx])
            )

        # The exiting satellite's trajectory data terminates
        # early. After termination, the morton codes must be -1.

        # Fetch all the aabbs of the exiting satellite.
        exit_aabbs = c.aabbs[:, self.exiting_idx, :, :]

        # Check that not all are finite.
        self.assertFalse(np.all(np.isfinite(exit_aabbs)))

        # Compute the indices of the conjunction steps
        # in which infinite aabbs show up.
        inf_idx = np.any(np.isinf(exit_aabbs), axis=(1, 2)).nonzero()[0]

        # Check the Morton codes.
        self.assertTrue(np.all(c.mcodes[inf_idx, self.exiting_idx] == ((1 << 64) - 1)))

        # Similarly, the number of objects reported in the root
        # node of the bvh trees must be tot_n_objs - 2.
        # NOTE: -3 (rather than -1) because 2 other satellites generated
        # infinite aabbs.
        for idx in inf_idx:
            t = c.get_bvh_tree(idx)
            self.assertEqual(t[0]["end"] - t[0]["begin"], tot_n_objs - 3)

    def test_custom_data_dir(self):
        # Test for custom data dir passed to the constructor.
        import tempfile
        from pathlib import Path
        import os
        from .. import (
            conjunctions as conj,
            polyjectory,
        )
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        # Single planar circular orbit case.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        # Test failure with already-existing dir.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = Path(tmpdirname) / "data_dir"
            os.mkdir(data_dir)

            with self.assertRaises(RuntimeError) as cm:
                conj(pj, conj_thresh=0.1, conj_det_interval=0.01, data_dir=data_dir)
            self.assertTrue("Error while creating the directory" in str(cm.exception))

        # Check proper creation and usage of custom data dir.
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = (Path(tmpdirname) / "data_dir").resolve()

            self.assertFalse(data_dir.exists())

            c = conj(pj, conj_thresh=0.1, conj_det_interval=0.01, data_dir=data_dir)

            self.assertTrue(data_dir.exists())
            self.assertTrue(data_dir.is_dir())

            del c

        # Check that an empty data dir path is treated as if not provided.
        # NOTE: here we only check the lack of throwing.
        conj(pj, conj_thresh=0.1, conj_det_interval=0.01, data_dir="")

    def test_tmpdir(self):
        # A test checking custom setting for tmpdir in the constructors.
        import tempfile
        from pathlib import Path
        from .. import conjunctions as conj, polyjectory, set_tmpdir, get_tmpdir
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        # Single planar circular orbit case.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            # NOTE: this checks that the dir is empty.
            self.assertTrue(not any(tmpdir.iterdir()))

            c = conj(pj, conj_thresh=0.1, conj_det_interval=0.01, tmpdir=tmpdir)

            self.assertTrue(any(tmpdir.iterdir()))

            del c

        # A test to check that a custom tmpdir overrides
        # the global tmpdir.
        orig_global_tmpdir = get_tmpdir()
        set_tmpdir(__file__)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            # NOTE: this checks that the dir is empty.
            self.assertTrue(not any(tmpdir.iterdir()))

            c = conj(pj, conj_thresh=0.1, conj_det_interval=0.01, tmpdir=tmpdir)

            self.assertTrue(any(tmpdir.iterdir()))

            del c

        # Restore the original global temp dir.
        set_tmpdir(orig_global_tmpdir)

        # Check that we cannot specify both tmpdir and data_dir at the same time.
        with self.assertRaises(ValueError) as cm:
            conj(pj, conj_thresh=0.1, conj_det_interval=0.01, tmpdir="", data_dir="")
        self.assertTrue(
            "The 'data_dir' and 'tmpdir' construction arguments cannot be provided at the same time"
            in str(cm.exception)
        )

    def test_zero_aabbs(self):
        # Test to check behaviour with aabbs of zero size.
        import numpy as np
        from .. import conjunctions as conj, polyjectory

        # Trajectory data for a single step.
        tdata = np.zeros((7, 6))
        # Make the object fixed in Cartesian space with x,y,z coordinates all 1.
        tdata[:3, 0] = 1.0
        # Set the radius.
        tdata[6, 0] = np.sqrt(3.0)
        tdata = np.ascontiguousarray(tdata.transpose())

        pj = polyjectory([[tdata, tdata, tdata]], [[0.0, 1.0, 2.0, 3.0]], [0])

        # Use epsilon as conj thresh so that it does not influence
        # the computation of the aabb.
        c = conj(pj, conj_thresh=np.finfo(float).eps, conj_det_interval=0.1)

        self.assertTrue(
            np.all(
                c.aabbs[:, :, 0, :3] == np.nextafter(np.single(1), np.single("-inf"))
            )
        )
        self.assertTrue(
            np.all(
                c.aabbs[:, :, 1, :3] == np.nextafter(np.single(1), np.single("+inf"))
            )
        )
        self.assertTrue(
            np.all(
                c.aabbs[:, :, 0, 3]
                == np.nextafter(np.single(np.sqrt(3.0)), np.single("-inf"))
            )
        )
        self.assertTrue(
            np.all(
                c.aabbs[:, :, 1, 3]
                == np.nextafter(np.single(np.sqrt(3.0)), np.single("+inf"))
            )
        )

    def test_no_traj_data(self):
        # This is a test to verify that when an object lacks
        # trajectory data it is always placed at the end of
        # the srt_* data.
        import numpy as np
        from .. import conjunctions, polyjectory

        # The goal here is to generate trajectory
        # data for which the aabb centre's morton code
        # is all ones (this will be tdata7). This will allow
        # us to verify that missing traj data is placed
        # after tdata7.
        # x.
        tdata0 = np.zeros((6, 7))
        tdata0[0, 0] = 1.0
        tdata1 = np.zeros((6, 7))
        tdata1[0, 0] = -1.0

        # y.
        tdata2 = np.zeros((6, 7))
        tdata2[0, 1] = 1.0
        tdata3 = np.zeros((6, 7))
        tdata3[0, 1] = -1.0

        # z.
        tdata4 = np.zeros((6, 7))
        tdata4[0, 2] = 1.0
        tdata5 = np.zeros((6, 7))
        tdata5[0, 2] = -1.0

        # Center.
        tdata6 = np.zeros((6, 7))

        # All ones.
        tdata7 = np.zeros((6, 7))
        tdata7[0:1] = 1

        # NOTE: the first 10 objects will have traj
        # data only for the first step, not the second.
        pj = polyjectory(
            [[tdata0]] * 10
            + [
                [tdata0] * 2,
                [tdata1] * 2,
                [tdata2] * 2,
                [tdata3] * 2,
                [tdata4] * 2,
                [tdata5] * 2,
                [tdata6] * 2,
                [tdata7] * 2,
            ],
            [[0.0, 1.0]] * 10 + [[0.0, 1.0, 2.0]] * 8,
            [0] * 18,
        )

        conjs = conjunctions(pj, 1e-16, 1.0)

        # Verify that at the second step all
        # inf aabbs are at the end of srt_aabbs
        # and the morton codes are all -1.
        self.assertTrue(np.all(np.isinf(conjs.aabbs[1, :10])))
        self.assertTrue(np.all(np.isinf(conjs.srt_aabbs[1, -11:-1])))
        self.assertTrue(np.all(conjs.mcodes[1, :10] == (2**64 - 1)))
        self.assertTrue(conjs.mcodes[1:, -1] == (2**64 - 1))
        self.assertTrue(np.all(conjs.srt_mcodes[1, -11:] == (2**64 - 1)))

    def test_bvh(self):
        # NOTE: most of the validation of bvh
        # trees is done within the C++ code
        # during construction in debug mode.
        # Here we instantiate several corner cases.
        import numpy as np
        from .. import conjunctions, polyjectory

        # Polyjectory with a single object.
        tdata = np.zeros((6, 7))
        tdata[1, :] = 0.1

        pj = polyjectory([[tdata]], [[0.0, 1.0]], [0])
        conjs = conjunctions(pj, 1e-16, 1.0)

        with self.assertRaises(IndexError) as cm:
            conjs.get_bvh_tree(1)
        self.assertTrue(
            "Cannot fetch the BVH tree for the conjunction timestep at index 1: the"
            " total number of conjunction steps is only 1" in str(cm.exception)
        )

        t = conjs.get_bvh_tree(0)
        self.assertEqual(len(t), 1)

        # Polyjectory with two identical objects.
        # This will result in exhausting all bits
        # in the morton codes for splitting.
        pj = polyjectory([[tdata], [tdata]], [[0.0, 1.0], [0.0, 1.0]], [0, 0])
        conjs = conjunctions(pj, 1e-16, 1.0)
        t = conjs.get_bvh_tree(0)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["begin"], 0)
        self.assertEqual(t[0]["end"], 2)
        self.assertEqual(t[0]["left"], -1)
        self.assertEqual(t[0]["right"], -1)

        # Polyjectory in which the morton codes
        # of two objects differ at the last bit.
        # x.
        tdata0 = np.zeros((6, 7))
        tdata0[0, 0] = 1.0
        tdata1 = np.zeros((6, 7))
        tdata1[0, 0] = -1.0

        # y.
        tdata2 = np.zeros((6, 7))
        tdata2[0, 1] = 1.0
        tdata3 = np.zeros((6, 7))
        tdata3[0, 1] = -1.0

        # z.
        tdata4 = np.zeros((6, 7))
        tdata4[0, 2] = 1.0
        tdata5 = np.zeros((6, 7))
        tdata5[0, 2] = -1.0

        # Center.
        tdata6 = np.zeros((6, 7))

        # All ones.
        tdata7 = np.zeros((6, 7))
        tdata7[0, :] = 1

        # All ones but last.
        tdata8 = np.zeros((6, 7))
        tdata8[0, :] = 1
        tdata8[0, 0] = 1.0 - 2.1 / 2**16

        pj = polyjectory(
            [
                [tdata0],
                [tdata1],
                [tdata2],
                [tdata3],
                [tdata4],
                [tdata5],
                [tdata6],
                [tdata7],
                [tdata8],
            ],
            [[0.0, 1.0]] * 9,
            [0] * 9,
        )

        conjs = conjunctions(pj, 1e-16, 1.0)
        self.assertEqual(conjs.mcodes[0, -2], 2**64 - 1)
        self.assertEqual(conjs.mcodes[0, -1], 2**64 - 2)
        t = conjs.get_bvh_tree(0)
        self.assertEqual(conjs.srt_idx[0, -1], 7)
        self.assertEqual(conjs.srt_idx[0, -2], 8)

    def test_broad_narrow_phase(self):
        # NOTE: for the broad-phase, we are relying
        # on internal debug checks implemented in C++.

        # We rely on sgp4 data for this test.
        if not hasattr(type(self), "sparse_sat_list"):
            return

        from .. import (
            make_sgp4_polyjectory,
            conjunctions as conj,
            otype,
        )
        import numpy as np

        sat_list = self.half_sat_list

        begin_jd = 2460496.5

        # Build the polyjectory. Run it for only 15 minutes.
        duration = 15.0 / 1440.0
        pt = make_sgp4_polyjectory(sat_list, begin_jd, begin_jd + duration)[0]

        # Build a list of object types that excludes two satellites
        # that we know undergo a conjunction.
        otypes = [otype.PRIMARY] * pt.n_objs
        otypes[6746] = otype.SECONDARY
        otypes[4549] = otype.SECONDARY

        # Run several tests using several conjunction detection intervals.
        # Store the conjunctions arrays for more testing later.
        c_arrays = []

        for cdet_interval in [1.0 / 1440, 5.0 / 1440.0, 1.0]:
            # Build the conjunctions object. This will trigger
            # the internal C++ sanity checks in debug mode.
            c = conj(
                pt, conj_thresh=10.0, conj_det_interval=cdet_interval, otypes=otypes
            )

            c_arrays.append(c.conjunctions)

            self.assertTrue(
                all(len(c.get_aabb_collisions(_)) > 0 for _ in range(c.n_cd_steps))
            )

            with self.assertRaises(IndexError) as cm:
                c.get_aabb_collisions(c.n_cd_steps)
            self.assertTrue(
                "Cannot fetch the list of AABB collisions for the conjunction timestep at"
                f" index {c.n_cd_steps}: the total number of conjunction steps is only"
                f" {c.n_cd_steps}" in str(cm.exception)
            )

            # The conjunctions must be sorted according
            # to the TCA.
            self.assertTrue(np.all(np.diff(c.conjunctions["tca"]) >= 0))

            # All conjunctions must happen before the polyjectory end time.
            self.assertTrue(c.conjunctions["tca"][-1] < duration)

            # No conjunction must be at or above the threshold.
            self.assertTrue(np.all(np.diff(c.conjunctions["dca"]) < 10))

            # Objects cannot have conjunctions with themselves.
            self.assertTrue(np.all(c.conjunctions["i"] != c.conjunctions["j"]))

            # DCA must be consistent with state vectors.
            self.assertTrue(
                np.all(
                    np.isclose(
                        np.linalg.norm(
                            c.conjunctions["ri"] - c.conjunctions["rj"], axis=1
                        ),
                        c.conjunctions["dca"],
                        rtol=1e-14,
                        atol=0.0,
                    )
                )
            )

            # Conjunctions cannot happen between secondaries.
            self.assertFalse(
                (4549, 6746) in list(tuple(_) for _ in c.conjunctions[["i", "j"]])
            )

            # Verify the conjunctions with the sgp4 python module.
            sl_array = np.array(sat_list)
            for cj in c.conjunctions:
                # Fetch the conjunction data.
                tca = cj["tca"]
                i, j = cj["i"], cj["j"]
                ri, rj = cj["ri"], cj["rj"]
                vi, vj = cj["vi"], cj["vj"]

                # Fetch the satellites.
                sat_i = sl_array[i]
                sat_j = sl_array[j]

                ei, sri, svi = sat_i.sgp4(begin_jd, tca)
                ej, srj, svj = sat_j.sgp4(begin_jd, tca)

                diff_ri = np.linalg.norm(sri - ri)
                diff_rj = np.linalg.norm(srj - rj)

                diff_vi = np.linalg.norm(svi - vi)
                diff_vj = np.linalg.norm(svj - vj)

                # NOTE: unit of measurement here is [km], vs typical
                # values of >1e3 km in the coordinates. Thus, relative
                # error is 1e-11, absolute error is ~10µm.
                self.assertLess(diff_ri, 1e-8)
                self.assertLess(diff_rj, 1e-8)

                # NOTE: unit of measurement here is [km/s], vs typicial
                # velocity values of >1 km/s.
                self.assertLess(diff_vi, 1e-11)
                self.assertLess(diff_vj, 1e-11)

        # Run consistency checks on c_arrays.
        for i in range(len(c_arrays)):
            cj_i = c_arrays[i]

            for j in range(i + 1, len(c_arrays)):
                cj_j = c_arrays[j]

                self.assertEqual(cj_i.shape, cj_j.shape)
                self.assertTrue(np.all(cj_i["i"] == cj_j["i"]))
                self.assertTrue(np.all(cj_i["j"] == cj_j["j"]))
                self.assertLess(
                    np.max(np.abs((cj_i["tca"] - cj_j["tca"]) / cj_j["tca"])), 1e-11
                )
                self.assertLess(
                    np.max(np.linalg.norm(cj_i["ri"] - cj_j["ri"], axis=1)), 1e-7
                )
                self.assertLess(
                    np.max(np.linalg.norm(cj_i["rj"] - cj_j["rj"], axis=1)), 1e-7
                )

        # Build a conjunctions object with all masked otypes.
        # There cannot be aabb collisions or conjunctions.
        c = conj(
            pt,
            conj_thresh=10.0,
            conj_det_interval=1.0,
            otypes=[otype.MASKED] * pt.n_objs,
        )

        self.assertEqual(len(c.otypes), pt.n_objs)

        for i in range(c.n_cd_steps):
            self.assertEqual(len(c.get_aabb_collisions(i)), 0)

        self.assertEqual(len(c.conjunctions), 0)

        # Same with all secondaries.
        c = conj(
            pt,
            conj_thresh=10.0,
            conj_det_interval=1.0,
            otypes=[otype.SECONDARY] * pt.n_objs,
        )

        self.assertEqual(len(c.otypes), pt.n_objs)

        for i in range(c.n_cd_steps):
            self.assertEqual(len(c.get_aabb_collisions(i)), 0)

        self.assertEqual(len(c.conjunctions), 0)

        # Try with a mix of secondaries and masked.
        otypes = [otype.SECONDARY] * (pt.n_objs // 2)
        otypes += [otype.MASKED] * (pt.n_objs - pt.n_objs // 2)

        c = conj(
            pt,
            conj_thresh=10.0,
            conj_det_interval=1.0,
            otypes=otypes,
        )

        self.assertEqual(len(c.otypes), pt.n_objs)

        for i in range(c.n_cd_steps):
            self.assertEqual(len(c.get_aabb_collisions(i)), 0)

        self.assertEqual(len(c.conjunctions), 0)

    def test_empty_traj(self):
        # Test to check that a polyjectory containing one or more
        # empty trajectories works as expected.
        from .. import conjunctions as conj, polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times
        import numpy as np

        # Construct a trajectory with zero steps.
        tcs_shape = list(_planar_circ_tcs.shape)
        tcs_shape[0] = 0
        tcs_no_steps = np.zeros(tuple(tcs_shape), dtype=float)

        pj = polyjectory(
            [_planar_circ_tcs, tcs_no_steps], [_planar_circ_times, []], [0, 0]
        )
        c = conj(pj, conj_thresh=1.0, conj_det_interval=0.1)
        self.assertEqual(len(c.conjunctions), 0)

        pj = polyjectory(
            [_planar_circ_tcs, tcs_no_steps, tcs_no_steps],
            [_planar_circ_times, [], []],
            [0, 0, 0],
        )
        c = conj(pj, conj_thresh=1.0, conj_det_interval=0.1)
        self.assertEqual(len(c.conjunctions), 0)

        pj = polyjectory(
            [tcs_no_steps, _planar_circ_tcs, tcs_no_steps],
            [[], _planar_circ_times, []],
            [0, 0, 0],
        )
        c = conj(pj, conj_thresh=1.0, conj_det_interval=0.1)
        self.assertEqual(len(c.conjunctions), 0)

        pj = polyjectory(
            [tcs_no_steps, tcs_no_steps, _planar_circ_tcs],
            [[], [], _planar_circ_times],
            [0, 0, 0],
        )
        c = conj(pj, conj_thresh=1.0, conj_det_interval=0.1)
        self.assertEqual(len(c.conjunctions), 0)

    def test_nonzero_tbegin(self):
        # Simple test for a single object
        # whose trajectory begins at t > 0.
        from .. import conjunctions as conj, polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times
        import numpy as np

        # Deterministic seeding.
        rng = np.random.default_rng(420)

        # Shift up the times.
        _planar_circ_times = _planar_circ_times + 1.0

        # Single planar circular orbit case.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        # Run the test for several conjunction detection intervals.
        for conj_det_interval in [0.01, 0.1, 0.5, 2.0, 5.0, 7.0, 10.0]:
            c = conj(pj, conj_thresh=0.1, conj_det_interval=conj_det_interval)

            # Shape checks.
            self.assertEqual(c.aabbs.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_aabbs.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_aabbs.shape, c.aabbs.shape)
            self.assertEqual(c.mcodes.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_mcodes.shape[0], c.cd_end_times.shape[0])
            self.assertEqual(c.srt_idx.shape[0], c.cd_end_times.shape[0])

            # The conjunction detection end time must coincide
            # with the trajectory end time.
            self.assertEqual(c.cd_end_times[-1], pj[0][1][-1])

            # The global aabbs must all coincide
            # exactly with the only object's aabbs.
            self.assertTrue(np.all(c.aabbs[:, 0] == c.aabbs[:, 1]))
            # With only one object, aabbs and srt_aabbs must be identical.
            self.assertTrue(np.all(c.aabbs == c.srt_aabbs))

            # In the z and r coordinates, all aabbs
            # should be of size circa 0.1 accounting for the
            # conjunction threshold.
            self.assertTrue(np.all(c.aabbs[:, 0, 0, 2] >= -0.05001))
            self.assertTrue(np.all(c.aabbs[:, 0, 1, 2] <= 0.05001))

            self.assertTrue(np.all(c.aabbs[:, 0, 0, 3] >= 1 - 0.05001))
            self.assertTrue(np.all(c.aabbs[:, 0, 1, 3] <= 1 + 0.05001))

            # Verify the aabbs.
            self._verify_conj_aabbs(c, pj, rng)

            # No aabb collisions or conjunctions expected.
            for i in range(c.n_cd_steps):
                self.assertEqual(len(c.get_aabb_collisions(i)), 0)
            self.assertEqual(len(c.conjunctions), 0)

    def test_cd_begin_end(self):
        # Test to check for correctness with trajectories
        # beginning and ending within a conjunction step.
        from .. import conjunctions, polyjectory
        import numpy as np

        # NOTE: in these tests, we have 2 objects initially
        # placed on the x axis at +-1. The two objects
        # move with uniform unitary speed towards the origin,
        # where they will meet at t = 1.

        # The overall time data runs at regular 0.1 intervals from 0 to 2.
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

        # First case: no overlap between the two trajectories.

        # The time data for object 1 goes up to 0.9
        tm_data_0 = tm_data[:10]
        # The time data for object 2 goes from 1.1 to the end.
        tm_data_1 = tm_data[11:]

        # Construct the trajectory data for the first object, moving
        # right to left.
        traj_data_0 = []
        for tm in tm_data_0[1:]:
            tdata = np.zeros((7, 4))
            tdata[0, 0] = 1.0 - (tm - 0.1)
            tdata[0, 1] = -1.0
            tdata[3, 0] = -1.0

            traj_data_0.append(tdata)
        traj_data_0 = np.ascontiguousarray(np.array(traj_data_0).transpose((0, 2, 1)))

        # Construct the trajectory data for the second object, moving
        # left to right.
        traj_data_1 = []
        for tm in tm_data_1[1:]:
            tdata = np.zeros((7, 4))
            tdata[0, 0] = -(1.0 - (tm - 0.1))
            tdata[0, 1] = 1.0
            tdata[3, 0] = 1.0

            traj_data_1.append(tdata)
        traj_data_1 = np.ascontiguousarray(np.array(traj_data_1).transpose((0, 2, 1)))

        # Construct the polyjectory.
        pj = polyjectory([traj_data_0, traj_data_1], [tm_data_0, tm_data_1], [0, 0])

        # Run conjunction detection.
        cj = conjunctions(pj=pj, conj_thresh=1e-6, conj_det_interval=2.0 / 3.0)

        # We must not detect any conjunction because the conjunection happens
        # when there is no data for both trajectories.
        self.assertEqual(len(cj.conjunctions), 0)

        # Second case: overlap between the two trajectories. The overlap occurs
        # in the second conjunction step.
        tm_data_0 = tm_data[:12]
        tm_data_1 = tm_data[9:]

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
        cj = conjunctions(pj=pj, conj_thresh=1e-6, conj_det_interval=2.0 / 3.0)
        # Here we must detect the conjunction.
        conjs = cj.conjunctions
        self.assertEqual(len(conjs), 1)
        self.assertTrue(np.all(conjs["i"] == 0))
        self.assertTrue(np.all(conjs["j"] == 1))
        self.assertAlmostEqual(conjs["tca"][0], 1.0, places=15)
        self.assertAlmostEqual(conjs["dca"][0], 0.0, delta=1e-15)
        self.assertTrue(np.allclose(conjs["ri"][0], [0, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["rj"][0], [0, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["vi"][0], [-1, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["vj"][0], [1, 0, 0], atol=1e-15, rtol=0.0))

        # Third case: both trajectories beginning staggered within the
        # conjunction step, no conjunction.
        tm_data_0 = tm_data[11:]
        tm_data_1 = tm_data[12:]

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
        cj = conjunctions(pj=pj, conj_thresh=1e-6, conj_det_interval=2.0 / 3.0)
        self.assertEqual(len(cj.conjunctions), 0)

        # Fourth case: both trajectories beginning staggered within the
        # conjunction step, with conjunction.
        tm_data_0 = tm_data[9:]
        tm_data_1 = tm_data[8:]

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
        cj = conjunctions(pj=pj, conj_thresh=1e-6, conj_det_interval=2.0 / 3.0)
        conjs = cj.conjunctions
        self.assertEqual(len(conjs), 1)
        self.assertTrue(np.all(conjs["i"] == 0))
        self.assertTrue(np.all(conjs["j"] == 1))
        self.assertAlmostEqual(conjs["tca"][0], 1.0, places=15)
        self.assertAlmostEqual(conjs["dca"][0], 0.0, delta=1e-15)
        self.assertTrue(np.allclose(conjs["ri"][0], [0, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["rj"][0], [0, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["vi"][0], [-1, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["vj"][0], [1, 0, 0], atol=1e-15, rtol=0.0))

        # Fifth case: both trajectories ending staggered within the conjunction step,
        # no conjunction.
        tm_data_0 = tm_data[:9]
        tm_data_1 = tm_data[:8]

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
        cj = conjunctions(pj=pj, conj_thresh=1e-6, conj_det_interval=2.0 / 3.0)
        self.assertEqual(len(cj.conjunctions), 0)

        # Sixth case: both trajectories ending staggered within the conjunction step,
        # with conjunction.
        tm_data_0 = tm_data[:11]
        tm_data_1 = tm_data[:12]

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
        cj = conjunctions(pj=pj, conj_thresh=1e-6, conj_det_interval=2.0 / 3.0)
        conjs = cj.conjunctions
        # NOTE: we assert >=1 conjunctions are detected,
        # as here we are at the limits of the numerics and a second
        # spurious conjunction might be detected.
        self.assertGreaterEqual(len(conjs), 1)
        self.assertTrue(np.all(conjs["i"] == 0))
        self.assertTrue(np.all(conjs["j"] == 1))
        self.assertAlmostEqual(conjs["tca"][0], 1.0, places=15)
        self.assertAlmostEqual(conjs["dca"][0], 0.0, delta=1e-15)
        self.assertTrue(np.allclose(conjs["ri"][0], [0, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["rj"][0], [0, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["vi"][0], [-1, 0, 0], atol=1e-15, rtol=0.0))
        self.assertTrue(np.allclose(conjs["vj"][0], [1, 0, 0], atol=1e-15, rtol=0.0))
