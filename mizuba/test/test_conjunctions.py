# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class conjunctions_test_case(_ut.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from skyfield.api import load
            from skyfield.iokit import parse_tle_file
        except ImportError:
            return

        from ._sgp4_test_data_202407 import sgp4_test_tle

        # Load the test TLEs.
        ts = load.timescale()
        sat_list = list(
            parse_tle_file(
                (bytes(_, "ascii") for _ in sgp4_test_tle.split("\n")),
                ts,
            )
        )

        # A sparse list of satellites.
        # NOTE: we manually include an object for which the
        # trajectory data terminates early (but only if the exit_radius
        # is set to 12000).
        cls.sparse_sat_list = sat_list[::2000] + [sat_list[220]]

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
    def _verify_conj_aabbs(self, c, rng):
        import numpy as np

        # Fetch the polyjectory.
        pj = c.polyjectory

        # For every conjunction step, pick random times within,
        # evaluate the polyjectory at the corresponding times and
        # assert that the positions are within the aabbs.
        for cd_idx, end_time in enumerate(c.cd_end_times):
            begin_time = 0.0 if cd_idx == 0 else c.cd_end_times[cd_idx - 1]

            # Pick 5 random times.
            random_times = rng.uniform(begin_time, end_time, (5,))

            # Fetch the global aabb for this conjunction step.
            global_lb = c.aabbs[cd_idx, pj.nobjs, 0]
            global_ub = c.aabbs[cd_idx, pj.nobjs, 1]

            # Iterate over all objects.
            for obj_idx in range(pj.nobjs):
                # Fetch the polyjectory data for the current object.
                traj, traj_times, status = pj[obj_idx]

                # Fetch the AABB of the object.
                aabb = c.aabbs[cd_idx, obj_idx]

                if begin_time >= traj_times[-1]:
                    # The trajectory data for the current object
                    # ends before the beginning of the current conjunction
                    # step. Skip the current object and assert that its
                    # aabb is infinite.
                    self.assertTrue(np.all(np.isinf(aabb)))
                    continue
                else:
                    self.assertTrue(np.all(np.isfinite(aabb)))

                # The aabb must be included in the global one.
                self.assertTrue(np.all(aabb[0] >= global_lb))
                self.assertTrue(np.all(aabb[1] <= global_ub))

                # Iterate over the random times.
                for time in random_times:
                    # Look for the trajectory step which ends at
                    # or after 'time'.
                    step_idx = np.searchsorted(traj_times, time)

                    # Skip the current time if there's no corresponding
                    # trajectory data.
                    if step_idx == len(traj_times):
                        continue

                    # Fetch the polynomials for all state variables
                    # in the trajectory step.
                    traj_polys = traj[step_idx]

                    # Compute the poly evaluation interval.
                    # This is the time elapsed since the beginning
                    # of the trajectory step.
                    h = time - (0.0 if step_idx == 0 else traj_times[step_idx - 1])

                    # Evaluate the polynomials and check that
                    # the results fit in the aabb.
                    for coord_idx, aabb_idx in zip([0, 1, 2, 6], range(4)):
                        pval = np.polyval(traj_polys[coord_idx, ::-1], h)
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
            "The conjunction threshold must be finite and positive, but instead a value of"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            conj(conj_det_interval=0.0, pj=pj, conj_thresh=float(np.finfo(float).max))
        self.assertTrue(
            "is too large and results in an overflow error" in str(cm.exception)
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

        # Test accessors.
        c = conj(pj, conj_thresh=1.0, conj_det_interval=0.1)

        self.assertEqual(c.n_cd_steps, len(c.cd_end_times))
        self.assertTrue(isinstance(c.bvh_node, np.dtype))
        self.assertTrue(isinstance(c.aabb_collision, np.dtype))
        self.assertTrue(isinstance(c.conj, np.dtype))

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

        # polyjectory.
        rc = sys.getrefcount(c)
        pj = c.polyjectory
        self.assertEqual(sys.getrefcount(c), rc + 1)
        with self.assertRaises(AttributeError) as cm:
            c.polyjectory = pj

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

    def test_main(self):
        import numpy as np
        from .. import conjunctions as conj, polyjectory
        from ._planar_circ import _planar_circ_tcs, _planar_circ_times

        # Deterministic seeding.
        rng = np.random.default_rng(42)

        # Single planar circular orbit case.
        pj = polyjectory([_planar_circ_tcs], [_planar_circ_times], [0])

        # Run the test for several conjunction detection intervals.
        for conj_det_interval in [0.01, 0.1, 0.5, 2.0, 5.0, 7.0]:
            c = conj(
                pj, conj_thresh=0.1, conj_det_interval=conj_det_interval, whitelist=[]
            )

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
            self._verify_conj_aabbs(c, rng)

            # No aabb collisions or conjunctions expected.
            for i in range(c.n_cd_steps):
                self.assertEqual(len(c.get_aabb_collisions(i)), 0)
            self.assertEqual(len(c.conjunctions), 0)

            # Test whitelist initialisation.
            c = conj(
                pj, conj_thresh=0.1, conj_det_interval=conj_det_interval, whitelist=[0]
            )

            with self.assertRaises(ValueError) as cm:
                conj(
                    pj,
                    conj_thresh=0.1,
                    conj_det_interval=conj_det_interval,
                    whitelist=[1],
                )
            self.assertTrue(
                "Invalid whitelist detected: the whitelist contains the object index 1, but the total number of objects is only 1"
                in str(cm.exception)
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

        from .. import sgp4_polyjectory

        # Use the sparse satellite list.
        sat_list = self.sparse_sat_list

        begin_jd = 2460496.5

        # Build the polyjectory.
        pt, mask = sgp4_polyjectory(
            sat_list, begin_jd, begin_jd + 0.25, exit_radius=12000.0
        )
        tot_nobjs = pt.nobjs

        # Build the conjunctions object. Keep a small threshold not to interfere
        # with aabb checking.
        c = conj(pt, conj_thresh=1e-8, conj_det_interval=1.0)

        # Verify the aabbs.
        self._verify_conj_aabbs(c, rng)

        # Shape checks.
        self.assertEqual(c.aabbs.shape, c.srt_aabbs.shape)
        self.assertEqual(c.mcodes.shape, c.srt_mcodes.shape)
        self.assertEqual(c.srt_idx.shape, (c.n_cd_steps, c.polyjectory.nobjs))

        # The global aabbs must be the same in srt_aabbs.
        self.assertTrue(
            np.all(
                c.aabbs[:, c.polyjectory.nobjs, :, :]
                == c.srt_aabbs[:, c.polyjectory.nobjs, :, :]
            )
        )

        # The individual aabbs for the objects will differ.
        self.assertFalse(
            np.all(
                c.aabbs[:, : c.polyjectory.nobjs, :, :]
                == c.srt_aabbs[:, : c.polyjectory.nobjs, :, :]
            )
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
            self.assertEqual(
                sorted(c.srt_idx[cd_idx]), list(range(c.polyjectory.nobjs))
            )

            self.assertTrue(
                np.all(
                    c.aabbs[cd_idx, c.srt_idx[cd_idx], :, :]
                    == c.srt_aabbs[cd_idx, : c.polyjectory.nobjs, :, :]
                )
            )

            self.assertTrue(
                np.all(c.mcodes[cd_idx, c.srt_idx[cd_idx]] == c.srt_mcodes[cd_idx])
            )

        # The last satellite's trajectory data terminates
        # early. After termination, the morton codes must be -1.
        last_aabbs = c.aabbs[:, c.polyjectory.nobjs - 1, :, :]
        self.assertFalse(np.all(np.isfinite(last_aabbs)))
        inf_idx = np.isinf(last_aabbs).nonzero()[0]
        self.assertTrue(np.all(c.mcodes[inf_idx, -1] == ((1 << 64) - 1)))

        # Similarly, the number of objects reported in the root
        # node of the bvh trees must be tot_nobjs - 1.
        for idx in inf_idx:
            t = c.get_bvh_tree(idx)
            self.assertEqual(t[0]["end"] - t[0]["begin"], tot_nobjs - 1)

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

        pj = polyjectory([[tdata, tdata, tdata]], [[1.0, 2.0, 3.0]], [0])

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
        tdata0 = np.zeros((7, 6))
        tdata0[0, 0] = 1.0
        tdata1 = np.zeros((7, 6))
        tdata1[0, 0] = -1.0

        # y.
        tdata2 = np.zeros((7, 6))
        tdata2[1, 0] = 1.0
        tdata3 = np.zeros((7, 6))
        tdata3[1, 0] = -1.0

        # z.
        tdata4 = np.zeros((7, 6))
        tdata4[2, 0] = 1.0
        tdata5 = np.zeros((7, 6))
        tdata5[2, 0] = -1.0

        # Center.
        tdata6 = np.zeros((7, 6))

        # All ones.
        tdata7 = np.zeros((7, 6))
        tdata7[:, 0] = 1

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
            [[1.0]] * 10 + [[1.0, 2.0]] * 8,
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
        tdata = np.zeros((7, 6))
        tdata[:, 1] = 0.1

        pj = polyjectory([[tdata]], [[1.0]], [0])
        conjs = conjunctions(pj, 1e-16, 1.0)

        with self.assertRaises(IndexError) as cm:
            conjs.get_bvh_tree(1)
        self.assertTrue(
            "Cannot fetch the BVH tree for the conjunction timestep at index 1: the total number of conjunction steps is only 1"
            in str(cm.exception)
        )

        t = conjs.get_bvh_tree(0)
        self.assertEqual(len(t), 1)

        # Polyjectory with two identical objects.
        # This will result in exhausting all bits
        # in the morton codes for splitting.
        pj = polyjectory([[tdata], [tdata]], [[1.0], [1.0]], [0, 0])
        conjs = conjunctions(pj, 1e-16, 1.0)
        t = conjs.get_bvh_tree(0)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["begin"], 0)
        self.assertEqual(t[0]["end"], 2)
        self.assertEqual(t[0]["parent"], -1)
        self.assertEqual(t[0]["left"], -1)
        self.assertEqual(t[0]["right"], -1)

        # Polyjectory in which the morton codes
        # of two objects differ at the last bit.
        # x.
        tdata0 = np.zeros((7, 6))
        tdata0[0, 0] = 1.0
        tdata1 = np.zeros((7, 6))
        tdata1[0, 0] = -1.0

        # y.
        tdata2 = np.zeros((7, 6))
        tdata2[1, 0] = 1.0
        tdata3 = np.zeros((7, 6))
        tdata3[1, 0] = -1.0

        # z.
        tdata4 = np.zeros((7, 6))
        tdata4[2, 0] = 1.0
        tdata5 = np.zeros((7, 6))
        tdata5[2, 0] = -1.0

        # Center.
        tdata6 = np.zeros((7, 6))

        # All ones.
        tdata7 = np.zeros((7, 6))
        tdata7[:, 0] = 1

        # All ones but last.
        tdata8 = np.zeros((7, 6))
        tdata8[:, 0] = 1
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
            [[1.0]] * 9,
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

        from .. import sgp4_polyjectory, conjunctions as conj
        import numpy as np

        sat_list = self.half_sat_list

        begin_jd = 2460496.5

        # Build the polyjectory. Run it for only 15 minutes.
        pt, mask = sgp4_polyjectory(sat_list, begin_jd, begin_jd + 15.0 / 1440.0)

        # Build the conjunctions object. This will trigger
        # the internal C++ sanity checks in debug mode.
        c = conj(pt, conj_thresh=10.0, conj_det_interval=1.0)

        self.assertTrue(
            all(len(c.get_aabb_collisions(_)) > 0 for _ in range(c.n_cd_steps))
        )

        with self.assertRaises(IndexError) as cm:
            c.get_aabb_collisions(c.n_cd_steps)
        self.assertTrue(
            f"Cannot fetch the list of AABB collisions for the conjunction timestep at index {c.n_cd_steps}: the total number of conjunction steps is only {c.n_cd_steps}"
            in str(cm.exception)
        )

        # The conjunctions must be sorted according
        # to the TCA.
        self.assertTrue(np.all(np.diff(c.conjunctions["tca"]) >= 0))

        # All conjunctions must happen before the polyjectory end time.
        self.assertTrue(c.conjunctions["tca"][-1] < 15.0)

        # No conjunction must be at or above the threshold.
        self.assertTrue(np.all(np.diff(c.conjunctions["dca"]) < 10))

        # Objects cannot have conjunctions with themselves.
        self.assertTrue(np.all(c.conjunctions["i"] != c.conjunctions["j"]))

        # DCA must be consistent with state vectors.
        self.assertTrue(
            np.all(
                np.isclose(
                    np.linalg.norm(c.conjunctions["ri"] - c.conjunctions["rj"], axis=1),
                    c.conjunctions["dca"],
                    rtol=1e-13,
                )
            )
        )

        # Verify the conjunctions with the sgp4 python module.
        sl_array = np.array(sat_list)[mask]
        for cj in c.conjunctions:
            # Fetch the conjunction data.
            tca = cj["tca"]
            dca = cj["dca"]
            i, j = cj["i"], cj["j"]
            ri, rj = cj["ri"], cj["rj"]
            vi, vj = cj["vi"], cj["vj"]

            # Fetch the satellite models.
            sat_i = sl_array[i].model
            sat_j = sl_array[j].model

            ei, sri, svi = sat_i.sgp4(begin_jd, tca / 1440.0)
            ej, srj, svj = sat_j.sgp4(begin_jd, tca / 1440.0)

            diff_ri = np.linalg.norm(sri - ri)
            diff_rj = np.linalg.norm(srj - rj)

            diff_vi = np.linalg.norm(svi - vi)
            diff_vj = np.linalg.norm(svj - vj)

            # NOTE: unit of measure here is [km], vs typical
            # values of >1e3 km in the coordinates. Thus, relative
            # error is 1e-11, absolute error is ~10µm.
            self.assertLess(diff_ri, 1e-8)
            self.assertLess(diff_rj, 1e-8)

            # NOTE: unit of measure here is [km/s], vs typicial
            # velocity values of >1 km/s.
            self.assertLess(diff_vi, 1e-11)
            self.assertLess(diff_vj, 1e-11)
