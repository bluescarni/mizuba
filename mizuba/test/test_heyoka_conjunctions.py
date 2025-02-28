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


class heyoka_conjunctions_test_case(_ut.TestCase):
    def _make_kep_ta(self, G, M, N):
        import heyoka as hy
        import numpy as np
        from copy import copy

        # The original dynamical model - Keplerian centre of attraction.
        orig_dyn = hy.model.fixed_centres(G, [M], [[0.0, 0.0, 0.0]])
        orig_vars = ["x", "y", "z", "vx", "vy", "vz"]

        # Create a dynamical model corresponding to N objects
        # non interacting with each other, attracted by the Keplerian
        # Earth. Also add an equation for the distance from the centre
        # of attraction.
        var_list = []
        dyn = []
        for i in range(N):
            new_vars = [hy.expression(_ + f"_{i}") for _ in orig_vars]
            new_vars += [hy.make_vars(f"r_{i}")]
            xi, yi, zi, vxi, vyi, vzi, ri = new_vars

            dsub = dict(zip(orig_vars, new_vars))
            new_dyn = hy.subs([_[1] for _ in orig_dyn], dsub)
            new_dyn += [(xi * vxi + yi * vyi + zi * vzi) / ri]

            var_list += new_vars
            dyn += new_dyn

        # List of conjunctions detected by heyoka.
        hy_conj_list = []

        # The conjunction event callback.
        class conj_cb:
            def __init__(self, i, j):
                self.i = i
                self.j = j

            def __call__(self, ta, time, d_sgn):
                # Compute the state of the system
                # at the point of minimum distance.
                # between objects i and j.
                ta.update_d_output(time)

                # Extract the state vectors
                # for objects i and j.
                st = ta.d_output.reshape(-1, 7)
                ri = st[self.i, 0:3]
                rj = st[self.j, 0:3]
                vi = st[self.i, 3:6]
                vj = st[self.j, 3:6]

                # Append to hy_conj_list:
                # - tca and dca,
                # - the indices of the objects,
                # - the state vectors.
                hy_conj_list.append(
                    (
                        self.i,
                        self.j,
                        time,
                        np.linalg.norm(ri - rj),
                        copy(ri),
                        copy(vi),
                        copy(rj),
                        copy(vj),
                    )
                )

        # Create the events for detecting conjunctions.
        ev_list = []
        svs_arr = np.array(var_list).reshape((-1, 7))
        for i in range(N):
            xi, yi, zi, vxi, vyi, vzi = svs_arr[i, :6]
            for j in range(i + 1, N):
                xj, yj, zj, vxj, vyj, vzj = svs_arr[j, :6]

                # The event equation.
                eq = (
                    (xi - xj) * (vxi - vxj)
                    + (yi - yj) * (vyi - vyj)
                    + (zi - zj) * (vzi - vzj)
                )

                # Create the event.
                ev_list.append(
                    hy.nt_event(
                        eq, conj_cb(i, j), direction=hy.event_direction.positive
                    )
                )

        # Build the integrator.
        ta = hy.taylor_adaptive(
            list(zip(var_list, dyn)), compact_mode=True, nt_events=ev_list
        )

        return ta, hy_conj_list

    # A test case in which we compare the results of conjunction
    # detection with a heyoka simulation in which we keep
    # track of the minimum distances between the objects.
    def test_tle(self):
        from .. import _have_sgp4_deps, _have_heyoka_deps, otype

        if not _have_sgp4_deps() or not _have_heyoka_deps():
            return

        import heyoka as hy
        from .. import conjunctions as conj, polyjectory
        import numpy as np
        import pathlib
        import polars as pl
        from sgp4.api import SatrecArray
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

        # Select around 10 objects.
        sat_list = sat_list[:: len(sat_list) // 10]
        N = len(sat_list)

        # Compute their positions at some date.
        jd = 2460496.5
        sat_arr = SatrecArray(sat_list)
        e, r, v = sat_arr.sgp4(np.array([jd]), np.array([0.0]))
        self.assertTrue(np.all(e == 0))

        # NOTE: gravitational constant in km**3/(kg * s**2).
        ta, hy_conj_list = self._make_kep_ta(6.67430e-20, 5.972168e24, N)

        # Setup the initial conditions.
        ic_rs = ta.state.reshape((-1, 7))
        for i in range(N):
            ic_rs[i, 0:3] = r[i][0]
            ic_rs[i, 3:6] = v[i][0]
            ic_rs[i, 6] = np.linalg.norm(ic_rs[i, 0:3])

        # Run the propagation and fetch the continuous output object.
        res = ta.propagate_for(86400.0, c_output=True)
        c_out = res[4]

        # Sort the heyoka conjunction list according to tca and transform it into the
        # appropriate structured dtype.
        hy_conj_list = np.sort(np.array(hy_conj_list, dtype=conj.conj), order="tca")

        # Build the polyjectory.
        trajs = []
        for i in range(N):
            trajs.append(
                np.ascontiguousarray(
                    c_out.tcs[:, i * 7 : (i + 1) * 7, :].transpose((0, 2, 1))
                )
            )
        pj = polyjectory(trajs, [c_out.times] * N, [0] * N)

        # Run first a conjunction detection with stupidly large conjunction threshold,
        # so that we detect all conjunctions.
        cj = conj(pj, 1e6, 60.0)

        # Filter out from the conjunctions list the conjunctions which happen
        # at the time boundaries of the polyjectory, since heyoka cannot detect these.
        cjc = cj.conjunctions[
            np.logical_and.reduce(
                (
                    np.abs(cj.conjunctions["tca"] - 86400.0) > 86400.0 * 1e-14,
                    cj.conjunctions["tca"] > 0,
                )
            )
        ]

        # Compare the results.
        self.assertEqual(len(cjc), len(hy_conj_list))
        self.assertTrue(np.all(np.isclose(cjc["tca"], hy_conj_list["tca"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["dca"], hy_conj_list["dca"], rtol=1e-12)))
        self.assertTrue(np.all(cjc["i"] == hy_conj_list["i"]))
        self.assertTrue(np.all(cjc["j"] == hy_conj_list["j"]))
        self.assertTrue(np.all(np.isclose(cjc["ri"], hy_conj_list["ri"], rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(cjc["rj"], hy_conj_list["rj"], rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(cjc["vi"], hy_conj_list["vi"], rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(cjc["vj"], hy_conj_list["vj"], rtol=1e-10)))

        # Re-run the same conjunction but with only 0 and 1 as primaries.
        otypes = [otype.SECONDARY] * pj.n_objs
        otypes[0] = otype.PRIMARY
        otypes[1] = otype.PRIMARY
        cj = conj(pj, 1e6, 60.0, otypes=otypes)
        flist = hy_conj_list[
            np.logical_or.reduce(
                (
                    hy_conj_list["i"] == 0,
                    hy_conj_list["i"] == 1,
                    hy_conj_list["j"] == 0,
                    hy_conj_list["j"] == 1,
                )
            )
        ]

        cjc = cj.conjunctions[
            np.logical_and.reduce(
                (
                    np.abs(cj.conjunctions["tca"] - 86400.0) > 86400.0 * 1e-14,
                    cj.conjunctions["tca"] > 0,
                )
            )
        ]

        self.assertEqual(len(cjc), len(flist))
        self.assertTrue(np.all(np.isclose(cjc["tca"], flist["tca"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["dca"], flist["dca"], rtol=1e-12)))
        self.assertTrue(np.all(cjc["i"] == flist["i"]))
        self.assertTrue(np.all(cjc["j"] == flist["j"]))
        self.assertTrue(np.all(np.isclose(cjc["ri"], flist["ri"], rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(cjc["rj"], flist["rj"], rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(cjc["vi"], flist["vi"], rtol=1e-10)))
        self.assertTrue(np.all(np.isclose(cjc["vj"], flist["vj"], rtol=1e-10)))

        # Run another conjunction detection, this time conjunction
        # thresh 500km.
        cj = conj(pj, 500.0, 60.0)

        hy_conj_list = hy_conj_list[hy_conj_list["dca"] < 500.0]

        cjc = cj.conjunctions[
            np.logical_and.reduce(
                (
                    np.abs(cj.conjunctions["tca"] - 86400.0) > 86400.0 * 1e-14,
                    cj.conjunctions["tca"] > 0,
                )
            )
        ]

        self.assertEqual(len(cjc), len(hy_conj_list))
        self.assertTrue(np.all(np.isclose(cjc["tca"], hy_conj_list["tca"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["dca"], hy_conj_list["dca"], rtol=1e-12)))
        self.assertTrue(np.all(cjc["i"] == hy_conj_list["i"]))
        self.assertTrue(np.all(cjc["j"] == hy_conj_list["j"]))
        self.assertTrue(np.all(np.isclose(cjc["ri"], hy_conj_list["ri"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["rj"], hy_conj_list["rj"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vi"], hy_conj_list["vi"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vj"], hy_conj_list["vj"], rtol=1e-12)))

        # Re-run the same conjunction but with a only 3 and 8 as primaries.
        otypes = [otype.SECONDARY] * pj.n_objs
        otypes[3] = otype.PRIMARY
        otypes[8] = otype.PRIMARY
        cj = conj(pj, 500.0, 60.0, otypes=otypes)
        flist = hy_conj_list[
            np.logical_or.reduce(
                (
                    hy_conj_list["i"] == 3,
                    hy_conj_list["i"] == 8,
                    hy_conj_list["j"] == 3,
                    hy_conj_list["j"] == 8,
                )
            )
        ]

        cjc = cj.conjunctions[
            np.logical_and.reduce(
                (
                    np.abs(cj.conjunctions["tca"] - 86400.0) > 86400.0 * 1e-14,
                    cj.conjunctions["tca"] > 0,
                )
            )
        ]

        self.assertEqual(len(cjc), len(flist))
        self.assertTrue(np.all(np.isclose(cjc["tca"], flist["tca"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["dca"], flist["dca"], rtol=1e-12)))
        self.assertTrue(np.all(cjc["i"] == flist["i"]))
        self.assertTrue(np.all(cjc["j"] == flist["j"]))
        self.assertTrue(np.all(np.isclose(cjc["ri"], flist["ri"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["rj"], flist["rj"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vi"], flist["vi"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vj"], flist["vj"], rtol=1e-12)))

        # Same with 200km.
        cj = conj(pj, 200.0, 60.0)

        hy_conj_list = hy_conj_list[hy_conj_list["dca"] < 200.0]

        cjc = cj.conjunctions[
            np.logical_and.reduce(
                (
                    np.abs(cj.conjunctions["tca"] - 86400.0) > 86400.0 * 1e-14,
                    cj.conjunctions["tca"] > 0,
                )
            )
        ]

        self.assertEqual(len(cjc), len(hy_conj_list))
        self.assertTrue(np.all(np.isclose(cjc["tca"], hy_conj_list["tca"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["dca"], hy_conj_list["dca"], rtol=1e-12)))
        self.assertTrue(np.all(cjc["i"] == hy_conj_list["i"]))
        self.assertTrue(np.all(cjc["j"] == hy_conj_list["j"]))
        self.assertTrue(np.all(np.isclose(cjc["ri"], hy_conj_list["ri"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["rj"], hy_conj_list["rj"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vi"], hy_conj_list["vi"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vj"], hy_conj_list["vj"], rtol=1e-12)))

        # Re-run the same conjunction but only with 9 and 2 as primaries.
        otypes = [otype.SECONDARY] * pj.n_objs
        otypes[9] = otype.PRIMARY
        otypes[2] = otype.PRIMARY
        cj = conj(pj, 200.0, 60.0, otypes=otypes)
        flist = hy_conj_list[
            np.logical_or.reduce(
                (
                    hy_conj_list["i"] == 9,
                    hy_conj_list["i"] == 2,
                    hy_conj_list["j"] == 9,
                    hy_conj_list["j"] == 2,
                )
            )
        ]

        cjc = cj.conjunctions[
            np.logical_and.reduce(
                (
                    np.abs(cj.conjunctions["tca"] - 86400.0) > 86400.0 * 1e-14,
                    cj.conjunctions["tca"] > 0,
                )
            )
        ]

        self.assertEqual(len(cjc), len(flist))
        self.assertTrue(np.all(np.isclose(cjc["tca"], flist["tca"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["dca"], flist["dca"], rtol=1e-12)))
        self.assertTrue(np.all(cjc["i"] == flist["i"]))
        self.assertTrue(np.all(cjc["j"] == flist["j"]))
        self.assertTrue(np.all(np.isclose(cjc["ri"], flist["ri"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["rj"], flist["rj"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vi"], flist["vi"], rtol=1e-12)))
        self.assertTrue(np.all(np.isclose(cjc["vj"], flist["vj"], rtol=1e-12)))

    def test_close_conjunction(self):
        # Test keplerian orbits leading to collisions.
        from .. import _have_heyoka_deps, otype
        from copy import copy

        if not _have_heyoka_deps():
            return

        import heyoka as hy
        from .. import conjunctions as conj, polyjectory
        import numpy as np

        orig_dyn = hy.model.fixed_centres(1.0, [1.0], [[0.0, 0.0, 0.0]])
        orig_vars = ["x", "y", "z", "vx", "vy", "vz"]

        N = 2

        # NOTE: run the same test with different
        # initial times for the trajectories.
        for init_time in [0.0, 0.2, 0.31]:
            ta, hy_conj_list = self._make_kep_ta(1.0, 1.0, N)
            ta.time = init_time

            # Setup the initial conditions.
            ic_rs = ta.state.reshape((-1, 7))

            ic_rs[0, 0] = 1.0
            ic_rs[0, 4] = 1.0
            ic_rs[0, 6] = 1.0

            ic_rs[1, 0] = -1.0
            ic_rs[1, 4] = 1.0
            ic_rs[1, 6] = 1.0

            c_out = ta.propagate_for(4.8, c_output=True)[4]

            # Build the polyjectory.
            trajs = []
            for i in range(N):
                trajs.append(
                    np.ascontiguousarray(
                        c_out.tcs[:, i * 7 : (i + 1) * 7, :].transpose((0, 2, 1))
                    )
                )
            pj = polyjectory(trajs, [c_out.times] * N, [0] * N)

            cj = conj(pj, 1e-4, 0.1)

            # Store the original conjunctions array for later use.
            orig_conj = copy(cj.conjunctions)

            hy_conj_arr = np.sort(np.array(hy_conj_list, dtype=conj.conj), order="tca")

            self.assertEqual(len(hy_conj_arr), 2)

            # Compare the results.
            self.assertEqual(len(cj.conjunctions), len(hy_conj_arr))
            self.assertTrue(
                np.all(
                    np.isclose(cj.conjunctions["tca"], hy_conj_arr["tca"], rtol=1e-12)
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(cj.conjunctions["dca"], hy_conj_arr["dca"], rtol=1e-12)
                )
            )
            self.assertTrue(np.all(cj.conjunctions["i"] == hy_conj_arr["i"]))
            self.assertTrue(np.all(cj.conjunctions["j"] == hy_conj_arr["j"]))
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["ri"], hy_conj_arr["ri"], rtol=1e-12))
            )
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["rj"], hy_conj_arr["rj"], rtol=1e-12))
            )
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["vi"], hy_conj_arr["vi"], rtol=1e-12))
            )
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["vj"], hy_conj_arr["vj"], rtol=1e-12))
            )

            # Try primary-secondary as otypes.
            cj = conj(pj, 1e-4, 0.1, otypes=[otype.PRIMARY, otype.SECONDARY])
            self.assertTrue(np.all(orig_conj == cj.conjunctions))

            cj = conj(pj, 1e-4, 0.1, otypes=[otype.SECONDARY, otype.PRIMARY])
            self.assertTrue(np.all(orig_conj == cj.conjunctions))

            # Try with several otype combination that must result
            # in no detected conjunctions.
            cj = conj(pj, 1e-4, 0.1, otypes=[otype.MASKED, otype.MASKED])
            self.assertEqual(len(cj.conjunctions), 0)

            cj = conj(pj, 1e-4, 0.1, otypes=[otype.SECONDARY, otype.SECONDARY])
            self.assertEqual(len(cj.conjunctions), 0)

            cj = conj(pj, 1e-4, 0.1, otypes=[otype.PRIMARY, otype.MASKED])
            self.assertEqual(len(cj.conjunctions), 0)

            cj = conj(pj, 1e-4, 0.1, otypes=[otype.MASKED, otype.PRIMARY])
            self.assertEqual(len(cj.conjunctions), 0)

            cj = conj(pj, 1e-4, 0.1, otypes=[otype.SECONDARY, otype.MASKED])
            self.assertEqual(len(cj.conjunctions), 0)

            cj = conj(pj, 1e-4, 0.1, otypes=[otype.MASKED, otype.SECONDARY])
            self.assertEqual(len(cj.conjunctions), 0)

            # Try an equatorial collision too.
            ta.time = init_time
            ta.state[:] = 0.0
            ta.reset_cooldowns()
            hy_conj_list.clear()

            ic_rs[0, 0] = 1.0
            ic_rs[0, 4] = 1.0
            ic_rs[0, 6] = 1.0

            ic_rs[1, 0] = -1.0
            ic_rs[1, 4] = 1.0
            ic_rs[1, 6] = 1.0

            c_out = ta.propagate_for(4.8, c_output=True)[4]

            # Build the polyjectory.
            trajs = []
            for i in range(N):
                trajs.append(
                    np.ascontiguousarray(
                        c_out.tcs[:, i * 7 : (i + 1) * 7, :].transpose((0, 2, 1))
                    )
                )
            pj = polyjectory(trajs, [c_out.times] * N, [0] * N)

            cj = conj(pj, 1e-4, 0.1)

            hy_conj_arr = np.sort(np.array(hy_conj_list, dtype=conj.conj), order="tca")

            self.assertEqual(len(hy_conj_arr), 2)

            # Compare the results.
            self.assertEqual(len(cj.conjunctions), len(hy_conj_arr))
            self.assertTrue(
                np.all(
                    np.isclose(cj.conjunctions["tca"], hy_conj_arr["tca"], rtol=1e-12)
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(cj.conjunctions["dca"], hy_conj_arr["dca"], rtol=1e-12)
                )
            )
            self.assertTrue(np.all(cj.conjunctions["i"] == hy_conj_arr["i"]))
            self.assertTrue(np.all(cj.conjunctions["j"] == hy_conj_arr["j"]))
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["ri"], hy_conj_arr["ri"], rtol=1e-12))
            )
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["rj"], hy_conj_arr["rj"], rtol=1e-12))
            )
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["vi"], hy_conj_arr["vi"], rtol=1e-12))
            )
            self.assertTrue(
                np.all(np.isclose(cj.conjunctions["vj"], hy_conj_arr["vj"], rtol=1e-12))
            )

            # Try two identical trajectories. This should produce no conjunctions.
            ta.time = init_time
            ta.state[:] = 0.0
            ta.reset_cooldowns()
            hy_conj_list.clear()

            ic_rs[0, 0] = 1.0
            ic_rs[0, 4] = 1.0
            ic_rs[0, 6] = 1.0

            ic_rs[1, 0] = 1.0
            ic_rs[1, 4] = 1.0
            ic_rs[1, 6] = 1.0

            c_out = ta.propagate_for(4.8, c_output=True)[4]

            # Build the polyjectory.
            trajs = []
            for i in range(N):
                trajs.append(
                    np.ascontiguousarray(
                        c_out.tcs[:, i * 7 : (i + 1) * 7, :].transpose((0, 2, 1))
                    )
                )
            pj = polyjectory(trajs, [c_out.times] * N, [0] * N)

            cj = conj(pj, 1e4, 0.1)

            self.assertEqual(len(cj.conjunctions), 0)
            self.assertEqual(len(hy_conj_list), 0)

    def test_boundary(self):
        # A test similar to the first test in test_boundary_conjunctions, but using Keplerian
        # orbits integrated with heyoka.
        from .. import _have_heyoka_deps

        if not _have_heyoka_deps():
            return

        import heyoka as hy
        from math import sqrt, pi
        import numpy as np
        from .. import polyjectory, conjunctions as conj

        # Setup a fixed-centre problem with two non-interacting
        # objects following Keplerian orbits. The first orbit
        # is circular, the second elliptic. The two objects will experience
        # conjunctions at the peri/apo centre of the elliptic orbit.
        N = 2
        ta, hy_conj_list = self._make_kep_ta(1.0, 1.0, N)

        # Setup the initial conditions.
        ic_rs = ta.state.reshape((-1, 7))

        # Eccentricity of the second orbit.
        ecc = 0.3

        ic_rs[0, 0] = 1.0
        ic_rs[0, 4] = 1.0
        ic_rs[0, 6] = 1.0

        ic_rs[1, 0] = 1.0 + ecc
        ic_rs[1, 4] = -1.0 * sqrt((1 - ecc) / (1 + ecc))
        ic_rs[1, 6] = 1.0 + ecc

        # Propagate until a short time before the second conjunction.
        c_out = ta.propagate_for(pi - 1e-6, c_output=True)[4]

        # NOTE: heyoka must detect only the initial conjunction at t == 0.
        self.assertEqual(len(hy_conj_list), 1)

        # Build the polyjectory.
        trajs = []
        for i in range(N):
            trajs.append(
                np.ascontiguousarray(
                    c_out.tcs[:, i * 7 : (i + 1) * 7, :].transpose((0, 2, 1))
                )
            )
        pj = polyjectory(trajs, [c_out.times] * N, [0] * N)

        # Run conjunction detection with a very large threshold.
        cj = conj(pj, 10000.0, 0.1)

        # NOTE: we must detect 2 conjunctions: the initial one at t=0
        # and the second one which does not correspond to a minimum
        # of the mutual distance.
        self.assertEqual(len(cj.conjunctions), 2)
        self.assertEqual(cj.conjunctions["tca"][0], 0.0)
        self.assertAlmostEqual(cj.conjunctions["tca"][1], pi - 1e-6, places=15)
        self.assertAlmostEqual(cj.conjunctions["dca"][0], ecc, places=15)

        # A second test in which we repeat the integration of the circular
        # orbit, but starting from (0, 1) rather than (1, 0), and we stop
        # again the integration right before the second conjunction. The objective
        # is to generate trajectory data for the circular orbit that begins
        # later than the other orbit.
        ta.reset_cooldowns()
        ta.time = pi / 2
        ic_rs[:] = 0.0

        ic_rs[0, 1] = 1.0
        ic_rs[0, 3] = -1.0
        ic_rs[0, 6] = 1.0
        ic_rs[1, 0] = 1.0 + ecc
        ic_rs[1, 4] = -1.0 * sqrt((1 - ecc) / (1 + ecc))
        ic_rs[1, 6] = 1.0 + ecc

        new_c_out = ta.propagate_for(pi / 2 - 1e-6, c_output=True)[4]

        # NOTE: no conjunctions must have been detected by heyoka in this
        # second integration.
        self.assertEqual(len(hy_conj_list), 1)

        # Build the trajectory data.
        trajs = []
        # Circular orbit.
        trajs.append(np.ascontiguousarray(new_c_out.tcs[:, :7, :].transpose((0, 2, 1))))
        # Elliptic orbit.
        trajs.append(np.ascontiguousarray(c_out.tcs[:, 7:14, :].transpose((0, 2, 1))))

        # Build the time data.
        times = []
        # Circular orbit.
        times.append(new_c_out.times)
        # Elliptic orbit.
        times.append(c_out.times)

        pj = polyjectory(trajs, times, [0] * N)

        # Run conjunction detection with a very large threshold.
        cj = conj(pj, 10000.0, 0.1)

        # NOTE: in addition to the conjunction at the end, we also have a conjunction
        # at the beginning of the circular trajectory (as the distance square
        # between the two objects at t = pi/2 is still increasing).
        self.assertEqual(len(cj.conjunctions), 2)
        self.assertAlmostEqual(cj.conjunctions["tca"][0], pi / 2, places=15)
        self.assertAlmostEqual(cj.conjunctions["tca"][1], pi - 1e-6, places=15)
