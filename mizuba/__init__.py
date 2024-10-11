# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Version setup.
from ._version import __version__

# We import the sub-modules into the root namespace.
from .core import *

del core

from . import test


class stopwatch:
    # A minimal stopwatch class, inspired by spdlog.
    def __init__(self):
        import time

        self._timestamp = time.monotonic_ns()

    def reset(self):
        import time

        self._timestamp = time.monotonic_ns()

    def __repr__(self):
        import time

        return f"{(time.monotonic_ns() - self._timestamp) / 1e9}"


def _sgp4_pre_filter_sat_list(sat_list, jd_begin, exit_radius, reentry_radius):
    try:
        from sgp4.api import Satrec, SatrecArray
    except ImportError:
        raise ImportError(
            "The 'sgp4' module is required in order to use the sgp4_polyjectory() function"
        )
    import numpy as np
    import logging

    # Fetch the logger.
    logger = logging.getLogger("mizuba")

    # Init the stopwatch.
    sw = stopwatch()

    if len(sat_list) == 0:
        raise ValueError(
            "The sgp4_polyjectory() function requires a non-empty list of satellites in input"
        )

    # Supported types for the objects in sat_list.
    supported_types = [Satrec]
    try:
        from skyfield.sgp4lib import EarthSatellite

        supported_types.append(EarthSatellite)
    except ImportError:
        pass
    supported_types = tuple(supported_types)

    if not all(isinstance(sat, supported_types) for sat in sat_list):
        raise TypeError(
            "The sgp4_polyjectory() function requires in input a list of Satrec objects from the 'sgp4' module or EarthSatellite objects from the 'skyfield' module"
        )

    # Turn into a list of Satrec, if necessary.
    sat_list = [sat if isinstance(sat, Satrec) else sat.model for sat in sat_list]

    # Turn into an array.
    sat_list = np.array(sat_list)

    # NOTE: these constants are taken from the wgs72 model and they are only
    # used to detect deep-space objects. I do not think it is necessary to allow
    # for customisability here.
    KE = 0.07436691613317342
    J2 = 1.082616e-3

    # Helper to compute the un-Kozaied mean motion from
    # the TLE elements. This is used to detect deep-space objects.
    # NOTE: this is ported directly from the official C++ source code.
    def no_unkozai(sat):
        from math import cos, sqrt

        no_kozai = sat.no_kozai
        ecco = sat.ecco
        inclo = sat.inclo

        cosio = cos(inclo)
        cosio2 = cosio * cosio
        eccsq = ecco * ecco
        omeosq = 1 - eccsq
        rteosq = sqrt(omeosq)

        d1 = 0.75 * J2 * (3 * cosio2 - 1) / (rteosq * omeosq)
        ak = (KE / no_kozai) ** (2.0 / 3)
        del_ = d1 / (ak * ak)
        adel = ak * (1 - del_ * del_ - del_ * (1.0 / 3 + 134 * del_ * del_ / 81))
        del_ = d1 / (adel * adel)
        ret = no_kozai / (1 + del_)

        return ret

    # Filter out deep-space objects.
    mask = 2 * np.pi / np.array([no_unkozai(sat) for sat in sat_list]) < 225.0

    # Build an sgp4 propagator.
    sat_arr = SatrecArray(sat_list)

    # Propagate the state of all satellites at jd_begin.
    e, r, v = sat_arr.sgp4(np.array([jd_begin]), np.array([0.0]))

    # Mask out the satellites that:
    # - generated an error code, or
    # - ended up at an invalid distance, or
    # - contain non-finite data.
    dist = np.linalg.norm(r[:, 0], axis=1)
    mask = np.logical_and.reduce(
        (
            mask,
            e[:, 0] == 0,
            dist < exit_radius,
            dist > reentry_radius,
            np.all(np.isfinite(r[:, 0]), axis=1),
            np.all(np.isfinite(v[:, 0]), axis=1),
        )
    )

    ret_list = list(sat_list[mask])

    if len(ret_list) == 0:
        raise ValueError(
            "Pre-filtering the satellite list during the construction of an sgp4_polyjectory resulted in an empty list - that is, the propagation of all satellites at jd_begin resulted in either an error or an invalid state vector"
        )

    logger.info(f"SGP4 satellite list pre-filter time: {sw}s")

    return ret_list, mask


# Logger setup.
def _setup_logger():
    import logging

    # Create the logger.
    logger = logging.getLogger("mizuba")

    # Set up the formatter.
    formatter = logging.Formatter(
        fmt=r"%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S",
    )

    # Create a handler.
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)

    # Link handler to logger.
    logger.addHandler(c_handler)


_setup_logger()
