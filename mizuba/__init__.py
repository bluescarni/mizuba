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

# Version setup.
from ._version import __version__

# We import the sub-modules into the root namespace.
from .core import *

del core

from . import test, data_sources
from ._sgp4_polyjectory import make_sgp4_polyjectory

from enum import IntEnum


def _astropy_trigger_utc_tai():
    # NOTE: this is a workaround for an issue
    # in erfa, which could in principle lead to
    # crashes in case astropy UTC/TAI conversions
    # are performed in a multithreaded context:
    #
    # https://github.com/liberfa/erfa/issues/103
    #
    # By triggering a UTC->TAI conversion at import
    # time, we are at least ensuring that the builtin
    # leap seconds table has been correctly initialised.
    #
    # Note that changing the leap seconds table at runtime
    # is also not thread safe, but we are never doing that.
    from astropy.time import Time

    Time(2460669.0, format="jd", scale="utc").tai


_astropy_trigger_utc_tai()

del _astropy_trigger_utc_tai


class sgp4_pj_status(IntEnum):
    # NOTE: the numbering here is set up to match the status
    # codes coming out of the sgp4 polyjectory ctor.
    OK = 0
    REENTRY = 1
    EXIT = 2
    NONFINITE_STATE = 3
    DEEP_SPACE = 4
    DUPLICATE = 5
    SGP4_ERROR_MEAN_ECC = 11
    SGP4_ERROR_MEAN_MEAN_MOTION = 12
    SGP4_ERROR_PERT_ECC = 13
    SGP4_ERROR_SEMI_LATUS_RECTUM = 14
    # NOTE: here we are skipping SGP4 error code 5, which should not be in use
    # any more in the latest SGP4 implementations.
    SGP4_DECAY = 16


class otype(IntEnum):
    # NOTE: the numbering here is set up to match the codes
    # used in C++.
    PRIMARY = 1
    SECONDARY = 2
    MASKED = 4


del IntEnum


def _have_sgp4_deps():
    # Helper to check if we have all the dependencies
    # necessary to support TLE propagation via sgp4.
    try:
        import skyfield
        import sgp4
        import pandas

        return True
    except ImportError:
        return False


def _check_sgp4_deps():
    # Throwing variant of the previous function.
    if not _have_sgp4_deps():
        raise ImportError(
            "Support for TLE propagation via SGP4 requires the following Python"
            " modules: sgp4, skyfield, pandas and astropy"
        )


def _have_heyoka_deps():
    # Helper to check if we have all the dependencies
    # necessary to support propagation via heyoka.
    try:
        import heyoka

        return True
    except ImportError:
        return False


def _sgp4_detect_duplicates(sat_list):
    # A function to detect duplicates that are sometimes
    # present in the satellite catalogues downloaded from
    # space-track.org. We identify duplicates based on exact
    # matching of the TLE data. A boolean flag vector flagging
    # the unique satellites in sat_list is returned.
    #
    # In order to detect duplicates, we put the TLE elements,
    # the epoch and the bstar in a structured numpy array and then
    # use the unique() function.
    import numpy as np

    # Prepare the dtype.
    dtype_fields = [
        "epochyr",
        "epochdays",
        "ndot",
        "nddot",
        "bstar",
        "inclo",
        "nodeo",
        "ecco",
        "argpo",
        "mo",
        "no_kozai",
    ]
    dtype = np.dtype([(f, float) for f in dtype_fields])

    # Create the structured array.
    sat_arr = np.array(
        [tuple(getattr(sat, f) for f in dtype_fields) for sat in sat_list], dtype=dtype
    )

    # Run unique().
    unique_idx = np.unique(sat_arr, return_index=True)[1]

    # Turn unique_idx into a mask array.
    unique_mask = np.zeros(
        (
            len(
                sat_list,
            )
        ),
        dtype=bool,
    )
    unique_mask[unique_idx] = True

    return unique_mask


def _sgp4_pre_filter_sat_list(orig_sat_list, jd_begin, exit_radius, reentry_radius):
    # This helper is invoked during the creation of an sgp4 polyjectory.
    # It will pre-filter the lists of satellites orig_sat_list, removing:
    #
    # - duplicates,
    # - deep-space objects,
    # - objects whose sgp4 propagation at jd_begin:
    #   - returns an error code, or
    #   - results in a nonfinite state, or
    #   - results in the object being outside the domain exit
    #     radius or inside the reentry radius.
    #
    # The return values are:
    #
    # - the filtered-out version of orig_sat_list,
    # - a numpy boolean mask that can be used to turn orig_sat_list
    #   into its filtered-out version,
    # - a pandas dataframe containing information about all the satellites
    #   in orig_sat_list, including their IDs, names, TLEs and status at
    #   jd_begin.

    from sgp4.api import Satrec, SatrecArray
    import pandas as pd
    from skyfield.sgp4lib import EarthSatellite
    import numpy as np

    if len(orig_sat_list) == 0:
        raise ValueError(
            "The sgp4_polyjectory() function requires a non-empty list of satellites in"
            " input"
        )

    # Supported types for the objects in orig_sat_list.
    supported_types = (Satrec, EarthSatellite)

    if not all(isinstance(sat, supported_types) for sat in orig_sat_list):
        raise TypeError(
            "The sgp4_polyjectory() function requires in input a list of Satrec objects"
            " from the 'sgp4' module or EarthSatellite objects from the 'skyfield'"
            " module"
        )

    # Turn orig_sat_list into an array of Satrec for advanced indexing.
    sat_list = np.array(
        [sat if isinstance(sat, Satrec) else sat.model for sat in orig_sat_list]
    )

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

    # Detect deep-space objects.
    dp_mask = 2 * np.pi / np.array([no_unkozai(sat) for sat in sat_list]) < 225.0

    # Detect duplicate satellites.
    unique_mask = _sgp4_detect_duplicates(sat_list)

    # Build an sgp4 propagator.
    sat_arr = SatrecArray(sat_list)

    # Propagate the state of all satellites at jd_begin.
    e, r, v = sat_arr.sgp4(np.array([jd_begin]), np.array([0.0]))

    # Detect objects for which sgp4 propagation errored out.
    sgp4_error_mask = e[:, 0] == 0

    # Compute the distance of all objects from the centre of the Earth.
    dist = np.linalg.norm(r[:, 0], axis=1)

    # Detect objects that exited the simulation domain.
    exit_mask = dist < exit_radius

    # Detect objects that had a re-entry.
    reentry_mask = dist > reentry_radius

    # Detect objects for which the propagation
    # resulted in a non-finite state.
    nonfinite_mask = np.logical_and(
        np.all(np.isfinite(r[:, 0]), axis=1),
        np.all(np.isfinite(v[:, 0]), axis=1),
    )

    # Put all masks together.
    mask = np.logical_and.reduce(
        (
            dp_mask,
            unique_mask,
            sgp4_error_mask,
            exit_mask,
            reentry_mask,
            nonfinite_mask,
        )
    )

    # Construct the filtered satellite list.
    ret_list = list(sat_list[mask])

    if len(ret_list) == 0:
        raise ValueError(
            "Pre-filtering the satellite list during the construction of an"
            " sgp4_polyjectory resulted in an empty list - that is, the propagation of"
            " all satellites at jd_begin resulted in either an error or an invalid"
            " state vector"
        )

    # Construct the dataframe.

    # The primary index, corresponding to the satellite's TLE number.
    idx = [
        sat.satnum if isinstance(sat, Satrec) else sat.model.satnum
        for sat in orig_sat_list
    ]

    # Create the name column, if possible.
    name_col = [
        (
            "name",
            [None if isinstance(sat, Satrec) else sat.name for sat in orig_sat_list],
        )
    ]

    # Labels of the data columns.
    col_labels = [
        "jdsatepoch",
        "jdsatepochF",
        "bstar",
        "inclo",
        "nodeo",
        "ecco",
        "argpo",
        "mo",
        "no_kozai",
    ]

    # Create the data columns.
    data_cols = [
        (
            c_name,
            [
                (
                    getattr(sat, c_name)
                    if isinstance(sat, Satrec)
                    else getattr(sat.model, c_name)
                )
                for sat in orig_sat_list
            ],
        )
        for c_name in col_labels
    ]

    # Dictionary to transform an sgp4 int error code into an sgp4_pj_status.
    # NOTE: here we are skipping error code 5, which should not be in use
    # any more in the latest SGP4 implementations.
    err_status_dict = {
        1: sgp4_pj_status.SGP4_ERROR_MEAN_ECC,
        2: sgp4_pj_status.SGP4_ERROR_MEAN_MEAN_MOTION,
        3: sgp4_pj_status.SGP4_ERROR_PERT_ECC,
        4: sgp4_pj_status.SGP4_ERROR_SEMI_LATUS_RECTUM,
    }

    # Create the status columns and the model column.
    status_codes = []
    status_msgs = []
    models = []
    for i, sat in enumerate(orig_sat_list):
        # Fetch and append the satellite's model.
        md = sat if isinstance(sat, Satrec) else sat.model
        models.append(md)

        # Init the status with OK.
        status = sgp4_pj_status.OK

        # First we check if the satellite is a deep-space
        # one or a duplicate. These properties do not depend
        # on the propagation at jd_begin, they are inferred directly
        # from the TLEs.
        if not dp_mask[i]:
            status = sgp4_pj_status.DEEP_SPACE
        elif not unique_mask[i]:
            status = sgp4_pj_status.DUPLICATE
        # Next, we check the results of propagation at jd_begin.
        #
        # NOTE: the logic here mirrors, for consistency, the logic implemented
        # when constructing the sgp4 polyjectory. Specifically, we check, in order:
        #
        # - the SGP4 error codes,
        # - nonfinite state being generated by the propagation at jd_begin,
        # - SGP4 decay,
        # - reentry or exit conditions.
        elif e[i, 0] != 0 and e[i, 0] != 6:
            status = err_status_dict[e[i, 0]]
        elif not nonfinite_mask[i]:
            status = sgp4_pj_status.NONFINITE_STATE
        elif e[i, 0] == 6:
            status = sgp4_pj_status.SGP4_DECAY
        elif not exit_mask[i]:
            status = sgp4_pj_status.EXIT
        elif not reentry_mask[i]:
            status = sgp4_pj_status.REENTRY

        # Append the status code and message.
        status_codes.append(status)
        status_msgs.append(status.name)

    # Create the dataframe.
    df = pd.DataFrame(
        dict(
            name_col
            + data_cols
            + [("model", models)]
            + [("init_code", status_codes)]
            + [("init_msg", status_msgs)]
        ),
        index=idx,
    )

    return ret_list, mask, df


def _sgp4_set_final_status(pj, df):
    # This function is called at the end of the sgp4_polyjectory()
    # function, after the propagation of the polyjectory pj. It will
    # add 2 extra columns to the dataframe df, which was generated
    # by the _sgp4_pre_filter_sat_list() function. The two new columns
    # represent the status codes and messages for the objects at the
    # end of the polyjectory.

    import numpy as np

    # Fetch the status vector from the polyjectory.
    pj_status = pj.status

    # Duplicate the 'init_code' and 'init_msg' columns
    # into the new 'final_code' and 'final_msg' columns.
    df["final_code"] = df["init_code"]
    df["final_msg"] = df["init_msg"]

    # Fetch the mask of the objects in df which also
    # appear in the polyjectory.
    mask = df["init_code"] == sgp4_pj_status.OK

    # Overwrite the original status codes and error
    # messages with the statuses from the polyjectory.
    df.loc[mask, "final_code"] = pj_status
    df.loc[mask, "final_msg"] = [sgp4_pj_status(_).name for _ in pj_status]

    return df


def make_sgp4_conjunctions_df(cj, df, jd_begin):
    _check_sgp4_deps()

    import pandas as pd
    from astropy import time

    # Fetch the conjunctions.
    conjs = cj.conjunctions

    # Mask out df, we will be operating on the list
    # of objects which appear in the polyjectory.
    df = df[df["init_code"] == sgp4_pj_status.OK]

    # Fetch the name column from df.
    name_col = df["name"]

    # Fetch the names of the objects.
    i_names = name_col.iloc[conjs["i"]]
    j_names = name_col.iloc[conjs["j"]]

    # Init the return value.
    retval = pd.DataFrame(
        {
            "i_name": i_names.reset_index(drop=True),
            "j_name": j_names.reset_index(drop=True),
        }
    )

    # Add the satellite numbers.
    retval["i_satnum"] = i_names.index
    retval["j_satnum"] = j_names.index

    # Convert the tcas to julian dates.
    tca_jd = time.Time(
        val=jd_begin, val2=conjs["tca"] / 1440.0, format="jd", scale="utc", precision=9
    )

    # Compute, and add to the dataframe, the days since epoch
    # for all the objects involved in conjunctions.

    # First we fetch the epochs of all objects in the polyjectory
    # as julian dates.
    jd_epochs = time.Time(
        val=df["jdsatepoch"],
        val2=df["jdsatepochF"],
        format="jd",
        scale="utc",
        precision=9,
    )

    # Then, for each conjunction, we compute the days since epoch
    # for the objects involved in the conjunction.
    retval["i_epoch_days"] = (tca_jd - jd_epochs[conjs["i"]]).jd
    retval["j_epoch_days"] = (tca_jd - jd_epochs[conjs["j"]]).jd

    # Add the tcas.
    retval["tca"] = conjs["tca"]
    retval["tca (UTC)"] = tca_jd.datetime

    # Add the dca.
    retval["dca"] = conjs["dca"]

    # Add ri/rj/vi/vj.
    retval["ri"] = list(conjs["ri"])
    retval["rj"] = list(conjs["rj"])
    retval["vi"] = list(conjs["vi"])
    retval["vj"] = list(conjs["vj"])

    return retval
