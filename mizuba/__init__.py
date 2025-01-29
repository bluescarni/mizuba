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

# Version setup.
from ._version import __version__

# We import the sub-modules into the root namespace.
from .core import *

del core

from . import test, data_sources
from ._sgp4_polyjectory import make_sgp4_polyjectory, sgp4_pj_status
from enum import IntEnum
from ._logging import _setup_logger


def _astropy_trigger_utc_tai() -> None:
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

# Setup the logger.
_setup_logger()

del _setup_logger


class otype(IntEnum):
    # NOTE: the numbering here is set up to match the codes
    # used in C++.
    PRIMARY = 1
    SECONDARY = 2
    MASKED = 4


del IntEnum


def _have_sgp4_deps() -> bool:
    # Helper to check if we have all the dependencies
    # necessary to support TLE propagation via sgp4.
    try:
        import sgp4

        return True
    except ImportError:
        return False


def _check_sgp4_deps() -> None:
    # Throwing variant of the previous function.
    if not _have_sgp4_deps():
        raise ImportError(
            "Support for TLE propagation via SGP4 requires the following Python modules: sgp4"
        )


def _have_heyoka_deps() -> bool:
    # Helper to check if we have all the dependencies
    # necessary to support propagation via heyoka.
    try:
        import heyoka

        return True
    except ImportError:
        return False
