# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class heyoka_conjunctions_test_case(_ut.TestCase):
    # A test case in which we compare the results of conjunction
    # detection with a heyoka simulation in which we keep
    # track of the minimum distances between the objects.
    def test_main(self):
        try:
            import heyoka as hy
            from sgp4.api import Satrec
        except ImportError:
            return
