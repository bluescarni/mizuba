# Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the mizuba library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


def run_test_suite():
    import unittest as _ut

    from . import (
        test_sgp4_polyjectory,
        test_conjunctions,
        test_polyjectory,
        test_heyoka_conjunctions,
    )

    retval = 0

    tl = _ut.TestLoader()

    suite = tl.loadTestsFromTestCase(test_conjunctions.conjunctions_test_case)
    suite.addTest(
        tl.loadTestsFromTestCase(test_heyoka_conjunctions.heyoka_conjunctions_test_case)
    )
    suite.addTest(
        tl.loadTestsFromTestCase(test_sgp4_polyjectory.sgp4_polyjectory_test_case)
    )
    suite.addTest(tl.loadTestsFromTestCase(test_polyjectory.polyjectory_test_case))

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1

    if retval != 0:
        raise RuntimeError("One or more tests failed.")
