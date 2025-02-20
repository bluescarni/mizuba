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


class tmpdir_test_case(_ut.TestCase):
    def test_setget(self):
        from .. import set_tmpdir, get_tmpdir, polyjectory
        from pathlib import Path
        import tempfile
        import numpy as np

        state_data = np.zeros((1, 8, 7))

        # Fetch the original tmpdir.
        orig_tmpdir = get_tmpdir()

        # Try to set to None or empty string.
        set_tmpdir(None)
        self.assertIsNone(get_tmpdir())
        set_tmpdir("")
        self.assertIsNone(get_tmpdir())

        # Try to set to non-empty string.
        set_tmpdir("aaa")
        self.assertEqual(get_tmpdir(), Path("aaa"))

        # Use the polyjectory class to check that creation
        # of temp dirs respects the tmpdir setting.
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname).resolve()
            set_tmpdir(path)

            # This is for checking that the dir is empty.
            self.assertTrue(not any(path.iterdir()))

            pj = polyjectory(
                trajs=[state_data, state_data[1:]],
                times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                status=np.array([0, 0], dtype=np.int32),
            )

            # Now the directory must not be empty.
            self.assertTrue(any(path.iterdir()))

            # Remove the poljectory data.
            pj.detach()
            self.assertTrue(not any(path.iterdir()))

        # An example in which the specified tmpdir does not exit.
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname).resolve()
            set_tmpdir(path / "foo")

            with self.assertRaises(ValueError) as cm:
                polyjectory(
                    trajs=[state_data, state_data[1:]],
                    times=[np.array([0.0, 1.0]), np.array([], dtype=float)],
                    status=np.array([0, 0], dtype=np.int32),
                )
            self.assertTrue(
                "Unable to canonicalise the path to the temporary dir"
                in str(cm.exception)
            )

        # Restore the original tmpdir.
        set_tmpdir(orig_tmpdir)
