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


def _setup_logger() -> None:
    # Helper for the initial setup of the logger.
    import logging

    # Create the logger.
    logger = logging.getLogger("mizuba")

    # Set up the formatter.
    formatter = logging.Formatter(
        fmt=r"[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S",
    )

    # Create a handler.
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)

    # Link handler to logger.
    logger.addHandler(c_handler)
