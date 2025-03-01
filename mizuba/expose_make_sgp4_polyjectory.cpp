// Copyright 2024-2025 Francesco Biscani
//
// This file is part of the mizuba library.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "common_utils.hpp"
#include "expose_make_sgp4_polyjectory.hpp"
#include "expose_polyjectory.hpp"
#include "make_sgp4_polyjectory.hpp"
#include "mdspan.hpp"

namespace mizuba_py
{

void expose_make_sgp4_polyjectory(pybind11::module_ &m)
{
    namespace py = pybind11;
    namespace mz = mizuba;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace py::literals;

    // Register the GPE dtype.
    using gpe = mz::gpe;
    PYBIND11_NUMPY_DTYPE(gpe, norad_id, epoch_jd1, epoch_jd2, n0, e0, i0, node0, omega0, m0, bstar);
    m.attr("gpe_dtype") = py::dtype::of<gpe>();

    m.def(
        "_make_sgp4_polyjectory",
        [](const py::array_t<gpe> &gpes, const double jd_begin, const double jd_end, const double reentry_radius,
           const double exit_radius, std::optional<std::filesystem::path> data_dir, bool persist,
           std::optional<std::filesystem::path> tmpdir) {
            // Check the number of dimensions for gpes.
            if (gpes.ndim() != 1) [[unlikely]] {
                throw std::invalid_argument(fmt::format("The array of gpes passed to make_sgp4_polyjectory() must have "
                                                        "1 dimension, but the number of dimensions is {} instead",
                                                        gpes.ndim()));
            }

            // Check that gpes is C-contiguous and properly aligned.
            check_array_cc_aligned(
                gpes, "The array of gpes passed to make_sgp4_polyjectory() must be C contiguous and properly aligned");

            // Construct the span over the gpes.
            const auto gpes_span
                = mz::dspan_1d<const gpe>{gpes.data(), boost::numeric_cast<std::size_t>(gpes.shape(0))};

            // NOTE: release the GIL during the creation of the polyjectory.
            const py::gil_scoped_release release;

            auto ret = mz::make_sgp4_polyjectory(gpes_span, jd_begin, jd_end, reentry_radius, exit_radius,
                                                 std::move(data_dir), persist, std::move(tmpdir));

            // Register the polyjectory implementation in the cleanup machinery.
            add_pj_weak_ptr(mz::detail::fetch_pj_impl(ret));

            return ret;
        },
        "gpes"_a.noconvert(), "jd_begin"_a, "jd_end"_a, "reentry_radius"_a, "exit_radius"_a, "data_dir"_a, "persist"_a,
        "tmpdir"_a);
}

} // namespace mizuba_py
