// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <Python.h>

#include <heyoka/mdspan.hpp>

#include "common_utils.hpp"
#include "polyjectory.hpp"
#include "sgp4_polyjectory.hpp"

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    namespace mz = mizuba;
    namespace mzpy = mizuba_py;
    namespace hy = heyoka;

    using namespace py::literals;

    // Disable automatic function signatures in the docs.
    // NOTE: the 'options' object needs to stay alive
    // throughout the whole definition of the module.
    py::options options;
    options.disable_function_signatures();

    m.doc() = "The core mizuba module";

    // polyjectory.
    py::class_<mz::polyjectory> pt_cl(m, "polyjectory", py::dynamic_attr{});
    pt_cl.def(
        py::init([](py::iterable trajs, py::iterable times, py::iterable status_) {
            auto traj_trans = [](const auto &o) {
                // Cast o to a NumPy array.
                auto arr = o.template cast<py::array_t<double>>();

                // Check shape/dimension.
                if (arr.ndim() != 3) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "A trajectory array must have 3 dimensions, but instead {} dimension(s) were detected",
                        arr.ndim()));
                }
                if (arr.shape(1) != 7) [[unlikely]] {
                    throw std::invalid_argument(fmt::format("A trajectory array must have a size of 7 in the second "
                                                            "dimension, but instead a size of {} was detected",
                                                            arr.shape(1)));
                }

                // Check contiguousness/alignment.
                mzpy::check_array_cc_aligned(arr, "All trajectory arrays must be C contiguous and properly aligned");

                return mz::polyjectory::traj_span_t(arr.data(), boost::numeric_cast<py::ssize_t>(arr.shape(0)),
                                                    boost::numeric_cast<py::ssize_t>(arr.shape(2)));
            };

            auto time_trans = [](const auto &o) {
                // Cast o to a NumPy array.
                auto arr = o.template cast<py::array_t<double>>();

                // Check dimensions.
                if (arr.ndim() != 1) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "A time array must have 1 dimension, but instead {} dimension(s) were detected", arr.ndim()));
                }

                // Check contiguousness/alignment.
                mzpy::check_array_cc_aligned(arr, "All time arrays must be C contiguous and properly aligned");

                return mz::polyjectory::time_span_t(arr.data(), boost::numeric_cast<py::ssize_t>(arr.shape(0)));
            };

            // Cast status to a NumPy array.
            auto status = status_.cast<py::array_t<std::int32_t>>();

            // Check dimensions.
            if (status.ndim() != 1) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "A status array must have 1 dimension, but instead {} dimension(s) were detected", status.ndim()));
            }

            // Check contiguousness/alignment.
            mzpy::check_array_cc_aligned(status, "The status array must be C contiguous and properly aligned");

            const auto *status_ptr = status.data();

            return mz::polyjectory(trajs | std::views::transform(traj_trans), times | std::views::transform(time_trans),
                                   std::ranges::subrange(status_ptr, status_ptr + status.shape(0)));
        }),
        "trajs"_a.noconvert(), "times"_a.noconvert(), "status"_a.noconvert());
    pt_cl.def_property_readonly("nobjs", &mz::polyjectory::get_nobjs);
    pt_cl.def_property_readonly("file_path", &mz::polyjectory::get_file_path);
    pt_cl.def_property_readonly("maxT", &mz::polyjectory::get_maxT);
    pt_cl.def_property_readonly("poly_order", &mz::polyjectory::get_poly_order);
    pt_cl.def(
        "__getitem__",
        [](const py::object &self, std::size_t i) {
            const auto *p = py::cast<const mz::polyjectory *>(self);

            // Fetch the spans and the status.
            const auto [traj_span, time_span, status] = (*p)[i];

            // Trajectory data.
            auto traj_ret
                = py::array_t<double>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(traj_span.extent(0)),
                                                                boost::numeric_cast<py::ssize_t>(traj_span.extent(1)),
                                                                boost::numeric_cast<py::ssize_t>(traj_span.extent(2))},
                                      traj_span.data_handle(), self);

            // Ensure the returned array is read-only.
            traj_ret.attr("flags").attr("writeable") = false;

            // Time data.
            auto time_ret
                = py::array_t<double>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(time_span.extent(0))},
                                      time_span.data_handle(), self);

            // Ensure the returned array is read-only.
            time_ret.attr("flags").attr("writeable") = false;

            return py::make_tuple(std::move(traj_ret), std::move(time_ret), status);
        },
        "i"_a.noconvert());

    // sgp4 polyjectory.
    m.def(
        "sgp4_polyjectory",
        [](py::list sat_list, double jd_begin, double jd_end, double exit_radius, double reentry_radius) {
            // Check and pre-filter sat_list.
            py::tuple filter_res = py::module_::import("mizuba").attr("_sgp4_pre_filter_sat_list")(
                sat_list, jd_begin, exit_radius, reentry_radius);
            sat_list = filter_res[0];
            py::object mask = filter_res[1];

            // Turn sat_list into a data vector.
            const auto sat_data = mzpy::sat_list_to_vector(sat_list);
            assert(sat_data.size() % 9u == 0u);

            // Create the input span.
            using span_t = hy::mdspan<const double, hy::extents<std::size_t, 9, std::dynamic_extent>>;
            const span_t in(sat_data.data(), boost::numeric_cast<std::size_t>(sat_data.size()) / 9u);

            auto poly_ret = [&]() {
                // NOTE: release the GIL during propagation.
                py::gil_scoped_release release;

                return mz::sgp4_polyjectory(in, jd_begin, jd_end, exit_radius, reentry_radius);
            }();

            return py::make_tuple(std::move(poly_ret), std::move(mask));
        },
        "sat_list"_a.noconvert(), "jd_begin"_a.noconvert(), "jd_end"_a.noconvert(),
        "exit_radius"_a.noconvert() = mz::sgp4_exit_radius, "reentry_radius"_a.noconvert() = mz::sgp4_reentry_radius);
}
