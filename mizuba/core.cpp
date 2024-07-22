// Copyright 2024 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mizuba library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/cast.h>
#include <ranges>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "poly_trajectory.hpp"

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    using namespace py::literals;
    namespace mz = mizuba;

    // Disable automatic function signatures in the docs.
    // NOTE: the 'options' object needs to stay alive
    // throughout the whole definition of the module.
    py::options options;
    options.disable_function_signatures();

    m.doc() = "The core mizuba module";

    // poly_trajectory.
    py::class_<mz::poly_trajectory> pt_cl(m, "poly_trajectory", py::dynamic_attr{});
    pt_cl.def(
        py::init([](py::iterable trajs, py::iterable times) {
            auto traj_trans = [](const auto &o) {
                auto arr = o.template cast<py::array_t<double>>();

                if (arr.ndim() != 3) [[unlikely]] {
                    // TODO
                    throw;
                }

                if (arr.shape(1) != 7) [[unlikely]] {
                    // TODO
                    throw;
                }

                // TODO check C contiguous and aligned.

                return mz::poly_trajectory::traj_span_t(arr.data(), boost::numeric_cast<py::ssize_t>(arr.shape(0)),
                                                        boost::numeric_cast<py::ssize_t>(arr.shape(2)));
            };

            auto time_trans = [](const auto &o) {
                auto arr = o.template cast<py::array_t<double>>();

                if (arr.ndim() != 1) [[unlikely]] {
                    // TODO
                    throw;
                }

                // TODO check C contiguous and aligned.

                return mz::poly_trajectory::time_span_t(arr.data(), boost::numeric_cast<py::ssize_t>(arr.shape(0)));
            };

            return mz::poly_trajectory(trajs | std::views::transform(traj_trans),
                                       times | std::views::transform(time_trans));
        }),
        "trajs"_a.noconvert(), "times"_a.noconvert());
    pt_cl.def(
        "get_obj_data",
        [](const py::object &self, std::size_t i) {
            const auto *p = py::cast<const mz::poly_trajectory *>(self);
            const auto [traj_span, time_span] = p->get_obj_data(i);

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

            return py::make_tuple(std::move(traj_ret), std::move(time_ret));
        },
        "i"_a.noconvert());
}
