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

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "common_utils.hpp"
#include "expose_polyjectory.hpp"
#include "mdspan.hpp"
#include "polyjectory.hpp"

namespace mizuba_py
{

namespace detail
{

namespace
{

// Global data for use in add_weak_ptr_cleanup().
constinit std::mutex pj_weak_ptr_mutex;
constinit std::vector<std::weak_ptr<mizuba::detail::polyjectory_impl>> pj_weak_ptr_vector;
std::mt19937 pj_weak_ptr_rng;

// Helper to setup the selector in the exposition of polyjectory::state_(m)eval().
// NOTE: it is important that selector is passed in as a const reference, as the
// return value may return a reference to it.
auto eval_setup_selector(const auto &selector)
{
    namespace mz = mizuba;

    std::optional<mz::dspan_1d<const std::size_t>> sel;

    if (selector) {
        std::visit(
            [&sel]<typename V>(const V &v) {
                if constexpr (std::same_as<V, std::size_t>) {
                    // The selector is a single index, wrap it in a 1-element span.
                    sel.emplace(&v, 1);
                } else {
                    // The selector is an array, check it and wrap it in a span.
                    check_array_cc_aligned(
                        v, "The selector array passed to state_eval() must be C contiguous and properly aligned");

                    if (v.ndim() != 1) [[unlikely]] {
                        throw std::invalid_argument(
                            fmt::format("The selector array passed to state_eval() must have 1 dimension, but the "
                                        "number of dimensions is {} instead",
                                        v.ndim()));
                    }

                    sel.emplace(v.data(), boost::numeric_cast<std::size_t>(v.shape(0)));
                }
            },
            *selector);
    }

    return sel;
}

} // namespace

} // namespace detail

void expose_polyjectory(pybind11::module_ &m)
{
    namespace py = pybind11;
    namespace mz = mizuba;
    using namespace py::literals;

    // Register polyjectory::traj_offset as a structured NumPy datatype.
    using traj_offset = mz::polyjectory::traj_offset;
    PYBIND11_NUMPY_DTYPE(traj_offset, offset, n_steps);

    py::class_<mz::polyjectory> pt_cl(m, "polyjectory", py::dynamic_attr{});
    pt_cl.def(
        py::init([](py::iterable trajs_, py::iterable times_, py::iterable status_, double epoch, double epoch2) {
            auto traj_trans = [](const py::array_t<double> &arr) {
                // Check shape/dimension.
                if (arr.ndim() != 3) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "A trajectory array must have 3 dimensions, but instead {} dimension(s) were detected",
                        arr.ndim()));
                }
                if (arr.shape(1) != 7) [[unlikely]] {
                    // LCOV_EXCL_START
                    throw std::invalid_argument(fmt::format("A trajectory array must have a size of 7 in the second "
                                                            "dimension, but instead a size of {} was detected",
                                                            // LCOV_EXCL_STOP
                                                            arr.shape(1)));
                }

                // Check contiguousness/alignment.
                check_array_cc_aligned(arr, "All trajectory arrays must be C contiguous and properly aligned");

                return mz::polyjectory::traj_span_t(arr.data(), boost::numeric_cast<std::size_t>(arr.shape(0)),
                                                    boost::numeric_cast<std::size_t>(arr.shape(2)));
            };

            auto time_trans = [](const py::array_t<double> &arr) {
                // Check dimensions.
                if (arr.ndim() != 1) [[unlikely]] {
                    throw std::invalid_argument(fmt::format(
                        "A time array must have 1 dimension, but instead {} dimension(s) were detected", arr.ndim()));
                }

                // Check contiguousness/alignment.
                check_array_cc_aligned(arr, "All time arrays must be C contiguous and properly aligned");

                return mz::polyjectory::time_span_t(arr.data(), boost::numeric_cast<std::size_t>(arr.shape(0)));
            };

            // Cast status to a NumPy array.
            auto status = status_.cast<py::array_t<std::int32_t>>();

            // Check dimensions.
            if (status.ndim() != 1) [[unlikely]] {
                throw std::invalid_argument(fmt::format(
                    "A status array must have 1 dimension, but instead {} dimension(s) were detected", status.ndim()));
            }

            // Check contiguousness/alignment.
            check_array_cc_aligned(status, "The status array must be C contiguous and properly aligned");

            const auto *status_ptr = status.data();

            // NOTE: we need to pre-convert all the objects in traj_ and times_
            // into arrays. By doing this, we ensure that the spans we create in
            // traj/time_trans are referring to memory that stays alive during the
            // construction of the polyjectory.
            // NOTE: here it would be better to convert all objects to py::buffer
            // instead, which is more generic than numpy arrays. For instance, via
            // py::buffer we could ingest memory-mapped arrays without having to copy them.
            // The challenge seems to be in allowing for a good degree of flexibility: lists, for
            // instance, cannot be converted to py::buffer, so perhaps we should first
            // try a py::buffer conversion, and, if it fails, convert to py::array_t<double>
            // (to be stored in a separate vector) and then to py::buffer.
            std::vector<py::array_t<double>> trajs;
            for (auto o : trajs_) {
                trajs.push_back(o.cast<py::array_t<double>>());
            }

            std::vector<py::array_t<double>> times;
            for (auto o : times_) {
                times.push_back(o.cast<py::array_t<double>>());
            }

            auto ret
                = mz::polyjectory(trajs | std::views::transform(traj_trans), times | std::views::transform(time_trans),
                                  std::ranges::subrange(status_ptr, status_ptr + status.shape(0)), epoch, epoch2);

            // Register the polyjectory implementation in the cleanup machinery.
            add_pj_weak_ptr(mz::detail::fetch_pj_impl(ret));

            return ret;
        }),
        "trajs"_a.noconvert(), "times"_a.noconvert(), "status"_a.noconvert(), "epoch"_a.noconvert() = 0.,
        "epoch2"_a.noconvert() = 0.);
    pt_cl.def(py::init<const std::filesystem::path &, const std::filesystem::path &, std::uint32_t,
                       std::vector<traj_offset>, std::vector<std::int32_t>, double, double>(),
              "traj_file"_a.noconvert(), "time_file"_a.noconvert(), "order"_a.noconvert(), "traj_offsets"_a.noconvert(),
              "status"_a.noconvert(), "epoch"_a.noconvert() = 0., "epoch2"_a.noconvert() = 0.);
    pt_cl.def_property_readonly("nobjs", &mz::polyjectory::get_nobjs);
    pt_cl.def_property_readonly("maxT", &mz::polyjectory::get_maxT);
    pt_cl.def_property_readonly("epoch", &mz::polyjectory::get_epoch);
    pt_cl.def_property_readonly("poly_order", &mz::polyjectory::get_poly_order);
    pt_cl.def_property_readonly("status", [](const py::object &self) {
        const auto *p = py::cast<const mz::polyjectory *>(self);

        // Fetch the status span.
        const auto status_span = p->get_status();

        // Turn into an array and return.
        return mdspan_to_array(self, status_span);
    });
    pt_cl.def(
        "__getitem__",
        [](const py::object &self, std::size_t i) {
            const auto *p = py::cast<const mz::polyjectory *>(self);

            // Fetch the spans and the status.
            const auto [traj_span, time_span, status] = (*p)[i];

            // Trajectory data.
            auto traj_ret = mdspan_to_array(self, traj_span);

            // Time data.
            auto time_ret = mdspan_to_array(self, time_span);

            return py::make_tuple(std::move(traj_ret), std::move(time_ret), status);
        },
        "i"_a.noconvert());
    pt_cl.def(
        "state_eval",
        [](const mz::polyjectory &self, std::variant<double, py::array_t<double>> tm,
           std::optional<py::array_t<double>> out_,
           // NOTE: when documenting, we need to point out that in order to avoid
           // conversions or copies the numpy type to be used for indexing must be np.uintp.
           std::optional<std::variant<std::size_t, py::array_t<std::size_t>>> selector) {
            // Setup the selector argument.
            const auto sel = detail::eval_setup_selector(selector);

            // Check or setup the output array.
            auto out = [&out_, &sel, &self, &tm, &selector]() {
                if (out_) {
                    // Output array provided, check it.
                    auto ret = *out_;

                    check_array_cc_aligned(
                        ret, "The output array passed to state_eval() must be C contiguous and properly aligned");

                    // NOTE: we check the number of dimensions and the size in the second
                    // dimension. The size in the first dimension is checked within the C++ code.
                    if (ret.ndim() != 2) [[unlikely]] {
                        throw std::invalid_argument(
                            fmt::format("The output array passed to state_eval() must have 2 dimensions, but the "
                                        "number of dimensions is {} instead",
                                        ret.ndim()));
                    }
                    if (ret.shape(1) != 7) [[unlikely]] {
                        throw std::invalid_argument(
                            // LCOV_EXCL_START
                            fmt::format("The output array passed to state_eval() must have a size of 7 in the second "
                                        "dimension, but the size in the second dimension is {} instead",
                                        // LCOV_EXCL_STOP
                                        ret.shape(1)));
                    }

                    // If the output array is provided, we must ensure that it does not overlap
                    // with the time array or the selector. If it did, we may end up concurrently reading from
                    // and writing to the same memory areas during multithreaded operations.
                    //
                    // NOTE: overlaps between the time array and the selector are ok.
                    if (const auto *tm_arr = std::get_if<py::array_t<double>>(&tm)) {
                        if (may_share_memory(ret, *tm_arr)) [[unlikely]] {
                            throw std::invalid_argument("Potential memory overlap detected between the output array "
                                                        "passed to state_eval() and the time array");
                        }
                    }

                    if (selector) {
                        if (const auto *sel_arr = std::get_if<py::array_t<std::size_t>>(&*selector)) {
                            if (may_share_memory(ret, *sel_arr)) [[unlikely]] {
                                throw std::invalid_argument(
                                    "Potential memory overlap detected between the output array "
                                    "passed to state_eval() and the array of object indices");
                            }
                        }
                    }

                    return ret;
                } else {
                    // Create the output array.
                    return py::array_t<double>(py::array::ShapeContainer{
                        boost::numeric_cast<py::ssize_t>(sel ? sel->extent(0) : self.get_nobjs()),
                        static_cast<py::ssize_t>(7)});
                }
            }();

            // Prepare the output span.
            const auto out_span = mz::polyjectory::single_eval_span_t(out.mutable_data(),
                                                                      boost::numeric_cast<std::size_t>(out.shape(0)));

            // Visit on the time argument and run the evaluation.
            std::visit(
                [&self, &out_span, &sel]<typename V>(const V &v) {
                    // Prepare the time argument.
                    const auto tm_arg = [&v]() {
                        if constexpr (std::same_as<V, py::array_t<double>>) {
                            // Time provided in array form.

                            // Check contiguousness/alignment for the time array.
                            check_array_cc_aligned(
                                v, "The time array passed to state_eval() must be C contiguous and properly aligned");

                            // Check the number of dimensions for the time array.
                            // NOTE: the size in the first dimension will be checked in the C++ code.
                            if (v.ndim() != 1) [[unlikely]] {
                                throw std::invalid_argument(
                                    fmt::format("The time array passed to state_eval() must have 1 dimension, but the "
                                                "number of dimensions is {} instead",
                                                v.ndim()));
                            }

                            // Wrap the time array in a span.
                            return mz::dspan_1d<const double>(v.data(), boost::numeric_cast<std::size_t>(v.shape(0)));
                        } else {
                            // Time provided as a scalar, just return it.
                            return v;
                        }
                    }();

                    // NOTE: release the GIL during evaluation.
                    py::gil_scoped_release release;

                    self.state_eval(out_span, tm_arg, sel);
                },
                tm);

            return out;
        },
        "time"_a, "out"_a.noconvert() = py::none{}, "obj_idx"_a = py::none{});
    pt_cl.def(
        "state_meval",
        [](const mz::polyjectory &self, py::array_t<double> tm, std::optional<py::array_t<double>> out_,
           // NOTE: when documenting, we need to point out that in order to avoid
           // conversions or copies the numpy type to be used for indexing must be np.uintp.
           std::optional<std::variant<std::size_t, py::array_t<std::size_t>>> selector) {
            // Setup the selector argument.
            const auto sel = detail::eval_setup_selector(selector);

            // Check the time array.
            check_array_cc_aligned(tm,
                                   "The time array passed to state_meval() must be C contiguous and properly aligned");

            if (tm.ndim() != 1 && tm.ndim() != 2) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("The time array passed to state_meval() must have either 1 or 2 dimensions, but the "
                                "number of dimensions is {} instead",
                                tm.ndim()));
            }

            // Fetch the number of time evaluations.
            const auto n_time_evals = (tm.ndim() == 1) ? tm.shape(0) : tm.shape(1);

            // Check or setup the output array.
            auto out = [&out_, &sel, &self, n_time_evals, &tm, &selector]() {
                if (out_) {
                    // Output array provided, check it.
                    auto ret = *out_;

                    check_array_cc_aligned(
                        ret, "The output array passed to state_meval() must be C contiguous and properly aligned");

                    // NOTE: we check the number of dimensions and the size in the third
                    // dimension. The sizes in the first and second dimensions are checked within the C++ code.
                    if (ret.ndim() != 3) [[unlikely]] {
                        throw std::invalid_argument(
                            fmt::format("The output array passed to state_meval() must have 3 dimensions, but the "
                                        "number of dimensions is {} instead",
                                        ret.ndim()));
                    }
                    if (ret.shape(2) != 7) [[unlikely]] {
                        throw std::invalid_argument(
                            // LCOV_EXCL_START
                            fmt::format("The output array passed to state_meval() must have a size of 7 in the third "
                                        "dimension, but the size in the third dimension is {} instead",
                                        // LCOV_EXCL_STOP
                                        ret.shape(2)));
                    }

                    // If the output array is provided, we must ensure that it does not overlap
                    // with the time array or the selector. If it did, we may end up concurrently reading from
                    // and writing to the same memory areas during multithreaded operations.
                    //
                    // NOTE: overlaps between the time array and the selector are ok.
                    if (may_share_memory(ret, tm)) [[unlikely]] {
                        throw std::invalid_argument("Potential memory overlap detected between the output array "
                                                    "passed to state_meval() and the time array");
                    }

                    if (selector) {
                        if (const auto *sel_arr = std::get_if<py::array_t<std::size_t>>(&*selector)) {
                            if (may_share_memory(ret, *sel_arr)) [[unlikely]] {
                                throw std::invalid_argument(
                                    "Potential memory overlap detected between the output array "
                                    "passed to state_meval() and the array of object indices");
                            }
                        }
                    }

                    return ret;
                } else {
                    // Create the output array.
                    return py::array_t<double>(py::array::ShapeContainer{
                        boost::numeric_cast<py::ssize_t>(sel ? sel->extent(0) : self.get_nobjs()), n_time_evals,
                        static_cast<py::ssize_t>(7)});
                }
            }();

            // Prepare the output span.
            const auto out_span
                = mz::polyjectory::multi_eval_span_t(out.mutable_data(), boost::numeric_cast<std::size_t>(out.shape(0)),
                                                     boost::numeric_cast<std::size_t>(out.shape(1)));

            // Prepare the time array span and evaluate.
            if (tm.ndim() == 1) {
                const auto tm_span
                    = mz::dspan_1d<const double>(tm.data(), boost::numeric_cast<std::size_t>(tm.shape(0)));

                // NOTE: release the GIL during evaluation.
                py::gil_scoped_release release;

                self.state_meval(out_span, tm_span, sel);
            } else {
                const auto tm_span
                    = mz::dspan_2d<const double>(tm.data(), boost::numeric_cast<std::size_t>(tm.shape(0)),
                                                 boost::numeric_cast<std::size_t>(tm.shape(1)));

                // NOTE: release the GIL during evaluation.
                py::gil_scoped_release release;

                self.state_meval(out_span, tm_span, sel);
            }

            return out;
        },
        "time"_a, "out"_a.noconvert() = py::none{}, "obj_idx"_a = py::none{});
    pt_cl.def_property_readonly_static("traj_offset", [](const py::object &) { return py::dtype::of<traj_offset>(); });
}

// Add a weak pointer to a polyjectory implementation to pj_weak_ptr_vector.
void add_pj_weak_ptr(const std::shared_ptr<mizuba::detail::polyjectory_impl> &ptr)
{
    add_weak_ptr_cleanup(detail::pj_weak_ptr_mutex, detail::pj_weak_ptr_vector, detail::pj_weak_ptr_rng, ptr);
}

// Cleanup polyjectory implementations that are still alive. This is meant to be run
// at program shutdown.
void cleanup_pj_weak_ptrs()
{
    cleanup_weak_ptrs(detail::pj_weak_ptr_mutex, detail::pj_weak_ptr_vector, "polyjectory", &mizuba::detail::close_pj);
}

} // namespace mizuba_py
