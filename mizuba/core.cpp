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
#include <iostream>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <Python.h>

#include <heyoka/mdspan.hpp>

#include "common_utils.hpp"
#include "conjunctions.hpp"
#include "polyjectory.hpp"
#include "sgp4_polyjectory.hpp"

namespace mizuba_py::detail
{

namespace
{

// TLDR: machinery to clean up polyjectories at Python shutdown.
//
// Python does not guarantee that all objects are garbage-collected at
// shutdown. This means that we may find ourselves in a situation where
// the temporary memory-mapped files used internally by a polyjectory
// are not deleted when the program terminates.
//
// In order to avoid this, we adopt the following approach:
//
// - every time a new polyjectory is constructed, we grab a weak pointer
//   to its implementation and store it in a global vector;
// - we register a cleanup function that, at shutdown, goes through
//   the unexpired weak pointers and manually closes the polyjectories,
//   thus ensuring that the temporary files are removed.
//
// All newly-created polyjectories which end up exposed as a py::object
// are affected by this issue. This means that the weak pointer registration
// should be done every time a new polyjectory is created in C++ before it is
// wrapped and returned as a py::object.
//
// For instance, both the polyjectory __init__() and the sgp4_polyjectory() factory need
// to register a weak pointer for the new polyjectory they create. OTOH, the
// 'polyjectory' property getter of a conjunctions object does not, because:
//
// - it is returning a reference to an existing polyjectory and not creating a new one, and
// - the reference it returns originates from a polyjectory that was originally constructed
//   on the Python side and then passed to the conjunctions' __init__(), which means a weak pointer
//   to it had already been registered.
//
// This all sounds unfortunately complicated, let us hope it does not get too messy :/
//
// NOTE: we will have to re-examine this approach if/when we implement des11n, as that
// results in a creation of a new Python-wrapped polyjectory without any weak pointer
// registration. Probably we will need to enforce the registration at unpickling time?
//
// NOTE: this approach results in unbounded size for the vector of weak pointers.
// If this ever becomes an issue, we can think about a more sophisticated implementation
// (which for instance could periodically remove expired weak pointers from the
// vector).

// Vector of weak pointers plus mutex for safe multithreaded access.
constinit std::vector<std::weak_ptr<mizuba::detail::polyjectory_impl>> pj_weak_ptr_vector;
constinit std::mutex pj_weak_ptr_mutex;

// Add a weak pointer to a polyjectory implementation to pj_weak_ptr_vector.
void add_pj_weak_ptr(const std::shared_ptr<mizuba::detail::polyjectory_impl> &ptr)
{
    std::lock_guard lock(pj_weak_ptr_mutex);

    pj_weak_ptr_vector.emplace_back(ptr);
}

// Cleanup polyjectory implementations that are still alive. This is meant to be run
// at program shutdown.
void cleanup_pj_weak_ptrs()
{
    std::lock_guard lock(pj_weak_ptr_mutex);

#if !defined(NDEBUG)
    std::cout << "Running the polyjectory cleanup function" << std::endl;

    // NOTE: we want to make sure that all non-expired weak pointers
    // are unique, because otherwise we will be closing the same polyjectory
    // twice, which is not allowed. Uniqueness of the weak pointers should be
    // guaranteed by the fact that all items added to pj_weak_ptr_vector are
    // constructed ex-novo.
    std::unordered_set<std::shared_ptr<mizuba::detail::polyjectory_impl>> ptr_set;
    for (const auto &wptr : pj_weak_ptr_vector) {
        if (auto sptr = wptr.lock()) {
            assert(ptr_set.insert(sptr).second); // LCOV_EXCL_LINE
        }
    }
    ptr_set.clear();

#endif

    for (auto &wptr : pj_weak_ptr_vector) {
        if (auto sptr = wptr.lock()) {
            // LCOV_EXCL_START
#if !defined(NDEBUG)
            std::cout << "Cleaning up a polyjectory still alive at shutdown" << std::endl;
#endif
            mizuba::detail::close_pj(sptr);
            // LCOV_EXCL_STOP
        }
    }
}

// NOTE: same exact scheme for the conjunctions class.
constinit std::vector<std::weak_ptr<mizuba::detail::conjunctions_impl>> cj_weak_ptr_vector;
constinit std::mutex cj_weak_ptr_mutex;

void add_cj_weak_ptr(const std::shared_ptr<mizuba::detail::conjunctions_impl> &ptr)
{
    std::lock_guard lock(cj_weak_ptr_mutex);

    cj_weak_ptr_vector.emplace_back(ptr);
}

void cleanup_cj_weak_ptrs()
{
    std::lock_guard lock(cj_weak_ptr_mutex);

#if !defined(NDEBUG)
    std::cout << "Running the conjunctions cleanup function" << std::endl;

    std::unordered_set<std::shared_ptr<mizuba::detail::conjunctions_impl>> ptr_set;
    for (const auto &wptr : cj_weak_ptr_vector) {
        if (auto sptr = wptr.lock()) {
            assert(ptr_set.insert(sptr).second); // LCOV_EXCL_LINE
        }
    }
    ptr_set.clear();

#endif

    for (auto &wptr : cj_weak_ptr_vector) {
        if (auto sptr = wptr.lock()) {
            // LCOV_EXCL_START
#if !defined(NDEBUG)
            std::cout << "Cleaning up a conjunctions object still alive at shutdown" << std::endl;
#endif
            mizuba::detail::close_cj(sptr);
            // LCOV_EXCL_STOP
        }
    }
}

} // namespace

} // namespace mizuba_py::detail

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
        py::init([](py::iterable trajs_, py::iterable times_, py::iterable status_) {
            auto traj_trans = [](const py::array_t<double> &arr) {
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
                mzpy::check_array_cc_aligned(arr, "All time arrays must be C contiguous and properly aligned");

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
            mzpy::check_array_cc_aligned(status, "The status array must be C contiguous and properly aligned");

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
                                  std::ranges::subrange(status_ptr, status_ptr + status.shape(0)));

            // Register the polyjectory implementation in the cleanup machinery.
            mzpy::detail::add_pj_weak_ptr(mz::detail::fetch_pj_impl(ret));

            return ret;
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

            // Register the polyjectory implementation in the cleanup machinery.
            mzpy::detail::add_pj_weak_ptr(mz::detail::fetch_pj_impl(poly_ret));

            return py::make_tuple(std::move(poly_ret), std::move(mask));
        },
        "sat_list"_a.noconvert(), "jd_begin"_a.noconvert(), "jd_end"_a.noconvert(),
        "exit_radius"_a.noconvert() = mz::sgp4_exit_radius, "reentry_radius"_a.noconvert() = mz::sgp4_reentry_radius);

    // Register conjunctions::bvh_node as a structured NumPy datatype.
    using bvh_node = mz::conjunctions::bvh_node;
    PYBIND11_NUMPY_DTYPE(bvh_node, begin, end, parent, left, right, lb, ub);
    // Same for conjunctions::aabb_collision.
    using aabb_collision = mz::conjunctions::aabb_collision;
    PYBIND11_NUMPY_DTYPE(aabb_collision, i, j);
    // Same for conjunctions::conj.
    using conj = mz::conjunctions::conj;
    PYBIND11_NUMPY_DTYPE(conj, tca, dca, i, j, ri, vi, rj, vj);

    // Conjunctions.
    py::class_<mz::conjunctions> conj_cl(m, "conjunctions", py::dynamic_attr{});
    conj_cl.def(py::init([](mz::polyjectory pj, double conj_thresh, double conj_det_interval,
                            std::vector<std::uint32_t> whitelist) {
                    auto ret = mz::conjunctions(std::move(pj), conj_thresh, conj_det_interval, std::move(whitelist));

                    // Register the conjunctions implementation in the cleanup machinery.
                    mzpy::detail::add_cj_weak_ptr(mz::detail::fetch_cj_impl(ret));

                    return ret;
                }),
                "pj"_a.noconvert(), "conj_thresh"_a.noconvert(), "conj_det_interval"_a.noconvert(),
                "whitelist"_a.noconvert() = std::vector<std::uint32_t>{});
    conj_cl.def_property_readonly("n_cd_steps", &mz::conjunctions::get_n_cd_steps);
    conj_cl.def_property_readonly("aabbs", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto aabbs_span = p->get_aabbs();

        // Turn into an array.
        auto ret = py::array_t<float>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(aabbs_span.extent(0)),
                                                                boost::numeric_cast<py::ssize_t>(aabbs_span.extent(1)),
                                                                boost::numeric_cast<py::ssize_t>(aabbs_span.extent(2)),
                                                                boost::numeric_cast<py::ssize_t>(aabbs_span.extent(3))},
                                      aabbs_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def_property_readonly("cd_end_times", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto cd_end_times_span = p->get_cd_end_times();

        // Turn into an array.
        auto ret = py::array_t<double>(
            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(cd_end_times_span.extent(0))},
            cd_end_times_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def_property_readonly("polyjectory", &mz::conjunctions::get_polyjectory);
    conj_cl.def_property_readonly("srt_aabbs", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_aabbs_span = p->get_srt_aabbs();

        // Turn into an array.
        auto ret
            = py::array_t<float>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(srt_aabbs_span.extent(0)),
                                                           boost::numeric_cast<py::ssize_t>(srt_aabbs_span.extent(1)),
                                                           boost::numeric_cast<py::ssize_t>(srt_aabbs_span.extent(2)),
                                                           boost::numeric_cast<py::ssize_t>(srt_aabbs_span.extent(3))},
                                 srt_aabbs_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def_property_readonly("mcodes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto mcodes_span = p->get_mcodes();

        // Turn into an array.
        auto ret = py::array_t<std::uint64_t>(
            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(mcodes_span.extent(0)),
                                      boost::numeric_cast<py::ssize_t>(mcodes_span.extent(1))},
            mcodes_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def_property_readonly("srt_mcodes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_mcodes_span = p->get_srt_mcodes();

        // Turn into an array.
        auto ret = py::array_t<std::uint64_t>(
            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(srt_mcodes_span.extent(0)),
                                      boost::numeric_cast<py::ssize_t>(srt_mcodes_span.extent(1))},
            srt_mcodes_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def_property_readonly("srt_idx", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_idx_span = p->get_srt_idx();

        // Turn into an array.
        auto ret = py::array_t<std::uint32_t>(
            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(srt_idx_span.extent(0)),
                                      boost::numeric_cast<py::ssize_t>(srt_idx_span.extent(1))},
            srt_idx_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def("get_bvh_tree", [](const py::object &self, std::size_t i) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the tree span.
        const auto tree_span = p->get_bvh_tree(i);

        // Turn into an array.
        auto ret
            = py::array_t<bvh_node>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(tree_span.extent(0))},
                                    tree_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    conj_cl.def("get_aabb_collisions", [](const py::object &self, std::size_t i) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the aabb collisions span.
        const auto aabb_collision_span = p->get_aabb_collisions(i);

        // Turn into an array.
        auto ret = py::array_t<aabb_collision>(
            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(aabb_collision_span.extent(0))},
            aabb_collision_span.data_handle(), self);

        // Ensure the returned array is read-only.
        ret.attr("flags").attr("writeable") = false;

        return ret;
    });
    // Expose static getters for the structured types.
    conj_cl.def_property_readonly_static("bvh_node", [](const py::object &) { return py::dtype::of<bvh_node>(); });
    conj_cl.def_property_readonly_static("aabb_collision",
                                         [](const py::object &) { return py::dtype::of<aabb_collision>(); });
    conj_cl.def_property_readonly_static("conj", [](const py::object &) { return py::dtype::of<conj>(); });

    // Register the polyjectory/conjunctions cleanup machinery.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        mzpy::detail::cleanup_pj_weak_ptrs();
        mzpy::detail::cleanup_cj_weak_ptrs();
    }));
}
