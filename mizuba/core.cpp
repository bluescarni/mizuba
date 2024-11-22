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
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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
#include "logging.hpp"
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
// the temporary memory-mapped files used internally by the polyjectory
// and conjunction classes are not deleted when the program terminates.
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
//
// NOTE: in jupyterlab the cleanup functions registered to run at exit sometimes
// do not run to completion, thus leaving behind temporary files after shutdown.
// I think we are seeing this issue:
//
// https://github.com/jupyterlab/jupyterlab/issues/16276

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

// Wrapper to invoke the cleanup functions at shutdown.
extern "C" void cpp_atexit_wrapper()
{
#if !defined(NDEBUG)
    std::cout << "Running the C++ cleanup function" << std::endl;
#endif

    cleanup_pj_weak_ptrs();
    cleanup_cj_weak_ptrs();
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

    // Register polyjectory::traj_offset as a structured NumPy datatype.
    using traj_offset = mz::polyjectory::traj_offset;
    PYBIND11_NUMPY_DTYPE(traj_offset, offset, n_steps);

    // polyjectory.
    py::class_<mz::polyjectory> pt_cl(m, "polyjectory", py::dynamic_attr{});
    pt_cl.def(
        py::init([](py::iterable trajs_, py::iterable times_, py::iterable status_, double init_epoch) {
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
                                  std::ranges::subrange(status_ptr, status_ptr + status.shape(0)), init_epoch);

            // Register the polyjectory implementation in the cleanup machinery.
            mzpy::detail::add_pj_weak_ptr(mz::detail::fetch_pj_impl(ret));

            return ret;
        }),
        "trajs"_a.noconvert(), "times"_a.noconvert(), "status"_a.noconvert(), "init_epoch"_a.noconvert() = 0.);
    pt_cl.def(py::init<const std::filesystem::path &, const std::filesystem::path &, std::uint32_t,
                       std::vector<traj_offset>, std::vector<std::int32_t>, double>(),
              "traj_file"_a.noconvert(), "time_file"_a.noconvert(), "order"_a.noconvert(), "traj_offsets"_a.noconvert(),
              "status"_a.noconvert(), "init_epoch"_a.noconvert() = 0.);
    pt_cl.def_property_readonly("nobjs", &mz::polyjectory::get_nobjs);
    pt_cl.def_property_readonly("maxT", &mz::polyjectory::get_maxT);
    pt_cl.def_property_readonly("init_epoch", &mz::polyjectory::get_init_epoch);
    pt_cl.def_property_readonly("poly_order", &mz::polyjectory::get_poly_order);
    pt_cl.def_property_readonly("status", [](const py::object &self) {
        const auto *p = py::cast<const mz::polyjectory *>(self);

        // Fetch the status span.
        const auto status_span = p->get_status();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, status_span);
    });
    pt_cl.def(
        "__getitem__",
        [](const py::object &self, std::size_t i) {
            const auto *p = py::cast<const mz::polyjectory *>(self);

            // Fetch the spans and the status.
            const auto [traj_span, time_span, status] = (*p)[i];

            // Trajectory data.
            auto traj_ret = mzpy::mdspan_to_array(self, traj_span);

            // Time data.
            auto time_ret = mzpy::mdspan_to_array(self, time_span);

            return py::make_tuple(std::move(traj_ret), std::move(time_ret), status);
        },
        "i"_a.noconvert());

    // Expose static getters for the structured types.
    pt_cl.def_property_readonly_static("traj_offset", [](const py::object &) { return py::dtype::of<traj_offset>(); });

    // sgp4 polyjectory.
    m.def(
        "sgp4_polyjectory",
        [](py::list sat_list, double jd_begin, double jd_end, double exit_radius, double reentry_radius,
           double init_epoch) {
            // Check for the necessary dependencies.
            py::module_::import("mizuba").attr("_check_sgp4_deps")();

            // Check and pre-filter sat_list.
            py::tuple filter_res = py::module_::import("mizuba").attr("_sgp4_pre_filter_sat_list")(
                sat_list, jd_begin, exit_radius, reentry_radius);
            sat_list = filter_res[0];
            py::object mask = filter_res[1];
            py::object pd = filter_res[2];

            // Turn sat_list into a data vector.
            const auto sat_data = mzpy::sat_list_to_vector(sat_list);
            assert(sat_data.size() % 9u == 0u);

            // Create the input span.
            using span_t = hy::mdspan<const double, hy::extents<std::size_t, 9, std::dynamic_extent>>;
            const span_t in(sat_data.data(), boost::numeric_cast<std::size_t>(sat_data.size()) / 9u);

            auto poly_ret = [in, jd_begin, jd_end, exit_radius, reentry_radius, init_epoch]() {
                // NOTE: release the GIL during propagation.
                py::gil_scoped_release release;

                return mz::sgp4_polyjectory(in, jd_begin, jd_end, exit_radius, reentry_radius, init_epoch);
            }();

            // Register the polyjectory implementation in the cleanup machinery.
            mzpy::detail::add_pj_weak_ptr(mz::detail::fetch_pj_impl(poly_ret));

            // Convert the polyjectory into a Python object.
            py::object poly_obj = py::cast(std::move(poly_ret));

            // Amend the dataframe with the final statuses from the polyjectory.
            pd = py::module_::import("mizuba").attr("_sgp4_set_final_status")(poly_obj, pd);

            return py::make_tuple(std::move(poly_obj), std::move(pd), std::move(mask));
        },
        "sat_list"_a.noconvert(), "jd_begin"_a.noconvert(), "jd_end"_a.noconvert(),
        "exit_radius"_a.noconvert() = mz::sgp4_exit_radius, "reentry_radius"_a.noconvert() = mz::sgp4_reentry_radius,
        "init_epoch"_a.noconvert() = 0.);

    // Register conjunctions::bvh_node as a structured NumPy datatype.
    using bvh_node = mz::conjunctions::bvh_node;
    PYBIND11_NUMPY_DTYPE(bvh_node, begin, end, left, right, lb, ub);
    // Same for conjunctions::aabb_collision.
    using aabb_collision = mz::conjunctions::aabb_collision;
    PYBIND11_NUMPY_DTYPE(aabb_collision, i, j);
    // Same for conjunctions::conj.
    using conj = mz::conjunctions::conj;
    PYBIND11_NUMPY_DTYPE(conj, i, j, tca, dca, ri, vi, rj, vj);

    // Conjunctions.
    py::class_<mz::conjunctions> conj_cl(m, "conjunctions", py::dynamic_attr{});
    conj_cl.def(py::init([](mz::polyjectory pj, double conj_thresh, double conj_det_interval,
                            std::optional<std::vector<std::int32_t>> otypes) {
                    // NOTE: release the GIL during conjunction detection.
                    py::gil_scoped_release release;

                    auto ret = mz::conjunctions(std::move(pj), conj_thresh, conj_det_interval, std::move(otypes));

                    // Register the conjunctions implementation in the cleanup machinery.
                    mzpy::detail::add_cj_weak_ptr(mz::detail::fetch_cj_impl(ret));

                    return ret;
                }),
                "pj"_a.noconvert(), "conj_thresh"_a.noconvert(), "conj_det_interval"_a.noconvert(),
                "otypes"_a.noconvert() = py::none{});
    conj_cl.def_property_readonly("n_cd_steps", &mz::conjunctions::get_n_cd_steps);
    conj_cl.def_property_readonly("aabbs", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto aabbs_span = p->get_aabbs();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, aabbs_span);
    });
    conj_cl.def_property_readonly("cd_end_times", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto cd_end_times_span = p->get_cd_end_times();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, cd_end_times_span);
    });
    conj_cl.def_property_readonly("polyjectory", &mz::conjunctions::get_polyjectory);
    conj_cl.def_property_readonly("srt_aabbs", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_aabbs_span = p->get_srt_aabbs();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, srt_aabbs_span);
    });
    conj_cl.def_property_readonly("mcodes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto mcodes_span = p->get_mcodes();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, mcodes_span);
    });
    conj_cl.def_property_readonly("srt_mcodes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_mcodes_span = p->get_srt_mcodes();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, srt_mcodes_span);
    });
    conj_cl.def_property_readonly("srt_idx", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_idx_span = p->get_srt_idx();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, srt_idx_span);
    });
    conj_cl.def("get_bvh_tree", [](const py::object &self, std::size_t i) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the tree span.
        const auto tree_span = p->get_bvh_tree(i);

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, tree_span);
    });
    conj_cl.def("get_aabb_collisions", [](const py::object &self, std::size_t i) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the aabb collisions span.
        const auto aabb_collision_span = p->get_aabb_collisions(i);

        // Turn into an array.
        return mzpy::mdspan_to_array(self, aabb_collision_span);
    });
    conj_cl.def_property_readonly("conjunctions", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto conj_span = p->get_conjunctions();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, conj_span);
    }); // LCOV_EXCL_LINE
    conj_cl.def_property_readonly("otypes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto otypes_span = p->get_otypes();

        // Turn into an array and return.
        return mzpy::mdspan_to_array(self, otypes_span);
    }); // LCOV_EXCL_LINE
    conj_cl.def_property_readonly("conj_thresh", &mz::conjunctions::get_conj_thresh);
    conj_cl.def_property_readonly("conj_det_interval", &mz::conjunctions::get_conj_det_interval);

    // Expose static getters for the structured types.
    conj_cl.def_property_readonly_static("bvh_node", [](const py::object &) { return py::dtype::of<bvh_node>(); });
    conj_cl.def_property_readonly_static("aabb_collision",
                                         [](const py::object &) { return py::dtype::of<aabb_collision>(); });
    conj_cl.def_property_readonly_static("conj", [](const py::object &) { return py::dtype::of<conj>(); });

    // Logging utils.
    m.def("set_logger_level_info", &mz::set_logger_level_info);
    m.def("set_logger_level_trace", &mz::set_logger_level_trace);

    // Register the polyjectory/conjunctions cleanup machinery on the Python side.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Running the Python cleanup function" << std::endl;
#endif

        mzpy::detail::cleanup_pj_weak_ptrs();
        mzpy::detail::cleanup_cj_weak_ptrs();
    }));

    // Register the cleanup machinery also on the C++ side.
    //
    // NOTE: we do this also on the C++ side because we have run into some situations
    // in which, after a Ctrl+C signal, the Python interpreter would not invoke
    // the functions registered with atexit, leaving behind the temporary data on disk.
    //
    // NOTE: because we declared the structures used for cleanup as constinit, they are
    // constructed at compile time and they should be guaranteed to be destroyed *after*
    // the execution of cpp_atexit_wrapper, whose registration happens at runtime.
    //
    // NOTE: perhaps consider keeping only the C++-side cleanup in the future? Also,
    // if this is ever extracted as a separate C++ library, we probably want to keep
    // this cleanup mechanism for those cases in which the destructors of local
    // variables will not be called (e.g., if the user calls std::exit()).
    std::atexit(mzpy::detail::cpp_atexit_wrapper);
}
