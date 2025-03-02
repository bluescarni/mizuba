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

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common_utils.hpp"
#include "conjunctions.hpp"
#include "expose_conjunctions.hpp"
#include "polyjectory.hpp"

namespace mizuba_py
{

namespace detail
{

namespace
{

// Global data for use in add_weak_ptr_cleanup().
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
constinit std::mutex cj_weak_ptr_mutex;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
constinit std::vector<std::weak_ptr<mizuba::detail::conjunctions_impl>> cj_weak_ptr_vector;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,cert-err58-cpp,cert-msc51-cpp,cert-msc32-c)
std::mt19937 cj_weak_ptr_rng;

// Add a weak pointer to a conjunctions implementation to cj_weak_ptr_vector.
void add_cj_weak_ptr(const std::shared_ptr<mizuba::detail::conjunctions_impl> &ptr)
{
    add_weak_ptr_cleanup(cj_weak_ptr_mutex, cj_weak_ptr_vector, cj_weak_ptr_rng, ptr);
}

} // namespace

} // namespace detail

void expose_conjunctions(pybind11::module_ &m)
{
    namespace py = pybind11;
    namespace mz = mizuba;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace py::literals;

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
    conj_cl.def(py::init([](const mz::polyjectory &pj, double conj_thresh, double conj_det_interval,
                            std::optional<std::vector<std::int32_t>> otypes) {
                    // NOTE: release the GIL during conjunction detection.
                    const py::gil_scoped_release release;

                    auto ret = mz::conjunctions(pj, conj_thresh, conj_det_interval, std::move(otypes));

                    // Register the conjunctions implementation in the cleanup machinery.
                    detail::add_cj_weak_ptr(mz::detail::fetch_cj_impl(ret));

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
        return mdspan_to_array(self, aabbs_span);
    });
    conj_cl.def_property_readonly("cd_end_times", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto cd_end_times_span = p->get_cd_end_times();

        // Turn into an array and return.
        return mdspan_to_array(self, cd_end_times_span);
    });
    conj_cl.def_property_readonly("srt_aabbs", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_aabbs_span = p->get_srt_aabbs();

        // Turn into an array and return.
        return mdspan_to_array(self, srt_aabbs_span);
    });
    conj_cl.def_property_readonly("mcodes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto mcodes_span = p->get_mcodes();

        // Turn into an array and return.
        return mdspan_to_array(self, mcodes_span);
    });
    conj_cl.def_property_readonly("srt_mcodes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_mcodes_span = p->get_srt_mcodes();

        // Turn into an array and return.
        return mdspan_to_array(self, srt_mcodes_span);
    });
    conj_cl.def_property_readonly("srt_idx", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto srt_idx_span = p->get_srt_idx();

        // Turn into an array and return.
        return mdspan_to_array(self, srt_idx_span);
    });
    conj_cl.def("get_bvh_tree", [](const py::object &self, std::size_t i) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the tree span.
        const auto tree_span = p->get_bvh_tree(i);

        // Turn into an array and return.
        return mdspan_to_array(self, tree_span);
    });
    conj_cl.def("get_aabb_collisions", [](const py::object &self, std::size_t i) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the aabb collisions span.
        const auto aabb_collision_span = p->get_aabb_collisions(i);

        // Turn into an array.
        return mdspan_to_array(self, aabb_collision_span);
    });
    conj_cl.def_property_readonly("conjunctions", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto conj_span = p->get_conjunctions();

        // Turn into an array and return.
        return mdspan_to_array(self, conj_span);
    }); // LCOV_EXCL_LINE
    conj_cl.def_property_readonly("otypes", [](const py::object &self) {
        const auto *p = py::cast<const mz::conjunctions *>(self);

        // Fetch the span.
        const auto otypes_span = p->get_otypes();

        // Turn into an array and return.
        return mdspan_to_array(self, otypes_span);
    }); // LCOV_EXCL_LINE
    conj_cl.def_property_readonly("conj_thresh", &mz::conjunctions::get_conj_thresh);
    conj_cl.def_property_readonly("conj_det_interval", &mz::conjunctions::get_conj_det_interval);

    // Expose static getters for the structured types.
    conj_cl.def_property_readonly_static("bvh_node", [](const py::object &) { return py::dtype::of<bvh_node>(); });
    conj_cl.def_property_readonly_static("aabb_collision",
                                         [](const py::object &) { return py::dtype::of<aabb_collision>(); });
    conj_cl.def_property_readonly_static("conj", [](const py::object &) { return py::dtype::of<conj>(); });
}

// Cleanup conjunctions implementations that are still alive. This is meant to be run
// at program shutdown.
void cleanup_cj_weak_ptrs()
{
    cleanup_weak_ptrs(detail::cj_weak_ptr_mutex, detail::cj_weak_ptr_vector, "conjunctions", &mizuba::detail::close_cj);
}

} // namespace mizuba_py
