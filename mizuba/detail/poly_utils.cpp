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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/expression.hpp>

#include "conjunctions_jit.hpp"
#include "poly_utils.hpp"

namespace mizuba::detail
{

void pwrap::back_to_cache()
{
    // NOTE: v will be empty when this has been
    // moved-from. In that case, we do not want
    // to send v back to the cache.
    if (!v.empty()) {
        assert(pc.empty() || pc[0].size() == v.size());

        // Move v into the cache.
        pc.push_back(std::move(v));
    }
}

std::vector<double> pwrap::get_poly_from_cache(std::uint32_t n)
{
    if (pc.empty()) {
        // No polynomials are available, create a new one.
        return std::vector<double>(boost::numeric_cast<std::vector<double>::size_type>(n + 1u));
    } else {
        // Extract an existing polynomial from the cache.
        auto retval = std::move(pc.back());
        pc.pop_back();

        assert(retval.size() == n + 1u);

        return retval;
    }
}

pwrap::pwrap(poly_cache &cache, std::uint32_t n) : pc(cache), v(get_poly_from_cache(n)) {}

pwrap::pwrap(pwrap &&other) noexcept : pc(other.pc), v(std::move(other.v))
{
    // Make sure we moved from a valid pwrap.
    assert(!v.empty()); // LCOV_EXCL_LINE

    // NOTE: we must ensure that other.v is cleared out, because
    // otherwise, when other is destructed, we could end up
    // returning to the cache a polynomial of the wrong size.
    //
    // In basically every existing implementation, moving a std::vector
    // will leave the original object empty, but technically this
    // does not seem to be guaranteed by the standard, so, better
    // safe than sorry. Quick checks on godbolt indicate that compilers
    // are anyway able to elide this clearing out of the vector.
    other.v.clear();
}

// NOTE: this does not support self-move, and requires that
// the cache of other is the same as the cache of this.
pwrap &pwrap::operator=(pwrap &&other) noexcept
{
    // Disallow self move.
    assert(this != &other); // LCOV_EXCL_LINE

    // Make sure the polyomial caches match.
    assert(&pc == &other.pc); // LCOV_EXCL_LINE

    // Make sure we are not moving from a
    // moved-from pwrap.
    assert(!other.v.empty()); // LCOV_EXCL_LINE

    // Put the current v in the cache.
    back_to_cache();

    // Do the move-assignment.
    v = std::move(other.v);

    // NOTE: we must ensure that other.v is cleared out, because
    // otherwise, when other is destructed, we could end up
    // returning to the cache a polynomial of the wrong size.
    //
    // In basically every existing implementation, moving a std::vector
    // will leave the original object empty, but technically this
    // does not seem to be guaranteed by the standard, so, better
    // safe than sorry. Quick checks on godbolt indicate that compilers
    // are anyway able to elide this clearing out of the vector.
    other.v.clear();

    return *this;
}

pwrap::~pwrap()
{
#if !defined(NDEBUG)

    // Run consistency checks on the cache in debug mode.
    // The cache must not contain empty vectors
    // and all vectors in the cache must have the same size.
    if (!pc.empty()) {
        const auto op1 = pc[0].size();

        for (const auto &vec : pc) {
            assert(!vec.empty());
            assert(vec.size() == op1);
        }
    }

#endif

    // Put the current v in the cache.
    back_to_cache();
}

namespace
{

// Evaluate the first derivative of a polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval_1(InputIt a, T x, std::uint32_t n)
{
    assert(n >= 2u); // LCOV_EXCL_LINE

    // Init the return value.
    auto ret1 = a[n] * n;

    for (std::uint32_t i = 1; i < n; ++i) {
        ret1 = a[n - i] * (n - i) + ret1 * x;
    }

    return ret1;
}

// Generic branchless sign function.
// It will return:
// - 0 if val is 0,
// - +-1 if val is greater/less than zero.
template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

// Given an input polynomial a(x), substitute
// x with x_1 * scal and write to ret the resulting
// polynomial in the new variable x_1. Requires
// random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt, typename T>
void poly_rescale(OutputIt ret, InputIt a, const T &scal, std::uint32_t n)
{
    T cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = cur_f * a[i];
        cur_f *= scal;
    }
}

// Transform the polynomial a(x) into 2**n * a(x / 2).
// Requires random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt>
void poly_rescale_p2(OutputIt ret, InputIt a, std::uint32_t n)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    value_type cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[n - i] = cur_f * a[n - i];
        cur_f *= 2;
    }
}

// Find the only existing root for the polynomial poly of the given order
// existing in [lb, ub).
template <typename T>
std::tuple<T, int> bracketed_root_find(const T *poly, std::uint32_t order, T lb, T ub)
{
    using std::isfinite;
    using std::nextafter;

    // NOTE: the Boost root finding routine searches in a closed interval,
    // but the goal here is to find a root in [lb, ub). Thus, we move ub
    // one position down so that it is not considered in the root finding routine.
    if (isfinite(lb) && isfinite(ub) && ub > lb) {
        ub = nextafter(ub, lb);
    }

    // Set a maximum number of iterations.
    constexpr boost::uintmax_t iter_limit = 100;
    auto max_iter = iter_limit;

    // Ensure that root finding does not throw on error,
    // rather it will write something to errno instead.
    // https://www.boost.org/doc/libs/1_75_0/libs/math/doc/html/math_toolkit/pol_tutorial/namespace_policies.html
    using boost::math::policies::domain_error;
    using boost::math::policies::errno_on_error;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::pole_error;
    using boost::math::policies::policy;

    using pol = policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>,
                       evaluation_error<errno_on_error>>;

    // Clear out errno before running the root finding.
    errno = 0;

    // Run the root finder.
    const auto p = boost::math::tools::toms748_solve([poly, order](T x) { return horner_eval(poly, order, x); }, lb, ub,
                                                     boost::math::tools::eps_tolerance<T>(), max_iter, pol{});
    const auto ret = p.first / 2 + p.second / 2;

    if (errno > 0) {
        // Some error condition arose during root finding,
        // return zero and errno.
        return std::tuple{static_cast<T>(0), errno};
    }

    if (max_iter < iter_limit) {
        // Root finding terminated within the
        // iteration limit, return ret and success.
        return std::tuple{ret, 0};
    } else {
        // LCOV_EXCL_START
        // Root finding needed too many iterations,
        // return the (possibly wrong) result
        // and flag -1.
        return std::tuple{ret, -1};
        // LCOV_EXCL_STOP
    }
}

} // namespace

bool run_poly_root_finding(const double *poly, std::uint32_t order, double rf_int, isol_t &isol, wlist_t &wlist,
                           conj_jit_data::fex_check_t fex_check, conj_jit_data::rtscc_t rtscc, conj_jit_data::pt1_t pt1,
                           std::uint32_t i, std::uint32_t j, int direction,
                           std::vector<std::tuple<std::uint32_t, std::uint32_t, double>> &detected_roots,
                           poly_cache &r_iso_cache)
{
    // Sanity check on direction.
    assert(direction == 0 || direction == 1 || direction == -1);

    // Run the fast exclusion check.
    std::uint32_t fex_check_res = 0, back_flag = 0;
    fex_check(poly, &rf_int, &back_flag, &fex_check_res);

    if (fex_check_res != 0u) {
        return true;
    }

    // Fast exclusion check failed, we need to run the real root isolation algorithm.

    // Clear out the list of isolating intervals.
    isol.clear();

    // Reset the working list.
    wlist.clear();

    // Temporary polynomials used in the bisection loop.
    pwrap tmp1(r_iso_cache, order), tmp2(r_iso_cache, order), tmp(r_iso_cache, order);

    // Helper to add a detected root to detected_roots.
    // NOTE: the root here is expected to be already rescaled
    // to the [0, rf_int) range.
    const auto add_root = [i, j, direction, order, poly, &detected_roots](double root) {
        // NOTE: we do one last check on the root in order to
        // avoid non-finite times.
        if (!std::isfinite(root)) [[unlikely]] {
            // LCOV_EXCL_START
            throw std::invalid_argument(fmt::format(
                "Polynomial root finding produced a non-finite root of {} involving objects {} and {}", root, i, j));
            // LCOV_EXCL_STOP
        }

        // Check the direction, if needed.
        bool accept_root = true;
        if (direction != 0) {
            // Evaluate the derivative and its absolute value.
            const auto der = poly_eval_1(poly, root, order);

            // Check it before proceeding.
            if (!std::isfinite(der)) [[unlikely]] {
                // LCOV_EXCL_START
                throw std::invalid_argument(fmt::format("Polynomial root finding produced the root {} with "
                                                        "nonfinite derivative {} involving objects {} and {}",
                                                        root, der, i, j));
                // LCOV_EXCL_STOP
            }

            // Compute sign of the derivative.
            const auto d_sgn = sgn(der);

            // Accept the root only if the sign of the derivative
            // matches the direction.
            if (d_sgn != direction) {
                accept_root = false;
            }
        }

        if (accept_root) {
            // Add it.
            detected_roots.emplace_back(i, j, root);
        }
    };

    // Rescale poly so that the range [0, rf_int)
    // becomes [0, 1), and write the resulting polynomial into tmp.
    poly_rescale(tmp.v.data(), poly, rf_int, order);

    // Place the first element in the working list.
    // NOTE: after the move, tmp is in an invalid state. However,
    // it will be immediately revived after entering the do/while
    // loop below.
    wlist.emplace_back(0, 1, std::move(tmp));

    do {
        // Fetch the current interval and polynomial from the working list.
        // NOTE: from now on, tmp contains the polynomial referred
        // to as q(x) in the real-root isolation wikipedia page.
        // NOTE: q(x) is the transformed polynomial whose roots in the x range [0, 1) we
        // will be looking for. lb and ub represent what 0 and 1 correspond to in the
        // *original* [0, 1) range.
        const auto lb = std::get<0>(wlist.back());
        const auto ub = std::get<1>(wlist.back());
        // NOTE: this will either revive an invalid tmp (first iteration),
        // or it will replace it with one of the bisecting polynomials.
        tmp = std::move(std::get<2>(wlist.back()));
        wlist.pop_back();

        // Check for a root at the lower bound, which occurs
        // if the constant term of the polynomial is zero. We also
        // check for finiteness of all the other coefficients, otherwise
        // we cannot really claim to have detected a root.
        // When we do proper root finding below, the
        // algorithm should be able to detect non-finite
        // polynomials.
        if (tmp.v[0] == 0 // LCOV_EXCL_LINE
            && std::all_of(tmp.v.data() + 1, tmp.v.data() + 1 + order,
                           [](const auto &x) { return std::isfinite(x); })) {
            // NOTE: the original range had been rescaled wrt to rf_int.
            // Thus, we need to rescale back when adding the detected
            // root.
            add_root(lb * rf_int);
        }

        // Reverse tmp into tmp1, translate tmp1 by 1 with output
        // in tmp2, and count the sign changes in tmp2.
        std::uint32_t n_sc = 0;
        rtscc(tmp1.v.data(), tmp2.v.data(), &n_sc, tmp.v.data());

        if (n_sc == 1u) {
            // Found isolating interval, add it to isol.
            isol.emplace_back(lb, ub);
        } else if (n_sc > 1u) {
            // No isolating interval found, bisect.

            // First we transform q into 2**n * q(x/2) and store the result into tmp1.
            poly_rescale_p2(tmp1.v.data(), tmp.v.data(), order);
            // Then we take tmp1 and translate it to produce 2**n * q((x+1)/2).
            pt1(tmp2.v.data(), tmp1.v.data());

            // Finally we add tmp1 and tmp2 to the working list.
            const auto mid = lb / 2 + ub / 2;
            wlist.emplace_back(lb, mid, std::move(tmp1));
            wlist.emplace_back(mid, ub, std::move(tmp2));

            // Revive tmp1 and tmp2 after moving them into wlist.
            tmp1 = pwrap(r_iso_cache, order);
            tmp2 = pwrap(r_iso_cache, order);
        }

        // LCOV_EXCL_START
        // NOTE: we want to put limits in order to avoid an endless loop when the algorithm fails.
        // The first check is on the working list size and it is based
        // on heuristic observation of the algorithm's behaviour in pathological
        // cases. The second check is that we cannot possibly find more isolating
        // intervals than the degree of the polynomial.
        if (wlist.size() > 250u || isol.size() > order) [[unlikely]] {
            throw std::invalid_argument(fmt::format("The polynomial root isolation algorithm failed: the working list "
                                                    "size is {} and the number of isolating intervals is {}",
                                                    wlist.size(), isol.size()));
        }
        // LCOV_EXCL_STOP
    } while (!wlist.empty());

    // Skip root finding if the list of isolating intervals is empty.
    if (!isol.empty()) {
        // Reconstruct a version of the original polynomial
        // in which the range [0, rf_int) is rescaled to [0, 1). We need
        // to do root finding on the rescaled polynomial because the
        // isolating intervals are also rescaled to [0, 1).
        // NOTE: tmp1 was either created with the correct size outside this
        // function, or it was re-created in the bisection above.
        poly_rescale(tmp1.v.data(), poly, rf_int, order);

        // Run the root finding in the isolating intervals.
        for (const auto &[lb, ub] : isol) {
            // Run the root finding.
            const auto [root, cflag] = bracketed_root_find(tmp1.v.data(), order, lb, ub);

            if (cflag == 0) [[likely]] {
                // Root finding finished successfully, record the root.
                // The found root needs to be rescaled by h.
                add_root(root * rf_int);
            } else {
                // LCOV_EXCL_START
                // Root finding encountered some issue. Ignore the
                // root and log the issue.
                if (cflag == -1) {
                    throw std::invalid_argument("Polynomial root finding failed due to too many iterations");
                } else {
                    throw std::invalid_argument(
                        fmt::format("Polynomial root finding returned a nonzero errno with code '{}'", cflag));
                }
                // LCOV_EXCL_STOP
            }
        }
    }

    return false;
}

// Create the heyoka expression for Vandermonde polynomial interpolation. See:
//
// https://www.ams.org/journals/mcom/1970-24-112/S0025-5718-1970-0290541-1/S0025-5718-1970-0290541-1.pdf
//
// The mandatory input argument is the polynomial interpolation order.
std::pair<std::vector<heyoka::expression>, std::vector<heyoka::expression>> vm_interp(std::uint32_t order)
{
    namespace hy = heyoka;

    assert(order >= 1u);

    // Safely compute order + 1.
    const auto op1 = static_cast<std::uint32_t>(boost::safe_numerics::safe<std::uint32_t>(order) + 1u);

    // Construct the expressions representing
    // the evaluation points and the evaluation values.
    std::vector<hy::expression> alpha, f;
    alpha.reserve(op1);
    f.reserve(op1);
    for (std::uint32_t i = 0; i < op1; ++i) {
        alpha.emplace_back(fmt::format("alpha_{}", i));
        f.emplace_back(fmt::format("f_{}", i));
    }

    // Construct the c vectors.
    std::vector<std::vector<hy::expression>> c_vecs;
    c_vecs.reserve(op1);
    c_vecs.push_back(f);

    for (std::uint32_t k = 0; k < order; ++k) {
        std::vector<hy::expression> cur_c_vec;
        cur_c_vec.reserve(op1);

        for (std::uint32_t j = 0; j < op1; ++j) {
            if (j <= k) {
                cur_c_vec.push_back(c_vecs.back()[j]);
            } else {
                assert(j >= 1u);
                auto tmp = c_vecs.back()[j] - c_vecs.back()[j - 1u];
                assert(j >= k + 1u);
                tmp /= alpha[j] - alpha[j - k - 1u];
                cur_c_vec.push_back(std::move(tmp));
            }
        }

        c_vecs.push_back(std::move(cur_c_vec));
    }

    // Construct the a vectors.
    std::vector<std::vector<hy::expression>> a_vecs;
    a_vecs.resize(boost::numeric_cast<decltype(a_vecs.size())>(op1));
    a_vecs.back() = c_vecs.back();

    // NOTE: this is a backwards loop, in which the k index starts
    // from order - 1 and goes back all the way to 0.
    for (std::uint32_t idx = 0; idx < order; ++idx) {
        const auto k = order - idx - 1u;
        auto &cur_a_vec = a_vecs[k];

        for (std::uint32_t j = 0; j < op1; ++j) {
            if (j < k || j == order) {
                cur_a_vec.push_back(a_vecs[k + 1u][j]);
            } else {
                cur_a_vec.push_back(a_vecs[k + 1u][j] - alpha[k] * a_vecs[k + 1u][j + 1u]);
            }
        }
    }

    // Concatenate the evaluation values into alpha.
    alpha.insert(alpha.end(), f.begin(), f.end());

    // Build the return values.
    return std::make_pair(std::move(alpha), std::move(a_vecs[0]));
}

} // namespace mizuba::detail
