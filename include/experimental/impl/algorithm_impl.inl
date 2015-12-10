//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <numeric>

namespace std {
namespace experimental {
namespace parallel {
inline namespace v1 {

#include "type_utils.inl"
#include "kernel_launch.inl"
#include "reduce.inl"
#include "transform.inl"
#include "transform_reduce.inl"

namespace details {

// generate
// std::generate forwarder
template<typename ForwardIterator, typename Generator>
void generate_impl(ForwardIterator first, ForwardIterator last,
                   Generator g,
                   std::input_iterator_tag) {
  std::generate(first, last, g);
}

// parallel::generate
template<typename ForwardIterator, typename Generator>
void generate_impl(ForwardIterator first, ForwardIterator last,
                   Generator g,
                   std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    generate_impl(first, last, g, std::input_iterator_tag{});
    return;
  }

  // FIXME: [[hc]] will cause g() having ambient context,
  //        use restrict(amp) temporarily
  auto first_ = utils::get_pointer(first);
  kernel_launch(N, [first_, g](hc::index<1> idx) restrict(amp) {
    *(first_ + idx[0]) = g();
  });
}

// for_each
// std::for_each forwarder
template<typename InputIterator, typename Function>
void for_each_impl(InputIterator first, InputIterator last,
                   Function f,
                   std::input_iterator_tag) {
  std::for_each(first, last, f);
}

// parallel::for_each
template<typename InputIterator, typename Function>
void for_each_impl(InputIterator first, InputIterator last,
                   Function f,
                   std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    for_each_impl(first, last, f, std::input_iterator_tag{});
    return;
  }

  auto first_ = utils::get_pointer(first);
  kernel_launch(N, [first_, f](hc::index<1> idx) [[hc]] {
    f(*(first_ + idx[0]));
  });
}

// replace_if
// std::replace_if forwarder
template<typename ForwardIterator, typename Function, typename T>
void replace_if_impl(ForwardIterator first, ForwardIterator last,
                     Function f, const T& new_value,
                     std::input_iterator_tag) {
  std::replace_if(first, last, f, new_value);
}

// parallel::replace_if
template<typename ForwardIterator, typename Function, typename T>
void replace_if_impl(ForwardIterator first, ForwardIterator last,
                     Function f, const T& new_value,
                     std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    replace_if_impl(first, last, f, new_value, std::input_iterator_tag{});
    return;
  }

  auto first_ = utils::get_pointer(first);
  kernel_launch(N, [first_, f, new_value](hc::index<1> idx) [[hc]] {
    if (f(*(first_ + idx[0])))
      *(first_ + idx[0]) = new_value;
  });
}

// replace_copy_if
// std::replace_copy_if forwarder
template<typename InputIterator, typename OutputIterator,
         typename Function, typename T>
OutputIterator replace_copy_if_impl(InputIterator first, InputIterator last,
                                    OutputIterator d_first,
                                    Function f, const T& new_value,
                                    std::input_iterator_tag) {
  return std::replace_copy_if(first, last, d_first, f, new_value);
}

// parallel::replace_copy_if
template<typename InputIterator, typename OutputIterator,
         typename Function, typename T>
OutputIterator replace_copy_if_impl(InputIterator first, InputIterator last,
                                    OutputIterator d_first,
                                    Function f, const T& new_value,
                                    std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return replace_copy_if_impl(first, last, d_first, f, new_value,
             std::input_iterator_tag{});
  }

  if (N >= 0) {
    auto first_ = utils::get_pointer(first);
    auto d_first_ = utils::get_pointer(d_first);
    kernel_launch(N, [first_, d_first_, f, new_value](hc::index<1> idx) [[hc]] {
      if (f(*(first_ + idx[0])))
        *(d_first_ + idx[0]) = new_value;
      else
        *(d_first_ + idx[0]) = *(first_ + idx[0]);
    });
  }
  return (N < 0) ? d_first : d_first + N;
}

// adjacent_difference (with predicate version)
// std::adjacent_difference forwarder
template<typename InputIterator, typename OutputIterator, typename Function>
OutputIterator adjacent_difference_impl(InputIterator first, InputIterator last,
                                        OutputIterator d_first,
                                        Function f,
                                        std::input_iterator_tag) {
  return std::adjacent_difference(first, last, d_first, f);
}

// parallel::adjacent_difference (with predicate version)
template<typename InputIterator, typename OutputIterator, typename Function>
OutputIterator adjacent_difference_impl(InputIterator first, InputIterator last,
                                        OutputIterator d_first,
                                        Function f,
                                        std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return adjacent_difference_impl(first, last, d_first, f,
             std::input_iterator_tag{});
  }

  if (N >= 0) {
    auto first_ = utils::get_pointer(first);
    auto d_first_ = utils::get_pointer(d_first);
    kernel_launch(N, [first_, d_first_, f](hc::index<1> idx) [[hc]] {
      if (idx[0] == 0)
        *(d_first_ + idx[0]) = *(first_ + idx[0]);
      else
        *(d_first_ + idx[0]) = f(*(first_ + idx[0]), *(first_ + (idx[0] - 1)));
    });
  }
  return (N < 0) ? d_first : d_first + N;
}

// swap_ranges
// std::swap_ranges forwarder
template<typename InputIterator, typename OutputIterator>
OutputIterator swap_ranges_impl(InputIterator first, InputIterator last,
                                OutputIterator d_first,
                                std::input_iterator_tag) {
  return std::swap_ranges(first, last, d_first);
}

// parallel::swap_ranges
template<typename InputIterator, typename OutputIterator>
OutputIterator swap_ranges_impl(InputIterator first, InputIterator last,
                                OutputIterator d_first,
                                std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return swap_ranges_impl(first, last, d_first, std::input_iterator_tag{});
  }

  if (N >= 0) {
    auto first_ = utils::get_pointer(first);
    auto d_first_ = utils::get_pointer(d_first);
    kernel_launch(N, [first_, d_first_](hc::index<1> idx) [[hc]] {
      std::iter_swap(first_ + idx[0], d_first_ + idx[0]);
    });
  }
  return (N < 0) ? d_first : d_first + N;
}

// lexicographical_compare
// std::lexicographical_compare forwarder
template<class InputIt1, class InputIt2, class Compare>
bool lexicographical_compare_impl(InputIt1 first1, InputIt1 last1,
                                  InputIt2 first2, InputIt2 last2,
                                  Compare comp,
                                  std::input_iterator_tag) {
  return std::lexicographical_compare(first1, last1, first2, last2, comp);
}

//
// Transition function: check Lo first, if Lo is = then pick Hi
//
//                  Hi address
//                  <    =    >
//               +--------------------
//             < |  <    <    <
//               +--------------------
// Lo address  = |  <    =    >
//               +--------------------
//             > |  >    >    >
//               +--------------------
//
//   0: <
//   1: =
//   2: >
//
//

// Note: 1. the comparison needs both operator== and operator< (or a functor),
//       both of them should be [[hc]]
template<class InputIt1, class InputIt2, class Compare>
bool lexicographical_compare_impl(InputIt1 first1, InputIt1 last1,
                                  InputIt2 first2, InputIt2 last2,
                                  Compare comp,
                                  std::random_access_iterator_tag) {
  unsigned n1 = std::distance(first1, last1);
  unsigned n2 = std::distance(first2, last2);
  unsigned N = std::min(n1, n2);

  // An empty range is lexicographically less than any non-empty range.
  // Two empty ranges are lexicographically equal.
  if (N == 0) {
    return n1 < n2;
  }

  // call to std::lexicographical_compare when small data size
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return lexicographical_compare_impl(first1, last1, first2, last2, comp,
             std::input_iterator_tag{});
  }

  typedef typename std::iterator_traits<InputIt1>::value_type _Tp;

  auto first1_ = n1 <= n2 ? first1 : first2;
  auto last1_  = n1 <= n2 ?  last1 :  last2;
  auto first2_ = n1 <= n2 ? first2 : first1;

  // transform_reduce assumes two vectors are the same size,
  // use the smaller one as the first vector
  auto ans = transform_reduce(par, first1_, last1_, first2_, 1,
                              [](const int &a, const int &b) {
                                return a == 1 ? b : a;
                              },
                              [comp](const _Tp &a, const _Tp &b) {
                                return comp(a, b) ? 0 : a == b ? 1 : 2;
                              });

  return ans == 1 ? n1 < n2 : ans == 0;
}

template<typename InputIt1, typename InputIt2, typename BinaryPredicate>
bool equal_impl(InputIt1 first1, InputIt1 last1,
                InputIt2 first2,
                BinaryPredicate p,
                std::input_iterator_tag) {
  return std::equal(first1, last1, first2, p);
}


template<typename InputIt1, typename InputIt2, typename BinaryPredicate>
bool equal_impl(InputIt1 first1, InputIt1 last1,
                InputIt2 first2,
                BinaryPredicate p,
                std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first1, last1));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return equal_impl(first1, last1, first2, p, std::input_iterator_tag{});
  }

  return transform_reduce(first1, last1, first2, true,
                          std::logical_and<bool>(), p);
}


} // namespace details

} // inline namespace v1
} // namespace parallel
} // namespace experimental
} // namespace std

