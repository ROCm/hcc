//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include "../numeric"

namespace std {
namespace experimental {
namespace parallel {
inline namespace v1 {

namespace details {

// lexicographical_compare
// The parallized version requires random access iterators
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
//       both of them should be restrict(amp)
template<class InputIt1, class InputIt2, class Compare>
bool lexicographical_compare_impl(InputIt1 first1, InputIt1 last1,
                                  InputIt2 first2, InputIt2 last2,
                                  Compare comp,
                                  std::random_access_iterator_tag) {
  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

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
    return std::lexicographical_compare(first1, last1, first2, last2, comp);
  }

  const auto trans = [](const int &a, const int &b) restrict(amp, cpu) {
    return a == 1 ? b : a;
  };

  std::vector<int> tmp(N);
  int *tmp_ = tmp.data();

  typename std::iterator_traits<InputIt1>::pointer first1_ = &(*first1);
  typename std::iterator_traits<InputIt2>::pointer first2_ = &(*first2);
  parallel_for_each(extent<1>(N), tsa,
                    [tmp_, first1_, first2_, comp](index<1> idx) restrict(amp) {
    tmp_[idx[0]] = comp(first1_[idx[0]], first2_[idx[0]]) ? 0 :
                   first1_[idx[0]] == first2_[idx[0]] ? 1 : 2;
  });

  tmp[0] = reduce(tmp.begin(), tmp.end(), 1, trans);

  // If one range is a prefix of another, the shorter range is
  // lexicographically less than the other.
  bool ans = tmp[0] == 1 ? n1 < n2 : tmp[0] == 0;
  return ans;
}

} // namespace details

} // inline namespace v1
} // namespace parallel
} // namespace experimental
} // namespace std

