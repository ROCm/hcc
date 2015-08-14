//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <iostream>

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
  if (N <= 16) {
    return std::lexicographical_compare(first1, last1, first2, last2, comp);
  }

  auto trans = [](char &a, char &b) restrict(amp, cpu) {
    return a == 1 ? b : a;
  };

  char *tmp = new char [N];

  typename std::iterator_traits<InputIt1>::pointer first1_ = &(*first1);
  typename std::iterator_traits<InputIt2>::pointer first2_ = &(*first2);
  parallel_for_each(extent<1>(N), tsa,
                    [tmp, first1_, first2_, comp] (index<1> idx) restrict(amp) {
    tmp[idx[0]] = comp(first1_[idx[0]], first2_[idx[0]]) ? 0 :
                  first1_[idx[0]] == first2_[idx[0]] ? 1 : 2;
  });

  // Reduction kernel: apply the transition table for logN times
  unsigned s = N;
  for (s = N / 2; s > 3; s /= 2) {
    parallel_for_each(extent<1>(s), tsa,
                      [tmp, s, trans] (index<1> idx) restrict(amp) {
      const int i = idx[0];
      if (2*i+1 < s*2) {
        tmp[i] = trans(tmp[2*i], tmp[2*i+1]);
      } else {
        tmp[i] = tmp[2*i];
      }
    });
  }

  if (s == 3) {
    tmp[0] = trans(tmp[0], tmp[1]);
    tmp[1] = tmp[2];
    tmp[0] = trans(tmp[0], tmp[1]);
  } else if (s == 2) {
    tmp[0] = trans(tmp[0], tmp[1]);
  } else { }

  // If one range is a prefix of another, the shorter range is
  // lexicographically less than the other.
  bool ans = tmp[0] == 1 ? n1 < n2 : tmp[0] == 0;
  delete [] tmp;
  return ans;
}

} // namespace details

} // inline namespace v1
} // namespace parallel
} // namespace experimental
} // namespace std

