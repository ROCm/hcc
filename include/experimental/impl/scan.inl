//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace details {

template<class InputIterator, class BinaryOperation>
typename std::iterator_traits<InputIterator>::value_type *
scan_impl(InputIterator first, InputIterator last,
          BinaryOperation binary_op) {

  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
  _Tp *first_ = &(*first);

  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

  const size_t N = static_cast<size_t>(std::distance(first, last));
  _Tp *tmp = new _Tp [N];

  // initialize
  parallel_for_each(extent<1>(N), tsa,
    [tmp, first_](index<1> idx) restrict(amp) {
      tmp[idx[0]] = first_[idx[0]];
  });

  // reduction depth = log N
  int depth = 0;
  for (unsigned N_ = N/2; N_ > 0; N_ >>= 1, depth++);

  auto round = [tmp, N, binary_op](unsigned &i, unsigned &j) {
    if (i < N) {
      tmp[i] = binary_op(tmp[i], tmp[j]);
      tmp[j] = tmp[j];
    }
  };

  for (int d = 0; d < depth; d++) {
    parallel_for_each(extent<1>((N + 1) / 2), tsa,
                      [d, round](index<1> idx) restrict(amp) {
      unsigned i = (1<<(d+1)) * idx[0] + (1<<(d+1)) - 1;
      unsigned j = (1<<(d+1)) * idx[0] + (1<<d) - 1;
      round(i, j);
    });
  }

  for (int d = depth - 1; d >= 0; d--) {
    parallel_for_each(extent<1>((N + 1) / 2), tsa,
                      [d, round](index<1> idx) restrict(amp) {
      unsigned i = (1<<(d+1)) * idx[0] + (1<<(d)) + (1<<(d+1)) - 1;
      unsigned j = (1<<(d+1)) * idx[0] + (1<<(d+1)) - 1;
      round(i, j);
    });
  }

  return tmp;
}

} // namespace details

