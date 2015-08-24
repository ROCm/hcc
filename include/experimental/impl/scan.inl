//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace details {

template<class InputIterator, class BinaryOperation>
std::unique_ptr<typename std::iterator_traits<InputIterator>::value_type>
scan_impl(InputIterator first, InputIterator last,
          BinaryOperation binary_op) {

  auto first_ = utils::get_pointer(first);

  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

  const size_t N = static_cast<size_t>(std::distance(first, last));
  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
  _Tp *stride = new _Tp [N];

  // initialize the stride
  parallel_for_each(extent<1>(N), tsa,
    [stride, first_](index<1> idx) restrict(amp) {
      stride[idx[0]] = first_[idx[0]];
  });

  // reduction depth = log N
  int depth = 0;
  for (unsigned N_ = N/2; N_ > 0; N_ >>= 1, depth++);

  // calculate the new value for stride[i] and pass stride[j] down
  const auto round = [stride, N, binary_op](const unsigned &i,
                                            const unsigned &j) {
    if (i < N) {
      stride[i] = binary_op(stride[i], stride[j]);
      stride[j] = stride[j];
    }
  };

  // For the first reduction tree
  //
  // i = 2^(d+1) * index + 2^(d+1) - 1
  // j = 2^(d+1) * index + 2^d - 1
  //
  // For the second reduction tree
  //
  // i = 2^(d+1) * index + 2^(d+1) + 2^d - 1
  // j = 2^(d+1) * index + 2^(d+1) - 1
  //
  parallel_for_each(extent<1>((N + 1) / 2), tsa,
                    [round, depth](index<1> idx) restrict(amp) {
    for (int d = 0; d < depth; d++) {
      const unsigned i = (1<<(d+1)) * idx[0] + (1<<(d+1)) - 1;
      const unsigned j = (1<<(d+1)) * idx[0] + (1<<d) - 1;
      round(i, j);
    }

    for (int d = depth - 1; d >= 0; d--) {
        const unsigned i = (1<<(d+1)) * idx[0] + (1<<(d+1)) + (1<<(d)) - 1;
        const unsigned j = (1<<(d+1)) * idx[0] + (1<<(d+1)) - 1;
        round(i, j);
    }
  });

  return std::move(std::unique_ptr<_Tp>(stride));
}

} // namespace details

