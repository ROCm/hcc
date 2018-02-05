#pragma once

namespace details {

// std::transform forwarder
// transform (unary version)
template<class InputIterator, class OutputIterator,
         class UnaryOperation>
OutputIterator
transform_impl(InputIterator first, InputIterator last,
               OutputIterator d_first,
               UnaryOperation unary_op,
               std::input_iterator_tag) {
  return std::transform(first, last, d_first, unary_op);
}

// transform (binary version)
template<class InputIterator, class OutputIterator,
         class BinaryOperation>
OutputIterator
transform_impl(InputIterator first1, InputIterator last1,
               InputIterator first2, OutputIterator d_first,
               BinaryOperation binary_op,
               std::input_iterator_tag) {
  return std::transform(first1, last1, first2, d_first, binary_op);
}


// parallel::transform
// transform (unary version)
template <class RandomAccessIterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform_impl(RandomAccessIterator first,
                              RandomAccessIterator last,
                              OutputIterator d_first,
                              UnaryOperation unary_op,
                              std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return transform_impl(first, last, d_first, unary_op,
             std::input_iterator_tag{});
  }

  using _Ti = typename std::iterator_traits<RandomAccessIterator>::value_type;
  using _To = typename std::iterator_traits<RandomAccessIterator>::value_type;
  auto f_ = utils::get_pointer(first);
  auto d_ = utils::get_pointer(d_first);
  hc::array_view<_Ti> first_(hc::extent<1>(N), f_);
  hc::array_view<_To> d_first_(hc::extent<1>(N), d_);
  d_first_.discard_data();

  kernel_launch(N, [d_first_, first_, unary_op](hc::index<1> idx) [[hc]] {
    d_first_[idx[0]] = unary_op(first_[idx[0]]);
  });

  return d_first + N;
}

// transform (binary version)
template <class RandomAccessIterator, class OutputIterator,
          class BinaryOperation>
OutputIterator transform_impl(RandomAccessIterator first1,
                              RandomAccessIterator last1,
                              RandomAccessIterator first2,
                              OutputIterator d_first,
                              BinaryOperation binary_op,
                              std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first1, last1));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return transform_impl(first1, last1, first2, d_first, binary_op,
             std::input_iterator_tag{});
  }

  using _Ti = typename std::iterator_traits<RandomAccessIterator>::value_type;
  using _To = typename std::iterator_traits<RandomAccessIterator>::value_type;
  auto f1 = utils::get_pointer(first1);
  auto f2 = utils::get_pointer(first2);
  auto d_ = utils::get_pointer(d_first);
  hc::array_view<_Ti> first1_(hc::extent<1>(N), f1);
  hc::array_view<_Ti> first2_(hc::extent<1>(N), f2);
  hc::array_view<_To> d_first_(hc::extent<1>(N), d_);
  d_first_.discard_data();

  kernel_launch(N, [d_first_, first1_, first2_, binary_op](hc::index<1> idx) [[hc]] {
    d_first_[idx[0]] = binary_op(first1_[idx[0]], first2_[idx[0]]);
  });

  return d_first + N;
}

} // namespace details
