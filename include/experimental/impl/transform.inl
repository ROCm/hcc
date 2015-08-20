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
OutputIterator transform_impl(RandomAccessIterator first, RandomAccessIterator last,
                              OutputIterator d_first,
                              UnaryOperation unary_op,
                              std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return transform_impl(first, last, d_first, unary_op,
             std::input_iterator_tag{});
  }

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type _Tp1;
  typedef typename std::iterator_traits<OutputIterator>::value_type _Tp2;

  _Tp1 *first_ = &(*first);
  _Tp2 *d_first_ = &(*d_first);

  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

  // initialize the stride
  parallel_for_each(extent<1>(N), tsa,
    [d_first_, first_, unary_op](index<1> idx) restrict(amp) {
      d_first_[idx[0]] = unary_op(first_[idx[0]]);
  });

  return d_first + N;
}

// transform (binary version)
template <class RandomAccessIterator, class OutputIterator,
          class BinaryOperation>
OutputIterator transform_impl(RandomAccessIterator first1, RandomAccessIterator last1,
                              RandomAccessIterator first2,
                              OutputIterator d_first,
                              BinaryOperation binary_op,
                              std::random_access_iterator_tag) {
  const size_t N = static_cast<size_t>(std::distance(first1, last1));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return transform_impl(first1, last1, first2, d_first, binary_op,
             std::input_iterator_tag{});
  }

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type _Tp1;
  typedef typename std::iterator_traits<OutputIterator>::value_type _Tp2;

  _Tp1 *first1_ = &(*first1);
  _Tp1 *first2_ = &(*first2);
  _Tp2 *d_first_ = &(*d_first);

  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

  // initialize the stride
  parallel_for_each(extent<1>(N), tsa,
    [d_first_, first1_, first2_, binary_op](index<1> idx) restrict(amp) {
      d_first_[idx[0]] = binary_op(first1_[idx[0]], first2_[idx[0]]);
  });

  return d_first + N;
}

} // namespace details
