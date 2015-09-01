namespace details {
template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
exclusive_scan_impl(InputIterator first, InputIterator last,
                    OutputIterator result,
                    T init, BinaryOperation binary_op,
                    std::input_iterator_tag) {
  std::partial_sum(first, last, result, binary_op);
  const size_t N = static_cast<size_t>(std::distance(first, last));
  for (int i = N-2; i >= 0; i--)
    result[i+1] = binary_op(init, result[i]);
  result[0] = init;
  return result + N;
}

template<class RandomAccessIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
exclusive_scan_impl(RandomAccessIterator first, RandomAccessIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op,
               std::random_access_iterator_tag) {
  // call to std::partial_sum when small data size
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return exclusive_scan_impl(first, last, result, init, binary_op,
             std::input_iterator_tag{});
  }

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type _Tp;
  auto result_ = utils::get_pointer(result);
  std::unique_ptr<_Tp> stride(new _Tp [N]);
  details::scan_impl(first, last, binary_op, stride.get());
  auto stride_ = stride.get();

  // copy back the result
  kernel_launch((N-1), [stride_, result_, init, binary_op](hc::index<1> idx) __attribute((hc)) {
    result_[idx[0]+1] = binary_op(init, stride_[idx[0]]);
  });

  result[0] = init;

  return result + N;
}

} // namespace details


template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op) {
  return details::exclusive_scan_impl(first, last, result, init, binary_op,
           typename std::iterator_traits<InputIterator>::iterator_category());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename T, typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op) {
  if (utils::isParallel(exec)) {
    return exclusive_scan(first, last, result, init, binary_op);
  } else {
    return details::exclusive_scan_impl(first, last, result, init, binary_op,
             std::input_iterator_tag{});
  }
}

template<typename InputIterator, typename OutputIterator,
         typename T,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               T init) {
  return exclusive_scan(first, last, result, init, std::plus<T>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               T init) {
  return exclusive_scan(exec, first, last, result, init, std::plus<T>());
}


