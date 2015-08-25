namespace details {
template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
inclusive_scan_impl(InputIterator first, InputIterator last,
                    OutputIterator result,
                    BinaryOperation binary_op, T init,
                    std::input_iterator_tag) {
  return std::partial_sum(first, last, result, binary_op);
}

template<class RandomAccessIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
inclusive_scan_impl(RandomAccessIterator first, RandomAccessIterator last,
                    OutputIterator result,
                    BinaryOperation binary_op, T init,
                    std::random_access_iterator_tag) {

  // call to std::partial_sum when small data size
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return inclusive_scan_impl(first, last, result, binary_op, init,
             std::input_iterator_tag{});
  }

  typedef typename std::iterator_traits<RandomAccessIterator>::value_type _Tp;
  auto result_ = utils::get_pointer(result);
  std::unique_ptr<_Tp> stride(new _Tp [N]);
  details::scan_impl(first, last, binary_op, stride.get());
  auto stride_ = stride.get();

  // copy back the result
  kernel_launch(N, [stride_, result_, init, binary_op](hc::index<1> idx) __attribute((hc)) {
    result_[idx[0]] = binary_op(init, stride_[idx[0]]);
  });

  return result + N;
}

} // namespace details

template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op, T init) {
  return details::inclusive_scan_impl(first, last, result, binary_op, init,
           typename std::iterator_traits<InputIterator>::iterator_category());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename BinaryOperation, typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op, T init) {
  return inclusive_scan(first, last, result, binary_op, init);
}

template<typename InputIterator, typename OutputIterator,
         typename BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(first, last, result, binary_op, Type{});
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  return inclusive_scan(first, last, result, binary_op);
}

template<typename InputIterator, typename OutputIterator,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(first, last, result, std::plus<Type>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result) {
  return inclusive_scan(first, last, result);
}


