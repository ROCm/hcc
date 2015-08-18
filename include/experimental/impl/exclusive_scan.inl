// FIXME this is a SEQUENTIAL implementation of exclusive_scan!
template<typename InputIterator, typename OutputIterator,
         typename T, typename BinaryOperation>
OutputIterator
__exclusive_scan(InputIterator first, InputIterator last,
                 OutputIterator result,
                 T init, BinaryOperation binary_op) {
  T sum = init;
  OutputIterator iter_input = first;
  OutputIterator iter_output = result;
  for (; iter_input != last; ++iter_input, ++iter_output) {
    *iter_output = sum;
    sum = binary_op(sum, *iter_input);
  }
  return result;
}

template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op) {

  // call to std::partial_sum when small data size
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    std::partial_sum(first, last, result, binary_op);
    for (int i = N-2; i >= 0; i--)
      result[i+1] = binary_op(init, result[i]);
    result[0] = init;
    return result + N;
  }

  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
  _Tp *result_ = &(*result);
  std::unique_ptr<_Tp> stride = details::scan_impl(first, last, binary_op);
  _Tp *stride_ = stride.get();

  // copy back the result
  parallel_for_each(extent<1>(N-1), tsa,
    [stride_, result_, init, binary_op](index<1> idx) restrict(amp) {
    result_[idx[0]+1] = binary_op(init, stride_[idx[0]]);
  });

  result[0] = init;

  return result + N;
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename T, typename BinaryOperation>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
exclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op) {
  return exclusive_scan(first, last, result, init, binary_op);
}

template<typename InputIterator, typename OutputIterator,
         typename T>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               T init) {
  return exclusive_scan(first, last, result, init, std::plus<T>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename T>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
exclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               T init) {
  return exclusive_scan(first, last, result, init);
}


