// FIXME this is a SEQUENTIAL implementation of inclusive_scan!
template<typename InputIterator, typename OutputIterator,
         typename BinaryOperation, typename T>
OutputIterator
__inclusive_scan(InputIterator first, InputIterator last,
                 OutputIterator result,
                 BinaryOperation binary_op, T init) {
  T sum = init;
  OutputIterator iter_input = first;
  OutputIterator iter_output = result;
  for (; iter_input != last; ++iter_input, ++iter_output) {
    sum = binary_op(sum, *iter_input);
    *iter_output = sum;
  }
  return result;
}

template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op, T init) {

  // call to std::partial_sum when small data size
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return std::partial_sum(first, last, result, binary_op);
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
  parallel_for_each(extent<1>(N), tsa,
    [stride_, result_, init, binary_op](index<1> idx) restrict(amp) {
    result_[idx[0]] = binary_op(init, stride_[idx[0]]);
  });

  return result + N;
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename BinaryOperation, typename T>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op, T init) {
  return inclusive_scan(first, last, result, binary_op, init);
}

template<typename InputIterator, typename OutputIterator,
         typename BinaryOperation>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(first, last, result, binary_op, Type{});
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename BinaryOperation>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  return inclusive_scan(first, last, result, binary_op);
}

template<typename InputIterator, typename OutputIterator>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(first, last, result, std::plus<Type>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result) {
  return inclusive_scan(first, last, result);
}


