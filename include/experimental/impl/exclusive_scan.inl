// FIXME this is a SEQUENTIAL implementation of exclusive_scan!
template<typename InputIterator, typename OutputIterator,
         typename T, typename BinaryOperation>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
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


