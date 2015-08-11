// FIXME this is a SEQUENTIAL implementation of transform_exclusive_scan!
template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename T, typename BinaryOperation>
OutputIterator
transform_exclusive_scan(InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         T init, BinaryOperation binary_op) {
  T sum = init;
  OutputIterator iter_input = first;
  OutputIterator iter_output = result;
  for (; iter_input != last; ++iter_input, ++iter_output) {
    *iter_output = sum;
    sum = binary_op(sum, unary_op(*iter_input));
  }
  return result;
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename T, typename BinaryOperation>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
transform_exclusive_scan(ExecutionPolicy&& exec,
                         InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         T init, BinaryOperation binary_op) {
  return transform_exclusive_scan(first, last, result, unary_op, init, binary_op);
}

