template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename T, typename BinaryOperation>
OutputIterator
transform_exclusive_scan(InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         T init, BinaryOperation binary_op) {
  // invoke std::experimental::parallel::transform and
  //        std::experimental::parallel::exclusive_scan
  transform(first, last, result, unary_op);
  const size_t N = static_cast<size_t>(std::distance(first, last));
  return exclusive_scan(result, result + N, result, init, binary_op);
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

