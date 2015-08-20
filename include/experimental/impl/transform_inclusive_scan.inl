template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation, typename T>
OutputIterator
transform_inclusive_scan(InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         BinaryOperation binary_op, T init) {
  // invoke std::experimental::parallel::transform and
  //        std::experimental::parallel::inclusive_scan
  transform(first, last, result, unary_op);
  const size_t N = static_cast<size_t>(std::distance(first, last));
  return inclusive_scan(result, result + N, result, binary_op, init);
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation, typename T>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
transform_inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               UnaryOperation unary_op,
               BinaryOperation binary_op, T init) {
  return transform_inclusive_scan(first, last, result, unary_op, binary_op, init);
}

template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation>
OutputIterator
transform_inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               UnaryOperation unary_op,
               BinaryOperation binary_op) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return transform_inclusive_scan(first, last, result, unary_op, binary_op, Type{});
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, OutputIterator>::type
transform_inclusive_scan(ExecutionPolicy&& exec,
                         InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         BinaryOperation binary_op) {
  return transform_inclusive_scan(first, last, result, binary_op);
}

