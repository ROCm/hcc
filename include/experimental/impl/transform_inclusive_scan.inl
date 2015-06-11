// FIXME this is a SEQUENTIAL implementation of transform_inclusive_scan!
template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation, typename T>
OutputIterator
transform_inclusive_scan(InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         BinaryOperation binary_op, T init) {
  T sum = init;
  OutputIterator iter_input = first;
  OutputIterator iter_output = result;
  for (; iter_input != last; ++iter_input, ++iter_output) {
    sum = binary_op(sum, unary_op(*iter_input));
    *iter_output = sum;
  }
  return result;
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

