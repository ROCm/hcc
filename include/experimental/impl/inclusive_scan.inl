// FIXME this is a SEQUENTIAL implementation of inclusive_scan!
template<typename InputIterator, typename OutputIterator,
         typename BinaryOperation, typename T>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
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


