// FIXME, this is a SEQUENTIAL implementation of transform_reduce!
template<typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation>
T transform_reduce(InputIterator first, InputIterator last,
                   UnaryOperation unary_op,
                   T init, BinaryOperation binary_op) {
  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
  auto new_op = [&](const T& a, const _Tp& b) {
    return binary_op(a, unary_op(b));
  };

  return std::accumulate(first, last, init, new_op);
}

template<typename ExecutionPolicy,
         typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, T>::type
transform_reduce(ExecutionPolicy&& exec,
                 InputIterator first, InputIterator last,
                 UnaryOperation unary_op,
                 T init, BinaryOperation binary_op) {
  return transform_reduce(first, last, unary_op, init, binary_op);
}

