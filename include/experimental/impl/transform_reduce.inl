template<typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T transform_reduce(InputIterator first, InputIterator last,
                   UnaryOperation unary_op,
                   T init, BinaryOperation binary_op) {
  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
  auto new_op = [&](const T& a, const _Tp& b) {
    return binary_op(a, unary_op(b));
  };

  // invoke std::experimental::parallel::reduce
  return reduce(first, last, init, new_op);
}

template<typename ExecutionPolicy,
         typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, T>::type
transform_reduce(ExecutionPolicy&& exec,
                 InputIterator first, InputIterator last,
                 UnaryOperation unary_op,
                 T init, BinaryOperation binary_op) {
  if (utils::isParallel(exec)) {
    return transform_reduce(first, last, unary_op, init, binary_op);
  } else {
    typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
    auto new_op = [&](const T& a, const _Tp& b) {
      return binary_op(a, unary_op(b));
    };
    return std::accumulate(first, last, init, new_op);
  }
}

