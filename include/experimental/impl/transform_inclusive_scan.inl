template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation, typename T,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
transform_inclusive_scan(InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         BinaryOperation binary_op, T init) {
  // invoke std::experimental::parallel::transform and
  //        std::experimental::parallel::inclusive_scan
  transform(par, first, last, result, unary_op);
  const size_t N = static_cast<size_t>(std::distance(first, last));
  return inclusive_scan(par, result, result + N, result, binary_op, init);
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation, typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
transform_inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               UnaryOperation unary_op,
               BinaryOperation binary_op, T init) {
  if (utils::isParallel(exec)) {
    return transform_inclusive_scan(first, last, result, unary_op, binary_op, init);
  } else {
    details::transform_impl(first, last, result, unary_op,
      std::input_iterator_tag{});
    const size_t N = static_cast<size_t>(std::distance(first, last));
    return details::inclusive_scan_impl(result, result + N, result, binary_op, init,
             std::input_iterator_tag{});
  }
}

template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
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
         typename UnaryOperation, typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
transform_inclusive_scan(ExecutionPolicy&& exec,
                         InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         BinaryOperation binary_op) {
  if (utils::isParallel(exec)) {
    return transform_inclusive_scan(first, last, result, binary_op);
  } else {
    details::transform_impl(first, last, result, unary_op,
      std::input_iterator_tag{});
    const size_t N = static_cast<size_t>(std::distance(first, last));
    typedef typename std::iterator_traits<OutputIterator>::value_type Type;
    return details::inclusive_scan_impl(result, result + N, result, binary_op, Type{},
             std::input_iterator_tag{});
  }
}

