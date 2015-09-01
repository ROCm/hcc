template<typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename T, typename BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
transform_exclusive_scan(InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         T init, BinaryOperation binary_op) {
  // invoke std::experimental::parallel::transform and
  //        std::experimental::parallel::exclusive_scan
  transform(par, first, last, result, unary_op);
  const size_t N = static_cast<size_t>(std::distance(first, last));
  return exclusive_scan(par, result, result + N, result, init, binary_op);
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename UnaryOperation, typename T, typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
transform_exclusive_scan(ExecutionPolicy&& exec,
                         InputIterator first, InputIterator last,
                         OutputIterator result,
                         UnaryOperation unary_op,
                         T init, BinaryOperation binary_op) {
  if (utils::isParallel(exec)) {
    return transform_exclusive_scan(first, last, result, unary_op, init, binary_op);
  } else {
    details::transform_impl(first, last, result, unary_op,
      std::input_iterator_tag{});
    const size_t N = static_cast<size_t>(std::distance(first, last));
    return details::exclusive_scan_impl(result, result + N, result, init, binary_op,
             std::input_iterator_tag{});
  }
}

