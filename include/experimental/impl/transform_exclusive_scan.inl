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
  details::transform_impl(first, last, result, unary_op,
    typename std::iterator_traits<InputIterator>::iterator_category());
  const size_t N = static_cast<size_t>(std::distance(first, last));
  return exclusive_scan(result, result + N, result, init, binary_op);
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
  return transform_exclusive_scan(first, last, result, unary_op, init, binary_op);
}

