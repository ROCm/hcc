/**
 * Effects: Assigns through each iterator i in [result,result + (last - first))
 * the value of GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, unary_op(*first), ..., unary_op(*(first + (i * - result) - 1))).
 *
 * @return The end of the resulting range beginning at result.
 *
 * Requires: Neither unary_op nor binary_op shall invalidate iterators or
 * subranges, or modify elements in the ranges [first,last) or
 * [result,result + (last - first)).
 *
 * Complexity: O(last - first) applications each of unary_op and binary_op.
 *
 * Notes: The difference between transform_exclusive_scan and
 * transform_inclusive_scan is that transform_exclusive_scan excludes the ith
 * input element from the ith sum. If binary_op is not mathematically
 * associative, the behavior of transform_exclusive_scan may
 */
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

/**
 * Effects: Assigns through each iterator i in [result,result + (last - first))
 * the value of GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, unary_op(*first), ..., unary_op(*(first + (i * - result) - 1))).
 *
 * @return The end of the resulting range beginning at result.
 *
 * Requires: Neither unary_op nor binary_op shall invalidate iterators or
 * subranges, or modify elements in the ranges [first,last) or
 * [result,result + (last - first)).
 *
 * Complexity: O(last - first) applications each of unary_op and binary_op.
 *
 * Notes: The difference between transform_exclusive_scan and
 * transform_inclusive_scan is that transform_exclusive_scan excludes the ith
 * input element from the ith sum. If binary_op is not mathematically
 * associative, the behavior of transform_exclusive_scan may
 */
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

