/**
 * @file numeric
 * Numeric Parallel algorithms
 */
namespace details {
template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
exclusive_scan_impl(InputIterator first, InputIterator last,
                    OutputIterator result,
                    T init, BinaryOperation binary_op,
                    std::input_iterator_tag) {
  std::partial_sum(first, last, result, binary_op);
  const size_t N = static_cast<size_t>(std::distance(first, last));
  for (int i = N-2; i >= 0; i--)
    result[i+1] = binary_op(init, result[i]);
  result[0] = init;
  return result + N;
}

template<class RandomAccessIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
exclusive_scan_impl(RandomAccessIterator first, RandomAccessIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op,
               std::random_access_iterator_tag) {
  // call to std::partial_sum when small data size
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return exclusive_scan_impl(first, last, result, init, binary_op,
             std::input_iterator_tag{});
  }

  scan_impl(first, last, result, init, binary_op, false);
  return result + N;
}

} // namespace details


/**
 * Effects: Assigns through each iterator i in [result,result + (last - first))
 * the value of GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ..., *(first + (i - result) - 1)).
 *
 * Return: The end of the resulting range beginning at result.
 *
 * Requires: binary_op shall not invalidate iterators or subranges, nor modify
 * elements in the ranges [first,last) or [result,result + (last - first)).
 *
 * Complexity: O(last - first) applications of binary_op.
 *
 * Notes: The difference between exclusive_scan and inclusive_scan is that
 * exclusive_scan excludes the ith input element from the ith sum. If
 * binary_op is not mathematically associative, the behavior of exclusive_scan
 * may be non-deterministic.
 * @{
 */
template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op) {
  return details::exclusive_scan_impl(first, last, result, init, binary_op,
           typename std::iterator_traits<InputIterator>::iterator_category());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename T, typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               T init, BinaryOperation binary_op) {
  if (utils::isParallel(exec)) {
    return exclusive_scan(first, last, result, init, binary_op);
  } else {
    return details::exclusive_scan_impl(first, last, result, init, binary_op,
             std::input_iterator_tag{});
  }
}
/**@}*/


/**
 * Effects: Same as exclusive_scan(first, last, result, init, plus<>())
 * @{
 */
template<typename InputIterator, typename OutputIterator,
         typename T,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               T init) {
  return exclusive_scan(first, last, result, init, std::plus<T>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
exclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               T init) {
  return exclusive_scan(exec, first, last, result, init, std::plus<T>());
}
/**@}*/


