/**
 * @file numeric
 * Numeric Parallel algorithms
 */
namespace details {
template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
inclusive_scan_impl(InputIterator first, InputIterator last,
                    OutputIterator result,
                    BinaryOperation binary_op, T init,
                    std::input_iterator_tag) {
  return std::partial_sum(first, last, result, binary_op);
}

template<class RandomAccessIterator, class OutputIterator,
         class T, class BinaryOperation>
OutputIterator
inclusive_scan_impl(RandomAccessIterator first, RandomAccessIterator last,
                    OutputIterator result,
                    BinaryOperation binary_op, T init,
                    std::random_access_iterator_tag) {

  // call to std::partial_sum when small data size
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return inclusive_scan_impl(first, last, result, binary_op, init,
             std::input_iterator_tag{});
  }

  scan_impl(first, last, result, init, binary_op);
  return result + N;
}

} // namespace details

/**
 * Effects: Assigns through each iterator i in [result,result + (last - first))
 * the value of GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, *first, ..., *(first + (i - result)))
 * or GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ..., *(first + (i - result)))
 * if init is provided.
 *
 * Return: The end of the resulting range beginning at result.
 *
 * Requires: binary_op shall not invalidate iterators or subranges, nor modify
 * elements in the ranges [first,last) or [result,result + (last - first)).
 *
 * Complexity: O(last - first) applications of binary_op.
 *
 * Notes: The difference between exclusive_scan and inclusive_scan is that
 * inclusive_scan includes the ith input element in the ith sum. If binary_op
 * is not mathematically associative, the behavior of inclusive_scan may be
 * non-deterministic.
 * @{
 */
template<class InputIterator, class OutputIterator,
         class T, class BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op, T init) {
  return details::inclusive_scan_impl(first, last, result, binary_op, init,
           typename std::iterator_traits<InputIterator>::iterator_category());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename BinaryOperation, typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op, T init) {
  if (utils::isParallel(exec)) {
    return inclusive_scan(first, last, result, binary_op, init);
  } else {
    return details::inclusive_scan_impl(first, last, result, binary_op, init,
             std::input_iterator_tag{});
  }
}

template<typename InputIterator, typename OutputIterator,
         typename BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(first, last, result, binary_op, Type{});
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         typename BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(exec, first, last, result, binary_op, Type{});
}
/**@}*/

/**
 * Effects: Same as inclusive_scan(first, last, result, plus<>()).
 * @{
 */
template<typename InputIterator, typename OutputIterator,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(first, last, result, std::plus<Type>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename OutputIterator,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
OutputIterator
inclusive_scan(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last,
               OutputIterator result) {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
  return inclusive_scan(exec, first, last, result, std::plus<Type>(), Type{});
}
/**@}*/


