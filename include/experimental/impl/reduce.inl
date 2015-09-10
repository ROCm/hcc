/**
 * @file numeric
 * Numeric Parallel algorithms
 */
#pragma once

namespace details {
template<class InputIterator, class T, class BinaryOperation>
T reduce_impl(InputIterator first, InputIterator last,
         T init,
         BinaryOperation binary_op,
         std::input_iterator_tag) {
  return std::accumulate(first, last, init, binary_op);
}

template<class T, class U, class BinaryOperation>
inline void round(const unsigned &i, const unsigned &N,
                  T *tmp, U *src,
                  BinaryOperation binary_op) {
  if (2*i+1 < N) {
    tmp[i] = binary_op(src[2*i], src[2*i+1]);
  } else {
    tmp[i] = src[2*i];
  }
}

template<class RandomAccessIterator, class T, class BinaryOperation>
T reduce_impl(RandomAccessIterator first, RandomAccessIterator last,
              T init,
              BinaryOperation binary_op,
              std::random_access_iterator_tag) {

  size_t N = static_cast<size_t>(std::distance(first, last));

  // call to std::accumulate when small data size
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return reduce_impl(first, last, init, binary_op, std::input_iterator_tag{});
  }

  unsigned s = (N + 1) / 2;
  T *tmp = new T [s];
  auto first_ = utils::get_pointer(first);

  kernel_launch(s, [tmp, first_, N, &s, binary_op](hc::index<1> idx) __attribute((hc)) {
    // first round
    round(idx[0], N, tmp, first_, binary_op);

    // Reduction kernel: apply logN - 1 times
    do {
      round(idx[0], s, tmp, tmp, binary_op);
      s = (s + 1) / 2;
    } while (s > 1);
  });

  // apply initial value
  T ans  = binary_op(init, tmp[0]);

  delete [] tmp;
  return ans;
}
} // namespace details


/**
 *
 * Return: GENERALIZED_SUM(binary_op, init, *first, ..., *(first + (last - first) - 1)).
 *
 * Requires: binary_op shall not invalidate iterators or subranges, nor modify
 * elements in the range [first,last).
 *
 * Complexity: O(last - first) applications of binary_op.
 *
 * Notes: The primary difference between reduce and accumulate is that the
 * behavior of reduce may be non-deterministic for non-associative or
 * non-commutative binary_op.
 * @{
 */
template<class InputIterator, class T, class BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T reduce(InputIterator first, InputIterator last,
         T init,
         BinaryOperation binary_op) {
  return details::reduce_impl(first, last, init, binary_op,
           typename std::iterator_traits<InputIterator>::iterator_category());
}

template<class ExecutionPolicy, class InputIterator, class T, class BinaryOperation,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T
reduce(ExecutionPolicy&& exec,
               InputIterator first, InputIterator last, T init,
               BinaryOperation binary_op) {
  if (utils::isParallel(exec)) {
    return reduce(first, last, init, binary_op);
  } else {
    return details::reduce_impl(first, last, init, binary_op,
             std::input_iterator_tag{});
  }
}
/**@}*/

/**
 * Effects: Same as reduce(first, last, init, plus<>())
 * @{
 */
template<typename InputIterator, typename T,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T reduce(InputIterator first, InputIterator last, T init) {
  typedef typename std::iterator_traits<InputIterator>::value_type Type;
  return reduce(first, last, init, std::plus<Type>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T
reduce(ExecutionPolicy&& exec,
         InputIterator first, InputIterator last, T init) {
  typedef typename std::iterator_traits<InputIterator>::value_type Type;
  return reduce(exec, first, last, init, std::plus<Type>());
}
/**@}*/

/**
 * Effects: Same as reduce(first, last, typename iterator_traits<InputIterator>::value_type{})
 * @{
 */
template<typename InputIterator,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
typename std::iterator_traits<InputIterator>::value_type
reduce(InputIterator first, InputIterator last) {
  typedef typename std::iterator_traits<InputIterator>::value_type Type;
  return reduce(first, last, Type{});
}

template<typename ExecutionPolicy,
         typename InputIterator,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
typename std::iterator_traits<InputIterator>::value_type
reduce(ExecutionPolicy&& exec,
       InputIterator first, InputIterator last) {
  typedef typename std::iterator_traits<InputIterator>::value_type Type;
  return reduce(exec, first, last, Type{}, std::plus<Type>());
}
/**@}*/

