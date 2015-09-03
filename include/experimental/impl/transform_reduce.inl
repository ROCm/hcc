#pragma once

/**
 *
 * @return GENERALIZED_SUM(binary_op, init, unary_op(*first), ..., unary_op(*(first + (last - first) - * 1))).
 *
 * Requires: Neither unary_op nor binary_op shall invalidate subranges, or
 * modify elements in the range [first,last).
 *
 * Complexity: O(last - first) applications each of unary_op and binary_op.
 *
 * Notes: transform_reduce does not apply unary_op to init.
 */
template<typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T transform_reduce(InputIterator first, InputIterator last,
                   UnaryOperation unary_op,
                   T init, BinaryOperation binary_op) {
  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    auto new_op = [&](const T& a, const _Tp& b) {
      return binary_op(a, unary_op(b));
    };
    return std::accumulate(first, last, init, new_op);
  }

  std::unique_ptr<T> tmp(new T [N]);

  // implement by transform & reduce
  // Note: because reduce may apply the binary_op in any order, so
  // we can't just apply the unary_op inside that.
  transform(par, first, last, tmp.get(), unary_op);

  // inline the reduction kernel from reduce.inl
  // save the cost of creating another stride for internal usage
  auto tmp_ = tmp.get();
  unsigned s = (N + 1) / 2;

  details::kernel_launch(s, [tmp_, N, &s, binary_op](hc::index<1> idx) __attribute((hc)) {
    // first round
    details::round(idx[0], N, tmp_, tmp_, binary_op);

    // Reduction kernel: apply logN - 1 times
    do {
      details::round(idx[0], s, tmp_, tmp_, binary_op);
      s = (s + 1) / 2;
    } while (s > 1);
  });

  // apply initial value
  T ans  = binary_op(init, tmp_[0]);

  return ans;
}

/**
 *
 * @return GENERALIZED_SUM(binary_op, init, unary_op(*first), ..., unary_op(*(first + (last - first) - * 1))).
 *
 * Requires: Neither unary_op nor binary_op shall invalidate subranges, or
 * modify elements in the range [first,last).
 *
 * Complexity: O(last - first) applications each of unary_op and binary_op.
 *
 * Notes: transform_reduce does not apply unary_op to init.
 */
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


// inner_product is basically a transform_reduce (two vectors version)
// make an alias (perfect forwarding) for that
template <typename... Args>
auto transform_reduce(Args&&... args)
       -> decltype(inner_product(std::forward<Args>(args)...)) {
  return inner_product(std::forward<Args>(args)...);
}

/**
 * Parallel version of std::inner_product in <algorithm>
 */
template<typename ExecutionPolicy,
         typename InputIt1, typename InputIt2,
         typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt1>> = nullptr>
T inner_product(ExecutionPolicy&& exec,
              InputIt1 first1, InputIt1 last1,
              InputIt2 first2,
              T value) {
  typedef typename std::iterator_traits<InputIt1>::value_type _Tp;
  return inner_product(first1, last1, first2, value,
                       std::plus<_Tp>(), std::multiplies<_Tp>());
}

/**
 * Parallel version of std::inner_product in <algorithm>
 */
template<typename ExecutionPolicy,
         typename InputIt1, typename InputIt2,
         typename T,
         typename BinaryOperation1, typename BinaryOperation2,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt1>> = nullptr>
T inner_product(ExecutionPolicy&& exec,
              InputIt1 first1, InputIt1 last1,
              InputIt2 first2,
              T value,
              BinaryOperation1 op1,
              BinaryOperation2 op2) {
  const size_t N = static_cast<size_t>(std::distance(first1, last1));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return std::inner_product(first1, last1, first2, value, op1, op2);
  }

  typedef typename std::iterator_traits<InputIt1>::value_type _Tp;
  std::unique_ptr<_Tp> tmp(new _Tp [N]);

  // implement inner_product by transform & reduce
  transform(exec, first1, last1, first2, tmp.get(), op2);

  // inline the reduction kernel from reduce.inl
  // save the cost of creating another stride for internal usage
  auto tmp_ = tmp.get();
  unsigned s = (N + 1) / 2;

  details::kernel_launch(s, [tmp_, N, &s, op1](hc::index<1> idx) __attribute((hc)) {
    // first round
    details::round(idx[0], N, tmp_, tmp_, op1);

    // Reduction kernel: apply logN - 1 times
    do {
      details::round(idx[0], s, tmp_, tmp_, op1);
      s = (s + 1) / 2;
    } while (s > 1);
  });

  // apply initial value
  T ans  = op1(value, tmp_[0]);

  return ans;
}
