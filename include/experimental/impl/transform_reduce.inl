#pragma once

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
  return reduce(tmp.get(), tmp.get() + N, init, binary_op);
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


// inner_product is basically a transform_reduce (two vectors version)
// make an alias (perfect forwarding) for that
template <typename... Args>
auto transform_reduce(Args&&... args)
       -> decltype(inner_product(std::forward<Args>(args)...)) {
  return inner_product(std::forward<Args>(args)...);
}

// inner_product
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
  return reduce(exec, tmp.get(), tmp.get() + N, value, op1);
}
