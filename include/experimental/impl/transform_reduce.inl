/**
 * @file numeric
 * Numeric Parallel algorithms
 */
#pragma once

#define _T_REDUCE_WAVEFRONT_SIZE 512

#define _T_REDUCE_STEP(_LENGTH, _IDX, _W) \
    if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
      T mine = scratch[_IDX];\
      T other = scratch[_IDX + _W];\
      scratch[_IDX] = binary_op(mine, other); \
    }\
    t_idx.barrier.wait();

/**
 *
 * Return: GENERALIZED_SUM(binary_op, init, unary_op(*first), ..., unary_op(*(first + (last - first) - * 1))).
 *
 * Requires: Neither unary_op nor binary_op shall invalidate subranges, or
 * modify elements in the range [first,last).
 *
 * Complexity: O(last - first) applications each of unary_op and binary_op.
 *
 * Notes: transform_reduce does not apply unary_op to init.
 * @{
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

  int max_ComputeUnits = 32;
  int numTiles = max_ComputeUnits*32;
  int length = (_T_REDUCE_WAVEFRONT_SIZE * numTiles);
  length = N < length ? N : length;
  unsigned int residual = length % _T_REDUCE_WAVEFRONT_SIZE;
  length = residual ? (length + _T_REDUCE_WAVEFRONT_SIZE - residual): length ;
  numTiles = static_cast< int >((N/_T_REDUCE_WAVEFRONT_SIZE)>= numTiles?(numTiles):
                                (std::ceil( static_cast< float >( N ) / _T_REDUCE_WAVEFRONT_SIZE) ));

  std::unique_ptr<T[]> r(new T[numTiles]);
  auto f_ = utils::get_pointer(first);
  hc::array_view<T> result(hc::extent<1>(numTiles), r.get());
  hc::array_view<_Tp> first_(hc::extent<1>(N), f_);
  result.discard_data();
  auto transform_op = unary_op;
  details::kernel_launch(length, [first_, N, length, transform_op, result, binary_op] (hc::tiled_index<1> t_idx) [[hc]]
                {
                int gx = t_idx.global[0];
                int gloId = gx;
                tile_static T scratch[_T_REDUCE_WAVEFRONT_SIZE];
                //  Initialize local data store
                unsigned int tileIndex = t_idx.local[0];

                T accumulator;
                if (gloId < N)
                {
                accumulator = transform_op(first_[gx]);
                gx += length;
                }


                // Loop sequentially over chunks of input vector, reducing an arbitrary size input
                // length into a length related to the number of workgroups
                while (gx < N)
                {
                T element = transform_op(first_[gx]);
                accumulator = binary_op(accumulator, element);
                gx += length;
                }

                scratch[tileIndex] = accumulator;
                t_idx.barrier.wait();

                unsigned int tail = N - (t_idx.tile[0] * _T_REDUCE_WAVEFRONT_SIZE);

                _T_REDUCE_STEP(tail, tileIndex, 256);
                _T_REDUCE_STEP(tail, tileIndex, 128);
                _T_REDUCE_STEP(tail, tileIndex, 64);
                _T_REDUCE_STEP(tail, tileIndex, 32);
                _T_REDUCE_STEP(tail, tileIndex, 16);
                _T_REDUCE_STEP(tail, tileIndex, 8);
                _T_REDUCE_STEP(tail, tileIndex, 4);
                _T_REDUCE_STEP(tail, tileIndex, 2);
                _T_REDUCE_STEP(tail, tileIndex, 1);


                //  Abort threads that are passed the end of the input vector
                if (gloId >= N)
                    return;

                //  Write only the single reduced value for the entire workgroup
                if (tileIndex == 0)
                {
                    result[t_idx.tile[ 0 ]] = scratch[0];
                }
                }, _T_REDUCE_WAVEFRONT_SIZE);
  result.synchronize();
  auto ans = std::accumulate(r.get(), r.get() + numTiles, init, binary_op);
  return ans;
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
/**@}*/


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

  /// OPTIMIZE: remove the unnecessary buffer on cpu
  typedef typename std::iterator_traits<InputIt1>::value_type _Tp;
  std::vector<_Tp> dist(N, _Tp());

  // implement inner_product by transform & reduce
  transform(exec, first1, last1, first2, std::begin(dist), op2);
  return reduce(exec, std::begin(dist), std::end(dist), value, op1);
}
/**@}*/
