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

#define REDUCE_WAVEFRONT_SIZE 512
#define _REDUCE_STEP(_LENGTH, _IDX, _W) \
if ((_IDX < _W) && ((_IDX + _W) < _LENGTH)) {\
	T mine = scratch[_IDX]; \
	T other = scratch[_IDX + _W]; \
	scratch[_IDX] = binary_op(mine, other); \
}\
    t_idx.barrier.wait();

int reduce_lexi(std::vector<int>& v) {

    const int N = static_cast<int>(v.size());
    auto binary_op = [](const int& a, const int& b) [[hc]] [[cpu]] { return a == 1 ? b : a; };
    // call to std::accumulate when small data size
    if (N <= details::PARALLELIZE_THRESHOLD) {
        return reduce_impl(std::begin(v), std::end(v), 1, binary_op, std::input_iterator_tag{});
    }

    int max_ComputeUnits = 32;
    int numTiles = max_ComputeUnits*32;
    int length = (REDUCE_WAVEFRONT_SIZE*numTiles);
    length = N < length ? N : length;
    unsigned int residual = length % REDUCE_WAVEFRONT_SIZE;
    length = residual ? (length + REDUCE_WAVEFRONT_SIZE - residual): length ;
    numTiles = static_cast< int >((N/REDUCE_WAVEFRONT_SIZE)>= numTiles?(numTiles):
                                  (std::ceil( static_cast< float >( N ) / REDUCE_WAVEFRONT_SIZE) ));

    std::vector<int> r(numTiles);
    hc::array_view<int> result(numTiles, r);
    hc::array_view<const int> first_(N, v);
    result.discard_data();
    kernel_launch(length,
                  [ first_, N, length, result, binary_op ]
                  ( hc::tiled_index<1> t_idx ) [[hc]] [[cpu]]
                  {
                  using T = int;
                  int gx = t_idx.global[0];
                  int gloId = gx;
                  tile_static int scratch[REDUCE_WAVEFRONT_SIZE];
                  //  Initialize local data store
                  unsigned int tileIndex = t_idx.local[0];

                  int accumulator;
                  if (gloId < N)
                  {
                  accumulator = first_[gx];
                  gx += length;
                  }


                  // Loop sequentially over chunks of input vector, reducing an arbitrary size input
                  // length into a length related to the number of workgroups
                  while (gx < N)
                  {
                      T element = first_[gx];
                      accumulator = binary_op(accumulator, element);
                      gx += length;
                  }

                  scratch[tileIndex] = accumulator;
                  t_idx.barrier.wait();

                  unsigned int tail = N - (t_idx.tile[0] * REDUCE_WAVEFRONT_SIZE);

                  _REDUCE_STEP(tail, tileIndex, 1);
                  _REDUCE_STEP(tail, tileIndex, 2);
                  _REDUCE_STEP(tail, tileIndex, 4);
                  _REDUCE_STEP(tail, tileIndex, 8);
                  _REDUCE_STEP(tail, tileIndex, 16);
                  _REDUCE_STEP(tail, tileIndex, 32);
                  _REDUCE_STEP(tail, tileIndex, 64);
                  _REDUCE_STEP(tail, tileIndex, 128);
                  _REDUCE_STEP(tail, tileIndex, 256);


                  //  Abort threads that are passed the end of the input vector
                  if (gloId >= N)
                      return;

                  //  Write only the single reduced value for the entire workgroup
                  if (tileIndex == 0)
                  {
                      result[t_idx.tile[ 0 ]] = scratch[0];
                  }

                  }, REDUCE_WAVEFRONT_SIZE);

    result.synchronize();
    auto ans = std::accumulate(std::begin(r), std::end(r), 1, binary_op);
    return ans;
}

template<class RandomAccessIterator, class T, class BinaryOperation>
T reduce_impl(RandomAccessIterator first, RandomAccessIterator last,
              T init,
              BinaryOperation binary_op,
              std::random_access_iterator_tag) {

    const int N = static_cast<int>(std::distance(first, last));
    // call to std::accumulate when small data size
    if (N <= details::PARALLELIZE_THRESHOLD) {
        return reduce_impl(first, last, init, binary_op, std::input_iterator_tag{});
    }

    int max_ComputeUnits = 32;
    int numTiles = max_ComputeUnits*32;
    int length = (REDUCE_WAVEFRONT_SIZE*numTiles);
    length = N < length ? N : length;
    unsigned int residual = length % REDUCE_WAVEFRONT_SIZE;
    length = residual ? (length + REDUCE_WAVEFRONT_SIZE - residual): length ;
    numTiles = static_cast< int >((N/REDUCE_WAVEFRONT_SIZE)>= numTiles?(numTiles):
                                  (std::ceil( static_cast< float >( N ) / REDUCE_WAVEFRONT_SIZE) ));

    auto f_ = utils::get_pointer(first);
    using _Ty = typename std::iterator_traits<RandomAccessIterator>::value_type;
    std::vector<T> r(numTiles);
    hc::array_view<T> result(hc::extent<1>(numTiles), r);
    hc::array_view<const _Ty> first_(hc::extent<1>(N), f_);
    result.discard_data();
    kernel_launch(length,
                  [ first_, N, length, result, binary_op ]
                  ( hc::tiled_index<1> t_idx ) [[hc]]
                  {
                  int gx = t_idx.global[0];
                  int gloId = gx;
                  tile_static T scratch[REDUCE_WAVEFRONT_SIZE];
                  //  Initialize local data store
                  unsigned int tileIndex = t_idx.local[0];

                  T accumulator;
                  if (gloId < N)
                  {
                  accumulator = first_[gx];
                  gx += length;
                  }


                  // Loop sequentially over chunks of input vector, reducing an arbitrary size input
                  // length into a length related to the number of workgroups
                  while (gx < N)
                  {
                      T element = first_[gx];
                      accumulator = binary_op(accumulator, element);
                      gx += length;
                  }

                  scratch[tileIndex] = accumulator;
                  t_idx.barrier.wait();

                  unsigned int tail = N - (t_idx.tile[0] * REDUCE_WAVEFRONT_SIZE);

                  _REDUCE_STEP(tail, tileIndex, 256);
                  _REDUCE_STEP(tail, tileIndex, 128);
                  _REDUCE_STEP(tail, tileIndex, 64);
                  _REDUCE_STEP(tail, tileIndex, 32);
                  _REDUCE_STEP(tail, tileIndex, 16);
                  _REDUCE_STEP(tail, tileIndex, 8);
                  _REDUCE_STEP(tail, tileIndex, 4);
                  _REDUCE_STEP(tail, tileIndex, 2);
                  _REDUCE_STEP(tail, tileIndex, 1);


                  //  Abort threads that are passed the end of the input vector
                  if (gloId >= N)
                      return;

                  //  Write only the single reduced value for the entire workgroup
                  if (tileIndex == 0)
                  {
                      result[t_idx.tile[ 0 ]] = scratch[0];
                  }

                  }, REDUCE_WAVEFRONT_SIZE);

    result.synchronize();
    auto ans = std::accumulate(std::begin(r), std::end(r), init, binary_op);
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

