// FIXME, this is a implementation of transform_reduce based on C++AMP
// ideally we want to get rid of C++AMP dependency soon
template<typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation>
T transform_reduce(InputIterator first, InputIterator last,
                   UnaryOperation unary_op,
                   T init, BinaryOperation binary_op) {
  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;

  // reduce implementation based on C++ AMP
#define TILE_SIZE (64)

  size_t elementCount = static_cast<size_t>(std::distance(first, last));
  int numTiles = (elementCount / TILE_SIZE) ? (elementCount / TILE_SIZE) : 1;

  std::vector<_Tp> av_src(elementCount);
  std::copy(first, last, av_src.data());
  std::vector<_Tp> av_dst(numTiles);

  bool unary_op_processed = false;
  while((elementCount % TILE_SIZE) == 0) {
    concurrency::parallel_for_each(concurrency::extent<1>(elementCount).tile<TILE_SIZE>(), [&](concurrency::tiled_index<TILE_SIZE> tidx) restrict(amp) {
      tile_static _Tp scratch[TILE_SIZE];

      unsigned tileIndex = tidx.local[0];
      unsigned globalIndex = tidx.global[0];
      scratch[tileIndex] = unary_op(av_src[globalIndex]);
      tidx.barrier.wait();

      for (unsigned stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (tileIndex < stride) {
          scratch[tileIndex] = binary_op(scratch[tileIndex], scratch[tileIndex + stride]);
        }
        tidx.barrier.wait();
      }

      if (tileIndex == 0) {
        av_dst[tidx.tile[0]] = scratch[0];
      }
    });

    unary_op_processed = true;
    elementCount /= TILE_SIZE;
    std::swap(av_src, av_dst);
#if 0
    for (int i = 0; i < elementCount; ++i) {
      std::cout << av_src[i] << " ";
    }
    std::cout << "\n";
#endif
  }

  std::vector<_Tp> resultVector(elementCount);
  std::copy(av_src.data(), av_src.data() + elementCount, std::begin(resultVector));
  if (!unary_op_processed)
    std::transform(std::begin(resultVector), std::end(resultVector), std::begin(resultVector), unary_op);
  _Tp resultValue = std::accumulate(std::begin(resultVector), std::end(resultVector), init, binary_op);

  return resultValue;
}

template<typename ExecutionPolicy,
         typename InputIterator, typename UnaryOperation,
         typename T, typename BinaryOperation>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, T>::type
transform_reduce(ExecutionPolicy&& exec,
                 InputIterator first, InputIterator last,
                 UnaryOperation unary_op,
                 T init, BinaryOperation binary_op) {
  return transform_reduce(first, last, unary_op, init, binary_op);
}

