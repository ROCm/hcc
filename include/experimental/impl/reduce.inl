// FIXME, this is a SEQUENTIAL implementation of reduce!
// a special version of reduce which does NOT dereference the iterator
template<class InputIterator, class T, class BinaryOperation>
T __reduce(InputIterator first, InputIterator last, T init,
           BinaryOperation binary_op) {
  T result = init;
  for (; first != last; ++first) {
    result = binary_op(init, first);
  }
  return result;
}

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

  using hc::extent;
  using hc::index;
  using hc::parallel_for_each;
  hc::ts_allocator tsa;

  size_t N = static_cast<size_t>(std::distance(first, last));

  // call to std::accumulate when small data size
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return reduce_impl(first, last, init, binary_op, std::input_iterator_tag{});
  }

  unsigned s = (N + 1) / 2;
  T *tmp = new T [s];
  auto first_ = utils::get_pointer(first);

  parallel_for_each(extent<1>(s), tsa,
                    [tmp, first_, N, &s, binary_op](index<1> idx) restrict(amp) {
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

//// FIXME, this is a implementation of reduce based on C++AMP
//// ideally we want to drop all C++AMP stuffs in parallel STL
//template<class InputIterator, class T, class BinaryOperation>
//T reduce(InputIterator first, InputIterator last, T init,
//               BinaryOperation binary_op) {
//  typedef typename std::iterator_traits<InputIterator>::value_type _Tp;
//
//#define TILE_SIZE (64)
//  size_t elementCount = static_cast<size_t>(std::distance(first, last));
//  int numTiles = (elementCount / TILE_SIZE) ? (elementCount / TILE_SIZE) : 1;
//
//  std::vector<_Tp> av_src(elementCount);
//  std::copy(first, last, av_src.data());  // FIXME: how can we get rid of this copy?
//  std::vector<_Tp> av_dst(numTiles);
//
//  while((elementCount % TILE_SIZE) == 0) {
//    concurrency::parallel_for_each(concurrency::extent<1>(elementCount).tile<TILE_SIZE>(), [&](concurrency::tiled_index<TILE_SIZE> tidx) restrict(amp) {
//      tile_static _Tp scratch[TILE_SIZE];
//
//      unsigned tileIndex = tidx.local[0];
//      unsigned globalIndex = tidx.global[0];
//      scratch[tileIndex] = av_src[globalIndex];
//      tidx.barrier.wait();
//
//      for (unsigned stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
//        if (tileIndex < stride) {
//          scratch[tileIndex] = binary_op(scratch[tileIndex], scratch[tileIndex + stride]);
//        }
//        tidx.barrier.wait();
//      }
//
//      if (tileIndex == 0) {
//        av_dst[tidx.tile[0]] = scratch[0];
//      }
//    });
//
//
//    elementCount /= TILE_SIZE;
//    std::swap(av_src, av_dst);
//
//#if 0
//    for (int i = 0; i < elementCount; ++i) {
//      std::cout << av_src[i] << " ";
//    }
//    std::cout << "\n";
//#endif
//  }
//
//  std::vector<_Tp> resultVector(elementCount);
//  std::copy(av_src.data(), av_src.data() + elementCount, std::begin(resultVector));
//  _Tp resultValue = std::accumulate(std::begin(resultVector), std::end(resultVector), init, binary_op);
//
//  return resultValue;
//}

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
  return reduce(first, last, init, binary_op);
}

template<typename InputIterator, typename T,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T reduce(InputIterator first, InputIterator last, T init) {
  return reduce(first, last, init, std::plus<T>());
}

template<typename ExecutionPolicy,
         typename InputIterator, typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIterator>> = nullptr>
T
reduce(ExecutionPolicy&& exec,
         InputIterator first, InputIterator last, T init) {
  return reduce(first, last, init);
}

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
  return reduce(first, last);
}

