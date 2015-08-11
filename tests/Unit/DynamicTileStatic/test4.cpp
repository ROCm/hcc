// XFAIL: Linux
// RUN: %cxxamp %s -Xclang -fhsa-ext -o %t.out && %t.out

#include <amp.h>
#include <hc.hpp>

#include <iostream>

#define __GROUP__ __attribute__((address_space(3)))

/// each workgroup will allocate 1 element in the group segment for each workitem
template<typename T>
bool test1D(size_t grid_size, size_t tile_size) {
  using namespace hc;

  // array_view which will store the offset of allocated memory for each
  // work item, relative to the beginning of group segment
  Concurrency::array_view<T, 1> avOffset(grid_size);

  // initialize ts_allocator
  ts_allocator tsa;
  tsa.setDynamicGroupSegmentSize(tile_size * sizeof(T));

  // launch kernel in tiled fashion
  tiled_extent<1> ex(grid_size, tile_size);
  parallel_for_each(ex, tsa, [&, avOffset](tiled_index<1>& idx) restrict(amp) {

    // reset allocator
    tsa.reset();

    // call ts_allocator
    // allocate 1 element for each work item
    __GROUP__ T* p = (__GROUP__ T*) tsa.alloc(sizeof(T) * 1);

    p += idx.local[0];

    // get the beginning of dynamic group memory
    __GROUP__ T* lds = (__GROUP__ T*) getLDS(tsa.getStaticGroupSegmentSize());

    // write allocated offset to avOffset
    avOffset(idx) = (p - lds) * sizeof(T);
  });

#if 0
  // print offset
  for (int i = 0; i < grid_size; ++i) {
    std::cout << avOffset[i] << " ";
  }
  std::cout << "\n";
#endif

  // verify data
  bool ret = true;
  for (int i = 0; i < grid_size; i += tile_size) {
    int sum = 0;
    for (int j = 0; j < tile_size; ++j) {
      sum += avOffset[i + j];
    }
    // check if the sum of offsets allocated for each tile is correct
    if (sum != (sizeof(T) * (tile_size * (tile_size - 1) / 2))) {
      ret = false;
      break;
    }
  }
#if 0
  if (ret == true) {
    std::cout << "verify success\n";
  }
#endif

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<int>(1, 1);
  ret &= test1D<int>(8, 2);
  ret &= test1D<int>(4096, 64);
  ret &= test1D<int>(4096, 256);

  ret &= test1D<float>(1, 1);
  ret &= test1D<float>(8, 2);
  ret &= test1D<float>(4096, 64);
  ret &= test1D<float>(4096, 256);

  ret &= test1D<double>(1, 1);
  ret &= test1D<double>(8, 2);
  ret &= test1D<double>(4096, 64);
  ret &= test1D<double>(4096, 256);

  return !(ret == true);
}

