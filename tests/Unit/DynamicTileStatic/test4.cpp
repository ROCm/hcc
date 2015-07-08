// XFAIL: Linux
// RUN: %cxxamp %s -Xclang -fhsa-ext -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

/// each work item will allocate 1 plus its global index of elements
/// in the assigned host buffer
/// FIXME: use group segment instead of global segment
template<typename T>
bool test1D(size_t grid_size, size_t tile_size) {
  using namespace concurrency;

  size_t buffer_elements = ((grid_size + 1) * grid_size / 2);
  size_t sizeof_element = sizeof(T);
  size_t buffer_size = buffer_elements * sizeof_element;

  // dynamically allocate a host buffer
  // FIXME: use group segment instead of global segment
  T* table = static_cast<T*>(malloc(buffer_size));

  // initialize ts_allocator which uses the host buffer
  // FIXME: use group segment instead of global segment
  ts_allocator tsa(table);

  // array_view which will store the offset of allocated memory for each
  // work item, relative to the beginning of table
  array_view<T, 1> av(grid_size);

  // launch kernel in tiled fashion
  tiled_extent_1D ex(grid_size, tile_size);
  parallel_for_each(ex, 0, [&, av](tiled_index_1D& idx) restrict(amp) {

    // fetch work item global index
    index<1> global = idx.global;

    // call ts_allocator
    // allocate (global index + 1) of element for each work item
    T* p = static_cast<T*>(tsa.alloc((global[0] + 1) * sizeof(T)));

    // write offset to av
    av(idx) = (p - table);

    // fill in data into allocated buffer
    for (int i = 0; i < global[0] + 1; ++i) {
      p[i] = (global[0] + 1);
    }
  });

#if 0
  // print offset
  for (int i = 0; i < grid_size; ++i) {
    std::cout << av[i] << " ";
  }
  std::cout << "\n";
#endif

#if 0
  // print array data
  for (int i = 0; i < buffer_elements; ++i) {
    std::cout << table[i] << " ";
  }
  std::cout << "\n";
#endif

  // verify data
  bool ret = true;
  for (int i = 0; i < grid_size; ++i) {
    int offset = av[i];
    for (int j = 0; j < (i + 1); ++j) {
      if (table[offset + j] != (i + 1)) {
#if 0
        std::cout << "verify failed at: " << offset + j << "\n";
#endif
        ret = false;
        break;
      }
    }
  }
#if 0
  if (ret == true) {
    std::cout << "verify success\n";
  }
#endif

  // release dynamically allocated host buffer
  // FIXME: use group segment instead of global segment
  free(table);

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

