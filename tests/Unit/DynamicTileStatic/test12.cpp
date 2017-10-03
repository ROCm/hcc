
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

#define __KERNEL__ __attribute__((amp))

template<size_t GRID_SIZE, size_t TILE_SIZE>
bool test() {
  using namespace hc;


  array_view<int, 1> av(GRID_SIZE);
  tiled_extent<1> ex(GRID_SIZE, TILE_SIZE);
  ex.set_dynamic_group_segment_size(1024);
  
  completion_future fut = parallel_for_each(hc::accelerator().get_default_view(),
                    ex,
                    __KERNEL__ [=](tiled_index<1>& tidx) {
    tile_static int lds1[TILE_SIZE];

    // obtain workitem absolute index and workgroup index
    index<1> global = tidx.global;
    index<1> local = tidx.local;

    // fetch the address of a variable in group segment
    unsigned char* ptr = (unsigned char*)&lds1[local[0]];

    // fetch the address of the beginning of dynamic group segment
    unsigned char* dynamic_lds = (unsigned char*)get_dynamic_group_segment_base_pointer();

    // calculate the offset and set to the result global array_view
    av(global) = (dynamic_lds - ptr);
  });

  // wait for kernel to complete
  fut.wait();

  // verify data
  bool ret = true;
  for (int i = 0; i < GRID_SIZE; ++i) {
#if 0
    std::cout << av[i] << " ";
#endif

    if (av[i] != sizeof(int) * (TILE_SIZE - (i % TILE_SIZE))) {
      ret = false;
      break;
    }

#if 0
    if ((i + 1) % TILE_SIZE == 0) {
      std::cout << "\n";
    }
#endif
  } 
#if 0
  std::cout << "\n";
#endif
  return ret;
}

int main() {
  bool ret = true;

  // The test case is only workable on LC backend as of now
  // because on HSAIL backend there is no way to check the size of
  // group segment.

  // Skip the test in case we are not using LC backend
#if __hcc_backend__ == HCC_BACKEND_AMDGPU
  ret &= test<1, 1>();
  ret &= test<4, 2>();
  ret &= test<8, 4>();
  ret &= test<64, 16>();
  ret &= test<256, 32>();
  ret &= test<4096, 64>();
#endif

  return !(ret == true);
}
