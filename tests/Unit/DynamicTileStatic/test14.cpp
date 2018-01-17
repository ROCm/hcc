
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

template<size_t GRID_SIZE, size_t TILE_SIZE, size_t DG_SIZE>
bool test() {
  using namespace hc;

  constexpr size_t STATIC_GROUP_SEGMENT_SIZE = sizeof(int) * TILE_SIZE;
  // total group segment asked by the application
  // HSA runtime may assign more than this value
  constexpr size_t TOTAL_GROUP_SEGMENT_SIZE = (STATIC_GROUP_SEGMENT_SIZE + DG_SIZE);

  array_view<int, 1> av1(GRID_SIZE);
  array_view<int, 1> av2(GRID_SIZE);
  array_view<int, 1> av3(GRID_SIZE);
  tiled_extent<1> ex(GRID_SIZE, TILE_SIZE, DG_SIZE);
  
  completion_future fut = parallel_for_each(hc::accelerator().get_default_view(),
                    ex,
                    [=](tiled_index<1>& tidx) [[hc]] {
    tile_static int lds1[TILE_SIZE];

    // obtain workitem absolute index and workgroup index
    index<1> global = tidx.global;
    index<1> local = tidx.local;

    lds1[local[0]] = local[0];
    tidx.barrier.wait();

    // get the size of static group segment
    av1(global) = get_static_group_segment_size();

    // get the size of total group segment
    av2(global) = get_group_segment_size();

    // store tile index so lds won't be optimized away
    av3(global) = lds1[local[0]];
  });

  // wait for kernel to complete
  fut.wait();

  // verify data
  bool ret = true;
  for (int i = 0; i < GRID_SIZE; ++i) {
#if 0
    if (i == 0)
    std::cout << "kernel: static group size: " << av1[i] << "\n" 
              << "kernel: total group size: " << av2[i] << "\n"
              << "kernel: static lds correctly in use: " << (av3[i] == i) << "\n";
#endif

    // verify if static group segment size is correct
    if (av1[i] != STATIC_GROUP_SEGMENT_SIZE) {
      std::cerr << "static group size error: " << av1[i] << " , expected: " << STATIC_GROUP_SEGMENT_SIZE << "\n";
      ret = false;
      break;
    }

    // verify if total group segment size is correct
    // HSA runtime should assign more memory than the amount asked by the application
    if (av2[i] < TOTAL_GROUP_SEGMENT_SIZE) {
      std::cerr << "total group size error: " << av2[i] << " , expected: " << TOTAL_GROUP_SEGMENT_SIZE << "\n";
      ret = false;
      break;
    }

    // verify if static group segment has been correctly used
    if (av3[i] != (i % TILE_SIZE)) {
      std::cerr << "static group segment value error: " << av3[i] << " , expected: " << (i % TILE_SIZE);
      ret = false;
      break;
    }
  } 

  return ret;
}

int main() {
  bool ret = true;

  // The test case is only workable on LC backend as of now
  // because on HSAIL backend there is no way to check the size of
  // group segment.

  // Skip the test in case we are not using LC backend
#if __hcc_backend__ == HCC_BACKEND_AMDGPU
  ret &= test<1, 1, 1024>();
  ret &= test<4, 2, 1024>();
  ret &= test<8, 4, 1024>();
  ret &= test<64, 16, 1024>();
  ret &= test<64, 64, 1024>();
  ret &= test<4096, 256, 1024>();
#endif

  return !(ret == true);
}
