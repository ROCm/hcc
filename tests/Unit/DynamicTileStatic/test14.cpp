// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

template<size_t GRID_SIZE, size_t TILE_SIZE>
bool test() {
  using namespace hc;


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
    av1(global) = __hsail_get_groupstaticsize();

    // get the size of total group segment
    av2(global) = __hsail_get_grouptotalsize();

    // store tile index so lds won't be optimized away
    av3(global) = lds1[local[0]];
  });

  // wait for kernel to complete
  fut.wait();

  // verify data
  bool ret = true;
  for (int i = TILE_SIZE-1; i < TILE_SIZE/*GRID_SIZE*/; ++i) {
#if 1
    std::cout << "kernel: static group size: " << av1[i] << "\n" 
              << "kernel: total group size: " << av2[i] << "\n"
              << "kernel: static lds correctly in use: " << (av3[i] == i) << "\n";
#endif

#if 0
    if (av[i] != sizeof(int) * (TILE_SIZE - (i % TILE_SIZE))) {
      ret = false;
      break;
    }
#endif

#if 1
    if ((i + 1) % TILE_SIZE == 0) {
      std::cout << "\n";
    }
#endif
  } 
#if 1
  std::cout << "\n";
#endif
  return ret;
}

int main() {
  bool ret = true;

  // FIXME: uncomment these line back when we have an updated dynamic group
  // segment allocation routine
#if 0
  ret &= test<1, 1>();
#endif
#if 0
  ret &= test<4, 2>();
  ret &= test<8, 4>();
  ret &= test<64, 16>();
#endif
#if 1
  ret &= test<64, 64>();
#endif
#if 0
  ret &= test<4096, 64>();
#endif

  return !(ret == true);
}
