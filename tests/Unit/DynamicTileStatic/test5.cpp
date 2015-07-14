// XFAIL: Linux
// RUN: %cxxamp %s -Xclang -fhsa-ext -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

#define __KERNEL__ __attribute__((amp))
#define __GROUP__ __attribute__((address_space(3)))

#define ALLOC_ELEMENT (local[0] + 1)
#define ELEMENT_SIZE (sizeof(int))
#define ALLOC_SIZE (ALLOC_ELEMENT * ELEMENT_SIZE)

#define DYNAMIC_GROUP_SEGMENT_SIZE ((1 + TILE_SIZE) * TILE_SIZE / 2 * ELEMENT_SIZE)

template<size_t GRID_SIZE, size_t TILE_SIZE>
bool test() {
  using namespace hc;

  ts_allocator tsa;
  tsa.setDynamicGroupSegmentSize(DYNAMIC_GROUP_SEGMENT_SIZE);

  Concurrency::array_view<int, 1> av(GRID_SIZE);
  tiled_extent_1D ex(GRID_SIZE, TILE_SIZE);
  
  parallel_for_each(accelerator().get_default_view(),
                    ex,
                    tsa,
                    __KERNEL__ [=, &tsa](tiled_index_1D& tidx) {
    tile_static int lds1[TILE_SIZE];
    tile_static int lds2[TILE_SIZE];

    // obtain workitem absolute index and workgroup index
    index<1> global = tidx.global;
    index<1> local = tidx.local;

    // allocate dynamic group memory
    // each work item will allocate 1 plus its workgroup index
    __GROUP__ int* p = (__GROUP__ int*) tsa.alloc(ALLOC_SIZE);

    // fill in with value of 1 plus its workgroup index to the allocated buffer
    // for workgroup id 0: 1
    // for workgroup id 1: 2, 2
    // for workgroup id 2: 3, 3, 3
    // for workgroup id 3: 4, 4, 4, 4
    for (int i = 0; i < ALLOC_ELEMENT; ++i) {
      p[i] = local[0] + 1;
    }
    tidx.barrier.wait_with_tile_static_memory_fence();

    // write data from dynamic group memory to static group memory
    // lds1[local[0]] will contain sum of p[0 .. ALLOC_SIZE]
    // for workgroup id 0: 1
    // for workgroup id 1: 2 + 2 = 4
    // for workgroup id 2: 3 + 3 + 3 = 9
    // for workgroup id 3: 4 + 4 + 4 + 4 = 16
    lds1[local[0]] = 0;
    for (int i = 0; i < ALLOC_ELEMENT; ++i) {
      lds1[local[0]] += p[i];
    }
    tidx.barrier.wait_with_tile_static_memory_fence();

    // lds2[local[0]] will contain sum of lds1[0 .. local[0]]
    // for workgroup id 0: 1
    // for workgroup id 1: 1 + 4 = 5
    // for workgroup id 2: 1 + 4 + 9 = 14
    // for workgroup id 3: 1 + 4 + 9 + 16 = 30
    lds2[local[0]] = 0;
    int sum = 0;
    for (int i = 0; i <= local[0]; ++i) {
      sum += lds1[i];
    }
    lds2[local[0]] = sum;
    tidx.barrier.wait_with_tile_static_memory_fence();

    // write lds2 to global memory, plus static group segment size
    av(global) = tsa.getStaticGroupSegmentSize() + lds2[local[0]];
  });

  // overhead introduced in ts_allocator
  size_t overhead = tsa.getStaticGroupSegmentSize();

  bool ret = true;
  // for each item within each group
  // the value will be the sum of following:
  // - static group segment (lds1, lds2, ts_allocator)
  // - 1^2 + 2^2 + ... + (workgroup_id+1)^2 : value calculated
  for (int i = 0; i < GRID_SIZE; ++i) {
#if 0
    std::cout << av[i];
#endif

    int groupId = i % TILE_SIZE;
    int sum = 0;
    for (int j = 1; j <= groupId + 1; ++j) {
      sum += j * j;
    }
    if (av[i] != overhead + sum) {
      ret = false;
#if 0
      std::cout << "F ";
#endif
    } else {
#if 0
      std::cout << "  ";
#endif
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

  ret &= test<1, 1>();
  ret &= test<8, 2>();
  ret &= test<16, 4>();
  ret &= test<16, 16>();
  ret &= test<256, 32>();
  ret &= test<4096, 16>();

  return !(ret == true);
}
