
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <cstdlib>
#include <iostream>
#include <random>

#define TEST_DEBUG (0)

#define DYNAMIC_GROUP_SEGMENT_SIZE (4096)

// An example which shows how to use the following new builtin functions
//
// hc::get_static_group_segment_size()
// hc::get_group_segment_size()

bool test() {
  bool ret = true;

  // define grid size and group size
  const int vecSize = 2048;
  const int groupSize = 256;

  int table_a[vecSize] { 0 };
  int table_b[vecSize] { 0 };

  hc::array_view<int, 1> av_a(vecSize, table_a);
  hc::array_view<int, 1> av_b(vecSize, table_b);

  // launch kernel
  hc::tiled_extent<1> e(vecSize, groupSize);
  e.set_dynamic_group_segment_size(DYNAMIC_GROUP_SEGMENT_SIZE);

  hc::completion_future fut = hc::parallel_for_each(
    e, 
    [=](hc::index<1> idx) __HC__ {
      // create a tile_static array
      tile_static int group[groupSize];
      group[idx[0]] = 0;

      // av_a stores the size of group segment
      av_a(idx) = hc::get_group_segment_size();

      // av_b stores the size of static group segment
      av_b(idx) = hc::get_static_group_segment_size() + group[idx[0]]; // use group__HC__ so it won't be optimized away
  });

  // create a barrier packet
  hc::accelerator_view av = hc::accelerator().get_default_view();
  hc::completion_future fut2 = av.create_marker();

  // wait on the barrier packet
  // the barrier packet would ensure all previous packets were processed
  fut2.wait();

  av_a.synchronize();
  av_b.synchronize();

  // verify
  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    //std::cout << table_a[i] << " " << table_b[i] << "\n";
    error += std::abs(table_a[i] - ((int)(sizeof(int) * groupSize) + DYNAMIC_GROUP_SEGMENT_SIZE));
    error += std::abs(table_b[i] - (int)(sizeof(int) * groupSize));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  ret &= (error == 0);

  return ret;
}

int main() {
  bool ret = true;

  // The test case is only workable on LC backend as of now
  // because on HSAIL backend there is no way to check the size of
  // group segment.

  // Skip the test in case we are not using LC backend
#if __hcc_backend__ == HCC_BACKEND_AMDGPU
  ret &= test();
#endif

  return !(ret == true);
}

