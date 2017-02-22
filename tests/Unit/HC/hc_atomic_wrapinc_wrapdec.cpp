
// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>

#include <iostream>
#include <vector>

#define TEST_DEBUG (0)

#define GRID_SIZE (1024)
#define TILE_SIZE (64)
#define CLAMP_VALUE_GLOBAL (GRID_SIZE / 2)
#define CLAMP_VALUE_TILE (TILE_SIZE / 2)

using namespace hc;

bool test_atomic_wrapinc_global() {
  bool ret = true;

  array<unsigned int, 1> data1(GRID_SIZE);
  array<unsigned int, 1> data2(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    // initialize value
    data1(idx) = idx[0]; // data1 initialized as workitem index
    data2(idx) = 0;      // data2 initialized as 0

    // do atomic wrap inc
    data2(idx) = __atomic_wrapinc(&data1(idx), CLAMP_VALUE_GLOBAL);
  }).wait();

  std::vector<unsigned int> result1 = data1;
  std::vector<unsigned int> result2 = data2;

  for (int i = 0; i < GRID_SIZE; ++i) {
#if TEST_DEBUG
    std::cout << result1[i] << " " << result2[i] << "\n";
#endif

    // data1 should honor rules set forth by wrapinc
    ret &= (i < CLAMP_VALUE_GLOBAL) ? (result1[i] == i + 1) // for values smaller then CLAMP_VALUE_GLOBAL, they would be incremented
                                    : (result1[i] == 0);    // otherwise clamped to 0
    // data2 should hold old values from data1
    ret &= (result2[i] == i);
  }

  return ret;
}

bool test_atomic_wrapinc_local() {
  bool ret = true;

  array<unsigned int, 1> data1(GRID_SIZE);
  array<unsigned int, 1> data2(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  parallel_for_each(ex.tile(TILE_SIZE), [&](tiled_index<1>& tidx) [[hc]] {
    tile_static unsigned int lds[TILE_SIZE];

    int group_index = tidx.local[0];
    int global_index = tidx.global[0];

    lds[group_index] = group_index;

    tidx.barrier.wait();

    data2(global_index) = __atomic_wrapinc(&lds[group_index], CLAMP_VALUE_TILE);
    data1(global_index) = lds[group_index];
  }).wait();

  std::vector<unsigned int> result1 = data1;
  std::vector<unsigned int> result2 = data2;

  for (int i = 0; i < GRID_SIZE / TILE_SIZE; ++i) {
    for (int j = 0; j < TILE_SIZE; ++j) {
#if TEST_DEBUG
      std::cout << result1[i * TILE_SIZE + j] << " " << result2[i * TILE_SIZE + j] << "\n";
#endif

      // data1 should honor rules set forth by wrapinc
      ret &= (j < CLAMP_VALUE_TILE) ? (result1[i * TILE_SIZE + j] == j + 1) // for values smaller then CLAMP_VALUE_TILE, they would be incremented
                                    : (result1[i * TILE_SIZE + j] == 0);    // otherwise clamped to 0
      // data2 should hold old values from lds, which are group index value
      ret &= (result2[i * TILE_SIZE + j] == j);
    }
  }

  return ret;
}

bool test_atomic_wrapdec_global() {
  bool ret = true;

  array<unsigned int, 1> data1(GRID_SIZE);
  array<unsigned int, 1> data2(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    // initialize value
    data1(idx) = idx[0]; // data1 initialized as workitem index
    data2(idx) = 0;      // data2 initialized as 0

    // do atomic wrap dec
    data2(idx) = __atomic_wrapdec(&data1(idx), CLAMP_VALUE_GLOBAL);
  }).wait();

  std::vector<unsigned int> result1 = data1;
  std::vector<unsigned int> result2 = data2;

  for (int i = 0; i < GRID_SIZE; ++i) {
#if TEST_DEBUG
    std::cout << result1[i] << " " << result2[i] << "\n";
#endif

    // data1 should honor rules set forth by wrapdec
    ret &= (i == 0) ? (result1[i] == CLAMP_VALUE_GLOBAL) // if old value is 0, it should carry the clamp value
                    : (i > CLAMP_VALUE_GLOBAL) ? (result1[i] == CLAMP_VALUE_GLOBAL) // for old values larger than the clamp value
                                                                                    // they would be clamped
                                               : (result1[i] == (i - 1));           // otherwise they would be decremented by 1
    // data2 should hold old values from data1
    ret &= (result2[i] == i);
  }

  return ret;
}

bool test_atomic_wrapdec_local() {
  bool ret = true;

  array<unsigned int, 1> data1(GRID_SIZE);
  array<unsigned int, 1> data2(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  parallel_for_each(ex.tile(TILE_SIZE), [&](tiled_index<1>& tidx) [[hc]] {
    tile_static unsigned int lds[TILE_SIZE];

    int group_index = tidx.local[0];
    int global_index = tidx.global[0];

    lds[group_index] = group_index;

    tidx.barrier.wait();

    data2(global_index) = __atomic_wrapdec(&lds[group_index], CLAMP_VALUE_TILE);
    data1(global_index) = lds[group_index];
  }).wait();

  std::vector<unsigned int> result1 = data1;
  std::vector<unsigned int> result2 = data2;

  for (int i = 0; i < GRID_SIZE / TILE_SIZE; ++i) {
    for (int j = 0; j < TILE_SIZE; ++j) {
#if TEST_DEBUG
      std::cout << result1[i * TILE_SIZE + j] << " " << result2[i * TILE_SIZE + j] << "\n";
#endif

      // data1 should honor rules set forth by wrapdec
      ret &= (i == 0) ? (result1[i] == CLAMP_VALUE_TILE) // if old value is 0, it should carry the clamp value
                      : (i > CLAMP_VALUE_TILE) ? (result1[i] == CLAMP_VALUE_TILE) // for old values larger than the clamp value
                                                                                  // they would be clamped
                                               : (result1[i] == (i - 1));         // otherwise they would be decremented by 1
      // data2 should hold old values from lds, which are group index value
      ret &= (result2[i] == i);
    }
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test_atomic_wrapinc_global();
  ret &= test_atomic_wrapdec_global();

  ret &= test_atomic_wrapinc_local();
  ret &= test_atomic_wrapdec_local();

  return !(ret == true);
}

