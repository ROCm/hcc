
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

/**
 * @file test10.cpp
 * Test used hc::tiled_index.tile_dim member to access the extent of tile size
 * within kernels.
 */

template<size_t GRID_SIZE, size_t TILE_SIZE>
bool test1D() {
  static_assert(GRID_SIZE % TILE_SIZE == 0, "The test is known not working if tile size does not divide grid size.\n");

  bool ret = true;

  using namespace hc;

  extent<1> ex(GRID_SIZE);
  tiled_extent<1> tiled_ex = ex.tile(TILE_SIZE);

  array_view<int, 1> output1(GRID_SIZE);

  parallel_for_each(tiled_ex, [=](tiled_index<1>& idx) __attribute((hc)) {
    output1(idx.global[0]) = idx.tile_dim[0];
  });

  for (int i = 0; i < GRID_SIZE; ++i) {
    if (output1[i] != TILE_SIZE) {
      ret = false;
      break;
    }
  }

  return ret;
}

template<size_t GRID_SIZE_Y, size_t GRID_SIZE_X, size_t TILE_SIZE_Y, size_t TILE_SIZE_X>
bool test2D() {
  static_assert((GRID_SIZE_Y % TILE_SIZE_Y == 0) &&
                (GRID_SIZE_X % TILE_SIZE_X == 0), "The test is known not working if tile size does not divide grid size.\n");

  bool ret = true;

  using namespace hc;

  extent<2> ex(GRID_SIZE_Y, GRID_SIZE_X);
  tiled_extent<2> tiled_ex = ex.tile(TILE_SIZE_Y, TILE_SIZE_X);

  array_view<int, 1> output1(GRID_SIZE_Y * GRID_SIZE_X);
  array_view<int, 1> output2(GRID_SIZE_Y * GRID_SIZE_X);

  parallel_for_each(tiled_ex, [=](tiled_index<2>& idx) __attribute((hc)) {
    output1(idx.global[1] * GRID_SIZE_X + idx.global[0]) = idx.tile_dim[0];
    output2(idx.global[1] * GRID_SIZE_X + idx.global[0]) = idx.tile_dim[1];
  });

  for (int i = 0; i < GRID_SIZE_Y * GRID_SIZE_X; ++i) {
    if (output1[i] != TILE_SIZE_Y || output2[i] != TILE_SIZE_X) {
      ret = false;
      break;
    }
  }

  return ret;
}

template<size_t GRID_SIZE_Z, size_t GRID_SIZE_Y, size_t GRID_SIZE_X,
         size_t TILE_SIZE_Z, size_t TILE_SIZE_Y, size_t TILE_SIZE_X>
bool test3D() {
  static_assert((GRID_SIZE_Z % TILE_SIZE_Z == 0) &&
                (GRID_SIZE_Y % TILE_SIZE_Y == 0) &&
                (GRID_SIZE_X % TILE_SIZE_X == 0), "The test is known not working if tile size does not divide grid size.\n");

  bool ret = true;

  using namespace hc;

  extent<3> ex(GRID_SIZE_Z, GRID_SIZE_Y, GRID_SIZE_X);
  tiled_extent<3> tiled_ex = ex.tile(TILE_SIZE_Z, TILE_SIZE_Y, TILE_SIZE_X);

  array_view<int, 1> output1(GRID_SIZE_Z * GRID_SIZE_Y * GRID_SIZE_X);
  array_view<int, 1> output2(GRID_SIZE_Z * GRID_SIZE_Y * GRID_SIZE_X);
  array_view<int, 1> output3(GRID_SIZE_Z * GRID_SIZE_Y * GRID_SIZE_X);

  parallel_for_each(tiled_ex, [=](tiled_index<3>& idx) __attribute((hc)) {
    int global_index = idx.global[2] * GRID_SIZE_Y * GRID_SIZE_X + idx.global[1] * GRID_SIZE_X + idx.global[0];
    output1(global_index) = idx.tile_dim[0];
    output2(global_index) = idx.tile_dim[1];
    output3(global_index) = idx.tile_dim[2];
  });

  for (int i = 0; i < GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_X; ++i) {
    if (output1[i] != TILE_SIZE_Z || output2[i] != TILE_SIZE_Y || output3[i] != TILE_SIZE_X) {
      ret = false;
      break;
    }
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<16, 4>();
  ret &= test1D<35, 5>();

  ret &= test2D<8, 8, 2, 2>();
  ret &= test2D<35, 35, 5, 5>();

  ret &= test3D<8, 8, 8, 4, 4, 4>();
  ret &= test3D<15, 15, 15, 5, 5, 5>();

  return !(ret == true);
}

