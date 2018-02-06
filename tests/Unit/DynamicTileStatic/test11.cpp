
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

/**
 * @file test11.cpp
 * Test used hc::tiled_index.tile_dim member to access the extent of tile size
 * within kernels.
 * This test deliberately checks when tile sizes do not divide grid sizes.
 */

template<size_t GRID_SIZE, size_t TILE_SIZE>
bool test1D() {
  static_assert(GRID_SIZE % TILE_SIZE, "The test wants to check cases where tile sizes do NOT divide grid sizes.\n");

  bool ret = true;

  using namespace hc;

  extent<1> ex(GRID_SIZE);
  tiled_extent<1> tiled_ex = ex.tile(TILE_SIZE);

  array_view<int, 1> output1(GRID_SIZE);

  parallel_for_each(tiled_ex, [=](tiled_index<1>& idx) __attribute((hc)) {
    output1(idx.global[0]) = idx.tile_dim[0];
  });

  int full_tiles = GRID_SIZE / TILE_SIZE;
  int remainder = GRID_SIZE % TILE_SIZE;

  for (int i = 0; i < TILE_SIZE * full_tiles; ++i) {
    if (output1[i] != TILE_SIZE) {
      ret = false;
      break;
    }
  }
  for (int i = TILE_SIZE * full_tiles; i < GRID_SIZE; ++i) {
    if (output1[i] != remainder) {
      ret = false;
      break;
    }
  }

  return ret;
}

template<size_t GRID_SIZE_Y, size_t GRID_SIZE_X, size_t TILE_SIZE_Y, size_t TILE_SIZE_X>
bool test2D() {
  static_assert((GRID_SIZE_Y % TILE_SIZE_Y) || 
                (GRID_SIZE_X % TILE_SIZE_X), "The test wants to check cases where tile sizes do NOT divide grid sizes.\n");

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

  int full_tiles_y = GRID_SIZE_Y / TILE_SIZE_Y;
  int full_tiles_x = GRID_SIZE_X / TILE_SIZE_X;
  int remainder_y = GRID_SIZE_Y % TILE_SIZE_Y;
  int remainder_x = GRID_SIZE_X % TILE_SIZE_X;

  for (int i = 0; i < GRID_SIZE_Y; ++i) {
    for (int j = 0; j < GRID_SIZE_X; ++j) {
      int gid = i * GRID_SIZE_X + j;
      if (i < full_tiles_y * TILE_SIZE_Y) {
        if (output2[gid] != TILE_SIZE_Y) {
          ret = false;
          break;
        }
      } else {
        if (output2[gid] != remainder_y) {
          ret = false;
          break;
        }
      }

      if (j < full_tiles_x * TILE_SIZE_X) {
        if (output1[gid] != TILE_SIZE_X) {
          ret = false;
          break;
        }
      } else {
        if (output1[gid] != remainder_x) {
          ret = false;
          break;
        }
      }
    }
  }

  return ret;
}

template<size_t GRID_SIZE_Z, size_t GRID_SIZE_Y, size_t GRID_SIZE_X,
         size_t TILE_SIZE_Z, size_t TILE_SIZE_Y, size_t TILE_SIZE_X>
bool test3D() {
  static_assert((GRID_SIZE_Z % TILE_SIZE_Z) || 
                (GRID_SIZE_Y % TILE_SIZE_Y) || 
                (GRID_SIZE_X % TILE_SIZE_X), "The test wants to check cases where tile sizes do NOT divide grid sizes.\n");

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

  int full_tiles_z = GRID_SIZE_Z / TILE_SIZE_Z;
  int full_tiles_y = GRID_SIZE_Y / TILE_SIZE_Y;
  int full_tiles_x = GRID_SIZE_X / TILE_SIZE_X;
  int remainder_z = GRID_SIZE_Z % TILE_SIZE_Z;
  int remainder_y = GRID_SIZE_Y % TILE_SIZE_Y;
  int remainder_x = GRID_SIZE_X % TILE_SIZE_X;

  for (int i = 0; i < GRID_SIZE_Z; ++i) {
    for (int j = 0; j < GRID_SIZE_Y; ++j) {
      for (int k = 0; k < GRID_SIZE_X; ++k) {
        int gid = i * GRID_SIZE_Y * GRID_SIZE_X + j * GRID_SIZE_X + k;
        if (i < full_tiles_z * TILE_SIZE_Z) {
          if (output3[gid] != TILE_SIZE_Z) {
            ret = false;
            break;
          }
        } else {
          if (output3[gid] != remainder_z) {
            ret = false;
            break;
          }
        }

        if (j < full_tiles_y * TILE_SIZE_Y) {
          if (output2[gid] != TILE_SIZE_Y) {
            ret = false;
            break;
          }
        } else {
          if (output2[gid] != remainder_y) {
            ret = false;
            break;
          }
        }

        if (k < full_tiles_x * TILE_SIZE_X) {
          if (output1[gid] != TILE_SIZE_X) {
            ret = false;
            break;
          }
        } else {
          if (output1[gid] != remainder_x) {
            ret = false;
            break;
          }
        }
      }
    }
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<5, 2>();
  ret &= test1D<7, 3>();
  ret &= test1D<19, 4>();
  ret &= test1D<1023, 17>();

  ret &= test2D<5, 5, 2, 2>();
  ret &= test2D<7, 7, 5, 5>();

  ret &= test3D<5, 5, 5, 2, 2, 2>();
  ret &= test3D<7, 7, 7, 5, 5, 5>();

  return !(ret == true);
}

