
// RUN: %hc %s -o %t.out && %t.out

#include<hc.hpp>

#include<iostream>

/// test where grid size can't be evenly divisible by tile size
/// on HSA it shall work fine

template<size_t GRID_SIZE, size_t TILE_SIZE>
bool test1D() {
  bool ret = true;

  using namespace hc;

  extent<1> ex(GRID_SIZE);
  tiled_extent<1> tiled_ex = ex.tile(TILE_SIZE);

  array_view<int, 1> table(GRID_SIZE);

  completion_future fut = parallel_for_each(tiled_ex, [=](tiled_index<1>& idx) __HC__ {
    table(idx) = idx.global[0];
  });

  fut.wait();

  for (int i = 0; i < GRID_SIZE; ++i) {
    if (table[i] != i) {
      ret = false;
      break;
    }
  }

  return ret;
}

template<size_t GRID_SIZE_Y, size_t GRID_SIZE_X, size_t TILE_SIZE_Y, size_t TILE_SIZE_X>
bool test2D() {
  bool ret = true;

  using namespace hc;

  extent<2> ex(GRID_SIZE_Y, GRID_SIZE_X);
  tiled_extent<2> tiled_ex = ex.tile(TILE_SIZE_Y, TILE_SIZE_X);

  array_view<int, 1> table(GRID_SIZE_Y * GRID_SIZE_X);

  completion_future fut = parallel_for_each(tiled_ex, [=](tiled_index<2>& idx) __HC__ {
    size_t index = idx.global[0] * GRID_SIZE_X + idx.global[1];
    table(index) = index;
  });

  fut.wait();

  for (int i = 0; i < GRID_SIZE_Y * GRID_SIZE_X; ++i) {
    if (table[i] != i) {
      ret = false;
      break;
    }
  }

  return ret;
}

template<size_t GRID_SIZE_Z, size_t GRID_SIZE_Y, size_t GRID_SIZE_X,
         size_t TILE_SIZE_Z, size_t TILE_SIZE_Y, size_t TILE_SIZE_X>
bool test3D() {
  bool ret = true;

  using namespace hc;

  extent<3> ex(GRID_SIZE_Z, GRID_SIZE_Y, GRID_SIZE_X);
  tiled_extent<3> tiled_ex = ex.tile(TILE_SIZE_Z, TILE_SIZE_Y, TILE_SIZE_X);

  array_view<int, 1> table(GRID_SIZE_Z * GRID_SIZE_Y * GRID_SIZE_X);

  completion_future fut = parallel_for_each(tiled_ex, [=](tiled_index<3>& idx) __HC__ {
    size_t index = idx.global[0] * GRID_SIZE_X * GRID_SIZE_Y + idx.global[1] * GRID_SIZE_X + idx.global[2];
    table(index) = index;
  });

  fut.wait();

  for (int i = 0; i < GRID_SIZE_Z * GRID_SIZE_Y * GRID_SIZE_X; ++i) {
    if (table[i] != i) {
      ret = false;
      break;
    }
  }

  return ret;
}
int main() {
  bool ret = true;

  ret &= test1D<5, 2>();
  ret &= test1D<1024, 7>();
  ret &= test1D<4096, 31>();

  ret &= test2D<5, 7, 2, 3>();
  ret &= test2D<32, 48, 5, 9>();
  ret &= test2D<64, 73, 7, 11>();

  ret &= test3D<5, 7, 9, 2, 2, 2>();
  ret &= test3D<16, 19, 11, 5, 5, 5>();
  ret &= test3D<17, 11, 5, 6, 3, 2>();

  return !(ret == true);
}

