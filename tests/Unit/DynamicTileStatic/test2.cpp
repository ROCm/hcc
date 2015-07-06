// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

/// test tiled_extent_1D
bool test1D() {
  bool ret = true;

  using namespace concurrency;

  // test default constructor
  tiled_extent_1D tile1;

  ret &= (tile1.tile_dim0 == 0);
  ret &= (tile1[0] == 0);

  // test constructor (int, int)
  tiled_extent_1D tile2(1, 1);

  ret &= (tile2.tile_dim0 == 1);
  ret &= (tile2[0] == 1);

  tiled_extent_1D tile3(64, 16);

  ret &= (tile3.tile_dim0 == 16);
  ret &= (tile3[0] == 64);

  // test copy consutrctor
  tiled_extent_1D tile4(tile3);

  ret &= (tile4.tile_dim0 == 16);
  ret &= (tile4[0] == 64);

  // test constructor (extent<1>&, int)
  extent<1> e1(16);
  tiled_extent_1D tile5(e1, 4);

  ret &= (tile5.tile_dim0 == 4);
  ret &= (tile5[0] == 16);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, 1024);

  // test create tiled_extent_1D where tile size is determined at runtime
  for (int i = 0; i < 10; ++i) {
    int extentSize = dis(gen);
    int tileSize = dis(gen);

    tiled_extent_1D tile6(extentSize, tileSize);

    ret &= (tile6.tile_dim0 == tileSize);
    ret &= (tile6[0] == extentSize);
  }

  return ret;
}

/// a test which checks if tiled_extent_1D is implemented correctly
int main() {
  bool ret = true;

  ret &= test1D();

  return !(ret == true);
}

