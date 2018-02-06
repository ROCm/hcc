
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

/// test tiled_extent<1>
bool test1D() {
  bool ret = true;

  using namespace hc;

  tiled_extent<1> tile1;

  // test default constructor
  ret &= (tile1.rank == 1);
  ret &= (tile1.tile_dim[0] == 0);
  ret &= (tile1[0] == 0);

  // test constructor (int, int)
  tiled_extent<1> tile2(1, 1);

  ret &= (tile2.rank == 1);
  ret &= (tile2.tile_dim[0] == 1);
  ret &= (tile2[0] == 1);

  tiled_extent<1> tile3(64, 16);

  ret &= (tile3.rank == 1);
  ret &= (tile3.tile_dim[0] == 16);
  ret &= (tile3[0] == 64);

  // test copy consutrctor
  tiled_extent<1> tile4(tile3);

  ret &= (tile4.rank == 1);
  ret &= (tile4.tile_dim[0] == 16);
  ret &= (tile4[0] == 64);

  // test constructor (extent<1>&, int)
  extent<1> e1(16);
  tiled_extent<1> tile5(e1, 4);

  ret &= (tile5.rank == 1);
  ret &= (tile5.tile_dim[0] == 4);
  ret &= (tile5[0] == 16);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, 1024);

  // test create tiled_extent<1> where tile size is determined at runtime
  for (int i = 0; i < 10; ++i) {
    int extentSize = dis(gen);
    int tileSize = dis(gen);

    tiled_extent<1> tile6(extentSize, tileSize);

    ret &= (tile6.rank == 1);
    ret &= (tile6.tile_dim[0] == tileSize);
    ret &= (tile6[0] == extentSize);
  }

  return ret;
}

/// test tiled_extent<2>
bool test2D() {
  bool ret = true;

  using namespace hc;

  // test default constructor
  tiled_extent<2> tile1;

  ret &= (tile1.rank == 2);
  ret &= (tile1.tile_dim[0] == 0);
  ret &= (tile1.tile_dim[1] == 0);
  ret &= (tile1[0] == 0);
  ret &= (tile1[1] == 0);

  // test constructor (int, int, int, int)
  tiled_extent<2> tile2(1, 1, 1, 1);

  ret &= (tile2.rank == 2);
  ret &= (tile2.tile_dim[0] == 1);
  ret &= (tile2.tile_dim[1] == 1);
  ret &= (tile2[0] == 1);
  ret &= (tile2[1] == 1);

  tiled_extent<2> tile3(64, 16, 8, 4);

  ret &= (tile3.rank == 2);
  ret &= (tile3.tile_dim[0] == 8);
  ret &= (tile3.tile_dim[1] == 4);
  ret &= (tile3[0] == 64);
  ret &= (tile3[1] == 16);

  // test copy consutrctor
  tiled_extent<2> tile4(tile3);

  ret &= (tile4.rank == 2);
  ret &= (tile4.tile_dim[0] == 8);
  ret &= (tile4.tile_dim[1] == 4);
  ret &= (tile4[0] == 64);
  ret &= (tile4[1] == 16);

  // test constructor (extent<2>&, int, int)
  extent<2> e1(16, 16);
  tiled_extent<2> tile5(e1, 4, 4);

  ret &= (tile5.rank == 2);
  ret &= (tile5.tile_dim[0] == 4);
  ret &= (tile5.tile_dim[1] == 4);
  ret &= (tile5[0] == 16);
  ret &= (tile5[1] == 16);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, 1024);

  // test create tiled_extent<2> where tile size is determined at runtime
  for (int i = 0; i < 10; ++i) {
    int extentSize0 = dis(gen);
    int extentSize1 = dis(gen);
    int tileSize0 = dis(gen);
    int tileSize1 = dis(gen);

    tiled_extent<2> tile6(extentSize0, extentSize1, tileSize0, tileSize1);

    ret &= (tile6.rank == 2);
    ret &= (tile6.tile_dim[0] == tileSize0);
    ret &= (tile6.tile_dim[1] == tileSize1);
    ret &= (tile6[0] == extentSize0);
    ret &= (tile6[1] == extentSize1);
  }

  return ret;
}

/// test tiled_extent<3>
bool test3D() {
  bool ret = true;

  using namespace hc;

  // test default constructor
  tiled_extent<3> tile1;

  ret &= (tile1.rank == 3);
  ret &= (tile1.tile_dim[0] == 0);
  ret &= (tile1.tile_dim[1] == 0);
  ret &= (tile1.tile_dim[2] == 0);
  ret &= (tile1[0] == 0);
  ret &= (tile1[1] == 0);
  ret &= (tile1[2] == 0);

  // test constructor (int, int, int, int, int, int)
  tiled_extent<3> tile2(1, 1, 1, 1, 1, 1);

  ret &= (tile2.rank == 3);
  ret &= (tile2.tile_dim[0] == 1);
  ret &= (tile2.tile_dim[1] == 1);
  ret &= (tile2.tile_dim[2] == 1);
  ret &= (tile2[0] == 1);
  ret &= (tile2[1] == 1);
  ret &= (tile2[2] == 1);

  tiled_extent<3> tile3(64, 16, 8, 8, 4, 2);

  ret &= (tile3.rank == 3);
  ret &= (tile3.tile_dim[0] == 8);
  ret &= (tile3.tile_dim[1] == 4);
  ret &= (tile3.tile_dim[2] == 2);
  ret &= (tile3[0] == 64);
  ret &= (tile3[1] == 16);
  ret &= (tile3[2] == 8);

  // test copy consutrctor
  tiled_extent<3> tile4(tile3);

  ret &= (tile3.rank == 3);
  ret &= (tile4.tile_dim[0] == 8);
  ret &= (tile4.tile_dim[1] == 4);
  ret &= (tile4.tile_dim[2] == 2);
  ret &= (tile4[0] == 64);
  ret &= (tile4[1] == 16);
  ret &= (tile4[2] == 8);

  // test constructor (extent<3>&, int, int, int)
  extent<3> e1(16, 16, 16);
  tiled_extent<3> tile5(e1, 4, 4, 4);

  ret &= (tile5.rank == 3);
  ret &= (tile5.tile_dim[0] == 4);
  ret &= (tile5.tile_dim[1] == 4);
  ret &= (tile5.tile_dim[2] == 4);
  ret &= (tile5[0] == 16);
  ret &= (tile5[1] == 16);
  ret &= (tile5[2] == 16);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, 1024);

  // test create tiled_extent<3> where tile size is determined at runtime
  for (int i = 0; i < 10; ++i) {
    int extentSize0 = dis(gen);
    int extentSize1 = dis(gen);
    int extentSize2 = dis(gen);
    int tileSize0 = dis(gen);
    int tileSize1 = dis(gen);
    int tileSize2 = dis(gen);

    tiled_extent<3> tile6(extentSize0, extentSize1, extentSize2, tileSize0, tileSize1, tileSize2);

    ret &= (tile6.rank == 3);
    ret &= (tile6.tile_dim[0] == tileSize0);
    ret &= (tile6.tile_dim[1] == tileSize1);
    ret &= (tile6.tile_dim[2] == tileSize2);
    ret &= (tile6[0] == extentSize0);
    ret &= (tile6[1] == extentSize1);
    ret &= (tile6[2] == extentSize2);
  }

  return ret;
}

/// a test which checks if tiled_extent<> is implemented correctly
int main() {
  bool ret = true;

  ret &= test1D();
  ret &= test2D();
  ret &= test3D();

  return !(ret == true);
}

