
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

/// test tiled_extent<1> built from extent<1>.tile_with_dynamic()
bool test1D() {
  bool ret = true;

  using namespace hc;

  extent<1> ex1;
  tiled_extent<1> tile1 = ex1.tile_with_dynamic(0, 0);

  ret &= (tile1.rank == 1);
  ret &= (tile1.tile_dim[0] == 0);
  ret &= (tile1[0] == 0);
  ret &= (tile1.get_dynamic_group_segment_size() == 0);

  extent<1> ex2(1);
  tiled_extent<1> tile2 = ex2.tile_with_dynamic(1, 1);

  ret &= (tile2.rank == 1);
  ret &= (tile2.tile_dim[0] == 1);
  ret &= (tile2[0] == 1);
  ret &= (tile2.get_dynamic_group_segment_size() == 1);

  extent<1> ex3(64);
  tiled_extent<1> tile3 = ex3.tile_with_dynamic(16, 16);

  ret &= (tile3.rank == 1);
  ret &= (tile3.tile_dim[0] == 16);
  ret &= (tile3[0] == 64);
  ret &= (tile3.get_dynamic_group_segment_size() == 16);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(1, 1024);

  // test create tiled_extent<1> where tile size is determined at runtime
  for (int i = 0; i < 10; ++i) {
    int extentSize = dis(gen);
    int tileSize = dis(gen);
    int dynamicGroupSegmentSize = dis(gen);

    extent<1> ex4(extentSize);
    tiled_extent<1> tile4 = ex4.tile_with_dynamic(tileSize, dynamicGroupSegmentSize);

    ret &= (tile4.rank == 1);
    ret &= (tile4.tile_dim[0] == tileSize);
    ret &= (tile4[0] == extentSize);
    ret &= (tile4.get_dynamic_group_segment_size() == dynamicGroupSegmentSize);
  }

  return ret;
}

/// test tiled_extent<2> built from extent<2>.tile()
bool test2D() {
  bool ret = true;

  using namespace hc;

  extent<2> ex1;
  tiled_extent<2> tile1 = ex1.tile_with_dynamic(0, 0, 0);

  ret &= (tile1.rank == 2);
  ret &= (tile1.tile_dim[0] == 0);
  ret &= (tile1.tile_dim[1] == 0);
  ret &= (tile1[0] == 0);
  ret &= (tile1[1] == 0);
  ret &= (tile1.get_dynamic_group_segment_size() == 0);

  extent<2> ex2(1, 1);
  tiled_extent<2> tile2 = ex2.tile_with_dynamic(1, 1, 1);

  ret &= (tile2.rank == 2);
  ret &= (tile2.tile_dim[0] == 1);
  ret &= (tile2.tile_dim[1] == 1);
  ret &= (tile2[0] == 1);
  ret &= (tile2[1] == 1);
  ret &= (tile2.get_dynamic_group_segment_size() == 1);

  extent<2> ex3(64, 16);
  tiled_extent<2> tile3 = ex3.tile_with_dynamic(8, 4, 4);

  ret &= (tile3.rank == 2);
  ret &= (tile3.tile_dim[0] == 8);
  ret &= (tile3.tile_dim[1] == 4);
  ret &= (tile3[0] == 64);
  ret &= (tile3[1] == 16);
  ret &= (tile3.get_dynamic_group_segment_size() == 4);

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
    int dynamicGroupSegmentSize = dis(gen);

    extent<2> ex4(extentSize0, extentSize1);
    tiled_extent<2> tile4 = ex4.tile_with_dynamic(tileSize0, tileSize1, dynamicGroupSegmentSize);

    ret &= (tile4.rank == 2);
    ret &= (tile4.tile_dim[0] == tileSize0);
    ret &= (tile4.tile_dim[1] == tileSize1);
    ret &= (tile4[0] == extentSize0);
    ret &= (tile4[1] == extentSize1);
    ret &= (tile4.get_dynamic_group_segment_size() == dynamicGroupSegmentSize);
  }

  return ret;
}

/// test tiled_extent<3>
bool test3D() {
  bool ret = true;

  using namespace hc;

  extent<3> ex1;
  tiled_extent<3> tile1 = ex1.tile_with_dynamic(0, 0, 0, 0);

  ret &= (tile1.rank == 3);
  ret &= (tile1.tile_dim[0] == 0);
  ret &= (tile1.tile_dim[1] == 0);
  ret &= (tile1.tile_dim[2] == 0);
  ret &= (tile1[0] == 0);
  ret &= (tile1[1] == 0);
  ret &= (tile1[2] == 0);
  ret &= (tile1.get_dynamic_group_segment_size() == 0);

  extent<3> ex2(1, 1, 1);
  tiled_extent<3> tile2 = ex2.tile_with_dynamic(1, 1, 1, 1);

  ret &= (tile2.rank == 3);
  ret &= (tile2.tile_dim[0] == 1);
  ret &= (tile2.tile_dim[1] == 1);
  ret &= (tile2.tile_dim[2] == 1);
  ret &= (tile2[0] == 1);
  ret &= (tile2[1] == 1);
  ret &= (tile2[2] == 1);
  ret &= (tile2.get_dynamic_group_segment_size() == 1);

  extent<3> ex3(64, 16, 8);
  tiled_extent<3> tile3 = ex3.tile_with_dynamic(8, 4, 2, 8);

  ret &= (tile3.rank == 3);
  ret &= (tile3.tile_dim[0] == 8);
  ret &= (tile3.tile_dim[1] == 4);
  ret &= (tile3.tile_dim[2] == 2);
  ret &= (tile3[0] == 64);
  ret &= (tile3[1] == 16);
  ret &= (tile3[2] == 8);
  ret &= (tile3.get_dynamic_group_segment_size() == 8);

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
    int dynamicGroupSegmentSize = dis(gen);

    extent<3> ex4(extentSize0, extentSize1, extentSize2);
    tiled_extent<3> tile4 = ex4.tile_with_dynamic(tileSize0, tileSize1, tileSize2, dynamicGroupSegmentSize);

    ret &= (tile4.rank == 3);
    ret &= (tile4.tile_dim[0] == tileSize0);
    ret &= (tile4.tile_dim[1] == tileSize1);
    ret &= (tile4.tile_dim[2] == tileSize2);
    ret &= (tile4[0] == extentSize0);
    ret &= (tile4[1] == extentSize1);
    ret &= (tile4[2] == extentSize2);
    ret &= (tile4.get_dynamic_group_segment_size() == dynamicGroupSegmentSize);
  }

  return ret;
}

/// a test which checks if extent::tile() is implemented correctly
int main() {
  bool ret = true;

  ret &= test1D();
  ret &= test2D();
  ret &= test3D();

  return !(ret == true);
}

