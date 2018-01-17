
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <algorithm>

#define TABLE_X (32)
#define TABLE_Y (32)
#define TABLE_Z (4)

// test various parallel_for_each with extent size 0
// they should all silently return without raising any exceptions

// 1D test case
bool test1D() {
  bool ret = true;

  using namespace hc;

  int table[TABLE_X] { 0 };

  // 1D non-tiled
  extent<1> ex1d(0);
  completion_future fut1 = parallel_for_each(ex1d, [&](index<1>& idx) __HC__ {
    table[idx[0]] = 1;
  });

  // wait on the kernel
  fut1.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X);

  // 1D tiled
  tiled_extent<1> tiled_ex1d = ex1d.tile(0);
  completion_future fut2 = parallel_for_each(tiled_ex1d, [&](tiled_index<1>& idx) __HC__ {
    table[idx.global[0]] = 1;
  });

  // wait on the kernel
  fut2.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X);

  // 1D non-tiled
  completion_future fut3 = parallel_for_each(ex1d, [&](index<1>& idx) __HC__ {
    table[idx[0]] = 1;
  });

  // wait on the kernel
  fut3.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X);

  // 1D tiled
  completion_future fut4 = parallel_for_each(tiled_ex1d, [&](tiled_index<1>& idx) __HC__ {
    table[idx.global[0]] = 1;
  });

  // wait on the kernel
  fut4.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X);

  return ret;
}

// 2D test case
bool test2D() {
  bool ret = true;

  using namespace hc;

  int table[TABLE_Y * TABLE_X] { 0 };

  // 2D non-tiled
  extent<2> ex2d(0, 0);
  completion_future fut1 = parallel_for_each(ex2d, [&](index<2>& idx) __HC__ {
    table[idx[0] * TABLE_Y + idx[1]] = 1;
  });

  // wait on the kernel
  fut1.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y);

  // 2D tiled
  tiled_extent<2> tiled_ex2d = ex2d.tile(0, 0);
  completion_future fut2 = parallel_for_each(tiled_ex2d, [&](tiled_index<2>& idx) __HC__ {
    table[idx.global[0] * TABLE_Y + idx.global[1]] = 1;
  });

  // wait on the kernel
  fut2.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y);

  // 2D non-tiled
  completion_future fut3 = parallel_for_each(ex2d, [&](index<2>& idx) __HC__ {
    table[idx[0] * TABLE_Y + idx[1]] = 1;
  });

  // wait on the kernel
  fut3.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y);

  // 2D tiled
  completion_future fut4 = parallel_for_each(tiled_ex2d, [&](tiled_index<2>& idx) __HC__ {
    table[idx.global[0] * TABLE_Y + idx.global[1]] = 1;
  });

  // wait on the kernel
  fut4.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y);

  return ret;
}

// 3D test case
bool test3D() {
  bool ret = true;

  using namespace hc;

  int table[TABLE_Z * TABLE_Y * TABLE_X] { 0 };

  // 3D non-tiled
  extent<3> ex3d(0, 0, 0);
  completion_future fut1 = parallel_for_each(ex3d, [&](index<3>& idx) __HC__ {
    table[idx[0] * TABLE_X * TABLE_Y + idx[1] * TABLE_Y + idx[2]] = 1;
  });

  // wait on the kernel
  fut1.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y * TABLE_Z);

  // 3D tiled
  tiled_extent<3> tiled_ex3d = ex3d.tile(0, 0, 0);
  completion_future fut2 = parallel_for_each(tiled_ex3d, [&](tiled_index<3>& idx) __HC__ {
    table[idx.global[0] * TABLE_X * TABLE_Y + idx.global[1] * TABLE_Y + idx.global[2]] = 1;
  });

  // wait on the kernel
  fut2.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y * TABLE_Z);

  // 2D non-tiled
  completion_future fut3 = parallel_for_each(ex3d, [&](index<3>& idx) __HC__ {
    table[idx[0] * TABLE_X * TABLE_Y + idx[1] * TABLE_Y + idx[2]] = 1;
  });

  // wait on the kernel
  fut3.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y * TABLE_Z);

  // 2D tiled
  completion_future fut4 = parallel_for_each(tiled_ex3d, [&](tiled_index<3>& idx) __HC__ {
    table[idx.global[0] * TABLE_X * TABLE_Y + idx.global[1] * TABLE_Y + idx.global[2]] = 1;
  });

  // wait on the kernel
  fut4.wait();

  // verify data
  // nothing shall be changed the kernel
  ret &= (std::count(std::begin(table), std::end(table), 0) == TABLE_X * TABLE_Y * TABLE_Z);

  return ret;
}
int main() {
  bool ret = true;

  ret &= test1D();
  ret &= test2D();
  ret &= test3D();

  return !(ret == true);
}

