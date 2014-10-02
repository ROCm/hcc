// RUN: %cxxamp %s -o %t.out && %t.out
#include <iostream> 
#include <amp.h>
#include <vector>
using namespace concurrency;
int test_1d() {
  std::vector<int> vv(100);
  for (int i = 0; i<100; i++)
    vv[i] = i;

  extent<1> e(100);
  {
    array_view<int, 1> av(e, vv.data()); 
    parallel_for_each(av.get_extent().tile<5>(),
      [=](tiled_index<5> idx) restrict(amp) { 
	av(idx) = 
          idx.tile[0] +
          idx.tile_origin[0] * 100;
      });
    assert(av.get_extent() == e);
    for(unsigned int i = 0; i < av.get_extent()[0]; i++)
      assert(i/5 + 100*(i-i%5) == av[i]);
  }
  return 0;
}
int test_2d() 
{
  std::vector<int> vv(200);
  for (int i = 0; i<200; i++)
    vv[i] = i;

  extent<2> e(10, 20);
  {
    array_view<int, 2> av(e, vv.data()); 
    parallel_for_each(av.get_extent().tile<5,5>(),
      [=](tiled_index<5,5> idx) restrict(amp) { 
	av(idx) = 
          idx.tile[0] +
          idx.tile[1] * 10 +
          idx.tile_origin[0] * 100 +
          idx.tile_origin[1] * 1000 +
          idx.tile_extent[0] * 10000 +
          idx.tile_extent[1] * 100000;
      });
    assert(av.get_extent() == e);
    for(unsigned int i = 0; i < av.get_extent()[0]; i++)
      for(unsigned int j = 0; j < av.get_extent()[1]; j++)
	assert(    i/5 +
               10*(j/5)+
               100*(i-i%5)+
               1000*(j-j%5)+
               10000*5 +
               100000*5 == av(i, j));
  }
  return 0;
}

int test_tiled_extent_1d(void) {
  extent<1> e(123);
  tiled_extent<10> myTileExtent(e);
  auto padded = myTileExtent.pad();
  assert(padded[0] == 130);

  auto trunc = myTileExtent.truncate();
  assert(trunc[0] == 120);
  return 0;
}

int test_tiled_extent_2d(void) {
  extent<2> e(123, 456);
  tiled_extent<10,30> myTileExtent(e);
  auto padded = myTileExtent.pad();
  assert(padded[0] == 130);
  assert(padded[1] == 480);

  auto trunc = myTileExtent.truncate();
  assert(trunc[0] == 120);
  assert(trunc[1] == 450);
  return 0;
}

int test_tiled_extent_3d(void) {
  extent<3> e(123, 456, 789);
  tiled_extent<10, 30, 40> myTileExtent(e);
  auto padded = myTileExtent.pad();
  assert(padded[0] == 130);
  assert(padded[1] == 480);
  assert(padded[2] == 800);

  auto trunc = myTileExtent.truncate();
  assert(trunc[0] == 120);
  assert(trunc[1] == 450);
  assert(trunc[2] == 760);
  return 0;
}

int main() {
  test_1d();
  test_2d();
  test_tiled_extent_1d();
  test_tiled_extent_2d();
  test_tiled_extent_3d();
  return 0;
}
