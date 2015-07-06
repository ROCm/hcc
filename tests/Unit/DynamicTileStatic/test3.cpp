// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>

/// test HC parallel_for_each interface
template<size_t grid_size, size_t tile_size>
bool test1D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each

  using namespace Concurrency;

  std::vector<int> table1(grid_size);
  std::vector<int> table2(grid_size);
  std::vector<int> table3(grid_size);
  std::vector<int> table4(grid_size);
  array_view<int, 1> av1(grid_size, table1);
  array_view<int, 1> av2(grid_size, table2);
  array_view<int, 1> av3(grid_size, table3);
  array_view<int, 1> av4(grid_size, table4);

  parallel_for_each(extent<1>(grid_size).tile<tile_size>(), [=](tiled_index<tile_size>& idx) restrict(amp) {
    av1(idx) = idx.global[0];
    av2(idx) = idx.local[0];
    av3(idx) = idx.tile[0];
    av4(idx) = idx.tile_origin[0];
  });


  // next run HC parallel_for_each

  std::vector<int> table5(grid_size);
  std::vector<int> table6(grid_size);
  std::vector<int> table7(grid_size);
  std::vector<int> table8(grid_size);
  array_view<int, 1> av5(grid_size, table5);
  array_view<int, 1> av6(grid_size, table6);
  array_view<int, 1> av7(grid_size, table7);
  array_view<int, 1> av8(grid_size, table8);

  // set dynamic tile size as 0 for now as we don't test this feature in this test yet
  parallel_for_each(tiled_extent_1D(grid_size, tile_size), 0, [=](tiled_index_1D& idx) restrict(amp) {
    av5(idx) = idx.global[0];
    av6(idx) = idx.local[0];
    av7(idx) = idx.tile[0];
    av8(idx) = idx.tile_origin[0];
  });

#define SHOW_CONTENT(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < GRID_SIZE / TILE_SIZE; ++i) { \
      for (int j = 0; j < TILE_SIZE; ++j) { \
        std::cout << table[i * TILE_SIZE + j] << " "; \
      } \
      std::cout << "\n"; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT("global", av1, table1)
  SHOW_CONTENT("local", av2, table2)
  SHOW_CONTENT("tile", av3, table3)
  SHOW_CONTENT("tile_origin", av4, table4)
#endif

#if 0
  SHOW_CONTENT("global", av5, table5)
  SHOW_CONTENT("local", av6, table6)
  SHOW_CONTENT("tile", av7, table7)
  SHOW_CONTENT("tile_origin", av8, table8)
#endif

#define VERIFY_CONTENT(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT(av1, table1, av5, table5)
  VERIFY_CONTENT(av2, table2, av6, table6)
  VERIFY_CONTENT(av3, table3, av7, table7)
  VERIFY_CONTENT(av4, table4, av8, table8)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<1,1>();
  ret &= test1D<32, 4>();
  ret &= test1D<1024, 16>();
  ret &= test1D<4096, 256>();

  return !(ret == true);
}

