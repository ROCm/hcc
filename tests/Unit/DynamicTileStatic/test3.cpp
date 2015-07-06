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

#define SHOW_CONTENT_1D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size / tile_size; ++i) { \
      for (int j = 0; j < tile_size; ++j) { \
        std::cout << table[i * tile_size + j] << " "; \
      } \
      std::cout << "\n"; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_1D("global", av1, table1)
  SHOW_CONTENT_1D("local", av2, table2)
  SHOW_CONTENT_1D("tile", av3, table3)
  SHOW_CONTENT_1D("tile_origin", av4, table4)
#endif

#if 0
  SHOW_CONTENT_1D("global", av5, table5)
  SHOW_CONTENT_1D("local", av6, table6)
  SHOW_CONTENT_1D("tile", av7, table7)
  SHOW_CONTENT_1D("tile_origin", av8, table8)
#endif

#define VERIFY_CONTENT_1D(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT_1D(av1, table1, av5, table5)
  VERIFY_CONTENT_1D(av2, table2, av6, table6)
  VERIFY_CONTENT_1D(av3, table3, av7, table7)
  VERIFY_CONTENT_1D(av4, table4, av8, table8)

  return ret;
}

/// test HC parallel_for_each interface
template<size_t grid_size_0, size_t grid_size_1, size_t tile_size_0, size_t tile_size_1>
bool test2D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each

  using namespace Concurrency;

  std::vector<int> table1(grid_size_0 * grid_size_1);
  std::vector<int> table2(grid_size_0 * grid_size_1);
  std::vector<int> table3(grid_size_0 * grid_size_1);
  std::vector<int> table4(grid_size_0 * grid_size_1);
  std::vector<int> table5(grid_size_0 * grid_size_1);
  std::vector<int> table6(grid_size_0 * grid_size_1);
  std::vector<int> table7(grid_size_0 * grid_size_1);
  std::vector<int> table8(grid_size_0 * grid_size_1);
  array_view<int, 2> av1(grid_size_0, grid_size_1, table1);
  array_view<int, 2> av2(grid_size_0, grid_size_1, table2);
  array_view<int, 2> av3(grid_size_0, grid_size_1, table3);
  array_view<int, 2> av4(grid_size_0, grid_size_1, table4);
  array_view<int, 2> av5(grid_size_0, grid_size_1, table5);
  array_view<int, 2> av6(grid_size_0, grid_size_1, table6);
  array_view<int, 2> av7(grid_size_0, grid_size_1, table7);
  array_view<int, 2> av8(grid_size_0, grid_size_1, table8);

  parallel_for_each(extent<2>(grid_size_0, grid_size_1).tile<tile_size_0, tile_size_1>(), [=](tiled_index<tile_size_0, tile_size_1>& idx) restrict(amp) {
    av1(idx) = idx.global[0];
    av2(idx) = idx.global[1];
    av3(idx) = idx.local[0];
    av4(idx) = idx.local[1];
    av5(idx) = idx.tile[0];
    av6(idx) = idx.tile[1];
    av7(idx) = idx.tile_origin[0];
    av8(idx) = idx.tile_origin[1];
  });

  // next run HC parallel_for_each

  std::vector<int> table9(grid_size_0 * grid_size_1);
  std::vector<int> table10(grid_size_0 * grid_size_1);
  std::vector<int> table11(grid_size_0 * grid_size_1);
  std::vector<int> table12(grid_size_0 * grid_size_1);
  std::vector<int> table13(grid_size_0 * grid_size_1);
  std::vector<int> table14(grid_size_0 * grid_size_1);
  std::vector<int> table15(grid_size_0 * grid_size_1);
  std::vector<int> table16(grid_size_0 * grid_size_1);
  array_view<int, 2> av9(grid_size_0, grid_size_1, table9);
  array_view<int, 2> av10(grid_size_0, grid_size_1, table10);
  array_view<int, 2> av11(grid_size_0, grid_size_1, table11);
  array_view<int, 2> av12(grid_size_0, grid_size_1, table12);
  array_view<int, 2> av13(grid_size_0, grid_size_1, table13);
  array_view<int, 2> av14(grid_size_0, grid_size_1, table14);
  array_view<int, 2> av15(grid_size_0, grid_size_1, table15);
  array_view<int, 2> av16(grid_size_0, grid_size_1, table16);

  // set dynamic tile size as 0 for now as we don't test this feature in this test yet
  parallel_for_each(tiled_extent_2D(grid_size_0, grid_size_1, tile_size_0, tile_size_1), 0, [=](tiled_index_2D& idx) restrict(amp) {
    av9(idx) = idx.global[0];
    av10(idx) = idx.global[1];
    av11(idx) = idx.local[0];
    av12(idx) = idx.local[1];
    av13(idx) = idx.tile[0];
    av14(idx) = idx.tile[1];
    av15(idx) = idx.tile_origin[0];
    av16(idx) = idx.tile_origin[1];
  });

#define SHOW_CONTENT_2D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1; ++i) { \
      std::cout << table[i] << " "; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_2D("global", av1, table1)
  SHOW_CONTENT_2D("global", av2, table2)
  SHOW_CONTENT_2D("local", av3, table3)
  SHOW_CONTENT_2D("local", av4, table4)
  SHOW_CONTENT_2D("tile", av5, table5)
  SHOW_CONTENT_2D("tile", av6, table6)
  SHOW_CONTENT_2D("tile_origin", av7, table7)
  SHOW_CONTENT_2D("tile_origin", av8, table8)
#endif

#if 0
  SHOW_CONTENT_2D("global", av9, table9)
  SHOW_CONTENT_2D("global", av10, table10)
  SHOW_CONTENT_2D("local", av11, table11)
  SHOW_CONTENT_2D("local", av12, table12)
  SHOW_CONTENT_2D("tile", av13, table13)
  SHOW_CONTENT_2D("tile", av14, table14)
  SHOW_CONTENT_2D("tile_origin", av15, table15)
  SHOW_CONTENT_2D("tile_origin", av16, table16)
#endif

#define VERIFY_CONTENT_2D(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT_2D(av1, table1, av9, table9)
  VERIFY_CONTENT_2D(av2, table2, av10, table10)
  VERIFY_CONTENT_2D(av3, table3, av11, table11)
  VERIFY_CONTENT_2D(av4, table4, av12, table12)
  VERIFY_CONTENT_2D(av5, table5, av13, table13)
  VERIFY_CONTENT_2D(av6, table6, av14, table14)
  VERIFY_CONTENT_2D(av7, table7, av15, table15)
  VERIFY_CONTENT_2D(av8, table8, av16, table16)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<1,1>();
  ret &= test1D<32, 4>();
  ret &= test1D<1024, 16>();
  ret &= test1D<4096, 256>();

  ret &= test2D<1, 1, 1, 1>();
  ret &= test2D<8, 8, 2, 2>();
  ret &= test2D<1024, 16, 32, 4>();
  ret &= test2D<4096, 256, 16, 16>();

  return !(ret == true);
}

