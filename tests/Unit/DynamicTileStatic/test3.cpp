
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>
#include <hc.hpp>

#include <iostream>

/// test HC parallel_for_each interface
template<size_t grid_size, size_t tile_size>
bool test1D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each
  std::vector<int> table1(grid_size);
  std::vector<int> table2(grid_size);
  std::vector<int> table3(grid_size);
  std::vector<int> table4(grid_size);
  Concurrency::array_view<int, 1> av1(grid_size, table1);
  Concurrency::array_view<int, 1> av2(grid_size, table2);
  Concurrency::array_view<int, 1> av3(grid_size, table3);
  Concurrency::array_view<int, 1> av4(grid_size, table4);

  Concurrency::parallel_for_each(Concurrency::extent<1>(grid_size).tile<tile_size>(), [=](Concurrency::tiled_index<tile_size>& idx) restrict(amp) {
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
  hc::array_view<int, 1> av5(grid_size, table5);
  hc::array_view<int, 1> av6(grid_size, table6);
  hc::array_view<int, 1> av7(grid_size, table7);
  hc::array_view<int, 1> av8(grid_size, table8);

  hc::completion_future fut = hc::parallel_for_each(hc::tiled_extent<1>(grid_size, tile_size), [=](hc::tiled_index<1>& idx) restrict(amp) {
    av5(idx) = idx.global[0];
    av6(idx) = idx.local[0];
    av7(idx) = idx.tile[0];
    av8(idx) = idx.tile_origin[0];
  });

  // wait for kernel completion
  fut.wait();

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

  std::vector<int> table1(grid_size_0 * grid_size_1);
  std::vector<int> table2(grid_size_0 * grid_size_1);
  std::vector<int> table3(grid_size_0 * grid_size_1);
  std::vector<int> table4(grid_size_0 * grid_size_1);
  std::vector<int> table5(grid_size_0 * grid_size_1);
  std::vector<int> table6(grid_size_0 * grid_size_1);
  std::vector<int> table7(grid_size_0 * grid_size_1);
  std::vector<int> table8(grid_size_0 * grid_size_1);
  Concurrency::array_view<int, 2> av1(grid_size_0, grid_size_1, table1);
  Concurrency::array_view<int, 2> av2(grid_size_0, grid_size_1, table2);
  Concurrency::array_view<int, 2> av3(grid_size_0, grid_size_1, table3);
  Concurrency::array_view<int, 2> av4(grid_size_0, grid_size_1, table4);
  Concurrency::array_view<int, 2> av5(grid_size_0, grid_size_1, table5);
  Concurrency::array_view<int, 2> av6(grid_size_0, grid_size_1, table6);
  Concurrency::array_view<int, 2> av7(grid_size_0, grid_size_1, table7);
  Concurrency::array_view<int, 2> av8(grid_size_0, grid_size_1, table8);

  Concurrency::parallel_for_each(Concurrency::extent<2>(grid_size_0, grid_size_1).tile<tile_size_0, tile_size_1>(), [=](Concurrency::tiled_index<tile_size_0, tile_size_1>& idx) restrict(amp) {
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
  hc::array_view<int, 2> av9(grid_size_0, grid_size_1, table9);
  hc::array_view<int, 2> av10(grid_size_0, grid_size_1, table10);
  hc::array_view<int, 2> av11(grid_size_0, grid_size_1, table11);
  hc::array_view<int, 2> av12(grid_size_0, grid_size_1, table12);
  hc::array_view<int, 2> av13(grid_size_0, grid_size_1, table13);
  hc::array_view<int, 2> av14(grid_size_0, grid_size_1, table14);
  hc::array_view<int, 2> av15(grid_size_0, grid_size_1, table15);
  hc::array_view<int, 2> av16(grid_size_0, grid_size_1, table16);

  hc::completion_future fut = hc::parallel_for_each(hc::tiled_extent<2>(grid_size_0, grid_size_1, tile_size_0, tile_size_1), [=](hc::tiled_index<2>& idx) restrict(amp) {
    av9(idx) = idx.global[0];
    av10(idx) = idx.global[1];
    av11(idx) = idx.local[0];
    av12(idx) = idx.local[1];
    av13(idx) = idx.tile[0];
    av14(idx) = idx.tile[1];
    av15(idx) = idx.tile_origin[0];
    av16(idx) = idx.tile_origin[1];
  });

  // wait for kernel completion
  fut.wait();

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

/// test HC parallel_for_each interface
template<size_t grid_size_0, size_t grid_size_1, size_t grid_size_2, size_t tile_size_0, size_t tile_size_1, size_t tile_size_2>
bool test3D() {

  bool ret = true;

  // first run normal C++AMP parallel_for_each

  std::vector<int> table1(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table2(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table3(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table4(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table5(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table6(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table7(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table8(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table9(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table10(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table11(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table12(grid_size_0 * grid_size_1 * grid_size_2);
  Concurrency::array_view<int, 3> av1(grid_size_0, grid_size_1, grid_size_2, table1);
  Concurrency::array_view<int, 3> av2(grid_size_0, grid_size_1, grid_size_2, table2);
  Concurrency::array_view<int, 3> av3(grid_size_0, grid_size_1, grid_size_2, table3);
  Concurrency::array_view<int, 3> av4(grid_size_0, grid_size_1, grid_size_2, table4);
  Concurrency::array_view<int, 3> av5(grid_size_0, grid_size_1, grid_size_2, table5);
  Concurrency::array_view<int, 3> av6(grid_size_0, grid_size_1, grid_size_2, table6);
  Concurrency::array_view<int, 3> av7(grid_size_0, grid_size_1, grid_size_2, table7);
  Concurrency::array_view<int, 3> av8(grid_size_0, grid_size_1, grid_size_2, table8);
  Concurrency::array_view<int, 3> av9(grid_size_0, grid_size_1, grid_size_2, table9);
  Concurrency::array_view<int, 3> av10(grid_size_0, grid_size_1, grid_size_2, table10);
  Concurrency::array_view<int, 3> av11(grid_size_0, grid_size_1, grid_size_2, table11);
  Concurrency::array_view<int, 3> av12(grid_size_0, grid_size_1, grid_size_2, table12);

  Concurrency::parallel_for_each(Concurrency::extent<3>(grid_size_0, grid_size_1, grid_size_2).tile<tile_size_0, tile_size_1, tile_size_2>(), [=](Concurrency::tiled_index<tile_size_0, tile_size_1, tile_size_2>& idx) restrict(amp) {
    av1(idx) = idx.global[0];
    av2(idx) = idx.global[1];
    av3(idx) = idx.global[2];
    av4(idx) = idx.local[0];
    av5(idx) = idx.local[1];
    av6(idx) = idx.local[2];
    av7(idx) = idx.tile[0];
    av8(idx) = idx.tile[1];
    av9(idx) = idx.tile[2];
    av10(idx) = idx.tile_origin[0];
    av11(idx) = idx.tile_origin[1];
    av12(idx) = idx.tile_origin[2];
  });

  // next run HC parallel_for_each

  std::vector<int> table13(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table14(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table15(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table16(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table17(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table18(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table19(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table20(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table21(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table22(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table23(grid_size_0 * grid_size_1 * grid_size_2);
  std::vector<int> table24(grid_size_0 * grid_size_1 * grid_size_2);
  hc::array_view<int, 3> av13(grid_size_0, grid_size_1, grid_size_2, table13);
  hc::array_view<int, 3> av14(grid_size_0, grid_size_1, grid_size_2, table14);
  hc::array_view<int, 3> av15(grid_size_0, grid_size_1, grid_size_2, table15);
  hc::array_view<int, 3> av16(grid_size_0, grid_size_1, grid_size_2, table16);
  hc::array_view<int, 3> av17(grid_size_0, grid_size_1, grid_size_2, table17);
  hc::array_view<int, 3> av18(grid_size_0, grid_size_1, grid_size_2, table18);
  hc::array_view<int, 3> av19(grid_size_0, grid_size_1, grid_size_2, table19);
  hc::array_view<int, 3> av20(grid_size_0, grid_size_1, grid_size_2, table20);
  hc::array_view<int, 3> av21(grid_size_0, grid_size_1, grid_size_2, table21);
  hc::array_view<int, 3> av22(grid_size_0, grid_size_1, grid_size_2, table22);
  hc::array_view<int, 3> av23(grid_size_0, grid_size_1, grid_size_2, table23);
  hc::array_view<int, 3> av24(grid_size_0, grid_size_1, grid_size_2, table24);

  hc::completion_future fut = hc::parallel_for_each(hc::tiled_extent<3>(grid_size_0, grid_size_1, grid_size_2, tile_size_0, tile_size_1, tile_size_2), [=](hc::tiled_index<3>& idx) restrict(amp) {
    av13(idx) = idx.global[0];
    av14(idx) = idx.global[1];
    av15(idx) = idx.global[2];
    av16(idx) = idx.local[0];
    av17(idx) = idx.local[1];
    av18(idx) = idx.local[2];
    av19(idx) = idx.tile[0];
    av20(idx) = idx.tile[1];
    av21(idx) = idx.tile[2];
    av22(idx) = idx.tile_origin[0];
    av23(idx) = idx.tile_origin[1];
    av24(idx) = idx.tile_origin[2];
  });

  // wait for kernel completion
  fut.wait();

#define SHOW_CONTENT_3D(str,av,table) \
  { \
    std::cout << str << "\n"; \
    av.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1 * grid_size_2; ++i) { \
      std::cout << table[i] << " "; \
    } \
    std::cout << "\n"; \
  } \

#if 0
  SHOW_CONTENT_3D("global", av1, table1)
  SHOW_CONTENT_3D("global", av2, table2)
  SHOW_CONTENT_3D("global", av3, table3)
  SHOW_CONTENT_3D("local", av4, table4)
  SHOW_CONTENT_3D("local", av5, table5)
  SHOW_CONTENT_3D("local", av6, table6)
  SHOW_CONTENT_3D("tile", av7, table7)
  SHOW_CONTENT_3D("tile", av8, table8)
  SHOW_CONTENT_3D("tile", av9, table9)
  SHOW_CONTENT_3D("tile_origin", av10, table10)
  SHOW_CONTENT_3D("tile_origin", av11, table11)
  SHOW_CONTENT_3D("tile_origin", av12, table12)
#endif

#if 0
  SHOW_CONTENT_3D("global", av13, table13)
  SHOW_CONTENT_3D("global", av14, table14)
  SHOW_CONTENT_3D("global", av15, table15)
  SHOW_CONTENT_3D("local", av16, table16)
  SHOW_CONTENT_3D("local", av17, table17)
  SHOW_CONTENT_3D("local", av18, table18)
  SHOW_CONTENT_3D("tile", av19, table19)
  SHOW_CONTENT_3D("tile", av20, table20)
  SHOW_CONTENT_3D("tile", av21, table21)
  SHOW_CONTENT_3D("tile_origin", av22, table22)
  SHOW_CONTENT_3D("tile_origin", av23, table23)
  SHOW_CONTENT_3D("tile_origin", av24, table24)
#endif

#define VERIFY_CONTENT_3D(av1, table1, av2, table2) \
  { \
    av1.synchronize(); av2.synchronize(); \
    for (int i = 0; i < grid_size_0 * grid_size_1 * grid_size_2; ++i) { \
      if (table1[i] != table2[i]) { \
        ret = false; \
        break; \
      } \
    } \
  }

  VERIFY_CONTENT_3D(av1, table1, av13, table13)
  VERIFY_CONTENT_3D(av2, table2, av14, table14)
  VERIFY_CONTENT_3D(av3, table3, av15, table15)
  VERIFY_CONTENT_3D(av4, table4, av16, table16)
  VERIFY_CONTENT_3D(av5, table5, av17, table17)
  VERIFY_CONTENT_3D(av6, table6, av18, table18)
  VERIFY_CONTENT_3D(av7, table7, av19, table19)
  VERIFY_CONTENT_3D(av8, table8, av20, table20)
  VERIFY_CONTENT_3D(av9, table9, av21, table21)
  VERIFY_CONTENT_3D(av10, table10, av22, table22)
  VERIFY_CONTENT_3D(av11, table11, av23, table23)
  VERIFY_CONTENT_3D(av12, table12, av24, table24)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test1D<1,1>();
  ret &= test1D<32, 4>();
  ret &= test1D<1024, 16>();
  ret &= test1D<4096, 256>();

#if 0
  ret &= test2D<1, 1, 1, 1>();
  ret &= test2D<8, 8, 2, 2>();
  ret &= test2D<1024, 16, 32, 4>();
  ret &= test2D<4096, 256, 16, 16>();
#endif

#if 0
  ret &= test3D<1, 1, 1, 1, 1, 1>();
  ret &= test3D<8, 8, 8, 2, 2, 2>();
  ret &= test3D<1024, 32, 16, 32, 4, 2>();
#endif

  return !(ret == true);
}

