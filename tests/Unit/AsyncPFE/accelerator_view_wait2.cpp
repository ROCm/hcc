
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <random>

#define LOOP_COUNT (10240)

/// test which checks the behavior of:
/// accelerator_view::wait()
template<size_t grid_size, size_t tile_size>
void execute(hc::array_view<const int, 1>& av1,
             hc::array_view<const int, 1>& av2,
             hc::array_view<int, 1>& av3) {
  // run HC parallel_for_each
  hc::parallel_for_each(hc::tiled_extent<1>(grid_size, tile_size), [=](hc::tiled_index<1>& idx) restrict(amp) {
    for (int i = 0; i < LOOP_COUNT; ++i) {
      av3(idx) = av1(idx) + av2(idx);
    }
  });
}

template<size_t grid_size>
bool verify(hc::array_view<const int, 1>& av1,
            hc::array_view<const int, 1>& av2,
            hc::array_view<int, 1>& av3) {
  for (int i = 0; i < grid_size; ++i) {
    if (av3[i] != av1[i] + av2[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  bool ret = true;

  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;

  // initialize test data
  std::vector<int> table1(1);
  std::vector<int> table2(1);
  std::vector<int> table3(1);
  for (int i = 0; i < 1; ++i) {
    table1[i] = int_dist(rd);
    table2[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av1(1, table1);
  hc::array_view<const int, 1> av2(1, table2);
  hc::array_view<int, 1> av3(1, table3);

  // launch kernel
  execute<1,1>(av1, av2, av3);

  // initialize test data
  std::vector<int> table4(32);
  std::vector<int> table5(32);
  std::vector<int> table6(32);
  for (int i = 0; i < 32; ++i) {
    table4[i] = int_dist(rd);
    table5[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av4(32, table4);
  hc::array_view<const int, 1> av5(32, table5);
  hc::array_view<int, 1> av6(32, table6);

  // launch kernel
  execute<32,4>(av4, av5, av6);

  // initialize test data
  std::vector<int> table7(1024);
  std::vector<int> table8(1024);
  std::vector<int> table9(1024);
  for (int i = 0; i < 1024; ++i) {
    table7[i] = int_dist(rd);
    table8[i] = int_dist(rd);
  }
  hc::array_view<const int, 1> av7(1024, table7);
  hc::array_view<const int, 1> av8(1024, table8);
  hc::array_view<int, 1> av9(1024, table9);

  // launch kernel
  execute<1024, 16>(av7, av8, av9);

  // wait on all commands on the default queue to finish
  hc::accelerator().get_default_view().wait();

  ret &= verify<1>(av1, av2, av3);
  ret &= verify<32>(av4, av5, av6);
  ret &= verify<1024>(av7, av8, av9);

  return !(ret == true);
}

