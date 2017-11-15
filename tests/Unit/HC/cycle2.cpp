
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <vector>
#include <algorithm>

#define GRID_SIZE (1024)

int main() {
  using namespace hc;
  array<uint64_t, 1> table(GRID_SIZE);
  array<uint64_t, 1> table2(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

  // launch a kernel, log current hardware cycle count
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table(idx) = __cycle_u64();
    table2(idx) = __cycle_u64();
  }).wait();

  std::vector<uint64_t> result = table;
  std::vector<uint64_t> result2 = table2;

  // The 1st and the 2nd cycle count must be different
  bool ret = std::lexicographical_compare(result.begin(), result.end(),
                                          result2.begin(), result2.end(),
                                          [](uint64_t &a, uint64_t &b) {
                                            return a != b;
                                          });
  return !(ret == true);
}

