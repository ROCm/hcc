// Parallel STL headers
#include <array_view>
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <array>

#define SIZE (16)

int main() {
  std::array<int, SIZE> table { 0 };

  // generate array_view
  std::bounds<1> bnd(SIZE);
  std::array_view<int, 1> av(table.data(), bnd);

  // range for
  for (auto idx : av.bounds()) {
    std::cout << idx[0] << "\n";
  }

  return 0;
}

