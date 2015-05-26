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

#define ROW (4)
#define COL (2)

int main() {
  std::array<int, ROW * COL> table { 0 };

  // generate array_view
  std::bounds<2> bnd({ROW, COL});
  std::array_view<int, 2> av(table.data(), bnd);

  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // range for
  for (auto idx : av.bounds()) {
    std::cout << idx[0] << " " << idx[1] << "\n";
  }

  return 0;
}

