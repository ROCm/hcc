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

#define Z (4)
#define Y (4)
#define X (2)

int main() {
  std::array<int, X * Y * Z> table { 0 };

  // generate array_view
  std::bounds<3> bnd({Z, Y, X});
  std::array_view<int, 3> av(table.data(), bnd);

  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // range for
  for (auto idx : av.bounds()) {
    std::cout << idx[0] << " " << idx[1] << " " << idx[2] << "\n";
  }

  return 0;
}

