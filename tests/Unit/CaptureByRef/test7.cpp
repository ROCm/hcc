// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (256)

int main() {
  using namespace Concurrency;

  int p = rand() % 15 + 1;

  int table[VECTOR_SIZE][VECTOR_SIZE];
  int table2[VECTOR_SIZE][VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      table[i][j] = rand() % 255 + 1;
    }
  }

  extent<2> ex(VECTOR_SIZE, VECTOR_SIZE);
  parallel_for_each(ex, [&](index<2> idx) restrict(amp) {
    // capture multiple 2D array types and scalar type by reference
    table2[idx[0]][idx[1]] = table[idx[0]][idx[1]] * p;
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      if (table2[i][j] != table[i][j] * p) {
        std::cout << "Failed at (" << i << "," << j << ")" << std::endl;
        return 1;
      }
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

