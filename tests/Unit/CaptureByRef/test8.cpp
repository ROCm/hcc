// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (64)

int main() {
  using namespace Concurrency;

  int p = rand() % 15 + 1;

  int table[VECTOR_SIZE][VECTOR_SIZE][VECTOR_SIZE];
  int table2[VECTOR_SIZE][VECTOR_SIZE][VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      for (int k = 0; k < VECTOR_SIZE; ++k) {
        table[i][j][k] = rand() % 255 + 1;
      }
    }
  }

  extent<3> ex(VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE);
  parallel_for_each(ex, [&](index<3> idx) restrict(amp) {
    // capture multiple 3D array types and scalar type by reference
    table2[idx[0]][idx[1]][idx[2]] = table[idx[0]][idx[1]][idx[2]] * p;
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      for (int k = 0; k < VECTOR_SIZE; ++k) {
        if (table2[i][j][k] != table[i][j][k] * p) {
          std::cout << "Failed at (" << i << "," << j << "," << k << ")" << std::endl;
          return 1;
        }
      }
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

