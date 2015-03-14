// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

#define VECTOR_SIZE (64)

class Cell {
public:
  int get() restrict(cpu,amp) { return value; }
  void set(int v) restrict(cpu,amp) { value = v; }
private:
  int value;
};

int main() {
  using namespace Concurrency;

  Cell matrixA[VECTOR_SIZE][VECTOR_SIZE];
  Cell matrixB[VECTOR_SIZE][VECTOR_SIZE];
  Cell matrixC[VECTOR_SIZE][VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      matrixA[i][j].set(rand() % 15 + 1);
      matrixB[i][j].set(rand() % 15 + 1);
    } 
  }

  extent<2> ex(VECTOR_SIZE, VECTOR_SIZE);
  parallel_for_each(ex, [&](index<2> idx) restrict(amp) {
    // capture array type, and POD type by reference
    // use member function to access POD type
    int result = 0;
    for (int k = 0; k < VECTOR_SIZE; ++k) {
      result += (matrixA[idx[0]][k].get() + matrixB[k][idx[1]].get());
    }
    matrixC[idx[0]][idx[1]].set(result);
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      int tmp = 0;
      for (int k = 0; k < VECTOR_SIZE; ++k) {
        tmp += (matrixA[i][k].get() + matrixB[k][j].get());
      }
      if (matrixC[i][j].get() != tmp) {
        std::cout << "Failed at (" << i << "," << j << ")" << std::endl;
        return 1;
      }
    }
  }

  std::cout << "Passed" << std::endl;
  return 0;
}

