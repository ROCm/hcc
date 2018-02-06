
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>
#include <iostream>
#include <cstdlib>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

#define VECTOR_SIZE (16)

bool test() {
  using namespace Concurrency;

  int p = rand() % 15 + 1;

  int table[VECTOR_SIZE][VECTOR_SIZE][VECTOR_SIZE][VECTOR_SIZE];
  int table2[VECTOR_SIZE][VECTOR_SIZE][VECTOR_SIZE][VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      for (int k = 0; k < VECTOR_SIZE; ++k) {
        for (int l = 0; l < VECTOR_SIZE; ++l) {
          table[i][j][k][l] = rand() % 255 + 1;
        }
      }
    }
  }

  int dim[4] { VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE, VECTOR_SIZE };
  extent<4> ex(dim);
  parallel_for_each(ex, [&](index<4> idx) restrict(amp) {
    // capture multiple 4D array types and scalar type by reference
    table2[idx[0]][idx[1]][idx[2]][idx[3]] = table[idx[0]][idx[1]][idx[2]][idx[3]] * p;
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      for (int k = 0; k < VECTOR_SIZE; ++k) {
        for (int l = 0; l < VECTOR_SIZE; ++l) {
          if (table2[i][j][k][l] != table[i][j][k][l] * p) {
            std::cout << "Failed at (" << i << "," << j << "," << k << "," << l << ")" << std::endl;
            return false;
          }
        }
      }
    }
  }

  std::cout << "Passed" << std::endl;
  return true;
}

int main() {
  bool ret = true;

  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::hcAgentProfileFull) {
    ret &= test();
  }

  return !(ret == true);
}

