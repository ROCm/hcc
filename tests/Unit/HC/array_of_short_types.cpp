
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <hc.hpp>

#define NUM_ARRAY 512
#define ARRAY_SIZE (1 * 1024)

// test whether array of char or short could be created in hc mode
template<typename T>
bool test() {
  bool ret = true;
  try {

    hc::array<T, 1>* arrays[NUM_ARRAY];
    for (int i = 0; i < NUM_ARRAY; ++i) {
      arrays[i] = new hc::array<T, 1>(ARRAY_SIZE);
    }

    for (int i = 0; i < NUM_ARRAY; ++i) {
      delete arrays[i];
    }
  } catch(std::exception e) {
    std::cout << e.what() << std::endl;
    ret = false;
  }
  return ret;
}

int main() {
  bool ret = true;

  ret &= test<char>();
  ret &= test<short>();

  return !(ret == true);
}

