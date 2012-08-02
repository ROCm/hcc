// RUN: %cxx11 %link %s -o %t1 && %t1

#include <iostream>

int main() {
  std::cout << "";
  return 0;
}

