// RUN: %clang_cppamp -c %s
// #include <iostream>

int func() restrict(amp) {
  return 0;
}

