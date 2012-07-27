// RUN: %clang_cppamp %s
// #include <iostream>

int main() {
  int a = 1;
  int b = 2;
  int c;
  [=, &c] ()
    restrict(cpu, amp)
    { c = a + b; } ();
  return c;
}

