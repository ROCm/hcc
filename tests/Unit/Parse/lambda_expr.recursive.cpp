// RUN: %clang_cppamp -c %s

int main() {
  int a = 1;
  int b = 2;
  int c;
  [=, &c] ()
    restrict(cpu) restrict(amp)
    { c = a + b; } ();
  return c;
}

