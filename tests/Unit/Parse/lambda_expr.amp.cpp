// RUN: %cxxamp -c %s

int main() {
  int a = 1;
  int b = 2;
  int c;
  [=, &c] ()
    restrict(amp)
    { c = a + b; } ();
  return c;
}

