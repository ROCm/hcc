// RUN: %cxxamp -c %s

int main() {
  int a = 1;
  int b = 2;
  int c;
  // capture-by-reference is not allowed in amp-restricted kernel lambda
  [=, &c] ()
    restrict(cpu)
    { c = a + b; } ();
  return c;
}

