// RUN: %cxxamp -c %s

int main() {
  int a = 1;
  int b = 2;
  int c;
  // Note that capture-by-reference in amp restricted codes is not allowed
  [=, &c] ()
    [[cpu]]
    { c = a + b; } ();
  return c;
}

