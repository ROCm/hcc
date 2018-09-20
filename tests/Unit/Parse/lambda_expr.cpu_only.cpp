// RUN: %cxxamp -c %s

int main() {
  int a = 1;
  int b = 2;
  int c;
  [=, &c] ()
    [[cpu]]
    { c = a + b; } ();
  return c;
}

