// RUN: %cxxamp %s -o %t.out && %t.out

#ifndef __HCC__
#error __HCC__ is not defined!
#endif

int main() {
  return 0;
}

