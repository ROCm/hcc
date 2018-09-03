// RUN: %cxxamp %s -o %t.out && %t.out

#ifndef __HC_CC__
  #error __HC_CC__ is not defined!
#endif

int main() {
  return 0;
}

