// RUN: %cxxamp %s -o %t.out && %t.out

#ifndef __HCC_AMP__
#error __HCC_AMP__ is not defined!
#endif

// __HCC_HC__ and __HCC_AMP__ are mutually exclusive
#ifdef __HCC_HC__
#error __HCC_HC__ is defined!
#endif

int main() {
  return 0;
}

