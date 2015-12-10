// XFAIL: Linux,boltzmann
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

// test if we allow volatile qualifier in HSA extension mode
// test if we allow union of volatile types in HSA extension mode

#define VOLATILE volatile

#define SIZE (128)

__attribute__((amp,cpu)) void p(VOLATILE float* fp) {
  *fp = 100.0f;
}

__attribute__((amp,cpu)) float foo1(float a) {
  union {
    VOLATILE float* fp;
    VOLATILE int* ip;
  } u;

  int i;
  u.ip = &i;
  p(u.fp);

  return *(u.fp);
}

__attribute__((amp,cpu)) float foo2(float a) {

  VOLATILE float* fp;
  VOLATILE int* ip;

  int i;
  ip = &i;
  fp = (VOLATILE float*)ip;
  p(fp);

  return *fp;
}

int main() {
  bool ret = true;

  using namespace concurrency;

  float table[SIZE] { 0.0f };

  // test foo1
  parallel_for_each(extent<1>(SIZE), [&table](index<1> idx) restrict(amp) {
    table[idx[0]] = foo1(0.0f);
  });

  for (int i = 0; i < SIZE; ++i) {
    if (table[i] != 100.0f) {
      ret = false;
      break;
    }
  }

  for (int i = 0; i < SIZE; ++i) {
    table[i] = 0.0f;
  }

  // test foo2
  parallel_for_each(extent<1>(SIZE), [&table](index<1> idx) restrict(amp) {
    table[idx[0]] = foo2(0.0f);
  });

  for (int i = 0; i < SIZE; ++i) {
    if (table[i] != 100.0f) {
      ret = false;
      break;
    }
  }

  return !(ret == true);
}

