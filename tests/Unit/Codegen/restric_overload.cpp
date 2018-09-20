// RUN: %gtest_amp %s -O2 -o %t && %t
#include <stdlib.h>
#ifndef __KALMAR_ACCELERATOR__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
class baz {
 public:
  void foo(void) [[hc]] {bar = 1;}
  void foo(void) [[cpu]] {bar = 2;}
  int bar;
};

int fake_use(void) [[cpu, hc]] {
  baz baz_cpu;
  baz_cpu.foo(); //call the one with [[cpu]]
  return baz_cpu.bar;
}
#ifndef __KALMAR_ACCELERATOR__
TEST(GPUCodeGen, Constructor) {
 EXPECT_EQ(2, fake_use());
}
#endif
