// RUN: %amp_device -O3 -D__GPU__=1 %s -c -o %t.device.o
// RUN: %gtest_amp %s %t.device.o -O3 -o %t && %t
#include <stdlib.h>
#ifndef __GPU__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
class baz {
 public:
  void foo(void) restrict(amp) {bar = 1;}
  void foo(void) restrict(cpu) {bar = 2;}
  int bar;
};

int fake_ampuse(void) restrict(amp) {
  baz baz_amp;
  baz_amp.foo(); //call the one with restrict(amp)
  return baz_amp.bar;
}
int fake_cpuuse(void) restrict(cpu) {
  baz baz_cpu;
  baz_cpu.foo(); //call the one with restrict(amp)
  return baz_cpu.bar;
}
#ifndef __GPU__
TEST(GPUCodeGen, Constructor) {
 EXPECT_EQ(2, fake_cpuuse());
 EXPECT_EQ(1, fake_ampuse());
}
#endif
