// RUN: %amp_device -D__GPU__=1 %s -c -o %t.device.o
// RUN: %gtest_amp %s %t.device.o -o %t && %t
#include <stdlib.h>
#ifndef __GPU__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
class baz {
 public:
  baz(void): foo(1234) {}
  __attribute__((used)) baz(int foo_, float bar_) restrict(amp);
  //:foo(foo_), bar(bar_) {}
  int foo;
  float bar;
};
int fake_use(void) restrict(amp) {
  baz bll(1, 2.0);
  return bll.foo;
}
#ifndef __GPU__
TEST(GPUCodeGen, Constructor) {
  baz bll(1, 2.0);
  EXPECT_EQ(bll.foo, 1);
}
#endif
