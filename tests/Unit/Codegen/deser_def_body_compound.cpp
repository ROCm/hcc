// XFAIL: *
// RUN: %gtest_amp %s -o %t && %t
#include <stdlib.h>
#ifndef __KALMAR_ACCELERATOR__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
class Member {
 public:
  // Compiler-generated constructor
  __attribute__((noinline))
  __attribute__((annotate("auto_deserialize"))) Member(float, int) restrict(amp);
  float bzzt;
  int zzz;
};

class baz {
 public:
  // Compiler-generated constructor
  __attribute__((annotate("auto_deserialize"))) baz(float m1, int m2,
    int foo_, float bar_) restrict(amp,cpu);

  Member m;
  int foo;
  float bar;
};

__attribute__((annotate("user_deserialize")))
int fake_use(void) restrict(amp) {
  baz bll(0.0, 0,  1, 2.0);
  return bll.foo;
}
#ifndef __KALMAR_ACCELERATOR__
TEST(GPUCodeGen, ConstructorCompound) {
  float local_float = 2.78f;
  baz bll(local_float, 2, 1, 2.0);
  EXPECT_EQ(bll.foo, 1);
  EXPECT_EQ(bll.m.bzzt, local_float);
  EXPECT_EQ(bll.m.zzz, 2);
}
#endif
