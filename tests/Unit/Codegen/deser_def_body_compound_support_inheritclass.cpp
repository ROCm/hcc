// RUN: %amp_device -D__KALMAR_ACCELERATOR__=1 %s -c -o %t.device.o
// RUN: %gtest_amp %s %t.device.o -o %t && %t

#include <stdlib.h>
#ifndef __KALMAR_ACCELERATOR__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif 
class Member {
 public:
  // Compiler-generated constructor
  __attribute__((annotate("auto_deserialize"))) Member(float, int) restrict(amp, cpu);
  float bzzt;
  int zzz;
};

class base {
 public:
  // Compiler-generated constructor
  __attribute__((annotate("auto_deserialize"))) base(float m1, int m2,
    int foo_, float bar_) restrict(amp, cpu);

  Member m;
  int foo;
  float bar;
};

class baz :public base {
 public:
  // Compiler-generated constructor
  __attribute__((annotate("auto_deserialize"))) baz(float m1, int m2,
    int foo_, float bar_, int bar_foo_) restrict(amp, cpu);
  int baz_foo;
};

__attribute__((annotate("user_deserialize")))
int fake_use(void) restrict(amp) {
  baz bll(0, 0,  1, 2.0, 1);
  return bll.foo;
}
#ifndef __KALMAR_ACCELERATOR__
TEST(GPUCodeGen, ConstructorCompound) {
  float local_float = 2.78f;
  baz bll(local_float, 2, 1, 2.0,1);
  EXPECT_EQ(bll.foo, 1);
  EXPECT_EQ(bll.m.bzzt, local_float);
  EXPECT_EQ(bll.m.zzz, 2);
  EXPECT_EQ(bll.baz_foo, 1);
}
#endif
