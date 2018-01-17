// RUN: %gtest_amp %s -o %t && %t
#include <stdlib.h>
#ifndef __KALMAR_ACCELERATOR__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
class baz {
 public:
  baz(void): foo(1234) {}
  __attribute__((annotate("auto_deserialize"))) baz(int foo_, float bar_) restrict(amp,cpu);
  //:foo(foo_), bar(bar_) {}
  int foo;
  float bar;
};

 __attribute__((annotate("user_deserialize")))
int fake_use(void)
  restrict(amp) {
  baz bll(1, 2.0);
  return bll.foo;
}
#ifndef __KALMAR_ACCELERATOR__
TEST(GPUCodeGen, Constructor) {
  baz bll(1, 2.0);
  EXPECT_EQ(bll.foo, 1);
}
#endif
