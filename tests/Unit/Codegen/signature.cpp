// XFAIL: *
// RUN: %amp_device -O2 -D__KALMAR_ACCELERATOR__=1 %s -c -o %t.device.o
// RUN: %gtest_amp %s %t.device.o -O2 -o %t && %t
#include <stdlib.h>
#ifndef __KALMAR_ACCELERATOR__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
class member {
 public:
   void cho(void) restrict(amp) {};
  member(int i) {
    _i = i+1;
  }
  int _i;
};
class base {
 public:
  void cho(void) restrict(amp) {};
  base(float f) {
    _f = f+1;
  }
  float _f;
};
class baz: public base {
 public:
  void cho(void) restrict(amp) {};
  // User-defined constructor with same signature as generated
  // deserializer
  baz(float f, int bar_, int i): base(f), bar(bar_), m(i){}
  int bar;
  member m;
};
#ifdef __KALMAR_ACCELERATOR__
__attribute__((annotate("user_deserialize")))
float fake_use(void) restrict(amp) {
  baz bll(1.1, 2, 1); // calls the deserializer
  return bll._f;
}
#else
extern float fake_use(void);
TEST(GPUCodeGen, Constructor) {
 baz user(1.1f, 2, 1); //calls user-defined constructor
 EXPECT_EQ(user._f, 2.1f);
 EXPECT_EQ(1.1f, fake_use()); //fake_use calls the generated constructor
}
#endif
