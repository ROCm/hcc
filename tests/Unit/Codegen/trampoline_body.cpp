// RUN: %amp_device -D__GPU__=1 %s -c -o %t.device.o
// RUN: %gtest_amp %s %t.device.o -o %t && %t
#include <stdlib.h>
#include <amp.h>
#ifndef __GPU__ //gtest requires rtti, but amp_device forbids rtti
#include <gtest/gtest.h>
#endif
#define MAGIC1 42

// To record results used in executable tests
class Counter {
 public:
  static void CalledGetGlobalId(int i) restrict(amp) {
    global_id_val = i;
  }
  static void CalledOperator(int i) restrict(amp) {
    operator_call_val = i; }
  static int global_id_val, operator_call_val;
};
#ifndef __GPU__
int Counter::global_id_val = 0;
int Counter::operator_call_val = 0;
#endif

// the functor to test
class baz {
 public:
  void operator()(Concurrency::index<1> idx) restrict(amp) {
    Counter::CalledOperator(idx[0] + MAGIC1 + foo);
  }
  // The generated trampoline code should look like:
#if 0
  static __attribute__((annotate("__cxxamp_trampoline")))
  void __cxxamp_trampoline(int foo, float bar) restrict(amp) {
    baz tmp(foo, bar);
    Concurrency::index<1> idx;
    // calls get_global_id(0);
    idx.__cxxamp_opencl_index();
    tmp(idx);
  }
#endif
  int foo;
  float bar;
};

// a fake implementation of get_global_id(0)
extern "C" int get_global_id(int) restrict(amp) {
  Counter::CalledGetGlobalId(MAGIC1);
  return MAGIC1;
}

// to trigger generation of __cxxamp_trampoline in device compilation mode
__attribute__((noinline))
void trigger(void) restrict(amp) {
  baz::__cxxamp_trampoline(1234, 56.78f);
}
#ifndef __GPU__
TEST(GPUCodeGen, Constructor) {
  // the generated implementation of trampoline should:
  // call the constructor of Concurrency::index<1> which
  // calls get_global_id. Our mock version of get_global_id
  // sets the Counter::global_id_val
  trigger();
  EXPECT_EQ(Counter::global_id_val, MAGIC1);
}
TEST(GPUCodeGen, FunctionBody) {
  // After calling index<1> constructor
  // the generated implementation of trampoline should
  // call the operator() with the constructed index<1> object
  // this test would succeed only if:
  // 1) trampoline calls operator() passing designated index object
  // 2) the functor is constructed correctly
  // 3) the operator() is called
  trigger();
  EXPECT_EQ(Counter::operator_call_val, MAGIC1+MAGIC1+1234);
}
#endif
