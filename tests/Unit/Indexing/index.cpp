// RUN: %amp_device -c -S -D__GPU__ -emit-llvm %s -O -o -|c++filt|%FileCheck %s
// RUN: %gtest_amp %s -o %t && %t
// Testing if an efficient (i.e. fully inlined version) of Concurrency::index
#include <amp.h>
#ifndef __GPU__ //Device mode compilation cannot have RTTI
#include <gtest/gtest.h>
#endif
#define N0 10

// Test code generation; operator[] should be inlined completely
// And there shouldn't be any load/stores!
int foo(int k) restrict(amp){
  Concurrency::index<1> i(k);
  return i[0];
}
//CHECK-NOT: load
//CHECK: define {{.*}} @foo(int)
//CHECK-NOT: call {{.*}}Concurrency::index<1>::operator[]
//CHECKL }

#ifndef __GPU__ //Device mode compilation cannot have RTTI
// Test correctness
TEST(ClassIndex, Index1D) {
  int n0 = N0;
  Concurrency::index<1> i(n0);
  EXPECT_EQ(n0, i[0]);
}
#endif
