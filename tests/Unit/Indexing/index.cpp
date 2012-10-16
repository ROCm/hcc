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
//CHECK: define {{.*}} @foo(int)
//CHECK-NOT: call {{.*}}Concurrency::index<1>::operator[]
//CHECK-NOT: load
//CHECKL }

#ifndef __GPU__ //Device mode compilation cannot have RTTI
// Test correctness
TEST(ClassIndex, Index1D) {
  int n0 = N0;
  Concurrency::index<1> i(n0);
  EXPECT_EQ(n0, i[0]);
}

TEST(ClassIndex, Def) {
  Concurrency::index<1> i(1234);
  // Test copy constructor
  Concurrency::index<1> j(i);
  EXPECT_EQ(i[0], j[0]);
  // Test prefix ++
  ++j;
  EXPECT_EQ(i[0]+1, j[0]);
  // Test postfix ++
  Concurrency::index<1> k(j++);
  EXPECT_EQ(i[0]+1, k[0]);
  EXPECT_EQ(i[0]+2, j[0]);
}

TEST(ClassIndex, Add) {
  Concurrency::index<2> i(1234, 5678);
  Concurrency::index<2> j(4321, 8765);
  Concurrency::index<2> k = i + j;
  EXPECT_EQ(1234+4321, k[0]);
  EXPECT_EQ(5678+8765, k[1]);
}
#endif
