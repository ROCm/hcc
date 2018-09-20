// RUN: %amp_device -c -S -D__KALMAR_ACCELERATOR__ -emit-llvm %s -O -o -|%cppfilt|%FileCheck %s
// RUN: %gtest_amp %s -o %t && %t
// Testing if an efficient (i.e. fully inlined version) of hc::index
#include <hc.hpp>
#ifndef __KALMAR_ACCELERATOR__ //Device mode compilation cannot have RTTI
#include <gtest/gtest.h>
#endif
#define N0 10

// Test code generation; operator[] should be inlined completely
// And there shouldn't be any load/stores!
int foo(int k) [[hc]]{
  hc::index<1> i(k);
  return i[0];
}
//CHECK: define {{.*}} @foo(int)
//CHECK-NOT: call {{.*}}hc::index<1>::operator[]
//CHECK-NOT: load
//CHECK: }

#ifndef __KALMAR_ACCELERATOR__ //Device mode compilation cannot have RTTI
// Test correctness
TEST(ClassIndex, Index1D) {
  int n0 = N0;
  hc::index<1> i(n0);
  EXPECT_EQ(n0, i[0]);
}

TEST(ClassIndex, Def) {
  hc::index<1> i(1234);
  // Test copy constructor
  hc::index<1> j(i);
  EXPECT_EQ(i[0], j[0]);
  // Test prefix ++
  ++j;
  EXPECT_EQ(i[0]+1, j[0]);
  // Test postfix ++
  hc::index<1> k(j++);
  EXPECT_EQ(i[0]+1, k[0]);
  EXPECT_EQ(i[0]+2, j[0]);
}

TEST(ClassIndex, Add) {
  hc::index<2> i(1234, 5678);
  hc::index<2> j(4321, 8765);
  hc::index<2> k = i + j;
  EXPECT_EQ(1234+4321, k[0]);
  EXPECT_EQ(5678+8765, k[1]);
}

TEST(ClassIndex, AddEqual) {
  hc::index<2> i(1234, 5678);
  hc::index<2> j(4321, 8765);
  i += j;
  EXPECT_EQ(1234+4321, i[0]);
  EXPECT_EQ(5678+8765, i[1]);
}

TEST(ClassIndex, SubEqual) {
  hc::index<2> i(5555, 9999);
  hc::index<2> j(4321, 8765);
  i -= j;
  EXPECT_EQ(1234, i[0]);
  EXPECT_EQ(1234, i[1]);
}
#endif
