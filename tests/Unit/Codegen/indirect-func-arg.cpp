// Make a unique directory to save the dumped IR.
// RUN: mkdir -p %T/indirect-func-arg
// RUN: %hc %s -o %t.out -Xlinker -dump-llvm -Xlinker -dump-dir=%T/indirect-func-arg
// RUN: %llvm-dis %T/indirect-func-arg/dump*.opt.bc
// RUN: cat %T/indirect-func-arg/dump*.opt.ll| %FileCheck %s
// RUN: %t.out

#include <hc.hpp>
#include <vector>

#define GRID_SIZE (1024)

// CHECK-LABEL: define weak_odr amdgpu_kernel void @"_ZZ4mainEN3$_019__cxxamp_trampolineEPjiiPi"(i32*, i32, i32, i32*)
struct A {
  int x[8];
  A()[[hc]] {
    x[0] = 1;
  }
};

int g(void *p) [[hc]] {
  return *((int*)p);
}
  
int f(A a, int i) [[hc]] {
  void *p[10];
  // CHECK-NOT:  bitcast [10 x i8*] addrspace(5)* %{{[^ ]+}} to %struct.A addrspace(5)* addrspace(5)*
  // The following addrspacecast and GEP are created by SROA.
  // CHECK: %[[a:[^ ]+]] = addrspacecast i8 addrspace(5)* %{{[^ ]+}} to i8*
  // CHECK: %[[p:[^ ]+]] = getelementptr inbounds [10 x i8*], [10 x i8*] addrspace(5)* %{{[^ ]+}}, i32 0, i32 0
  // CHECK: store i8* %[[a]], i8* addrspace(5)* %[[p]]
  p[0]  = (void*)&a;
  return g(p[i]);
} 
 
int main() {
  using namespace hc;
  array<unsigned int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  int i;
  auto k = [&](index<1>& idx) [[hc]]{
    A a;
    table(idx) = f(a, i);
  };
  i = 0;
  parallel_for_each(ex, k ).wait();

  // verify result
  bool ret = true;
  std::vector<unsigned int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (result[i] == 1);
  }

  return !(ret == true);
}

