// RUN: %cxxamp %s -emit-llvm -S -o -|%cppfilt|%FileCheck %s

__attribute__((hc_grid_launch)) void foo(int arg1, int arg2)
{
  return;
}

// CHECK: ; Function Attrs: {{hc_grid_launch * noinline}}
// CHECK-NEXT: define void @foo
