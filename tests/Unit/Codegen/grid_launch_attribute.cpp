// RUN: %cxxamp %s -emit-llvm -S -o -|%FileCheck %s

__attribute__((hc_grid_launch, always_inline, noinline)) void foo(int arg1, int arg2)
{
  return;
}

int bar()
{
  return 0;
}

// CHECK-NOT: alwaysinline
// CHECK: define void @_Z3fooii(i32 %arg1, i32 %arg2) [[HC:#[0-9]+]] {
// CHECK: define i32 @_Z3barv() [[NHC:#[0-9]+]] {
// CHECK: [[HC]] = { {{.*}} "hc_grid_launch" {{.*}} }
// CHECK-NOT: [[NHC]] = { {{.*}} "hc_grid_launch" {{.*}} }
