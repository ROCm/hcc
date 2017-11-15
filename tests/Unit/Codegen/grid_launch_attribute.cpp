// RUN: %cxxamp %s -emit-llvm -S -o -|%FileCheck %s

// dummy grid_launch_parm type used to satisfy hc_grid_launch requirement
struct grid_launch_parm {};
__attribute__((hc_grid_launch, always_inline, noinline)) void foo(grid_launch_parm lp, int arg1, int arg2)
{
  return;
}

int bar()
{
  return 0;
}

// CHECK-NOT: alwaysinline
// CHECK: define void @_Z3foo16grid_launch_parmii(i32{{.*}}, i32{{.*}}) [[HC:#[0-9]+]] {
// CHECK: define i32 @_Z3barv() [[NHC:#[0-9]+]] {
// CHECK: [[HC]] = { {{.*}} "hc_grid_launch" {{.*}} }
// CHECK-NOT: [[NHC]] = { {{.*}} "hc_grid_launch" {{.*}} }
