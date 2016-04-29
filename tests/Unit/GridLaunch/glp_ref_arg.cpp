// RUN: %hc -lhc_am %s -c -o %t.out 2>&1 | %FileCheck %s

#include "hc.hpp"
#include "grid_launch.hpp"

__attribute__((hc_grid_launch)) void kernel(grid_launch_parm &lp, int* x) {
}

template<typename T>
__attribute__((hc_grid_launch)) void kernel_tmpl(grid_launch_parm &lp, T* x) {
}

// CHECK: glp_ref_arg.cpp:6:63: error: hc_grid_launch function does not support passing by reference
// CHECK-NEXT: __attribute__((hc_grid_launch)) void kernel(grid_launch_parm &lp, int* x) {
// CHECK: glp_ref_arg.cpp:10:68: error: hc_grid_launch function does not support passing by reference
// CHECK-NEXT: __attribute__((hc_grid_launch)) void kernel_tmpl(grid_launch_parm &lp, T* x) {
