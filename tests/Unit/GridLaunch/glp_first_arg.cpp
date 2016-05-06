// RUN: %hc %s -c -o %t.out 2>&1 | %FileCheck %s

#include "hc.hpp"
#include "grid_launch.hpp"

__attribute__((hc_grid_launch)) void kernel(int* x) {
}

// CHECK: glp_first_arg.cpp:6:50: error: hc_grid_launch function must have grid_launch_parm type as first parameter
// CHECK-NEXT: __attribute__((hc_grid_launch)) void kernel(int* x) {
