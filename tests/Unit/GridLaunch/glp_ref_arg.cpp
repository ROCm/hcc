// RUN: %hc -lhc_am %s -o %t.out 2>&1 | %FileCheck %s

#include "hc.hpp"
#include "grid_launch.h"

__attribute__((hc_grid_launch)) void kernel(grid_launch_parm &lp, int* x) {
}

// CHECK: glp_ref_arg.cpp:6:63: error: hc_grid_launch function does not support passing by reference
// CHECK-NEXT: __attribute__((hc_grid_launch)) void kernel(grid_launch_parm &lp, int* x) {
