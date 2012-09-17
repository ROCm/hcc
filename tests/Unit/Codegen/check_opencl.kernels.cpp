// RUN: %cxxamp -c -S -emit-llvm %s -o-|c++filt|%FileCheck %s

__attribute__((opencl_kernel_function)) void kernel_function(int a) {
}

// CHECK: !opencl.kernels = !{!0}
// CHECK: !0 = metadata !{void (i32)* @kernel_function(int)}
