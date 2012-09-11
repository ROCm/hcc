; RUN: %llc -march=c < %s | %FileCheck %s

; ModuleID = 'check_kernel_IR_with_global.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:
  64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:
128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define void @_ZN9myFunctor9LaunchPadEPfjS0_jS0_j(float addrspace(1)* nocapture %a,
  i32 %sz_a, float addrspace(1)* nocapture %b, i32 %sz_b, float addrspace(1)*
  nocapture %c, i32 %sz_c) uwtable align 2 {
entry:

  ret void
}

!opencl.kernels = !{!0}
!0 = metadata !{void (float addrspace(1)*, i32, float addrspace(1)*, i32,
  float addrspace(1)*, i32)* @_ZN9myFunctor9LaunchPadEPfjS0_jS0_j}

;CHECK: __kernel
;CHECK: __global
;CHECK-NOT: long long
