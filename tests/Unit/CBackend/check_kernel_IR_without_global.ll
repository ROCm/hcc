; RUN: %llc -march=c < %s | %FileCheck %s

; ModuleID = 'check_kernel_IR_without_global.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define void @_ZN9myFunctor9LaunchPadEPfjS0_jS0_j(float* nocapture %a, i32 %sz_a,
  float* nocapture %b, i32 %sz_b, float* nocapture %c, i32 %sz_c) 
  uwtable align 2 {
entry:

  ret void
}

declare i32 @get_global_id(i32)

!opencl.kernels = !{!0}
!0 = metadata !{void (float*, i32, float*, i32, float*, i32)*
  @_ZN9myFunctor9LaunchPadEPfjS0_jS0_j}

;CHECK: __kernel
;CHECK: __global
;CHECK-NOT: long long
