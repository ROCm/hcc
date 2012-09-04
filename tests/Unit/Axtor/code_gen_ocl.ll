; RUN: %llvm-as %s -o %kernel.bc
; RUN: %axtor %kernel.bc -m OCL | %FileCheck %s

; ModuleID = 'kernel.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-
f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-
n8:16:32:64-S128"

target triple = "x86_64-unknown-linux-gnu"

define void @hellocl(i32* nocapture %buffer1, i32* nocapture %buffer2,
    i32* nocapture %buffer3) nounwind uwtable {
entry:
  %call = tail call i32 (...)* @get_global_id(i32 0) nounwind
  %call1 = tail call i32 (...)* @get_global_id(i32 1) nounwind
  ret void
}

declare i32 @get_global_id(...)

!opencl.kernels = !{!0}

!0 = metadata !{void (i32*, i32*, i32*)* @hellocl}

;CHECK: #pragma OPENCL EXTENSION cl_khr_fp64: enable
;CHECK: __kernel void hellocl(int* buffer1, int* buffer2, int* buffer3)

