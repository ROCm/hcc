; RUN: %llc -march=c < %s | %FileCheck %s

; ModuleID = 'check_kernel_IR_with_global.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define void @test(float addrspace(1)* nocapture %a, i32 %sz_a, float addrspace(1)* nocapture %b, i32 %sz_b, float addrspace(1)* nocapture %c, i32 %sz_c) #0 {
  ret void
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.kernels = !{!0}

!0 = metadata !{void (float addrspace(1)*, i32, float addrspace(1)*, i32, float addrspace(1)*, i32)* @test}
;CHECK: __kernel
;CHECK: __global
;CHECK-NOT: long long
