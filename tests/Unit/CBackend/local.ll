; RUN: %llc -march=c < %s | %FileCheck %s
; ModuleID = 'myfunctor_lambda.cc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

%"class.Concurrency::array_view" = type { float addrspace(1)*, i32 }

@_ZZ4mainEN1aE = internal addrspace(3) unnamed_addr global [16 x float] zeroinitializer, align 4

declare i32 @get_global_id(i32) nounwind readonly


define internal void @cho (float addrspace(1)*, i32, float addrspace(1)*, i32, float addrspace(1)*, i32) nounwind align 2 {
entry:
;CHECK: static void cho
;CHECK: __local {{.*}} _ZZ4mainEN1aE
  %call.i.i = tail call i32 @get_global_id(i32 0) nounwind readonly
  %call2.i.i = tail call i32 @get_local_id(i32 0) nounwind readonly
  %arrayidx.i21.i = getelementptr inbounds float addrspace(1)* %0, i32 %call.i.i
  %6 = load float addrspace(1)* %arrayidx.i21.i, align 4
  %arrayidx.i = getelementptr inbounds [16 x float] addrspace(3)* @_ZZ4mainEN1aE, i32 0, i32 %call2.i.i
  store float %6, float addrspace(3)* %arrayidx.i, align 4
  %arrayidx.i14.i = getelementptr inbounds float addrspace(1)* %4, i32 %call.i.i
  %7 = load float addrspace(1)* %arrayidx.i14.i, align 4
  %add.i = fadd float %6, %7
  %arrayidx.i.i = getelementptr inbounds float addrspace(1)* %2, i32 %call.i.i
  store float %add.i, float addrspace(1)* %arrayidx.i.i, align 4
  ret void
}

declare i32 @get_local_id(i32) nounwind readonly

