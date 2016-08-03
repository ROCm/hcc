; XFAIL: Darwin
; RUN: %spirify %s | tee %t | %FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@_ZZ4mainEN6localAE = internal global [10 x [10 x i32]] zeroinitializer, section "clamp_opencl_local", align 4
;CHECK-NOT:@0 = internal global [10 x [10 x i32]] zeroinitializer, section "clamp_opencl_local", align 4
;CHECK: ZZ4mainEN3_EC__019__cxxamp_trampolineEiiiiPiiiiiiiiii.ZZ4mainEN6localAE[[CLONE:[0-9]*]] = {{.+}} addrspace(3) global [10 x [10 x i32]]

define internal void @"_ZZ4mainEN3$_019__cxxamp_trampolineEiiiiPiiiiiiiiii"(i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32) #4 align 2 {
;CHECK: void @ZZ4mainEN3_EC__019__cxxamp_trampolineEiiiiPiiiiiiiiii(i32, i32, i32, i32, i32 addrspace(1)*
entry:
  %call.i.i15 = tail call i32 @get_global_id(i32 1) #6
  %call2.i.i = tail call i32 @get_global_id(i32 0) #6
  %call3.i.i = tail call i32 @get_local_id(i32 1) #6
  %call4.i.i = tail call i32 @get_local_id(i32 0) #6
  %arrayidx3.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %call3.i.i, i32 %call4.i.i
;CHECK:   %arrayidx3.i = getelementptr inbounds [10 x [10 x i32]] addrspace(3)* @ZZ4mainEN3_EC__019__cxxamp_trampolineEiiiiPiiiiiiiiii.ZZ4mainEN6localAE[[CLONE]],
  store i32 0, i32* %arrayidx3.i, align 4, !tbaa !1
  tail call void @barrier(i32 0)
  br label %for.cond4.preheader.i

for.cond4.preheader.i:                            ; preds = %for.cond4.preheader.i, %entry
  %i.041.i = phi i32 [ 0, %entry ], [ %inc11.i, %for.cond4.preheader.i ]
  %arrayidx8.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 0
  %14 = addrspacecast i32* %arrayidx8.i to i32 addrspace(3)*
  %call.i.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %14, i32 1)
  %arrayidx8.1.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 1
  %15 = addrspacecast i32* %arrayidx8.1.i to i32 addrspace(3)*
  %call.i.1.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %15, i32 1)
  %arrayidx8.2.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 2
  %16 = addrspacecast i32* %arrayidx8.2.i to i32 addrspace(3)*
  %call.i.2.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %16, i32 1)
  %arrayidx8.3.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 3
  %17 = addrspacecast i32* %arrayidx8.3.i to i32 addrspace(3)*
  %call.i.3.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %17, i32 1)
  %arrayidx8.4.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 4
  %18 = addrspacecast i32* %arrayidx8.4.i to i32 addrspace(3)*
  %call.i.4.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %18, i32 1)
  %arrayidx8.5.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 5
  %19 = addrspacecast i32* %arrayidx8.5.i to i32 addrspace(3)*
  %call.i.5.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %19, i32 1)
  %arrayidx8.6.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 6
  %20 = addrspacecast i32* %arrayidx8.6.i to i32 addrspace(3)*
  %call.i.6.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %20, i32 1)
  %arrayidx8.7.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 7
  %21 = addrspacecast i32* %arrayidx8.7.i to i32 addrspace(3)*
  %call.i.7.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %21, i32 1)
  %arrayidx8.8.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 8
  %22 = addrspacecast i32* %arrayidx8.8.i to i32 addrspace(3)*
  %call.i.8.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %22, i32 1)
  %arrayidx8.9.i = getelementptr inbounds [10 x [10 x i32]]* @_ZZ4mainEN6localAE, i32 0, i32 %i.041.i, i32 9
  %23 = addrspacecast i32* %arrayidx8.9.i to i32 addrspace(3)*
  %call.i.9.i = tail call i32 @atomic_add_local(i32 addrspace(3)* %23, i32 1)
  %inc11.i = add nsw i32 %i.041.i, 1
  %exitcond.i = icmp eq i32 %inc11.i, 10
  br i1 %exitcond.i, label %"_ZZ4mainENK3$_0clEN11Concurrency11tiled_indexILi10ELi10ELi0EEE.exit", label %for.cond4.preheader.i

"_ZZ4mainENK3$_0clEN11Concurrency11tiled_indexILi10ELi10ELi0EEE.exit": ; preds = %for.cond4.preheader.i
  tail call void @barrier(i32 0)
  %24 = load i32* %arrayidx3.i, align 4, !tbaa !1
  %add.i.i.i = add nsw i32 %call.i.i15, %9
  %mul.i.i.i = mul i32 %add.i.i.i, %7
  %add12.i.i.i = add i32 %13, %11
  %add.ptr.sum.i.i.i = add i32 %add12.i.i.i, %call2.i.i
  %add.ptr1.sum.i.i.i = add i32 %add.ptr.sum.i.i.i, %mul.i.i.i
  %add.ptr3.i.i.i = getelementptr inbounds i32* %4, i32 %add.ptr1.sum.i.i.i
  store i32 %24, i32* %add.ptr3.i.i.i, align 4, !tbaa !1
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @get_global_id(i32) #5

; Function Attrs: nounwind readonly
declare i32 @get_local_id(i32) #5

declare i32 @atomic_add_local(i32 addrspace(3)*, i32) #4

declare void @barrier(i32) #4

attributes #0 = { alwaysinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { alwaysinline nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind readonly }

!hcc.kernels = !{!0}

!0 = metadata !{void (i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @"_ZZ4mainEN3$_019__cxxamp_trampolineEiiiiPiiiiiiiiii"}
!1 = metadata !{metadata !"int", metadata !2}
!2 = metadata !{metadata !"omnipotent char", metadata !3}
!3 = metadata !{metadata !"Simple C/C++ TBAA"}
