; RUN: %spirify %s | tee %t | %FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.texture = type { %struct.textureReference, float* }
%struct.textureReference = type { i32, i8, %struct.hipChannelFormatDesc }
%struct.hipChannelFormatDesc = type { i32 }
%"class.Kalmar::index" = type { %"struct.Kalmar::index_impl" }
%"struct.Kalmar::index_impl" = type { %"class.Kalmar::__index_leaf" }
%"class.Kalmar::__index_leaf" = type { i32, i32 }
%"class.Kalmar::index.0" = type { %"struct.Kalmar::index_impl.1" }
%"struct.Kalmar::index_impl.1" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2" }
%"class.Kalmar::__index_leaf.2" = type { i32, i32 }
%"class.Kalmar::index.3" = type { %"struct.Kalmar::index_impl.4" }
%"struct.Kalmar::index_impl.4" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2", %"class.Kalmar::__index_leaf.5" }
%"class.Kalmar::__index_leaf.5" = type { i32, i32 }

@t_img = global %struct.texture zeroinitializer, align 8
; CHECK: @t_img = addrspace(1) global %struct.texture zeroinitializer, align 8
; CHECK-NOT: @t_img = global %struct.texture zeroinitializer, align 8
@t_grad_x = global %struct.texture zeroinitializer, align 8
; CHECK: @t_grad_x = addrspace(1) global %struct.texture zeroinitializer, align 8
; CHECK-NOT: @t_grad_x = global %struct.texture zeroinitializer, align 8
@t_grad_y = global %struct.texture zeroinitializer, align 8
; CHECK: @t_grad_y = addrspace(1) global %struct.texture zeroinitializer, align 8
; CHECK-NOT: @t_grad_y = global %struct.texture zeroinitializer, align 8

@"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF" = internal unnamed_addr global [3321 x float] zeroinitializer, section "clamp_opencl_local", align 16
; CHECK-NOT: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF" = internal unnamed_addr global [3321 x float] zeroinitializer, section "clamp_opencl_local", align 16
; CHECK-NOT: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF" = internal unnamed_addr addrspace(1) global [3321 x float] undef, section "clamp_opencl_local", align 16
; CHECK: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF" = internal unnamed_addr addrspace(3) global [3321 x float] undef, section "clamp_opencl_local", align 16
@"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer" = internal unnamed_addr global [320 x float] zeroinitializer, section "clamp_opencl_local", align 16
; CHECK-NOT: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer" = internal unnamed_addr global [320 x float] zeroinitializer, section "clamp_opencl_local", align 16
; CHECK-NOT: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer" = internal unnamed_addr addrspace(1) global [320 x float] undef, section "clamp_opencl_local", align 16
; CHECK: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer" = internal unnamed_addr addrspace(3) global [320 x float] undef, section "clamp_opencl_local", align 16
@"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged" = internal unnamed_addr global i1 false, section "clamp_opencl_local", align 4
; CHECK-NOT: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged" = internal unnamed_addr global i1 false, section "clamp_opencl_local", align 4
; CHECK-NOT: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged" = internal unnamed_addr addrspace(1) global i1 undef, section "clamp_opencl_local", align 4
; CHECK: @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged" = internal unnamed_addr addrspace(3) global i1 undef, section "clamp_opencl_local", align 4

; Function Attrs: alwaysinline nounwind uwtable
declare void @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind uwtable
declare void @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index.0"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind uwtable
declare void @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index.3"* nocapture) #0 align 2

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: nobuiltin
declare noalias i8* @_Znwm(i64) #2

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #3

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define internal spir_kernel void @"_ZZ9hipMemsetEN3$_019__cxxamp_trampolineEmPvi"(i64, i8*, i32) #4 align 2 {
  %4 = tail call i64 @amp_get_global_id(i32 0) #9
  %sext.i = shl i64 %4, 32
  %5 = ashr exact i64 %sext.i, 32
  %6 = icmp ult i64 %5, %0
  br i1 %6, label %7, label %"_ZZ9hipMemsetENK3$_0clEN2hc11tiled_indexILi1EEE.exit"

; <label>:7                                       ; preds = %3
  %8 = trunc i32 %2 to i8
  %9 = getelementptr inbounds i8* %1, i64 %5
  store i8 %8, i8* %9, align 1, !tbaa !21
  br label %"_ZZ9hipMemsetENK3$_0clEN2hc11tiled_indexILi1EEE.exit"

"_ZZ9hipMemsetENK3$_0clEN2hc11tiled_indexILi1EEE.exit": ; preds = %7, %3
  ret void
}

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare i64 @amp_get_global_id(i32) #5

declare float @opencl_sqrt(float) #6

; Function Attrs: nounwind readnone
declare i64 @amp_get_group_id(i32) #5

; Function Attrs: nounwind readnone
declare i64 @amp_get_local_size(i32) #5

; Function Attrs: nounwind readnone
declare i64 @amp_get_local_id(i32) #5

declare float @opencl_cos(float) #6

declare float @opencl_sin(float) #6

; Function Attrs: nounwind uwtable
define internal spir_kernel void @"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_EN3$_119__cxxamp_trampolineEiiiiS2_S2_"(i32, i32, i32, i32, float*, float*) #4 align 2 {
  %7 = tail call i64 @amp_get_local_id(i32 2) #9
  %8 = tail call i64 @amp_get_group_id(i32 2) #9
  %9 = tail call i64 @amp_get_local_size(i32 2) #9
  %10 = mul nsw i64 %9, %8
  %11 = add nsw i64 %10, %7
  %12 = trunc i64 %11 to i32
  %13 = srem i32 %12, %2
  %14 = sdiv i32 %12, %2
  %15 = icmp sgt i32 %0, 0
  br i1 %15, label %.lr.ph7.i, label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit"

.lr.ph7.i:                                        ; preds = %6
  %16 = load float** getelementptr inbounds (%struct.texture* @t_img, i64 0, i32 1), align 8, !tbaa !24
; CHECK-NOT: %16 = load float** getelementptr inbounds (%struct.texture* @t_img, i64 0, i32 1), align 8, !tbaa !24
; CHECK: %16 = getelementptr inbounds %struct.texture addrspace(1)* @t_img{{[0-9]*}}, i64 0, i32 1
; CHECK-NEXT: %17 = load float* addrspace(1)* %16, align 8, !tbaa !24

  %17 = icmp sgt i32 %1, 0
  br i1 %17, label %.lr.ph7.i.split.us, label %.loopexit.i.preheader

.loopexit.i.preheader:                            ; preds = %.lr.ph7.i
  %backedge.overflow = icmp eq i32 %0, 0
  br i1 %backedge.overflow, label %.loopexit.i.preheader17, label %overflow.checked

.loopexit.i.preheader17:                          ; preds = %middle.block, %.loopexit.i.preheader
  %el_i.03.i.ph = phi i32 [ 0, %.loopexit.i.preheader ], [ %resume.val, %middle.block ]
  %18 = sub i32 %0, %el_i.03.i.ph
  %xtraiter = and i32 %18, 7
  %lcmp.mod = icmp ne i32 %xtraiter, 0
  %lcmp.overflow = icmp eq i32 %18, 0
  %lcmp.or = or i1 %lcmp.overflow, %lcmp.mod
  br i1 %lcmp.or, label %unr.cmp44, label %.loopexit.i.preheader17.split

unr.cmp44:                                        ; preds = %.loopexit.i.preheader17
  %un.tmp45 = icmp eq i32 %xtraiter, 1
  br i1 %un.tmp45, label %.loopexit.i.unr41, label %unr.cmp39

unr.cmp39:                                        ; preds = %unr.cmp44
  %un.tmp40 = icmp eq i32 %xtraiter, 2
  br i1 %un.tmp40, label %.loopexit.i.unr36, label %unr.cmp34

unr.cmp34:                                        ; preds = %unr.cmp39
  %un.tmp35 = icmp eq i32 %xtraiter, 3
  br i1 %un.tmp35, label %.loopexit.i.unr31, label %unr.cmp29

unr.cmp29:                                        ; preds = %unr.cmp34
  %un.tmp30 = icmp eq i32 %xtraiter, 4
  br i1 %un.tmp30, label %.loopexit.i.unr26, label %unr.cmp24

unr.cmp24:                                        ; preds = %unr.cmp29
  %un.tmp25 = icmp eq i32 %xtraiter, 5
  br i1 %un.tmp25, label %.loopexit.i.unr21, label %unr.cmp

unr.cmp:                                          ; preds = %unr.cmp24
  %un.tmp = icmp eq i32 %xtraiter, 6
  br i1 %un.tmp, label %.loopexit.i.unr19, label %.loopexit.i.unr

.loopexit.i.unr:                                  ; preds = %unr.cmp
  %19 = add nsw i32 %el_i.03.i.ph, 1
  %exitcond.unr = icmp eq i32 %19, %0
  br label %.loopexit.i.unr19

.loopexit.i.unr19:                                ; preds = %.loopexit.i.unr, %unr.cmp
  %el_i.03.i.unr = phi i32 [ %19, %.loopexit.i.unr ], [ %el_i.03.i.ph, %unr.cmp ]
  %20 = add nsw i32 %el_i.03.i.unr, 1
  %exitcond.unr20 = icmp eq i32 %20, %0
  br label %.loopexit.i.unr21

.loopexit.i.unr21:                                ; preds = %.loopexit.i.unr19, %unr.cmp24
  %el_i.03.i.unr22 = phi i32 [ %20, %.loopexit.i.unr19 ], [ %el_i.03.i.ph, %unr.cmp24 ]
  %21 = add nsw i32 %el_i.03.i.unr22, 1
  %exitcond.unr23 = icmp eq i32 %21, %0
  br label %.loopexit.i.unr26

.loopexit.i.unr26:                                ; preds = %.loopexit.i.unr21, %unr.cmp29
  %el_i.03.i.unr27 = phi i32 [ %21, %.loopexit.i.unr21 ], [ %el_i.03.i.ph, %unr.cmp29 ]
  %22 = add nsw i32 %el_i.03.i.unr27, 1
  %exitcond.unr28 = icmp eq i32 %22, %0
  br label %.loopexit.i.unr31

.loopexit.i.unr31:                                ; preds = %.loopexit.i.unr26, %unr.cmp34
  %el_i.03.i.unr32 = phi i32 [ %22, %.loopexit.i.unr26 ], [ %el_i.03.i.ph, %unr.cmp34 ]
  %23 = add nsw i32 %el_i.03.i.unr32, 1
  %exitcond.unr33 = icmp eq i32 %23, %0
  br label %.loopexit.i.unr36

.loopexit.i.unr36:                                ; preds = %.loopexit.i.unr31, %unr.cmp39
  %el_i.03.i.unr37 = phi i32 [ %23, %.loopexit.i.unr31 ], [ %el_i.03.i.ph, %unr.cmp39 ]
  %24 = add nsw i32 %el_i.03.i.unr37, 1
  %exitcond.unr38 = icmp eq i32 %24, %0
  br label %.loopexit.i.unr41

.loopexit.i.unr41:                                ; preds = %.loopexit.i.unr36, %unr.cmp44
  %el_i.03.i.unr42 = phi i32 [ %24, %.loopexit.i.unr36 ], [ %el_i.03.i.ph, %unr.cmp44 ]
  %25 = add nsw i32 %el_i.03.i.unr42, 1
  %exitcond.unr43 = icmp eq i32 %25, %0
  br label %.loopexit.i.preheader17.split

.loopexit.i.preheader17.split:                    ; preds = %.loopexit.i.unr41, %.loopexit.i.preheader17
  %el_i.03.i.unr46 = phi i32 [ %el_i.03.i.ph, %.loopexit.i.preheader17 ], [ %25, %.loopexit.i.unr41 ]
  %26 = icmp ult i32 %18, 8
  br i1 %26, label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18", label %.loopexit.i.preheader17.split.split

.loopexit.i.preheader17.split.split:              ; preds = %.loopexit.i.preheader17.split
  br label %.loopexit.i

overflow.checked:                                 ; preds = %.loopexit.i.preheader
  %n.vec = and i32 %0, -32
  %cmp.zero = icmp eq i32 %n.vec, 0
  br i1 %cmp.zero, label %middle.block, label %vector.body.preheader

vector.body.preheader:                            ; preds = %overflow.checked
  %27 = lshr i32 %0, 5
  %28 = mul i32 %27, 32
  %29 = add i32 %28, -32
  %30 = lshr i32 %29, 5
  %31 = add i32 %30, 1
  %xtraiter47 = and i32 %31, 7
  %lcmp.mod48 = icmp ne i32 %xtraiter47, 0
  %lcmp.overflow49 = icmp eq i32 %31, 0
  %lcmp.or50 = or i1 %lcmp.overflow49, %lcmp.mod48
  br i1 %lcmp.or50, label %unr.cmp78, label %vector.body.preheader.split

unr.cmp78:                                        ; preds = %vector.body.preheader
  %un.tmp79 = icmp eq i32 %xtraiter47, 1
  br i1 %un.tmp79, label %vector.body.unr75, label %unr.cmp73

unr.cmp73:                                        ; preds = %unr.cmp78
  %un.tmp74 = icmp eq i32 %xtraiter47, 2
  br i1 %un.tmp74, label %vector.body.unr70, label %unr.cmp68

unr.cmp68:                                        ; preds = %unr.cmp73
  %un.tmp69 = icmp eq i32 %xtraiter47, 3
  br i1 %un.tmp69, label %vector.body.unr65, label %unr.cmp63

unr.cmp63:                                        ; preds = %unr.cmp68
  %un.tmp64 = icmp eq i32 %xtraiter47, 4
  br i1 %un.tmp64, label %vector.body.unr60, label %unr.cmp58

unr.cmp58:                                        ; preds = %unr.cmp63
  %un.tmp59 = icmp eq i32 %xtraiter47, 5
  br i1 %un.tmp59, label %vector.body.unr55, label %unr.cmp53

unr.cmp53:                                        ; preds = %unr.cmp58
  %un.tmp54 = icmp eq i32 %xtraiter47, 6
  br i1 %un.tmp54, label %vector.body.unr51, label %vector.body.unr

vector.body.unr:                                  ; preds = %unr.cmp53
  %index.next.unr = add i32 0, 32
  %32 = icmp eq i32 %index.next.unr, %n.vec
  br label %vector.body.unr51

vector.body.unr51:                                ; preds = %vector.body.unr, %unr.cmp53
  %index.unr = phi i32 [ %index.next.unr, %vector.body.unr ], [ 0, %unr.cmp53 ]
  %index.next.unr52 = add i32 %index.unr, 32
  %33 = icmp eq i32 %index.next.unr52, %n.vec
  br label %vector.body.unr55

vector.body.unr55:                                ; preds = %vector.body.unr51, %unr.cmp58
  %index.unr56 = phi i32 [ %index.next.unr52, %vector.body.unr51 ], [ 0, %unr.cmp58 ]
  %index.next.unr57 = add i32 %index.unr56, 32
  %34 = icmp eq i32 %index.next.unr57, %n.vec
  br label %vector.body.unr60

vector.body.unr60:                                ; preds = %vector.body.unr55, %unr.cmp63
  %index.unr61 = phi i32 [ %index.next.unr57, %vector.body.unr55 ], [ 0, %unr.cmp63 ]
  %index.next.unr62 = add i32 %index.unr61, 32
  %35 = icmp eq i32 %index.next.unr62, %n.vec
  br label %vector.body.unr65

vector.body.unr65:                                ; preds = %vector.body.unr60, %unr.cmp68
  %index.unr66 = phi i32 [ %index.next.unr62, %vector.body.unr60 ], [ 0, %unr.cmp68 ]
  %index.next.unr67 = add i32 %index.unr66, 32
  %36 = icmp eq i32 %index.next.unr67, %n.vec
  br label %vector.body.unr70

vector.body.unr70:                                ; preds = %vector.body.unr65, %unr.cmp73
  %index.unr71 = phi i32 [ %index.next.unr67, %vector.body.unr65 ], [ 0, %unr.cmp73 ]
  %index.next.unr72 = add i32 %index.unr71, 32
  %37 = icmp eq i32 %index.next.unr72, %n.vec
  br label %vector.body.unr75

vector.body.unr75:                                ; preds = %vector.body.unr70, %unr.cmp78
  %index.unr76 = phi i32 [ %index.next.unr72, %vector.body.unr70 ], [ 0, %unr.cmp78 ]
  %index.next.unr77 = add i32 %index.unr76, 32
  %38 = icmp eq i32 %index.next.unr77, %n.vec
  br label %vector.body.preheader.split

vector.body.preheader.split:                      ; preds = %vector.body.unr75, %vector.body.preheader
  %index.unr80 = phi i32 [ 0, %vector.body.preheader ], [ %index.next.unr77, %vector.body.unr75 ]
  %39 = icmp ult i32 %31, 8
  br i1 %39, label %middle.block.loopexit, label %vector.body.preheader.split.split

vector.body.preheader.split.split:                ; preds = %vector.body.preheader.split
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.body.preheader.split.split
  %index = phi i32 [ %index.unr80, %vector.body.preheader.split.split ], [ %index.next.7, %vector.body ]
  %index.next = add i32 %index, 32
  %index.next.1 = add i32 %index.next, 32
  %index.next.2 = add i32 %index.next.1, 32
  %index.next.3 = add i32 %index.next.2, 32
  %index.next.4 = add i32 %index.next.3, 32
  %index.next.5 = add i32 %index.next.4, 32
  %index.next.6 = add i32 %index.next.5, 32
  %index.next.7 = add i32 %index.next.6, 32
  %40 = icmp eq i32 %index.next.7, %n.vec
  br i1 %40, label %middle.block.loopexit.unr-lcssa, label %vector.body, !llvm.loop !27

middle.block.loopexit.unr-lcssa:                  ; preds = %vector.body
  br label %middle.block.loopexit

middle.block.loopexit:                            ; preds = %middle.block.loopexit.unr-lcssa, %vector.body.preheader.split
  br label %middle.block

middle.block:                                     ; preds = %middle.block.loopexit, %overflow.checked
  %resume.val = phi i32 [ 0, %overflow.checked ], [ %n.vec, %middle.block.loopexit ]
  %cmp.n = icmp eq i32 %resume.val, %0
  br i1 %cmp.n, label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit", label %.loopexit.i.preheader17

.lr.ph7.i.split.us:                               ; preds = %.lr.ph7.i
  %41 = sdiv i32 %1, 2
  %42 = sub nsw i32 %14, %41
  %43 = sdiv i32 %0, 2
  %44 = sub nsw i32 %13, %43
  %45 = zext i32 %42 to i64
  %46 = zext i32 %44 to i64
  br label %47

; <label>:47                                      ; preds = %.loopexit.i.us, %.lr.ph7.i.split.us
  %indvars.iv9 = phi i64 [ %indvars.iv.next10, %.loopexit.i.us ], [ 0, %.lr.ph7.i.split.us ]
  %max.06.i.us = phi float [ %max.3.i.us, %.loopexit.i.us ], [ 0.000000e+00, %.lr.ph7.i.split.us ]
  %48 = add nsw i64 %indvars.iv9, %46
  %49 = trunc i64 %48 to i32
  %50 = icmp sgt i32 %49, -1
  %51 = icmp slt i32 %49, %2
  %or.cond.i.us = and i1 %50, %51
  br i1 %or.cond.i.us, label %.lr.ph.i.preheader.us, label %.loopexit.i.us

.lr.ph.i.us:                                      ; preds = %.lr.ph.i.preheader.us, %69
  %indvars.iv = phi i64 [ 0, %.lr.ph.i.preheader.us ], [ %indvars.iv.next, %69 ]
  %max.12.i.us = phi float [ %max.06.i.us, %.lr.ph.i.preheader.us ], [ %max.2.i.us, %69 ]
  %52 = add nsw i64 %indvars.iv, %45
  %53 = trunc i64 %52 to i32
  %54 = icmp sgt i32 %53, -1
  %55 = icmp slt i32 %53, %3
  %or.cond.us = and i1 %54, %55
  br i1 %or.cond.us, label %56, label %69

; <label>:56                                      ; preds = %.lr.ph.i.us
  %57 = add nsw i64 %indvars.iv, %72
  %58 = getelementptr inbounds float* %4, i64 %57
  %59 = load float* %58, align 4, !tbaa !30
  %60 = fcmp une float %59, 0.000000e+00
  br i1 %60, label %61, label %69

; <label>:61                                      ; preds = %56
  %62 = mul nsw i32 %53, %2
  %63 = add nsw i32 %62, %49
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds float* %16, i64 %64
  %66 = load float* %65, align 4, !tbaa !30
  %67 = fcmp ogt float %66, %max.12.i.us
  br i1 %67, label %68, label %69

; <label>:68                                      ; preds = %61
  br label %69

; <label>:69                                      ; preds = %68, %61, %56, %.lr.ph.i.us
  %max.2.i.us = phi float [ %66, %68 ], [ %max.12.i.us, %61 ], [ %max.12.i.us, %56 ], [ %max.12.i.us, %.lr.ph.i.us ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond8 = icmp eq i32 %lftr.wideiv, %1
  br i1 %exitcond8, label %.loopexit.i.us.loopexit, label %.lr.ph.i.us

.loopexit.i.us.loopexit:                          ; preds = %69
  %max.2.i.us.lcssa = phi float [ %max.2.i.us, %69 ]
  br label %.loopexit.i.us

.loopexit.i.us:                                   ; preds = %.loopexit.i.us.loopexit, %47
  %max.3.i.us = phi float [ %max.06.i.us, %47 ], [ %max.2.i.us.lcssa, %.loopexit.i.us.loopexit ]
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1
  %lftr.wideiv11 = trunc i64 %indvars.iv.next10 to i32
  %exitcond12 = icmp eq i32 %lftr.wideiv11, %0
  br i1 %exitcond12, label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit", label %47

.lr.ph.i.preheader.us:                            ; preds = %47
  %70 = trunc i64 %indvars.iv9 to i32
  %71 = mul nsw i32 %70, %1
  %72 = sext i32 %71 to i64
  br label %.lr.ph.i.us

.loopexit.i:                                      ; preds = %.loopexit.i, %.loopexit.i.preheader17.split.split
  %el_i.03.i = phi i32 [ %el_i.03.i.unr46, %.loopexit.i.preheader17.split.split ], [ %80, %.loopexit.i ]
  %73 = add nsw i32 %el_i.03.i, 1
  %74 = add nsw i32 %73, 1
  %75 = add nsw i32 %74, 1
  %76 = add nsw i32 %75, 1
  %77 = add nsw i32 %76, 1
  %78 = add nsw i32 %77, 1
  %79 = add nsw i32 %78, 1
  %80 = add nsw i32 %79, 1
  %exitcond.7 = icmp eq i32 %80, %0
  br i1 %exitcond.7, label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18.unr-lcssa", label %.loopexit.i, !llvm.loop !32

"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit": ; preds = %.loopexit.i.us
  %max.3.i.us.lcssa = phi float [ %max.3.i.us, %.loopexit.i.us ]
  br label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit"

"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18.unr-lcssa": ; preds = %.loopexit.i
  br label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18"

"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18": ; preds = %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18.unr-lcssa", %.loopexit.i.preheader17.split
  br label %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit"

"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit": ; preds = %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18", %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit", %middle.block, %6
  %max.0.lcssa.i = phi float [ 0.000000e+00, %6 ], [ 0.000000e+00, %middle.block ], [ %max.3.i.us.lcssa, %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit" ], [ 0.000000e+00, %"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_ENK3$_1clEN2hc11tiled_indexILi3EEE.exit.loopexit18" ]
  %81 = mul nsw i32 %13, %3
  %82 = add nsw i32 %81, %14
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds float* %5, i64 %83
  store float %max.0.lcssa.i, float* %84, align 4, !tbaa !30
  ret void
}

; Function Attrs: nounwind uwtable
define internal spir_kernel void @"_ZZ12GICOV_kernelRK18grid_launch_parm_siPfS2_S2_PiS3_EN3$_019__cxxamp_trampolineES3_S3_iS2_S2_S2_"(i32*, i32*, i32, float*, float*, float*) #4 align 2 {
  %7 = tail call i64 @amp_get_local_id(i32 2) #9
  %8 = tail call i64 @amp_get_group_id(i32 2) #9
  %9 = add nsw i64 %8, 22
  %10 = trunc i64 %9 to i32
  %11 = add nsw i64 %7, 22
  %12 = trunc i64 %11 to i32
; CHECK: %12 = trunc i64 %11 to i32
  %13 = load float** getelementptr inbounds (%struct.texture* @t_grad_x, i64 0, i32 1), align 8, !tbaa !24
; CHECK: %13 = getelementptr inbounds %struct.texture addrspace(1)* @t_grad_x{{[0-9]}}, i64 0, i32 1
; CHECK-NOT: %13 = load float** getelementptr inbounds (%struct.texture* @t_grad_x, i64 0, i32 1), align 8, !tbaa !24
; CHECK-NEXT: %14 = load float* addrspace(1)* %13, align 8, !tbaa !24
  %14 = load float** getelementptr inbounds (%struct.texture* @t_grad_y, i64 0, i32 1), align 8, !tbaa !24
; CHECK: %15 = getelementptr inbounds %struct.texture addrspace(1)* @t_grad_y{{[0-9]}}, i64 0, i32 1
; CHECK-NOT: %14 = load float** getelementptr inbounds (%struct.texture* @t_grad_y, i64 0, i32 1), align 8, !tbaa !24
; CHECK-NEXT: %16 = load float* addrspace(1)* %15, align 8, !tbaa !24
  br label %.preheader.i

.preheader.i:                                     ; preds = %47, %6
  %indvars.iv8.i = phi i64 [ 0, %6 ], [ %indvars.iv.next9.i, %47 ]
  %max_GICOV.05.i = phi float [ 0.000000e+00, %6 ], [ %max_GICOV.1.i, %47 ]
  %15 = mul nsw i64 %indvars.iv8.i, 150
  br label %16

; <label>:16                                      ; preds = %16, %.preheader.i
  %indvars.iv.i = phi i64 [ 0, %.preheader.i ], [ %indvars.iv.next.i, %16 ]
  %sum.03.i = phi float [ 0.000000e+00, %.preheader.i ], [ %38, %16 ]
  %mean.02.i = phi float [ 0.000000e+00, %.preheader.i ], [ %43, %16 ]
  %M2.01.i = phi float [ 0.000000e+00, %.preheader.i ], [ %46, %16 ]
  %17 = add nsw i64 %indvars.iv.i, %15
  %18 = getelementptr inbounds i32* %0, i64 %17
  %19 = load i32* %18, align 4, !tbaa !33
  %20 = add nsw i32 %19, %12
  %21 = getelementptr inbounds i32* %1, i64 %17
  %22 = load i32* %21, align 4, !tbaa !33
  %23 = add nsw i32 %22, %10
  %24 = mul nsw i32 %23, %2
  %25 = add nsw i32 %20, %24
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds float* %13, i64 %26
  %28 = load float* %27, align 4, !tbaa !30
  %29 = getelementptr inbounds float* %3, i64 %indvars.iv.i
  %30 = load float* %29, align 4, !tbaa !30
  %31 = fmul float %28, %30
  %32 = getelementptr inbounds float* %14, i64 %26
  %33 = load float* %32, align 4, !tbaa !30
  %34 = getelementptr inbounds float* %4, i64 %indvars.iv.i
  %35 = load float* %34, align 4, !tbaa !30
  %36 = fmul float %33, %35
  %37 = fadd float %31, %36
  %38 = fadd float %sum.03.i, %37
  %39 = fsub float %37, %mean.02.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %40 = trunc i64 %indvars.iv.next.i to i32
  %41 = sitofp i32 %40 to float
  %42 = fdiv float %39, %41
  %43 = fadd float %mean.02.i, %42
  %44 = fsub float %37, %43
  %45 = fmul float %39, %44
  %46 = fadd float %M2.01.i, %45
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, 150
  br i1 %exitcond.i, label %47, label %16

; <label>:47                                      ; preds = %16
  %.lcssa10 = phi float [ %46, %16 ]
  %.lcssa = phi float [ %38, %16 ]
  %48 = fdiv float %.lcssa, 1.500000e+02
  %49 = fdiv float %.lcssa10, 1.490000e+02
  %50 = fmul float %48, %48
  %51 = fdiv float %50, %49
  %52 = fcmp ogt float %51, %max_GICOV.05.i
  %max_GICOV.1.i = select i1 %52, float %51, float %max_GICOV.05.i
  %indvars.iv.next9.i = add nuw nsw i64 %indvars.iv8.i, 1
  %exitcond10.i = icmp eq i64 %indvars.iv.next9.i, 7
  br i1 %exitcond10.i, label %"_ZZ12GICOV_kernelRK18grid_launch_parm_siPfS2_S2_PiS3_ENK3$_0clEN2hc11tiled_indexILi3EEE.exit", label %.preheader.i

"_ZZ12GICOV_kernelRK18grid_launch_parm_siPfS2_S2_PiS3_ENK3$_0clEN2hc11tiled_indexILi3EEE.exit": ; preds = %47
  %max_GICOV.1.i.lcssa = phi float [ %max_GICOV.1.i, %47 ]
  %53 = mul nsw i32 %10, %2
  %54 = add nsw i32 %53, %12
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds float* %5, i64 %55
  store float %max_GICOV.1.i.lcssa, float* %56, align 4, !tbaa !30
  ret void
}

; Function Attrs: noduplicate
declare void @hc_barrier(i32) #7

declare float @opencl_fabs(float) #6

declare float @opencl_atan(float) #6

; Function Attrs: uwtable
define internal spir_kernel void @"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifEN3$_019__cxxamp_trampolineES3_S3_S4_S4_fifff"(float**, float**, i32*, i32*, float, i32, float, float, float) #8 align 2 {
  %10 = tail call i64 @amp_get_local_id(i32 2) #9
  %11 = tail call i64 @amp_get_group_id(i32 2) #9
  %sext.i = shl i64 %11, 32
  %12 = ashr exact i64 %sext.i, 32
  %13 = getelementptr inbounds float** %0, i64 %12
  %14 = load float** %13, align 8, !tbaa !35
  %15 = getelementptr inbounds float** %1, i64 %12
  %16 = load float** %15, align 8, !tbaa !35
  %17 = getelementptr inbounds i32* %2, i64 %12
  %18 = load i32* %17, align 4, !tbaa !33
  %19 = getelementptr inbounds i32* %3, i64 %12
  %20 = load i32* %19, align 4, !tbaa !33
  %21 = mul nsw i32 %20, %18
  %22 = add nsw i32 %21, 319
  %23 = sdiv i32 %22, 320
  %24 = trunc i64 %10 to i32
  %25 = icmp sgt i32 %21, 0
  br i1 %25, label %.lr.ph23.i.preheader, label %._crit_edge24.i

.lr.ph23.i.preheader:                             ; preds = %9
  br label %.lr.ph23.i

.lr.ph23.i:                                       ; preds = %38, %.lr.ph23.i.preheader
  %thread_block.021.i = phi i32 [ %39, %38 ], [ 0, %.lr.ph23.i.preheader ]
  %26 = mul nsw i32 %thread_block.021.i, 320
  %27 = add nsw i32 %26, %24
  %28 = sdiv i32 %27, %20
  %29 = icmp slt i32 %28, %18
  br i1 %29, label %30, label %38

; <label>:30                                      ; preds = %.lr.ph23.i
  %31 = srem i32 %27, %20
  %32 = mul nsw i32 %28, %20
  %33 = add nsw i32 %31, %32
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds float* %14, i64 %34
  %36 = load float* %35, align 4, !tbaa !30
  %37 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %34
  store float %36, float* %37, align 4, !tbaa !30
  br label %38

; <label>:38                                      ; preds = %30, %.lr.ph23.i
  %39 = add nsw i32 %thread_block.021.i, 1
  %40 = icmp slt i32 %39, %23
  br i1 %40, label %.lr.ph23.i, label %._crit_edge24.i.loopexit

._crit_edge24.i.loopexit:                         ; preds = %38
  %.lcssa15 = phi i32 [ %28, %38 ]
  br label %._crit_edge24.i

._crit_edge24.i:                                  ; preds = %._crit_edge24.i.loopexit, %9
  %i.0.lcssa.i = phi i32 [ undef, %9 ], [ %.lcssa15, %._crit_edge24.i.loopexit ]
  tail call void @hc_barrier(i32 1) #10
  %41 = icmp eq i64 %10, 0
  br i1 %41, label %42, label %43

; <label>:42                                      ; preds = %._crit_edge24.i
  store i1 false, i1* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged", align 4
  br label %43

; <label>:43                                      ; preds = %42, %._crit_edge24.i
  tail call void @hc_barrier(i32 1) #10
  %44 = sitofp i32 %20 to float
  %45 = fdiv float 1.000000e+00, %44
  %46 = srem i32 320, %20
  %47 = fdiv float 1.000000e+00, %4
  %.b15.i = load i1* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged", align 4
  br i1 %.b15.i, label %.critedge.preheader.i, label %.lr.ph19.i

.lr.ph19.i:                                       ; preds = %43
  %48 = srem i32 %24, %20
  %49 = sub nsw i32 %48, %46
  %sext1.i = shl i64 %10, 32
  %50 = ashr exact i64 %sext1.i, 32
  %51 = getelementptr inbounds [320 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer", i64 0, i64 %50
  %52 = icmp sgt i32 %24, 255
  %sext2.i = add i64 %sext1.i, -1099511627776
  %53 = ashr exact i64 %sext2.i, 32
  %54 = getelementptr inbounds [320 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer", i64 0, i64 %53
  %55 = icmp eq i32 %24, 0
  %56 = sitofp i32 %21 to float
  %57 = add nsw i32 %18, -1
  %58 = add nsw i32 %20, -1
  %59 = add nsw i32 %23, -1
  %60 = fsub float %7, %6
  %61 = fadd float %6, %7
  %62 = fsub float -0.000000e+00, %7
  %63 = fsub float %62, %6
  %64 = fsub float %6, %7
  br label %65

; <label>:65                                      ; preds = %270, %.lr.ph19.i
  %i.117.i = phi i32 [ %i.0.lcssa.i, %.lr.ph19.i ], [ %i.2.lcssa.i, %270 ]
  %iterations.016.i = phi i32 [ 0, %.lr.ph19.i ], [ %271, %270 ]
  %66 = icmp slt i32 %iterations.016.i, %5
  br i1 %66, label %67, label %.critedge.preheader.i.loopexit

.critedge.preheader.i.loopexit:                   ; preds = %270, %65
  br label %.critedge.preheader.i

.critedge.preheader.i:                            ; preds = %.critedge.preheader.i.loopexit, %43
  br i1 %25, label %.lr.ph.i.preheader, label %"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEE.exit"

.lr.ph.i.preheader:                               ; preds = %.critedge.preheader.i
  br label %.lr.ph.i

; <label>:67                                      ; preds = %65
  br i1 %25, label %.lr.ph12.i.preheader, label %._crit_edge.i

.lr.ph12.i.preheader:                             ; preds = %67
  br label %.lr.ph12.i

.lr.ph12.i:                                       ; preds = %241, %.lr.ph12.i.preheader
  %thread_block.111.i = phi i32 [ %245, %241 ], [ 0, %.lr.ph12.i.preheader ]
  %i.210.i = phi i32 [ %72, %241 ], [ %i.117.i, %.lr.ph12.i.preheader ]
  %j.09.i = phi i32 [ %..i, %241 ], [ %49, %.lr.ph12.i.preheader ]
  %total_diff.08.i = phi float [ %244, %241 ], [ 0.000000e+00, %.lr.ph12.i.preheader ]
  %68 = mul nsw i32 %thread_block.111.i, 320
  %69 = add nsw i32 %68, %24
  %70 = sitofp i32 %69 to float
  %71 = fmul float %45, %70
  %72 = fptosi float %71 to i32
  %73 = add nsw i32 %j.09.i, %46
  %74 = icmp slt i32 %73, %20
  %75 = select i1 %74, i32 0, i32 %20
  %..i = sub nsw i32 %73, %75
  %76 = icmp slt i32 %72, %18
  br i1 %76, label %77, label %223

; <label>:77                                      ; preds = %.lr.ph12.i
  %78 = icmp eq i32 %72, 0
  %79 = add nsw i32 %72, -1
  %.5.i = select i1 %78, i32 0, i32 %79
  %80 = icmp eq i32 %72, %57
  %81 = add nsw i32 %72, 1
  %82 = select i1 %80, i32 %57, i32 %81
  %83 = icmp eq i32 %73, %75
  %84 = add nsw i32 %..i, -1
  %.6.i = select i1 %83, i32 0, i32 %84
  %85 = icmp eq i32 %..i, %58
  %86 = add nsw i32 %..i, 1
  %87 = select i1 %85, i32 %58, i32 %86
  %88 = mul nsw i32 %72, %20
  %89 = add nsw i32 %88, %..i
  %90 = sext i32 %89 to i64
  %91 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %90
  %92 = load float* %91, align 4, !tbaa !30
  %93 = mul nsw i32 %.5.i, %20
  %94 = add nsw i32 %93, %..i
  %95 = sext i32 %94 to i64
  %96 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %95
  %97 = load float* %96, align 4, !tbaa !30
  %98 = fsub float %97, %92
  %99 = mul nsw i32 %82, %20
  %100 = add nsw i32 %99, %..i
  %101 = sext i32 %100 to i64
  %102 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %101
  %103 = load float* %102, align 4, !tbaa !30
  %104 = fsub float %103, %92
  %105 = add nsw i32 %.6.i, %88
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %106
  %108 = load float* %107, align 4, !tbaa !30
  %109 = fsub float %108, %92
  %110 = add nsw i32 %87, %88
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %111
  %113 = load float* %112, align 4, !tbaa !30
  %114 = fsub float %113, %92
  %115 = add nsw i32 %93, %87
  %116 = sext i32 %115 to i64
  %117 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %116
  %118 = load float* %117, align 4, !tbaa !30
  %119 = fsub float %118, %92
  %120 = add nsw i32 %99, %87
  %121 = sext i32 %120 to i64
  %122 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %121
  %123 = load float* %122, align 4, !tbaa !30
  %124 = fsub float %123, %92
  %125 = add nsw i32 %93, %.6.i
  %126 = sext i32 %125 to i64
  %127 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %126
  %128 = load float* %127, align 4, !tbaa !30
  %129 = fsub float %128, %92
  %130 = add nsw i32 %99, %.6.i
  %131 = sext i32 %130 to i64
  %132 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %131
  %133 = load float* %132, align 4, !tbaa !30
  %134 = fsub float %133, %92
  %135 = fmul float %98, %6
  %136 = fmul float %47, %135
  %137 = fsub float -0.000000e+00, %136
  %138 = tail call float @opencl_atan(float %137) #11
  %139 = fpext float %138 to double
  %140 = fmul double %139, 0x3FD45F318E7ADAF5
  %141 = fadd double %140, 5.000000e-01
  %142 = fptrunc double %141 to float
  %143 = fmul float %104, %6
  %144 = fmul float %47, %143
  %145 = tail call float @opencl_atan(float %144) #11
  %146 = fpext float %145 to double
  %147 = fmul double %146, 0x3FD45F318E7ADAF5
  %148 = fadd double %147, 5.000000e-01
  %149 = fptrunc double %148 to float
  %150 = fmul float %109, %7
  %151 = fmul float %47, %150
  %152 = fsub float -0.000000e+00, %151
  %153 = tail call float @opencl_atan(float %152) #11
  %154 = fpext float %153 to double
  %155 = fmul double %154, 0x3FD45F318E7ADAF5
  %156 = fadd double %155, 5.000000e-01
  %157 = fptrunc double %156 to float
  %158 = fmul float %114, %7
  %159 = fmul float %47, %158
  %160 = tail call float @opencl_atan(float %159) #11
  %161 = fpext float %160 to double
  %162 = fmul double %161, 0x3FD45F318E7ADAF5
  %163 = fadd double %162, 5.000000e-01
  %164 = fptrunc double %163 to float
  %165 = fmul float %60, %119
  %166 = fmul float %47, %165
  %167 = tail call float @opencl_atan(float %166) #11
  %168 = fpext float %167 to double
  %169 = fmul double %168, 0x3FD45F318E7ADAF5
  %170 = fadd double %169, 5.000000e-01
  %171 = fptrunc double %170 to float
  %172 = fmul float %61, %124
  %173 = fmul float %47, %172
  %174 = tail call float @opencl_atan(float %173) #11
  %175 = fpext float %174 to double
  %176 = fmul double %175, 0x3FD45F318E7ADAF5
  %177 = fadd double %176, 5.000000e-01
  %178 = fptrunc double %177 to float
  %179 = fmul float %63, %129
  %180 = fmul float %47, %179
  %181 = tail call float @opencl_atan(float %180) #11
  %182 = fpext float %181 to double
  %183 = fmul double %182, 0x3FD45F318E7ADAF5
  %184 = fadd double %183, 5.000000e-01
  %185 = fptrunc double %184 to float
  %186 = fmul float %64, %134
  %187 = fmul float %47, %186
  %188 = tail call float @opencl_atan(float %187) #11
  %189 = fpext float %188 to double
  %190 = fmul double %189, 0x3FD45F318E7ADAF5
  %191 = fadd double %190, 5.000000e-01
  %192 = fptrunc double %191 to float
  %193 = fpext float %92 to double
  %194 = fmul float %98, %142
  %195 = fmul float %104, %149
  %196 = fadd float %194, %195
  %197 = fmul float %109, %157
  %198 = fadd float %196, %197
  %199 = fmul float %114, %164
  %200 = fadd float %198, %199
  %201 = fmul float %119, %171
  %202 = fadd float %200, %201
  %203 = fmul float %124, %178
  %204 = fadd float %202, %203
  %205 = fmul float %129, %185
  %206 = fadd float %204, %205
  %207 = fmul float %134, %192
  %208 = fadd float %206, %207
  %209 = fpext float %208 to double
  %210 = fmul double %209, 1.000000e-01
  %211 = fadd double %193, %210
  %212 = fptrunc double %211 to float
  %213 = getelementptr inbounds float* %16, i64 %90
  %214 = load float* %213, align 4, !tbaa !30
  %215 = fpext float %214 to double
  %216 = fmul double %215, 2.000000e-01
  %217 = fsub float %212, %214
  %218 = fpext float %217 to double
  %219 = fmul double %216, %218
  %220 = fpext float %212 to double
  %221 = fsub double %220, %219
  %222 = fptrunc double %221 to float
  br label %223

; <label>:223                                     ; preds = %77, %.lr.ph12.i
  %old_val.0.i = phi float [ %92, %77 ], [ 0.000000e+00, %.lr.ph12.i ]
  %new_val.0.i = phi float [ %222, %77 ], [ 0.000000e+00, %.lr.ph12.i ]
  %224 = icmp sgt i32 %thread_block.111.i, 0
  %225 = icmp slt i32 %i.210.i, %18
  %or.cond.i = and i1 %224, %225
  br i1 %or.cond.i, label %226, label %232

; <label>:226                                     ; preds = %223
  %227 = load float* %51, align 4, !tbaa !30
  %228 = mul nsw i32 %i.210.i, %20
  %229 = add nsw i32 %228, %j.09.i
  %230 = sext i32 %229 to i64
  %231 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %230
  store float %227, float* %231, align 4, !tbaa !30
  br label %232

; <label>:232                                     ; preds = %226, %223
  %233 = icmp slt i32 %thread_block.111.i, %59
  br i1 %233, label %234, label %235

; <label>:234                                     ; preds = %232
  store float %new_val.0.i, float* %51, align 4, !tbaa !30
  br label %241

; <label>:235                                     ; preds = %232
  br i1 %76, label %236, label %241

; <label>:236                                     ; preds = %235
  %237 = mul nsw i32 %72, %20
  %238 = add nsw i32 %237, %..i
  %239 = sext i32 %238 to i64
  %240 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %239
  store float %new_val.0.i, float* %240, align 4, !tbaa !30
  br label %241

; <label>:241                                     ; preds = %236, %235, %234
  %242 = fsub float %new_val.0.i, %old_val.0.i
  %243 = tail call float @opencl_fabs(float %242) #11
  %244 = fadd float %total_diff.08.i, %243
  tail call void @hc_barrier(i32 1) #10
  %245 = add nsw i32 %thread_block.111.i, 1
  %246 = icmp slt i32 %245, %23
  br i1 %246, label %.lr.ph12.i, label %._crit_edge.i.loopexit

._crit_edge.i.loopexit:                           ; preds = %241
  %.lcssa14 = phi float [ %244, %241 ]
  %.lcssa = phi i32 [ %72, %241 ]
  br label %._crit_edge.i

._crit_edge.i:                                    ; preds = %._crit_edge.i.loopexit, %67
  %i.2.lcssa.i = phi i32 [ %i.117.i, %67 ], [ %.lcssa, %._crit_edge.i.loopexit ]
  %total_diff.0.lcssa.i = phi float [ 0.000000e+00, %67 ], [ %.lcssa14, %._crit_edge.i.loopexit ]
  store float %total_diff.0.lcssa.i, float* %51, align 4, !tbaa !30
  tail call void @hc_barrier(i32 1) #10
  br i1 %52, label %247, label %251

; <label>:247                                     ; preds = %._crit_edge.i
  %248 = load float* %51, align 4, !tbaa !30
  %249 = load float* %54, align 4, !tbaa !30
  %250 = fadd float %248, %249
  store float %250, float* %54, align 4, !tbaa !30
  br label %251

; <label>:251                                     ; preds = %247, %._crit_edge.i
  tail call void @hc_barrier(i32 1) #10
  br label %252

; <label>:252                                     ; preds = %261, %251
  %th.014.i = phi i32 [ 128, %251 ], [ %262, %261 ]
  %253 = icmp slt i32 %24, %th.014.i
  br i1 %253, label %254, label %261

; <label>:254                                     ; preds = %252
  %255 = add nsw i32 %th.014.i, %24
  %256 = sext i32 %255 to i64
  %257 = getelementptr inbounds [320 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE6buffer", i64 0, i64 %256
  %258 = load float* %257, align 4, !tbaa !30
  %259 = load float* %51, align 4, !tbaa !30
  %260 = fadd float %258, %259
  store float %260, float* %51, align 4, !tbaa !30
  br label %261

; <label>:261                                     ; preds = %254, %252
  tail call void @hc_barrier(i32 1) #10
  %262 = sdiv i32 %th.014.i, 2
  %263 = icmp sgt i32 %th.014.i, 1
  br i1 %263, label %252, label %264

; <label>:264                                     ; preds = %261
  br i1 %55, label %265, label %270

; <label>:265                                     ; preds = %264
  %266 = load float* %51, align 4, !tbaa !30
  %267 = fdiv float %266, %56
  %268 = fcmp olt float %267, %8
  br i1 %268, label %269, label %270

; <label>:269                                     ; preds = %265
  store i1 true, i1* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged", align 4
  br label %270

; <label>:270                                     ; preds = %269, %265, %264
  tail call void @hc_barrier(i32 1) #10
  %271 = add nsw i32 %iterations.016.i, 1
  %.b.i = load i1* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE14cell_converged", align 4
  br i1 %.b.i, label %.critedge.preheader.i.loopexit, label %65

.lr.ph.i:                                         ; preds = %.critedge.i, %.lr.ph.i.preheader
  %thread_block.27.i = phi i32 [ %284, %.critedge.i ], [ 0, %.lr.ph.i.preheader ]
  %272 = mul nsw i32 %thread_block.27.i, 320
  %273 = add nsw i32 %272, %24
  %274 = sdiv i32 %273, %20
  %275 = icmp slt i32 %274, %18
  br i1 %275, label %276, label %.critedge.i

; <label>:276                                     ; preds = %.lr.ph.i
  %277 = srem i32 %273, %20
  %278 = mul nsw i32 %274, %20
  %279 = add nsw i32 %277, %278
  %280 = sext i32 %279 to i64
  %281 = getelementptr inbounds [3321 x float]* @"_ZZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEEE5IMGVF", i64 0, i64 %280
  %282 = load float* %281, align 4, !tbaa !30
  %283 = getelementptr inbounds float* %14, i64 %280
  store float %282, float* %283, align 4, !tbaa !30
  br label %.critedge.i

.critedge.i:                                      ; preds = %276, %.lr.ph.i
  %284 = add nsw i32 %thread_block.27.i, 1
  %285 = icmp slt i32 %284, %23
  br i1 %285, label %.lr.ph.i, label %"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEE.exit.loopexit"

"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEE.exit.loopexit": ; preds = %.critedge.i
  br label %"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEE.exit"

"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEE.exit": ; preds = %"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifENK3$_0clEN2hc11tiled_indexILi3EEE.exit.loopexit", %.critedge.preheader.i
  ret void
}

attributes #0 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { nobuiltin "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noduplicate "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nobuiltin nounwind readnone }
attributes #10 = { nobuiltin noduplicate }
attributes #11 = { nobuiltin }

!opencl.kernels = !{!0, !6, !12, !14}
!llvm.ident = !{!20, !20, !20, !20, !20}

!0 = metadata !{void (i64, i8*, i32)* @"_ZZ9hipMemsetEN3$_019__cxxamp_trampolineEmPvi", metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!1 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0}
!2 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none"}
!3 = metadata !{metadata !"kernel_arg_type", metadata !"size_t", metadata !"void*", metadata !"int"}
!4 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !""}
!5 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !""}
!6 = metadata !{void (i32, i32, i32, i32, float*, float*)* @"_ZZ13dilate_kernelRK18grid_launch_parm_siiiiPfS2_EN3$_119__cxxamp_trampolineEiiiiS2_S2_", metadata !7, metadata !8, metadata !9, metadata !10, metadata !11}
!7 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!8 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!9 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"float*", metadata !"float*"}
!10 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!11 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!12 = metadata !{void (i32*, i32*, i32, float*, float*, float*)* @"_ZZ12GICOV_kernelRK18grid_launch_parm_siPfS2_S2_PiS3_EN3$_019__cxxamp_trampolineES3_S3_iS2_S2_S2_", metadata !7, metadata !8, metadata !13, metadata !10, metadata !11}
!13 = metadata !{metadata !"kernel_arg_type", metadata !"int*", metadata !"int*", metadata !"int", metadata !"float*", metadata !"float*", metadata !"float*"}
!14 = metadata !{void (float**, float**, i32*, i32*, float, i32, float, float, float)* @"_ZZ12IMGVF_kernelRK18grid_launch_parm_sPPfS3_PiS4_fffifEN3$_019__cxxamp_trampolineES3_S3_S4_S4_fifff", metadata !15, metadata !16, metadata !17, metadata !18, metadata !19}
!15 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!16 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!17 = metadata !{metadata !"kernel_arg_type", metadata !"float **", metadata !"float **", metadata !"int*", metadata !"int*", metadata !"float", metadata !"int", metadata !"float", metadata !"float", metadata !"float"}
!18 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!19 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!20 = metadata !{metadata !"HCC clang version 3.5.0 (tags/RELEASE_350/final) (based on HCC 0.8.1542-c8cc1e2-348ebf7 LLVM 3.5.0svn)"}
!21 = metadata !{metadata !22, metadata !22, i64 0}
!22 = metadata !{metadata !"omnipotent char", metadata !23, i64 0}
!23 = metadata !{metadata !"Simple C/C++ TBAA"}
!24 = metadata !{metadata !25, metadata !26, i64 16}
!25 = metadata !{metadata !"_ZTS7textureIfLi1EL18hipTextureReadMode0EE", metadata !26, i64 16}
!26 = metadata !{metadata !"any pointer", metadata !22, i64 0}
!27 = metadata !{metadata !27, metadata !28, metadata !29}
!28 = metadata !{metadata !"llvm.loop.vectorize.width", i32 1}
!29 = metadata !{metadata !"llvm.loop.interleave.count", i32 1}
!30 = metadata !{metadata !31, metadata !31, i64 0}
!31 = metadata !{metadata !"float", metadata !22, i64 0}
!32 = metadata !{metadata !32, metadata !28, metadata !29}
!33 = metadata !{metadata !34, metadata !34, i64 0}
!34 = metadata !{metadata !"int", metadata !22, i64 0}
!35 = metadata !{metadata !26, metadata !26, i64 0}
