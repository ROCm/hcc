; ModuleID = 'hsail-amdgpu'

@"&hcc_static_group_segment_size" = addrspace(1) global i32 0, align 4
@"&hcc_dynamic_group_segment_size" = addrspace(1) global i32 0, align 4

; Function Attrs: alwaysinline nounwind readonly
define linkonce_odr spir_func i32 @__hsail_get_global_id(i32) #0 {
  %dispatch_ptr = call noalias nonnull dereferenceable(64) i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %size_xy_ptr = getelementptr inbounds i8, i8 addrspace(2)* %dispatch_ptr, i64 4
  %2 = bitcast i8 addrspace(2)* %size_xy_ptr to i32 addrspace(2)*
  %size_xy = load i32, i32 addrspace(2)* %2, align 4, !invariant.load !0
  switch i32 %0, label %22 [
    i32 0, label %3
    i32 1, label %9
    i32 2, label %15
  ]

; <label>:3:                                      ; preds = %1
  %4 = and i32 %size_xy, 65535
  %5 = call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = call i32 @llvm.amdgcn.workitem.id.x(), !range !1
  %7 = mul i32 %4, %5
  %8 = add i32 %6, %7
  ret i32 %8

; <label>:9:                                      ; preds = %1
  %10 = lshr i32 %size_xy, 16
  %11 = call i32 @llvm.amdgcn.workgroup.id.y()
  %12 = call i32 @llvm.amdgcn.workitem.id.y(), !range !1
  %13 = mul i32 %10, %11
  %14 = add i32 %12, %13
  ret i32 %14

; <label>:15:                                     ; preds = %1
  %size_z_ptr = getelementptr inbounds i8, i8 addrspace(2)* %dispatch_ptr, i64 8
  %16 = bitcast i8 addrspace(2)* %size_z_ptr to i32 addrspace(2)*
  %17 = load i32, i32 addrspace(2)* %16, align 4, !range !1, !invariant.load !0
  %18 = call i32 @llvm.amdgcn.workgroup.id.z()
  %19 = call i32 @llvm.amdgcn.workitem.id.z(), !range !1
  %20 = mul i32 %17, %18
  %21 = add i32 %19, %20
  ret i32 %21

; <label>:22:                                     ; preds = %1
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.z() #1

; Function Attrs: nounwind readnone
declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_abs_f32(float) #2 {
  %2 = call float @llvm.fabs.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.fabs.f32(float) #1

; Function Attrs: alwaysinline convergent nounwind
define linkonce_odr spir_func void @__hsail_barrier() #3 {
  tail call void @llvm.amdgcn.s.barrier() #4
  ret void
}

; Function Attrs: convergent nounwind
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_abs_f64(double) #2 {
  %2 = call double @llvm.fabs.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.fabs.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_ceil_f32(float) #2 {
  %2 = call float @llvm.ceil.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.ceil.f32(float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_ceil_f64(double) #2 {
  %2 = call double @llvm.ceil.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.ceil.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_copysign_f32(float, float) #2 {
  %3 = call float @llvm.copysign.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone
declare float @llvm.copysign.f32(float, float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_copysign_f64(double, double) #2 {
  %3 = call double @llvm.copysign.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone
declare double @llvm.copysign.f64(double, double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_floor_f32(float) #2 {
  %2 = call float @llvm.floor.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.floor.f32(float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_floor_f64(double) #2 {
  %2 = call double @llvm.floor.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.floor.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_fma_f32(float, float, float) #2 {
  %4 = call float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %4
}

; Function Attrs: nounwind readnone
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_fma_f64(double, double, double) #2 {
  %4 = call double @llvm.fma.f64(double %0, double %1, double %2)
  ret double %4
}

; Function Attrs: nounwind readnone
declare double @llvm.fma.f64(double, double, double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_max_f32(float, float) #2 {
  %3 = call float @llvm.maxnum.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_max_f64(double, double) #2 {
  %3 = call double @llvm.maxnum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone
declare double @llvm.maxnum.f64(double, double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_min_f32(float, float) #2 {
  %3 = call float @llvm.minnum.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone
declare float @llvm.minnum.f32(float, float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_min_f64(double, double) #2 {
  %3 = call double @llvm.minnum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone
declare double @llvm.minnum.f64(double, double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_nfma_f32(float, float, float) #2 {
  %4 = call float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %4
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_nfma_f64(double, double, double) #2 {
  %4 = call double @llvm.fma.f64(double %0, double %1, double %2)
  ret double %4
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_nrcp_f32(float) #2 {
  %2 = call float @llvm.amdgcn.rcp.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.rcp.f32(float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_nrsqrt_f32(float) #2 {
  %2 = call float @llvm.amdgcn.rsq.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.rsq.f32(float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_nrsqrt_f64(double) #2 {
  %2 = call double @llvm.amdgcn.rsq.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.amdgcn.rsq.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_nsqrt_f32(float) #2 {
  %2 = call float @llvm.sqrt.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hsail_round_f32(float) #5 {
  %2 = call float @llvm.round.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.round.f32(float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_round_f64(double) #2 {
  %2 = call double @llvm.round.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.round.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_sqrt_f32(float) #2 {
  %2 = call float @llvm.sqrt.f32(float %0)
  ret float %2
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_sqrt_f64(double) #2 {
  %2 = call double @llvm.sqrt.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hsail_trunc_f32(float) #2 {
  %2 = call float @llvm.trunc.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.trunc.f32(float) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hsail_trunc_f64(double) #2 {
  %2 = call double @llvm.trunc.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.trunc.f64(double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i32 @__hsail_class_f32(float, i32) #2 {
  %3 = call i1 @llvm.amdgcn.class.f32(float %0, i32 %1)
  %4 = sext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: nounwind readnone
declare i1 @llvm.amdgcn.class.f32(float, i32) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i32 @__hsail_class_f64(double, i32) #2 {
  %3 = call i1 @llvm.amdgcn.class.f64(double %0, i32 %1)
  %4 = sext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: nounwind readnone
declare i1 @llvm.amdgcn.class.f64(double, i32) #1

; Function Attrs: alwaysinline norecurse nounwind readnone
define linkonce_odr spir_func i32 @__hsail_mulhi_u32(i32, i32) #6 {
  %3 = zext i32 %0 to i64
  %4 = zext i32 %1 to i64
  %5 = mul nuw i64 %3, %4
  %6 = lshr i64 %5, 32
  %7 = trunc i64 %6 to i32
  ret i32 %7
}

; Function Attrs: alwaysinline norecurse nounwind readnone
define linkonce_odr spir_func i32 @__hsail_mad_u32(i32, i32, i32) #6 {
  %4 = mul i32 %0, %1
  %5 = add i32 %4, %2
  ret i32 %5
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i32 @__hsail_firstbit_u32(i32) #2 {
  %2 = call i32 @llvm.ctlz.i32(i32 %0, i1 true)
  %3 = icmp eq i32 %0, 0
  %4 = select i1 %3, i32 -1, i32 %2
  ret i32 %4
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) #1

; Function Attrs: norecurse nounwind readnone
define linkonce_odr spir_func i32 @__hsail_max_s32(i32, i32) #7 {
  %3 = icmp sgt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

; Function Attrs: alwaysinline norecurse nounwind readnone
define linkonce_odr spir_func i32 @__hsail_min_s32(i32, i32) #6 {
  %3 = icmp slt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

; Function Attrs: alwaysinline norecurse nounwind readnone
define linkonce_odr spir_func i32 @__hsail_bytealign_b32(i32, i32, i32) #6 {
  %4 = shl i32 %2, 3
  %5 = and i32 %4, 24
  %6 = zext i32 %5 to i64
  %7 = zext i32 %0 to i64
  %8 = zext i32 %1 to i64
  %9 = shl nuw i64 %8, 32
  %10 = or i64 %7, %9
  %11 = lshr i64 %10, %6
  %12 = trunc i64 %11 to i32
  ret i32 %12
}

; Function Attrs: alwaysinline norecurse nounwind readnone
define linkonce_odr spir_func float @__hsail_ftz_f32(float) #6 {
  %2 = bitcast float %0 to i32
  %3 = and i32 %2, 2139095040
  %4 = and i32 %2, 8388607
  %5 = icmp eq i32 %3, 0
  %6 = icmp ne i32 %4, 0
  %7 = and i1 %5, %6
  %8 = select i1 %7, float 0.000000e+00, float %0
  ret float %8
}

; Function Attrs: nounwind readnone
define linkonce_odr spir_func double @__hsail_fraction_f64(double) #1 {
  %2 = call double @llvm.floor.f64(double %0)
  %3 = fsub double %0, %2
  ret double %3
}

; Function Attrs: nounwind readnone
define linkonce_odr spir_func i32 @__hsail_get_local_id(i32) #1 {
  %2 = call i32 @llvm.amdgcn.workitem.id.x(), !range !1
  %3 = call i32 @llvm.amdgcn.workitem.id.y(), !range !1
  %4 = call i32 @llvm.amdgcn.workitem.id.z(), !range !1
  %5 = icmp eq i32 %0, 1
  %6 = select i1 %5, i32 %3, i32 %2
  %7 = icmp eq i32 %0, 2
  %8 = select i1 %7, i32 %4, i32 %6
  ret i32 %8
}

; Function Attrs: nounwind readnone
define linkonce_odr spir_func i32 @__hsail_get_group_id(i32) #1 {
  %2 = call i32 @llvm.amdgcn.workgroup.id.x()
  %3 = call i32 @llvm.amdgcn.workgroup.id.y()
  %4 = call i32 @llvm.amdgcn.workgroup.id.z()
  %5 = icmp eq i32 %0, 1
  %6 = select i1 %5, i32 %3, i32 %2
  %7 = icmp eq i32 %0, 2
  %8 = select i1 %7, i32 %4, i32 %6
  ret i32 %8
}

; Function Attrs: alwaysinline nounwind readonly
define linkonce_odr spir_func i32 @__hsail_currentworkgroup_size(i32) #0 {
  %dispatch_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %size_xy_ptr = getelementptr i8, i8 addrspace(2)* %dispatch_ptr, i64 4
  %2 = bitcast i8 addrspace(2)* %size_xy_ptr to i32 addrspace(2)*
  %size_xy = load i32, i32 addrspace(2)* %2, align 4, !invariant.load !0
  switch i32 %0, label %10 [
    i32 0, label %3
    i32 1, label %5
    i32 2, label %7
  ]

; <label>:3:                                      ; preds = %1
  %4 = and i32 %size_xy, 65535
  ret i32 %4

; <label>:5:                                      ; preds = %1
  %6 = lshr i32 %size_xy, 16
  ret i32 %6

; <label>:7:                                      ; preds = %1
  %size_z_ptr = getelementptr i8, i8 addrspace(2)* %dispatch_ptr, i64 8
  %8 = bitcast i8 addrspace(2)* %size_z_ptr to i32 addrspace(2)*
  %9 = load i32, i32 addrspace(2)* %8, align 4, !range !1, !invariant.load !0
  ret i32 %9

; <label>:10:                                     ; preds = %1
  ret i32 1
}

; Function Attrs: alwaysinline nounwind readonly
define linkonce_odr spir_func i32 @__hsail_get_global_size(i32) #0 {
  %dispatch_ptr = call noalias nonnull dereferenceable(64) i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  switch i32 %0, label %14 [
    i32 0, label %2
    i32 1, label %6
    i32 2, label %10
  ]

; <label>:2:                                      ; preds = %1
  %3 = getelementptr inbounds i8, i8 addrspace(2)* %dispatch_ptr, i64 12
  %4 = bitcast i8 addrspace(2)* %3 to i32 addrspace(2)*
  %5 = load i32, i32 addrspace(2)* %4, align 4, !invariant.load !0
  ret i32 %5

; <label>:6:                                      ; preds = %1
  %7 = getelementptr inbounds i8, i8 addrspace(2)* %dispatch_ptr, i64 16
  %8 = bitcast i8 addrspace(2)* %7 to i32 addrspace(2)*
  %9 = load i32, i32 addrspace(2)* %8, align 4, !invariant.load !0
  ret i32 %9

; <label>:10:                                     ; preds = %1
  %11 = getelementptr inbounds i8, i8 addrspace(2)* %dispatch_ptr, i64 20
  %12 = bitcast i8 addrspace(2)* %11 to i32 addrspace(2)*
  %13 = load i32, i32 addrspace(2)* %12, align 4, !invariant.load !0
  ret i32 %13

; <label>:14:                                     ; preds = %1
  ret i32 1
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_get_num_groups(i32 %i) #5 {
  %1 = call spir_func i32 @__hsail_get_global_size(i32 %i)
  %2 = call spir_func i32 @__hsail_currentworkgroup_size(i32 %i)
  %3 = sdiv i32 %1, %2
  %4 = srem i32 %1, %2
  %not. = icmp ne i32 %4, 0
  %5 = zext i1 %not. to i32
  %6 = add nsw i32 %3, %5
  ret i32 %6
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i32 @__hsail_get_lane_id() #2 {
  %1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1), !range !2
  ret i32 %2
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i32 @__bitrev_b32(i32 %input) #2 {
  %rev = call i32 @llvm.bitreverse.i32(i32 %input)
  ret i32 %rev
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i64 @__bitrev_b64(i64 %input) #2 {
  %rev = call i64 @llvm.bitreverse.i64(i64 %input)
  ret i64 %rev
}

; Function Attrs: nounwind readnone
declare i32 @llvm.bitreverse.i32(i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.bitreverse.i64(i64) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__activelanemask_v4_b64_b1(i32 %input) #5 {
  %a = tail call i64 asm "v_cmp_ne_i32_e64 $0, 0, $1", "=s,v"(i32 %input) #1
  ret i64 %a
}

define linkonce_odr spir_func i32 @amdgcn_wave_rshift_1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 312, i32 15, i32 15, i1 0)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_rshift_zero_1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 312, i32 15, i32 15, i1 1)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_rrotate_1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 316, i32 15, i32 15, i1 0)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_lshift_1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 304, i32 15, i32 15, i1 0)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_lshift_zero_1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 304, i32 15, i32 15, i1 1)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_lrotate_1(i32 %v) #3  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 308, i32 15, i32 15, i1 0)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_row_rshift(i32 %data, i32 %delta) #3 {
  switch i32 %delta, label %31 [
    i32 1, label %1
    i32 2, label %3
    i32 3, label %5
    i32 4, label %7
    i32 5, label %9
    i32 6, label %11
    i32 7, label %13
    i32 8, label %15
    i32 9, label %17
    i32 10, label %19
    i32 11, label %21
    i32 12, label %23
    i32 13, label %25
    i32 14, label %27
    i32 15, label %29
  ]

; <label>:1:                                              ; preds = %0                     
  %2 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 273, i32 15, i32 15, i1 0)
  ret i32 %2

; <label>:3:                                              ; preds = %0                    
  %4 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 274, i32 15, i32 15, i1 0)
  ret i32 %4

; <label>:5:                                              ; preds = %0                     
  %6 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 275, i32 15, i32 15, i1 0)
  ret i32 %6

; <label>:7:                                              ; preds = %0                     
  %8 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 276, i32 15, i32 15, i1 0)
  ret i32 %8

; <label>:9:                                              ; preds = %0                     
  %10 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 277, i32 15, i32 15, i1 0)
  ret i32 %10

; <label>:11:                                              ; preds = %0                     
  %12 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 278, i32 15, i32 15, i1 0)
  ret i32 %12

; <label>:13:                                              ; preds = %0                     
  %14 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 279, i32 15, i32 15, i1 0)
  ret i32 %14

; <label>:15:                                              ; preds = %0                     
  %16 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 280, i32 15, i32 15, i1 0)
  ret i32 %16

; <label>:17:                                              ; preds = %0                     
  %18 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 281, i32 15, i32 15, i1 0)
  ret i32 %18

; <label>:19:                                              ; preds = %0                     
  %20 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 282, i32 15, i32 15, i1 0)
  ret i32 %20

; <label>:21:                                              ; preds = %0                     
  %22 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 283, i32 15, i32 15, i1 0)
  ret i32 %22

; <label>:23:                                              ; preds = %0                     
  %24 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 284, i32 15, i32 15, i1 0)
  ret i32 %24

; <label>:25:                                              ; preds = %0                     
  %26 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 285, i32 15, i32 15, i1 0)
  ret i32 %26

; <label>:27:                                              ; preds = %0                     
  %28 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 286, i32 15, i32 15, i1 0)
  ret i32 %28

; <label>:29:                                              ; preds = %0                     
  %30 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 287, i32 15, i32 15, i1 0)
  ret i32 %30

; <label>:31:
  ret i32 %data
}

define linkonce_odr spir_func i32 @amdgcn_ds_permute(i32 %index, i32 %src) #3  {
  %call = call i32 @llvm.amdgcn.ds.permute(i32 %index, i32 %src)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_ds_bpermute(i32 %index, i32 %src) #3  {
  %call = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src)
  ret i32 %call
}

; llvm.amdgcn.mov.dpp.i32 <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i32, i32, i1) #4


;llvm.amdgcn.ds.permute <index> <src>
declare i32 @llvm.amdgcn.ds.permute(i32, i32) #4

;llvm.amdgcn.ds.bpermute <index> <src>
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4


; Function Attrs: nounwind argmemonly
define linkonce_odr spir_func i32 @__atomic_wrapinc_global(i32 addrspace(1)* nocapture %addr, i32 %val) #8 {
  %ret = tail call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* nocapture %addr, i32 %val) 
  ret i32 %ret
}

; Function Attrs: nounwind argmemonly
declare i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* nocapture, i32) #8

; Function Attrs: nounwind argmemonly
define linkonce_odr spir_func i32 @__atomic_wrapinc_local(i32 addrspace(3)* nocapture %addr, i32 %val) #8 {
  %ret = tail call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* nocapture %addr, i32 %val) 
  ret i32 %ret
}

; Function Attrs: nounwind argmemonly
declare i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* nocapture, i32) #8

; Function Attrs: nounwind argmemonly
define linkonce_odr spir_func i32 @__atomic_wrapdec_global(i32 addrspace(1)* nocapture %addr, i32 %val) #8 {
  %ret = tail call i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* nocapture %addr, i32 %val) 
  ret i32 %ret
}

; Function Attrs: nounwind argmemonly
declare i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* nocapture, i32) #8

; Function Attrs: nounwind argmemonly
define linkonce_odr spir_func i32 @__atomic_wrapdec_local(i32 addrspace(3)* nocapture %addr, i32 %val) #8 {
  %ret = tail call i32 @llvm.amdgcn.atomic.dec.i32.p3i32(i32 addrspace(3)* nocapture %addr, i32 %val) 
  ret i32 %ret
}

; Function Attrs: nounwind argmemonly
declare i32 @llvm.amdgcn.atomic.dec.i32.p3i32(i32 addrspace(3)* nocapture, i32) #8

; Function Attrs: nounwind readnone
define linkonce_odr spir_func i64 @__clock_u64() #1 {
  %ret = tail call i64 @llvm.amdgcn.s.memrealtime()
  ret i64 %ret
}

; Function Attrs: nounwind readnone
declare i64 @llvm.amdgcn.s.memrealtime() #1

; Function Attrs: nounwind readnone
define linkonce_odr spir_func i64 @__cycle_u64() #1 {
  %ret = tail call i64 @llvm.amdgcn.s.memtime()
  ret i64 %ret
}

; Function Attrs: nounwind readnone
declare i64 @llvm.amdgcn.s.memtime() #1

attributes #0 = { alwaysinline nounwind readonly }
attributes #1 = { nounwind readnone }
attributes #2 = { alwaysinline nounwind readnone }
attributes #3 = { alwaysinline convergent nounwind }
attributes #4 = { convergent nounwind }
attributes #5 = { alwaysinline nounwind }
attributes #6 = { alwaysinline norecurse nounwind readnone }
attributes #7 = { norecurse nounwind readnone }
attributes #8 = { nounwind argmemonly }

!0 = !{}
!1 = !{i32 0, i32 2048}
!2 = !{i32 0, i32 64}
