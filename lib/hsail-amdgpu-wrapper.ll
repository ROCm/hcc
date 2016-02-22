; ModuleID = 'hsail-amdgpu'

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_get_global_id(i32) #0 {
  %dispatch_ptr = call noalias nonnull dereferenceable(64) i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %dispatch_ptr_i32 = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %size_xy_ptr = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 1
  %size_xy = load i32, i32 addrspace(2)* %size_xy_ptr, align 4, !invariant.load !0
  switch i32 %0, label %20 [
    i32 0, label %2
    i32 1, label %8
    i32 2, label %14
  ]

; <label>:2                                       ; preds = %1
  %3 = and i32 %size_xy, 65535 ; 0xffff
  %4 = call i32 @llvm.r600.read.tgid.x()
  %5 = call i32 @llvm.r600.read.tidig.x()
  %6 = mul i32 %3, %4
  %7 = add i32 %5, %6
  ret i32 %7

; <label>:8                                       ; preds = %1
  %9 = lshr i32 %size_xy, 16
  %10 = call i32 @llvm.r600.read.tgid.y()
  %11 = call i32 @llvm.r600.read.tidig.y()
  %12 = mul i32 %9, %10
  %13 = add i32 %11, %12
  ret i32 %13

; <label>:14                                      ; preds = %1
  %size_z_ptr = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 2
  %15 = load i32, i32 addrspace(2)* %size_z_ptr, align 4, !invariant.load !0, !range !1
  %16 = call i32 @llvm.r600.read.tgid.z()
  %17 = call i32 @llvm.r600.read.tidig.z()
  %18 = mul i32 %15, %16
  %19 = add i32 %17, %18
  ret i32 %19

; <label>:20                                      ; preds = %1
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.z() #1

declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_abs_f32(float) #0 {
  %2 = call float @llvm.fabs.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.fabs.f32(float) #1

; Function Attrs: alwaysinline nounwind convergent
define linkonce_odr spir_func void @__hsail_barrier() #3 {
  tail call void @llvm.amdgcn.s.barrier() #2
  ret void
}

; Function Attrs: nounwind convergent
declare void @llvm.amdgcn.s.barrier() #2


; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_abs_f64(double) #0 {
  %2 = call double @llvm.fabs.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.fabs.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_ceil_f32(float) #0 {
  %2 = call float @llvm.ceil.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.ceil.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_ceil_f64(double) #0 {
  %2 = call double @llvm.ceil.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.ceil.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_copysign_f32(float, float) #0 {
  %3 = call float @llvm.copysign.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone
declare float @llvm.copysign.f32(float, float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_copysign_f64(double, double) #0 {
  %3 = call double @llvm.copysign.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone
declare double @llvm.copysign.f64(double, double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_floor_f32(float) #0 {
  %2 = call float @llvm.floor.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.floor.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_floor_f64(double) #0 {
  %2 = call double @llvm.floor.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.floor.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_fma_f32(float, float, float) #0 {
  %4 = call float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %4
}

; Function Attrs: nounwind readnone
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_fma_f64(double, double, double) #0 {
  %4 = call double @llvm.fma.f64(double %0, double %1, double %2)
  ret double %4
}

; Function Attrs: nounwind readnone
declare double @llvm.fma.f64(double, double, double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_max_f32(float, float) #0 {
  %3 = call float @llvm.maxnum.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_max_f64(double, double) #0 {
  %3 = call double @llvm.maxnum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone
declare double @llvm.maxnum.f64(double, double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_min_f32(float, float) #0 {
  %3 = call float @llvm.minnum.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone
declare float @llvm.minnum.f32(float, float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_min_f64(double, double) #0 {
  %3 = call double @llvm.minnum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone
declare double @llvm.minnum.f64(double, double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_nfma_f32(float, float, float) #0 {
  %4 = call float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %4
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_nfma_f64(double, double, double) #0 {
  %4 = call double @llvm.fma.f64(double %0, double %1, double %2)
  ret double %4
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_nrcp_f32(float) #0 {
  %2 = call float @llvm.amdgcn.rcp.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.rcp.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_nrsqrt_f32(float) #1 {
  %2 = call float @llvm.amdgcn.rsq.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.amdgcn.rsq.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_nrsqrt_f64(double) #1 {
  %2 = call double @llvm.amdgcn.rsq.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.amdgcn.rsq.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_nsqrt_f32(float) #1 {
  %2 = call float @llvm.sqrt.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_round_f32(float) #0 {
  %2 = call float @llvm.round.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.round.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_round_f64(double) #1 {
  %2 = call double @llvm.round.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.round.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_sqrt_f32(float) #1 {
  %2 = call float @llvm.sqrt.f32(float %0)
  ret float %2
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_sqrt_f64(double) #1 {
  %2 = call double @llvm.sqrt.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_trunc_f32(float) #1 {
  %2 = call float @llvm.trunc.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
declare float @llvm.trunc.f32(float) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_trunc_f64(double) #1 {
  %2 = call double @llvm.trunc.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
declare double @llvm.trunc.f64(double) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_class_f32(float, i32) #1 {
  %3 = call i1 @llvm.amdgcn.class.f32(float %0, i32 %1)
  %4 = sext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: nounwind readnone
declare i1 @llvm.amdgcn.class.f32(float, i32) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_class_f64(double, i32) #1 {
  %3 = call i1 @llvm.amdgcn.class.f64(double %0, i32 %1)
  %4 = sext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: nounwind readnone
declare i1 @llvm.amdgcn.class.f64(double, i32) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_mulhi_u32(i32, i32) #1 {
  %3 = zext i32 %0 to i64
  %4 = zext i32 %1 to i64
  %5 = mul i64 %3, %4
  %6 = lshr i64 %5, 32
  %7 = trunc i64 %6 to i32
  ret i32 %7
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_mad_u32(i32, i32, i32) #1 {
  %4 = mul i32 %0, %1
  %5 = add i32 %4, %2
  ret i32 %5
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_firstbit_u32(i32) #1 {
  %2 = call i32 @llvm.ctlz.i32(i32 %0, i1 true)
  %3 = icmp eq i32 %0, 0
  %4 = select i1 %3, i32 -1, i32 %2
  ret i32 %4
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) #1

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_max_s32(i32, i32) #1 {
  %3 = icmp sgt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_min_s32(i32, i32) #1 {
  %3 = icmp slt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_bytealign_b32(i32, i32, i32) #1 {
  %4 = and i32 %2, 3
  %5 = mul i32 %4, 8
  %6 = zext i32 %5 to i64
  %7 = zext i32 %0 to i64
  %8 = zext i32 %1 to i64
  %9 = shl i64 %8, 32
  %10 = or i64 %7, %9
  %11 = lshr i64 %10, %6
  %12 = trunc i64 %11 to i32
  ret i32 %12
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func float @__hsail_ftz_f32(float) #1 {
  %2 = bitcast float %0 to i32
  %3 = and i32 %2, 2139095040
  %4 = and i32 %2, 8388607
  %5 = icmp eq i32 %3, 0
  %6 = icmp ne i32 %4, 0
  %7 = and i1 %5, %6
  %8 = select i1 %7, float 0.000000e+00, float %0
  ret float %8
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func double @__hsail_fraction_f64(double) #1 {
  %2 = call double @llvm.floor.f64(double %0)
  %3 = fsub double -0.000000e+00, %2
  %4 = fadd double %0, %3
  ret double %4
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_get_local_id(i32) #1 {
  %2 = call i32 @llvm.r600.read.tidig.x()
  %3 = call i32 @llvm.r600.read.tidig.y()
  %4 = call i32 @llvm.r600.read.tidig.z()
  %5 = icmp eq i32 %0, 1
  %6 = select i1 %5, i32 %3, i32 %2
  %7 = icmp eq i32 %0, 2
  %8 = select i1 %7, i32 %4, i32 %6
  ret i32 %8
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_get_group_id(i32) #1 {
  %2 = call i32 @llvm.r600.read.tgid.x()
  %3 = call i32 @llvm.r600.read.tgid.y()
  %4 = call i32 @llvm.r600.read.tgid.z()
  %5 = icmp eq i32 %0, 1
  %6 = select i1 %5, i32 %3, i32 %2
  %7 = icmp eq i32 %0, 2
  %8 = select i1 %7, i32 %4, i32 %6
  ret i32 %8
}

; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_currentworkgroup_size(i32) #0 {
  %dispatch_ptr = call noalias i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %dispatch_ptr_i32 = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %size_xy_ptr = getelementptr i32, i32 addrspace(2)* %dispatch_ptr_i32, i32 1
  %size_xy = load i32, i32 addrspace(2)* %size_xy_ptr, align 4, !invariant.load !0
  switch i32 %0, label %8 [
    i32 0, label %2
    i32 1, label %4
    i32 2, label %6
  ]

; <label>:2                                       ; preds = %1
  %3 = and i32 %size_xy, 65535 ; 0xffff
  ret i32 %3

; <label>:4                                      ; preds = %1
  %5 = lshr i32 %size_xy, 16
  ret i32 %5

; <label>:6                                      ; preds = %1
  %size_z_ptr = getelementptr i32 ,i32 addrspace(2)* %dispatch_ptr_i32, i32 2
  %7 = load i32, i32 addrspace(2)* %size_z_ptr, align 4, !invariant.load !0, !range !1
  ret i32 %7

; <label>:8                                      ; preds = %1
  ret i32 1
}



; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_get_global_size(i32) #0 {
  %dispatch_ptr = call noalias nonnull dereferenceable(64) i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %dispatch_ptr_i32 = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  switch i32 %0, label %11 [
    i32 0, label %2
    i32 1, label %5
    i32 2, label %8
  ]

; <label>:2                                       ; preds = %1
  %3 = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 3
  %4 = load i32, i32 addrspace(2)* %3, align 4, !invariant.load !0
  ret i32 %4

; <label>:5                                       ; preds = %1
  %6 = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 4
  %7 = load i32, i32 addrspace(2)* %6, align 4, !invariant.load !0
  ret i32 %7

; <label>:8                                       ; preds = %1
  %9 = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 5
  %10 = load i32, i32 addrspace(2)* %9, align 4, !invariant.load !0
  ret i32 %10

; <label>:11                                      ; preds = %1
  ret i32 1
}


; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_get_num_groups(i32 %i) #0 {
  %1 = alloca i32, align 4
  %global_size = alloca i32, align 4
  %group_size = alloca i32, align 4
  %num_group = alloca i32, align 4
  store i32 %i, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  %3 = call spir_func i32 @__hsail_get_global_size(i32 %2)
  store i32 %3, i32* %global_size, align 4
  %4 = load i32, i32* %1, align 4
  %5 = call spir_func i32 @__hsail_currentworkgroup_size(i32 %4)
  store i32 %5, i32* %group_size, align 4
  %6 = load i32, i32* %global_size, align 4
  %7 = load i32, i32* %group_size, align 4
  %8 = sdiv i32 %6, %7
  store i32 %8, i32* %num_group, align 4
  %9 = load i32, i32* %global_size, align 4
  %10 = load i32, i32* %group_size, align 4
  %11 = srem i32 %9, %10
  %12 = icmp eq i32 %11, 0
  %13 = select i1 %12, i32 0, i32 1
  %14 = load i32, i32* %num_group, align 4
  %15 = add nsw i32 %14, %13
  store i32 %15, i32* %num_group, align 4
  %16 = load i32, i32* %num_group, align 4
  ret i32 %16
}


; Function Attrs: alwaysinline
define linkonce_odr spir_func i32 @__hsail_get_lane_id() #0  {
  %1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %2 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1)
  ret i32 %2
}
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #1

; global variable to store the size of static group segment
; the value would be set by Kalmar runtime prior to kernel dispatch
; define @hcc_static_group_segment_size as a module-level global variable
@"&hcc_static_group_segment_size" = addrspace(1) global i32 0, align 4

; global variable to store the size of dynamic group segment
; the value would be set by Kalmar runtime prior to kernel dispatch
; define @hcc_dynamic_group_segment_size as a module-level global variable
@"&hcc_dynamic_group_segment_size" = addrspace(1) global i32 0, align 4

!0 = !{}
!1 = !{i32 0, i32 257}

attributes #0 = { alwaysinline }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind convergent }
attributes #3 = { alwaysinline nounwind convergent }
