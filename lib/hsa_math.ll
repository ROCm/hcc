; ModuleID = 'hsa_math.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @amp_get_global_size(i32 %n) #0 {
entry:
  %0 = alloca i32, align 4
  store i32 %n, i32* %0, align 4
  %1 = load i32* %0, align 4
  %2 = call spir_func i64 @_Z15get_global_sizej(i32 %1) #1
  ret i64 %2
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z15get_global_sizej(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @amp_get_global_id(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @amp_get_local_id(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z12get_local_idj(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z12get_local_idj(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @amp_get_num_groups(i32 %n) #0 {
  %1 = alloca i32, align 4
  store i32 %n, i32* %1, align 4
  %2 = load i32* %1, align 4
  %3 = call spir_func i64 @_Z14get_num_groupsj(i32 %2) #1
  ret i64 %3
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z14get_num_groupsj(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @amp_get_group_id(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z12get_group_idj(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z12get_group_idj(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func void @amp_barrier(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  call spir_func void @_Z7barrierj(i32 %0)
  ret void
}

declare spir_func void @_Z7barrierj(i32)

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @amp_get_local_size(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z14get_local_sizej(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z14get_local_sizej(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @hc_get_grid_size(i32 %n) #0 {
entry:
  %0 = alloca i32, align 4
  store i32 %n, i32* %0, align 4
  %1 = load i32* %0, align 4
  %2 = call spir_func i64 @_Z15get_global_sizej(i32 %1) #1
  ret i64 %2
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @hc_get_workitem_absolute_id(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @hc_get_workitem_id(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z12get_local_idj(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @hc_get_num_groups(i32 %n) #0 {
  %1 = alloca i32, align 4
  store i32 %n, i32* %1, align 4
  %2 = load i32* %1, align 4
  %3 = call spir_func i64 @_Z14get_num_groupsj(i32 %2) #1
  ret i64 %3
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @hc_get_group_id(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z12get_group_idj(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @hc_barrier(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  call spir_func void @_Z7barrierj(i32 %0)
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @hc_get_group_size(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  %0 = load i32* %n.addr, align 4
  %call = call spir_func i64 @_Z14get_local_sizej(i32 %0) #1
  ret i64 %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_acos(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4acosf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4acosf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_acos_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4acosd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4acosd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_acosh(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5acoshf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5acoshf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_acosh_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5acoshd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5acoshd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_asin(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4asinf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4asinf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_asin_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4asind(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4asind(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_asinh(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5asinhf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5asinhf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_asinh_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5asinhd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5asinhd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_atan(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4atanf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4atanf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_atan_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4atand(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4atand(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_atanh(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5atanhf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5atanhf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_atanh_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5atanhd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5atanhd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_atan2(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z5atan2ff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5atan2ff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_atan2_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z5atan2dd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5atan2dd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_cbrt(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4cbrtf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4cbrtf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_cbrt_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4cbrtd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4cbrtd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_ceil(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4ceilf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4ceilf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_ceil_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4ceild(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4ceild(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_copysign(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z8copysignff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z8copysignff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_copysign_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z8copysigndd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z8copysigndd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_cos(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z3cosf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3cosf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_cos_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z3cosd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3cosd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_cosh(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4coshf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4coshf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_cosh_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4coshd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4coshd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_cospi(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5cospif(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5cospif(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_cospi_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5cospid(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5cospid(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_erf(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z3erff(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3erff(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_erf_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z3erfd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3erfd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_erfc(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4erfcf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4erfcf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_erfc_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4erfcd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4erfcd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_exp(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z3expf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3expf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_exp_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z3expd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3expd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_exp2(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4exp2f(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4exp2f(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_exp2_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4exp2d(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4exp2d(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_exp10(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5exp10f(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5exp10f(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_exp10_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5exp10d(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5exp10d(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_expm1(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5expm1f(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5expm1f(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_expm1_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5expm1d(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5expm1d(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fabs(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4fabsf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fabsf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_fabs_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4fabsd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4fabsd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fdim(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z4fdimff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fdimff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_fdim_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z4fdimdd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4fdimdd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_floor(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5floorf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5floorf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_floor_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5floord(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5floord(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fma(float %x, float %y, float %z) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  %z.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  store float %z, float* %z.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %2 = load float* %z.addr, align 4
  %call = call spir_func float @_Z3fmafff(float %0, float %1, float %2) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3fmafff(float, float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_fma_double(double %x, double %y, double %z) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  %z.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  store double %z, double* %z.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %2 = load double* %z.addr, align 8
  %call = call spir_func double @_Z3fmaddd(double %0, double %1, double %2) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3fmaddd(double, double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fmax(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z3maxff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3maxff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_fmax_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z3maxdd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3maxdd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fmin(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z3minff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3minff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_fmin_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z3mindd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3mindd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fmod(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z4fmodff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fmodff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_fmod_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z4fmoddd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4fmoddd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_frexpf_global(float %x, i32 addrspace(1)* %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32 addrspace(1)*, align 8
  store float %x, float* %x.addr, align 4
  store i32 addrspace(1)* %exp, i32 addrspace(1)** %exp.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load i32 addrspace(1)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(1)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %0, i32 addrspace(4)* %3) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5frexpfPU3AS4i(float, i32 addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_frexpf_local(float %x, i32 addrspace(3)* %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32 addrspace(3)*, align 8
  store float %x, float* %x.addr, align 4
  store i32 addrspace(3)* %exp, i32 addrspace(3)** %exp.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load i32 addrspace(3)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(3)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %0, i32 addrspace(4)* %3) #1
  ret float %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_frexpf(float %x, i32 addrspace(4)* %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32 addrspace(4)*, align 8
  store float %x, float* %x.addr, align 4
  store i32 addrspace(4)* %exp, i32 addrspace(4)** %exp.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load i32 addrspace(4)** %exp.addr, align 8
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %0, i32 addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_frexp_global(double %x, i32 addrspace(1)* %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32 addrspace(1)*, align 8
  store double %x, double* %x.addr, align 8
  store i32 addrspace(1)* %exp, i32 addrspace(1)** %exp.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load i32 addrspace(1)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(1)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %0, i32 addrspace(4)* %3) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5frexpdPU3AS4i(double, i32 addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_frexp_local(double %x, i32 addrspace(3)* %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32 addrspace(3)*, align 8
  store double %x, double* %x.addr, align 8
  store i32 addrspace(3)* %exp, i32 addrspace(3)** %exp.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load i32 addrspace(3)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(3)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %0, i32 addrspace(4)* %3) #1
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_frexp(double %x, i32 addrspace(4)* %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32 addrspace(4)*, align 8
  store double %x, double* %x.addr, align 8
  store i32 addrspace(4)* %exp, i32 addrspace(4)** %exp.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load i32 addrspace(4)** %exp.addr, align 8
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %0, i32 addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_hypot(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z5hypotff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5hypotff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_hypot_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z5hypotdd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5hypotdd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_ilogb(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func i32 @_Z5ilogbf(float %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5ilogbf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_ilogb_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func i32 @_Z5ilogbd(double %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5ilogbd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isfinite(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func i32 @_Z8isfinitef(float %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isfinitef(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isfinite_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func i32 @_Z8isfinited(double %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isfinited(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isinf(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func i32 @_Z5isinff(float %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isinff(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isinf_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func i32 @_Z5isinfd(double %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isinfd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isnan(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func i32 @_Z5isnanf(float %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isnanf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isnan_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func i32 @_Z5isnand(double %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isnand(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isnormal(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func i32 @_Z8isnormalf(float %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isnormalf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_isnormal_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func i32 @_Z8isnormald(double %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isnormald(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_lgammaf_global(float %x, i32 addrspace(1)* %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32 addrspace(1)*, align 8
  store float %x, float* %x.addr, align 4
  store i32 addrspace(1)* %exp, i32 addrspace(1)** %exp.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load i32 addrspace(1)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(1)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func float @_Z8lgamma_rfPU3AS4i(float %0, i32 addrspace(4)* %3) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z8lgamma_rfPU3AS4i(float, i32 addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_lgammaf_local(float %x, i32 addrspace(3)* %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32 addrspace(3)*, align 8
  store float %x, float* %x.addr, align 4
  store i32 addrspace(3)* %exp, i32 addrspace(3)** %exp.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load i32 addrspace(3)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(3)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func float @_Z8lgamma_rfPU3AS4i(float %0, i32 addrspace(4)* %3) #1
  ret float %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_lgammaf(float %x, i32 addrspace(4)* %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32 addrspace(4)*, align 8
  store float %x, float* %x.addr, align 4
  store i32 addrspace(4)* %exp, i32 addrspace(4)** %exp.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load i32 addrspace(4)** %exp.addr, align 8
  %call = call spir_func float @_Z8lgamma_rfPU3AS4i(float %0, i32 addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_lgamma_global(double %x, i32 addrspace(1)* %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32 addrspace(1)*, align 8
  store double %x, double* %x.addr, align 8
  store i32 addrspace(1)* %exp, i32 addrspace(1)** %exp.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load i32 addrspace(1)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(1)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func double @_Z8lgamma_rdPU3AS4i(double %0, i32 addrspace(4)* %3) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z8lgamma_rdPU3AS4i(double, i32 addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_lgamma_local(double %x, i32 addrspace(3)* %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32 addrspace(3)*, align 8
  store double %x, double* %x.addr, align 8
  store i32 addrspace(3)* %exp, i32 addrspace(3)** %exp.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load i32 addrspace(3)** %exp.addr, align 8
  %2 = ptrtoint i32 addrspace(3)* %1 to i64
  %3 = inttoptr i64 %2 to i32 addrspace(4)*
  %call = call spir_func double @_Z8lgamma_rdPU3AS4i(double %0, i32 addrspace(4)* %3) #1
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_lgamma(double %x, i32 addrspace(4)* %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32 addrspace(4)*, align 8
  store double %x, double* %x.addr, align 8
  store i32 addrspace(4)* %exp, i32 addrspace(4)** %exp.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load i32 addrspace(4)** %exp.addr, align 8
  %call = call spir_func double @_Z8lgamma_rdPU3AS4i(double %0, i32 addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_log(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z3logf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3logf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_log_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z3logd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3logd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_log10(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5log10f(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5log10f(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_log10_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5log10d(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5log10d(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_log2(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4log2f(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4log2f(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_log2_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4log2d(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4log2d(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_log1p(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5log1pf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5log1pf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_log1p_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5log1pd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5log1pd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_logb(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4logbf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4logbf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_logb_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4logbd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4logbd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_modff_global(float %x, float addrspace(1)* %iptr) #0 {
entry:
  %x.addr = alloca float, align 4
  %iptr.addr = alloca float addrspace(1)*, align 8
  store float %x, float* %x.addr, align 4
  store float addrspace(1)* %iptr, float addrspace(1)** %iptr.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load float addrspace(1)** %iptr.addr, align 8
  %2 = ptrtoint float addrspace(1)* %1 to i64
  %3 = inttoptr i64 %2 to float addrspace(4)*
  %call = call spir_func float @_Z4modffPU3AS4f(float %0, float addrspace(4)* %3) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4modffPU3AS4f(float, float addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_modff_local(float %x, float addrspace(3)* %iptr) #0 {
entry:
  %x.addr = alloca float, align 4
  %iptr.addr = alloca float addrspace(3)*, align 8
  store float %x, float* %x.addr, align 4
  store float addrspace(3)* %iptr, float addrspace(3)** %iptr.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load float addrspace(3)** %iptr.addr, align 8
  %2 = ptrtoint float addrspace(3)* %1 to i64
  %3 = inttoptr i64 %2 to float addrspace(4)*
  %call = call spir_func float @_Z4modffPU3AS4f(float %0, float addrspace(4)* %3) #1
  ret float %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_modff(float %x, float addrspace(4)* %iptr) #0 {
entry:
  %x.addr = alloca float, align 4
  %iptr.addr = alloca float addrspace(4)*, align 8
  store float %x, float* %x.addr, align 4
  store float addrspace(4)* %iptr, float addrspace(4)** %iptr.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load float addrspace(4)** %iptr.addr, align 8
  %call = call spir_func float @_Z4modffPU3AS4f(float %0, float addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_modf_global(double %x, double addrspace(1)* %iptr) #0 {
entry:
  %x.addr = alloca double, align 8
  %iptr.addr = alloca double addrspace(1)*, align 8
  store double %x, double* %x.addr, align 8
  store double addrspace(1)* %iptr, double addrspace(1)** %iptr.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double addrspace(1)** %iptr.addr, align 8
  %2 = ptrtoint double addrspace(1)* %1 to i64
  %3 = inttoptr i64 %2 to double addrspace(4)*
  %call = call spir_func double @_Z4modfdPU3AS4d(double %0, double addrspace(4)* %3) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4modfdPU3AS4d(double, double addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_modf_local(double %x, double addrspace(3)* %iptr) #0 {
entry:
  %x.addr = alloca double, align 8
  %iptr.addr = alloca double addrspace(3)*, align 8
  store double %x, double* %x.addr, align 8
  store double addrspace(3)* %iptr, double addrspace(3)** %iptr.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double addrspace(3)** %iptr.addr, align 8
  %2 = ptrtoint double addrspace(3)* %1 to i64
  %3 = inttoptr i64 %2 to double addrspace(4)*
  %call = call spir_func double @_Z4modfdPU3AS4d(double %0, double addrspace(4)* %3) #1
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_modf(double %x, double addrspace(4)* %iptr) #0 {
entry:
  %x.addr = alloca double, align 8
  %iptr.addr = alloca double addrspace(4)*, align 8
  store double %x, double* %x.addr, align 8
  store double addrspace(4)* %iptr, double addrspace(4)** %iptr.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double addrspace(4)** %iptr.addr, align 8
  %call = call spir_func double @_Z4modfdPU3AS4d(double %0, double addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_nan(i32 %tagp) #0 {
entry:
  %tagp.addr = alloca i32, align 4
  store i32 %tagp, i32* %tagp.addr, align 4
  %0 = load i32* %tagp.addr, align 4
  %call = call spir_func float @_Z3nanj(i32 %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3nanj(i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_nan_double(i32 %tagp) #0 {
entry:
  %tagp.addr = alloca i32, align 4
  store i32 %tagp, i32* %tagp.addr, align 4
  %0 = load i32* %tagp.addr, align 4
  %conv = sext i32 %0 to i64
  %call = call spir_func double @_Z3nanm(i64 %conv) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3nanm(i64) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_nearbyint(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4rintf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4rintf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_nearbyint_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4rintd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4rintd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_nextafter(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z9nextafterff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z9nextafterff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_nextafter_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z9nextafterdd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z9nextafterdd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_pow(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z3powff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3powff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_pow_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z3powdd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3powdd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_remainder(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z9remainderff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z9remainderff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_remainder_double(double %x, double %y) #0 {
entry:
  %x.addr = alloca double, align 8
  %y.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  store double %y, double* %y.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double* %y.addr, align 8
  %call = call spir_func double @_Z9remainderdd(double %0, double %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z9remainderdd(double, double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_remquof_global(float %x, float %y, i32 addrspace(1)* %quo) #0 {
entry:
  ret float 0.000000e+00
}

declare spir_func float @_Z6remquoffPU3AS4i(float, float, i32 addrspace(4)*)

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_remquof_local(float %x, float %y, i32 addrspace(3)* %quo) #0 {
entry:
  ret float 0.000000e+00
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_remquof(float %x, float %y, i32 addrspace(4)* %quo) #0 {
entry:
  ret float 0.000000e+00
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_remquo_global(double %x, double %y, i32 addrspace(1)* %quo) #0 {
entry:
  ret double 0.000000e+00
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z6remquoddPU3AS4i(double, double, i32 addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_remquo_local(double %x, double %y, i32 addrspace(3)* %quo) #0 {
entry:
  ret double 0.000000e+00
}

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_remquo(double %x, double %y, i32 addrspace(4)* %quo) #0 {
entry:
  ret double 0.000000e+00
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_round(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5roundf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5roundf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_round_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5roundd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5roundd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_rsqrt(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5rsqrtf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5rsqrtf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_rsqrt_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5rsqrtd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5rsqrtd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_sinpi(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5sinpif(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5sinpif(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_sinpi_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5sinpid(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5sinpid(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_ldexp(float %x, i32 %exp) #0 {
entry:
  %x.addr = alloca float, align 4
  %exp.addr = alloca i32, align 4
  store float %x, float* %x.addr, align 4
  store i32 %exp, i32* %exp.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load i32* %exp.addr, align 4
  %call = call spir_func float @_Z5ldexpfi(float %0, i32 %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5ldexpfi(float, i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_ldexp_double(double %x, i32 %exp) #0 {
entry:
  %x.addr = alloca double, align 8
  %exp.addr = alloca i32, align 4
  store double %x, double* %x.addr, align 8
  store i32 %exp, i32* %exp.addr, align 4
  %0 = load double* %x.addr, align 8
  %1 = load i32* %exp.addr, align 4
  %call = call spir_func double @_Z5ldexpdi(double %0, i32 %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5ldexpdi(double, i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_signbit(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func i32 @_Z7signbitf(float %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z7signbitf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_signbit_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func i32 @_Z7signbitd(double %0) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z7signbitd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_sin(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z3sinf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3sinf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_sin_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z3sind(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3sind(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func void @opencl_sincosf_global(float %x, float addrspace(1)* %s, float addrspace(1)* %c) #0 {
entry:
  %x.addr = alloca float, align 4
  %s.addr = alloca float addrspace(1)*, align 8
  %c.addr = alloca float addrspace(1)*, align 8
  store float %x, float* %x.addr, align 4
  store float addrspace(1)* %s, float addrspace(1)** %s.addr, align 8
  store float addrspace(1)* %c, float addrspace(1)** %c.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load float addrspace(1)** %c.addr, align 8
  %2 = addrspacecast float addrspace(1)* %1 to float addrspace(4)*
  %call = call spir_func float @_Z6sincosfPU3AS4f(float %0, float addrspace(4)* %2) #1
  %3 = load float addrspace(1)** %s.addr, align 8
  store float %call, float addrspace(1)* %3, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z6sincosfPU3AS4f(float, float addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func void @opencl_sincosf_local(float %x, float addrspace(3)* %s, float addrspace(3)* %c) #0 {
entry:
  %x.addr = alloca float, align 4
  %s.addr = alloca float addrspace(3)*, align 8
  %c.addr = alloca float addrspace(3)*, align 8
  store float %x, float* %x.addr, align 4
  store float addrspace(3)* %s, float addrspace(3)** %s.addr, align 8
  store float addrspace(3)* %c, float addrspace(3)** %c.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load float addrspace(3)** %c.addr, align 8
  %2 = addrspacecast float addrspace(3)* %1 to float addrspace(4)*
  %call = call spir_func float @_Z6sincosfPU3AS4f(float %0, float addrspace(4)* %2) #1
  %3 = load float addrspace(3)** %s.addr, align 8
  store float %call, float addrspace(3)* %3, align 4
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @opencl_sincosf(float %x, float addrspace(4)* %s, float addrspace(4)* %c) #0 {
entry:
  %x.addr = alloca float, align 4
  %s.addr = alloca float addrspace(4)*, align 8
  %c.addr = alloca float addrspace(4)*, align 8
  store float %x, float* %x.addr, align 4
  store float addrspace(4)* %s, float addrspace(4)** %s.addr, align 8
  store float addrspace(4)* %c, float addrspace(4)** %c.addr, align 8
  %0 = load float* %x.addr, align 4
  %1 = load float addrspace(4)** %c.addr, align 8
  %call = call spir_func float @_Z6sincosfPU3AS4f(float %0, float addrspace(4)* %1) #1
  %2 = load float addrspace(4)** %s.addr, align 8
  store float %call, float addrspace(4)* %2, align 4
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @opencl_sincos_global(double %x, double addrspace(1)* %s, double addrspace(1)* %c) #0 {
entry:
  %x.addr = alloca double, align 8
  %s.addr = alloca double addrspace(1)*, align 8
  %c.addr = alloca double addrspace(1)*, align 8
  store double %x, double* %x.addr, align 8
  store double addrspace(1)* %s, double addrspace(1)** %s.addr, align 8
  store double addrspace(1)* %c, double addrspace(1)** %c.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double addrspace(1)** %c.addr, align 8
  %2 = addrspacecast double addrspace(1)* %1 to double addrspace(4)*
  %call = call spir_func double @_Z6sincosdPU3AS4d(double %0, double addrspace(4)* %2) #1
  %3 = load double addrspace(1)** %s.addr, align 8
  store double %call, double addrspace(1)* %3, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z6sincosdPU3AS4d(double, double addrspace(4)*) #1

; Function Attrs: nounwind
define linkonce_odr spir_func void @opencl_sincos_local(double %x, double addrspace(3)* %s, double addrspace(3)* %c) #0 {
entry:
  %x.addr = alloca double, align 8
  %s.addr = alloca double addrspace(3)*, align 8
  %c.addr = alloca double addrspace(3)*, align 8
  store double %x, double* %x.addr, align 8
  store double addrspace(3)* %s, double addrspace(3)** %s.addr, align 8
  store double addrspace(3)* %c, double addrspace(3)** %c.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double addrspace(3)** %c.addr, align 8
  %2 = addrspacecast double addrspace(3)* %1 to double addrspace(4)*
  %call = call spir_func double @_Z6sincosdPU3AS4d(double %0, double addrspace(4)* %2) #1
  %3 = load double addrspace(3)** %s.addr, align 8
  store double %call, double addrspace(3)* %3, align 8
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @opencl_sincos(double %x, double addrspace(4)* %s, double addrspace(4)* %c) #0 {
entry:
  %x.addr = alloca double, align 8
  %s.addr = alloca double addrspace(4)*, align 8
  %c.addr = alloca double addrspace(4)*, align 8
  store double %x, double* %x.addr, align 8
  store double addrspace(4)* %s, double addrspace(4)** %s.addr, align 8
  store double addrspace(4)* %c, double addrspace(4)** %c.addr, align 8
  %0 = load double* %x.addr, align 8
  %1 = load double addrspace(4)** %c.addr, align 8
  %call = call spir_func double @_Z6sincosdPU3AS4d(double %0, double addrspace(4)* %1) #1
  %2 = load double addrspace(4)** %s.addr, align 8
  store double %call, double addrspace(4)* %2, align 8
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_sinh(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4sinhf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4sinhf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_sinh_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4sinhd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4sinhd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_sqrt(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4sqrtf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4sqrtf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_sqrt_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4sqrtd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4sqrtd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_tgamma(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z6tgammaf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z6tgammaf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_tgamma_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z6tgammad(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z6tgammad(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_tan(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z3tanf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3tanf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_tan_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z3tand(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3tand(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_tanh(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z4tanhf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4tanhf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_tanh_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z4tanhd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4tanhd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_tanpi(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5tanpif(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5tanpif(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_tanpi_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5tanpid(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5tanpid(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_trunc(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  %0 = load float* %x.addr, align 4
  %call = call spir_func float @_Z5truncf(float %0) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5truncf(float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func double @opencl_trunc_double(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double* %x.addr, align 8
  %call = call spir_func double @_Z5truncd(double %0) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5truncd(double) #1

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_sub_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_sub_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_sub_unsigned(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_int(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_sub_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_sub_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_sub_int(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @atomic_add_float_global(float addrspace(1)* %x, float %y) #0 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:
  %1 = load volatile float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:
  ret float %add
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @atomic_add_float_local(float addrspace(3)* %x, float %y) #0 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:
  %1 = load volatile float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:
  ret float %add
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @atomic_add_float(float addrspace(4)* %x, float %y) #0 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:
  ret float %add
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @atomic_sub_float_global(float addrspace(1)* %x, float %y) #0 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:
  %1 = load volatile float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:
  ret float %sub
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @atomic_sub_float_local(float addrspace(3)* %x, float %y) #0 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:
  %1 = load volatile float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:
  ret float %sub
}

; Function Attrs: nounwind
define linkonce_odr spir_func float @atomic_sub_float(float addrspace(4)* %x, float %y) #0 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:
  ret float %sub
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_int(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned_global(i32 addrspace(1)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_dec_unsigned_global(i32 addrspace(1)* %x) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned_local(i32 addrspace(3)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_dec_unsigned_local(i32 addrspace(3)* %x) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned(i32 addrspace(4)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_dec_unsigned(i32 addrspace(4)* %x) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_int_global(i32 addrspace(1)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_dec_int_global(i32 addrspace(1)* %x) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_int_local(i32 addrspace(3)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_dec_int_local(i32 addrspace(3)* %x) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_int(i32 addrspace(4)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_dec_int(i32 addrspace(4)* %x) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_add_int64_local(i64 addrspace(3)* %x, i64 %y) #0 {
entry:
  %ret = atomicrmw add i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_sub_int64_local(i64 addrspace(3)* %x, i64 %y) #0 {
entry:
  %ret = atomicrmw sub i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_add_int64_global(i64 addrspace(1)* %x, i64 %y) #0 {
entry:
  %ret = atomicrmw add i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_sub_int64_global(i64 addrspace(1)* %x, i64 %y) #0 {
entry:
  %ret = atomicrmw sub i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw xchg i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw xchg i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_exchange_int64_local(i64 addrspace(3)* %x, i64 %y) #0 {
entry:
  %ret = atomicrmw xchg i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw xchg i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw xchg i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_exchange_int64_global(i64 addrspace(1)* %x, i64 %y) #0 {
entry:
  %ret = atomicrmw xchg i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_int_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #0 {
entry:
  %val_success = cmpxchg i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #0 {
entry:
  %val_success = cmpxchg i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_compare_exchange_int64_local(i64 addrspace(3)* %x, i64 %y, i64 %z) #0 {
entry:
  %val_success = cmpxchg i64 addrspace(3)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_int_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #0 {
entry:
  %val_success = cmpxchg i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #0 {
entry:
  %val_success = cmpxchg i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_compare_exchange_int64_global(i64 addrspace(1)* %x, i64 %y, i64 %z) #0 {
entry:
  %val_success = cmpxchg i64 addrspace(1)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = metadata !{}
!1 = metadata !{i32 5}
