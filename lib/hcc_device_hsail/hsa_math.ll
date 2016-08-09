; ModuleID = 'hsa_math.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i64 @amp_get_global_size(i32 %n) #0 {
entry:
  %0 = call spir_func i64 @_Z15get_global_sizej(i32 %n) #1
  ret i64 %0
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z15get_global_sizej(i32) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i64 @amp_get_global_id(i32 %n) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 %n) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i64 @amp_get_local_id(i32 %n) #0 {
entry:
  %call = call spir_func i64 @_Z12get_local_idj(i32 %n) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z12get_local_idj(i32) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func i64 @amp_get_num_groups(i32 %n) #0 {
  %1 = call spir_func i64 @_Z14get_num_groupsj(i32 %n) #1
  ret i64 %1
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z14get_num_groupsj(i32) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @amp_get_group_id(i32 %n) #2 {
entry:
  %call = call spir_func i64 @_Z12get_group_idj(i32 %n) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z12get_group_idj(i32) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @amp_barrier(i32 %n) #2 {
entry:
  call spir_func void @_Z7barrierj(i32 %n) #3
  ret void
}

; Function Attrs: alwaysinline nounwind
declare spir_func void @_Z7barrierj(i32) #2

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @amp_get_local_size(i32 %n) #2 {
entry:
  %call = call spir_func i64 @_Z14get_local_sizej(i32 %n) #1
  ret i64 %call
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z14get_local_sizej(i32) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @hc_get_grid_size(i32 %n) #2 {
entry:
  %0 = call spir_func i64 @_Z15get_global_sizej(i32 %n) #1
  ret i64 %0
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @hc_get_workitem_absolute_id(i32 %n) #2 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 %n) #1
  ret i64 %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @hc_get_workitem_id(i32 %n) #2 {
entry:
  %call = call spir_func i64 @_Z12get_local_idj(i32 %n) #1
  ret i64 %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @hc_get_num_groups(i32 %n) #2 {
  %1 = call spir_func i64 @_Z14get_num_groupsj(i32 %n) #1
  ret i64 %1
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @hc_get_group_id(i32 %n) #2 {
entry:
  %call = call spir_func i64 @_Z12get_group_idj(i32 %n) #1
  ret i64 %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @hc_barrier(i32 %n) #2 {
entry:
  call spir_func void @_Z7barrierj(i32 %n) #3
  ret void
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @hc_get_group_size(i32 %n) #2 {
entry:
  %call = call spir_func i64 @_Z14get_local_sizej(i32 %n) #1
  ret i64 %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_acos(float %x) #2 {
entry:
  %call = call spir_func float @_Z4acosf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4acosf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_acos_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4acosd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4acosd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_acosh(float %x) #2 {
entry:
  %call = call spir_func float @_Z5acoshf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5acoshf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_acosh_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5acoshd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5acoshd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_asin(float %x) #2 {
entry:
  %call = call spir_func float @_Z4asinf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4asinf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_asin_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4asind(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4asind(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_asinh(float %x) #2 {
entry:
  %call = call spir_func float @_Z5asinhf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5asinhf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_asinh_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5asinhd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5asinhd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_atan(float %x) #2 {
entry:
  %call = call spir_func float @_Z4atanf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4atanf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_atan_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4atand(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4atand(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_atanh(float %x) #2 {
entry:
  %call = call spir_func float @_Z5atanhf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5atanhf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_atanh_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5atanhd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5atanhd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_atan2(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z5atan2ff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5atan2ff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_atan2_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z5atan2dd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5atan2dd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_cbrt(float %x) #2 {
entry:
  %call = call spir_func float @_Z4cbrtf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4cbrtf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_cbrt_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4cbrtd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4cbrtd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_ceil(float %x) #2 {
entry:
  %call = call spir_func float @_Z4ceilf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4ceilf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_ceil_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4ceild(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4ceild(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_copysign(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z8copysignff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z8copysignff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_copysign_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z8copysigndd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z8copysigndd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_cos(float %x) #2 {
entry:
  %call = call spir_func float @_Z3cosf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3cosf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_cos_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z3cosd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3cosd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_cosh(float %x) #2 {
entry:
  %call = call spir_func float @_Z4coshf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4coshf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_cosh_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4coshd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4coshd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_cospi(float %x) #2 {
entry:
  %call = call spir_func float @_Z5cospif(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5cospif(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_cospi_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5cospid(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5cospid(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_erf(float %x) #2 {
entry:
  %call = call spir_func float @_Z3erff(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3erff(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_erf_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z3erfd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3erfd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_erfc(float %x) #2 {
entry:
  %call = call spir_func float @_Z4erfcf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4erfcf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_erfc_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4erfcd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4erfcd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_exp(float %x) #2 {
entry:
  %call = call spir_func float @_Z3expf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3expf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_exp_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z3expd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3expd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_exp2(float %x) #2 {
entry:
  %call = call spir_func float @_Z4exp2f(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4exp2f(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_exp2_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4exp2d(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4exp2d(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_exp10(float %x) #2 {
entry:
  %call = call spir_func float @_Z5exp10f(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5exp10f(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_exp10_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5exp10d(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5exp10d(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_expm1(float %x) #2 {
entry:
  %call = call spir_func float @_Z5expm1f(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5expm1f(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_expm1_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5expm1d(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5expm1d(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_fabs(float %x) #2 {
entry:
  %call = call spir_func float @_Z4fabsf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fabsf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_fabs_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4fabsd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4fabsd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_fdim(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z4fdimff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fdimff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_fdim_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z4fdimdd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4fdimdd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_floor(float %x) #2 {
entry:
  %call = call spir_func float @_Z5floorf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5floorf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_floor_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5floord(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5floord(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_fma(float %x, float %y, float %z) #2 {
entry:
  %call = call spir_func float @_Z3fmafff(float %x, float %y, float %z) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3fmafff(float, float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_fma_double(double %x, double %y, double %z) #2 {
entry:
  %call = call spir_func double @_Z3fmaddd(double %x, double %y, double %z) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3fmaddd(double, double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_fmax(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z3maxff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3maxff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_fmax_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z3maxdd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3maxdd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_fmin(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z3minff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3minff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_fmin_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z3mindd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3mindd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_fmod(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z4fmodff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fmodff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_fmod_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z4fmoddd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4fmoddd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_frexpf_global(float %x, i32 addrspace(1)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(1)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %x, i32 addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5frexpfPU3AS4i(float, i32 addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_frexpf_local(float %x, i32 addrspace(3)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(3)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %x, i32 addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_frexpf(float %x, i32 addrspace(4)* readnone %exp) #2 {
entry:
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %x, i32 addrspace(4)* %exp) #1
  ret float %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_frexp_global(double %x, i32 addrspace(1)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(1)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %x, i32 addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5frexpdPU3AS4i(double, i32 addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_frexp_local(double %x, i32 addrspace(3)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(3)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %x, i32 addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_frexp(double %x, i32 addrspace(4)* readnone %exp) #2 {
entry:
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %x, i32 addrspace(4)* %exp) #1
  ret double %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_hypot(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z5hypotff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5hypotff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_hypot_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z5hypotdd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5hypotdd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_ilogb(float %x) #2 {
entry:
  %call = call spir_func i32 @_Z5ilogbf(float %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5ilogbf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_ilogb_double(double %x) #2 {
entry:
  %call = call spir_func i32 @_Z5ilogbd(double %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5ilogbd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isfinite(float %x) #2 {
entry:
  %call = call spir_func i32 @_Z8isfinitef(float %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isfinitef(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isfinite_double(double %x) #2 {
entry:
  %call = call spir_func i32 @_Z8isfinited(double %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isfinited(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isinf(float %x) #2 {
entry:
  %call = call spir_func i32 @_Z5isinff(float %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isinff(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isinf_double(double %x) #2 {
entry:
  %call = call spir_func i32 @_Z5isinfd(double %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isinfd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isnan(float %x) #2 {
entry:
  %call = call spir_func i32 @_Z5isnanf(float %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isnanf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isnan_double(double %x) #2 {
entry:
  %call = call spir_func i32 @_Z5isnand(double %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z5isnand(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isnormal(float %x) #2 {
entry:
  %call = call spir_func i32 @_Z8isnormalf(float %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isnormalf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_isnormal_double(double %x) #2 {
entry:
  %call = call spir_func i32 @_Z8isnormald(double %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z8isnormald(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_lgammaf_global(float %x, i32 addrspace(1)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(1)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func float @_Z8lgamma_rfPU3AS4i(float %x, i32 addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z8lgamma_rfPU3AS4i(float, i32 addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_lgammaf_local(float %x, i32 addrspace(3)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(3)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func float @_Z8lgamma_rfPU3AS4i(float %x, i32 addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_lgammaf(float %x, i32 addrspace(4)* readnone %exp) #2 {
entry:
  %call = call spir_func float @_Z8lgamma_rfPU3AS4i(float %x, i32 addrspace(4)* %exp) #1
  ret float %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_lgamma_global(double %x, i32 addrspace(1)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(1)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func double @_Z8lgamma_rdPU3AS4i(double %x, i32 addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z8lgamma_rdPU3AS4i(double, i32 addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_lgamma_local(double %x, i32 addrspace(3)* %exp) #2 {
entry:
  %0 = ptrtoint i32 addrspace(3)* %exp to i64
  %1 = inttoptr i64 %0 to i32 addrspace(4)*
  %call = call spir_func double @_Z8lgamma_rdPU3AS4i(double %x, i32 addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_lgamma(double %x, i32 addrspace(4)* readnone %exp) #2 {
entry:
  %call = call spir_func double @_Z8lgamma_rdPU3AS4i(double %x, i32 addrspace(4)* %exp) #1
  ret double %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_log(float %x) #2 {
entry:
  %call = call spir_func float @_Z3logf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3logf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_log_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z3logd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3logd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_log10(float %x) #2 {
entry:
  %call = call spir_func float @_Z5log10f(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5log10f(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_log10_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5log10d(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5log10d(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_log2(float %x) #2 {
entry:
  %call = call spir_func float @_Z4log2f(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4log2f(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_log2_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4log2d(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4log2d(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_log1p(float %x) #2 {
entry:
  %call = call spir_func float @_Z5log1pf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5log1pf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_log1p_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5log1pd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5log1pd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_logb(float %x) #2 {
entry:
  %call = call spir_func float @_Z4logbf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4logbf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_logb_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4logbd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4logbd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_modff_global(float %x, float addrspace(1)* %iptr) #2 {
entry:
  %0 = ptrtoint float addrspace(1)* %iptr to i64
  %1 = inttoptr i64 %0 to float addrspace(4)*
  %call = call spir_func float @_Z4modffPU3AS4f(float %x, float addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4modffPU3AS4f(float, float addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_modff_local(float %x, float addrspace(3)* %iptr) #2 {
entry:
  %0 = ptrtoint float addrspace(3)* %iptr to i64
  %1 = inttoptr i64 %0 to float addrspace(4)*
  %call = call spir_func float @_Z4modffPU3AS4f(float %x, float addrspace(4)* %1) #1
  ret float %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_modff(float %x, float addrspace(4)* readnone %iptr) #2 {
entry:
  %call = call spir_func float @_Z4modffPU3AS4f(float %x, float addrspace(4)* %iptr) #1
  ret float %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_modf_global(double %x, double addrspace(1)* %iptr) #2 {
entry:
  %0 = ptrtoint double addrspace(1)* %iptr to i64
  %1 = inttoptr i64 %0 to double addrspace(4)*
  %call = call spir_func double @_Z4modfdPU3AS4d(double %x, double addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4modfdPU3AS4d(double, double addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_modf_local(double %x, double addrspace(3)* %iptr) #2 {
entry:
  %0 = ptrtoint double addrspace(3)* %iptr to i64
  %1 = inttoptr i64 %0 to double addrspace(4)*
  %call = call spir_func double @_Z4modfdPU3AS4d(double %x, double addrspace(4)* %1) #1
  ret double %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_modf(double %x, double addrspace(4)* readnone %iptr) #2 {
entry:
  %call = call spir_func double @_Z4modfdPU3AS4d(double %x, double addrspace(4)* %iptr) #1
  ret double %call
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_nan(i32 %tagp) #2 {
entry:
  %call = call spir_func float @_Z3nanj(i32 %tagp) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3nanj(i32) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_nan_double(i32 %tagp) #2 {
entry:
  %conv = sext i32 %tagp to i64
  %call = call spir_func double @_Z3nanm(i64 %conv) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3nanm(i64) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_nearbyint(float %x) #2 {
entry:
  %call = call spir_func float @_Z4rintf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4rintf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_nearbyint_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4rintd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4rintd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_nextafter(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z9nextafterff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z9nextafterff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_nextafter_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z9nextafterdd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z9nextafterdd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_pow(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z3powff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3powff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_pow_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z3powdd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3powdd(double, double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_remainder(float %x, float %y) #2 {
entry:
  %call = call spir_func float @_Z9remainderff(float %x, float %y) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z9remainderff(float, float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_remainder_double(double %x, double %y) #2 {
entry:
  %call = call spir_func double @_Z9remainderdd(double %x, double %y) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z9remainderdd(double, double) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hc_remquof_global(float %x, float %y, i32 addrspace(1)* nocapture readnone %quo) #0 {
entry:
  ret float 0.000000e+00
}

declare spir_func float @_Z6remquoffPU3AS4i(float, float, i32 addrspace(4)*)

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hc_remquof_local(float %x, float %y, i32 addrspace(3)* nocapture readnone %quo) #0 {
entry:
  ret float 0.000000e+00
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func float @__hc_remquof(float %x, float %y, i32 addrspace(4)* nocapture readnone %quo) #0 {
entry:
  ret float 0.000000e+00
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hc_remquo_global(double %x, double %y, i32 addrspace(1)* nocapture readnone %quo) #0 {
entry:
  ret double 0.000000e+00
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z6remquoddPU3AS4i(double, double, i32 addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hc_remquo_local(double %x, double %y, i32 addrspace(3)* nocapture readnone %quo) #0 {
entry:
  ret double 0.000000e+00
}

; Function Attrs: alwaysinline nounwind readnone
define linkonce_odr spir_func double @__hc_remquo(double %x, double %y, i32 addrspace(4)* nocapture readnone %quo) #0 {
entry:
  ret double 0.000000e+00
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_round(float %x) #2 {
entry:
  %call = call spir_func float @_Z5roundf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5roundf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_round_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5roundd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5roundd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_rsqrt(float %x) #2 {
entry:
  %call = call spir_func float @_Z5rsqrtf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5rsqrtf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_rsqrt_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5rsqrtd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5rsqrtd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_sinpi(float %x) #2 {
entry:
  %call = call spir_func float @_Z5sinpif(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5sinpif(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_sinpi_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5sinpid(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5sinpid(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_ldexp(float %x, i32 %exp) #2 {
entry:
  %call = call spir_func float @_Z5ldexpfi(float %x, i32 %exp) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5ldexpfi(float, i32) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_ldexp_double(double %x, i32 %exp) #2 {
entry:
  %call = call spir_func double @_Z5ldexpdi(double %x, i32 %exp) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5ldexpdi(double, i32) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_signbit(float %x) #2 {
entry:
  %call = call spir_func i32 @_Z7signbitf(float %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z7signbitf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hc_signbit_double(double %x) #2 {
entry:
  %call = call spir_func i32 @_Z7signbitd(double %x) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z7signbitd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_sin(float %x) #2 {
entry:
  %call = call spir_func float @_Z3sinf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3sinf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_sin_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z3sind(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3sind(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @__hc_sincosf_global(float %x, float addrspace(1)* nocapture %s, float addrspace(1)* %c) #2 {
entry:
  %0 = addrspacecast float addrspace(1)* %c to float addrspace(4)*
  %call = call spir_func float @_Z6sincosfPU3AS4f(float %x, float addrspace(4)* %0) #1
  store float %call, float addrspace(1)* %s, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z6sincosfPU3AS4f(float, float addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @__hc_sincosf_local(float %x, float addrspace(3)* nocapture %s, float addrspace(3)* %c) #2 {
entry:
  %0 = addrspacecast float addrspace(3)* %c to float addrspace(4)*
  %call = call spir_func float @_Z6sincosfPU3AS4f(float %x, float addrspace(4)* %0) #1
  store float %call, float addrspace(3)* %s, align 4
  ret void
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @__hc_sincosf(float %x, float addrspace(4)* nocapture %s, float addrspace(4)* %c) #2 {
entry:
  %call = call spir_func float @_Z6sincosfPU3AS4f(float %x, float addrspace(4)* %c) #1
  store float %call, float addrspace(4)* %s, align 4
  ret void
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @__hc_sincos_global(double %x, double addrspace(1)* nocapture %s, double addrspace(1)* %c) #2 {
entry:
  %0 = addrspacecast double addrspace(1)* %c to double addrspace(4)*
  %call = call spir_func double @_Z6sincosdPU3AS4d(double %x, double addrspace(4)* %0) #1
  store double %call, double addrspace(1)* %s, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z6sincosdPU3AS4d(double, double addrspace(4)*) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @__hc_sincos_local(double %x, double addrspace(3)* nocapture %s, double addrspace(3)* %c) #2 {
entry:
  %0 = addrspacecast double addrspace(3)* %c to double addrspace(4)*
  %call = call spir_func double @_Z6sincosdPU3AS4d(double %x, double addrspace(4)* %0) #1
  store double %call, double addrspace(3)* %s, align 8
  ret void
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func void @__hc_sincos(double %x, double addrspace(4)* nocapture %s, double addrspace(4)* %c) #2 {
entry:
  %call = call spir_func double @_Z6sincosdPU3AS4d(double %x, double addrspace(4)* %c) #1
  store double %call, double addrspace(4)* %s, align 8
  ret void
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_sinh(float %x) #2 {
entry:
  %call = call spir_func float @_Z4sinhf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4sinhf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_sinh_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4sinhd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4sinhd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_sqrt(float %x) #2 {
entry:
  %call = call spir_func float @_Z4sqrtf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4sqrtf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_sqrt_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4sqrtd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4sqrtd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_tgamma(float %x) #2 {
entry:
  %call = call spir_func float @_Z6tgammaf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z6tgammaf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_tgamma_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z6tgammad(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z6tgammad(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_tan(float %x) #2 {
entry:
  %call = call spir_func float @_Z3tanf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z3tanf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_tan_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z3tand(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z3tand(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_tanh(float %x) #2 {
entry:
  %call = call spir_func float @_Z4tanhf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4tanhf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_tanh_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z4tanhd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z4tanhd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_tanpi(float %x) #2 {
entry:
  %call = call spir_func float @_Z5tanpif(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5tanpif(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_tanpi_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5tanpid(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5tanpid(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @__hc_trunc(float %x) #2 {
entry:
  %call = call spir_func float @_Z5truncf(float %x) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z5truncf(float) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func double @__hc_trunc_double(double %x) #2 {
entry:
  %call = call spir_func double @_Z5truncd(double %x) #1
  ret double %call
}

; Function Attrs: nounwind readnone
declare spir_func double @_Z5truncd(double) #1

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_exchange_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_compare_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_compare_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_compare_exchange_unsigned(i32 addrspace(4)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(4)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_sub_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_sub_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_sub_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_exchange_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_exchange_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_exchange_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_compare_exchange_int_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_compare_exchange_int_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_compare_exchange_int(i32 addrspace(4)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(4)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_add_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_add_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_add_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_sub_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_sub_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_sub_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_exchange_float_global(float addrspace(1)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %xchg = fadd float %y, 0.000000e+00
  %3 = bitcast float %xchg to i32
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %xchg
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_exchange_float_local(float addrspace(3)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %xchg = fadd float %y, 0.000000e+00
  %3 = bitcast float %xchg to i32
  %val_success = cmpxchg i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %xchg
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_exchange_float(float addrspace(4)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %xchg = fadd float %y, 0.000000e+00
  %3 = bitcast float %xchg to i32
  %val_success = cmpxchg i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %xchg
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_add_float_global(float addrspace(1)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %add
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_add_float_local(float addrspace(3)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %add
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_add_float(float addrspace(4)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %add = fadd float %1, %y
  %3 = bitcast float %add to i32
  %val_success = cmpxchg i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %add
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_sub_float_global(float addrspace(1)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(1)* %x to i32 addrspace(1)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(1)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg i32 addrspace(1)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %sub
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_sub_float_local(float addrspace(3)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(3)* %x to i32 addrspace(3)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(3)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg i32 addrspace(3)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %sub
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func float @atomic_sub_float(float addrspace(4)* %x, float %y) #2 {
entry:
  %0 = bitcast float addrspace(4)* %x to i32 addrspace(4)*
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %1 = load volatile float, float addrspace(4)* %x, align 4
  %2 = bitcast float %1 to i32
  %sub = fsub float %1, %y
  %3 = bitcast float %sub to i32
  %val_success = cmpxchg i32 addrspace(4)* %0, i32 %2, i32 %3 seq_cst seq_cst, !mem.scope !1
  %success = extractvalue { i32, i1 } %val_success, 1
  br i1 %success, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret float %sub
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_exchange_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_exchange_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_exchange_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_compare_exchange_uint64_global(i64 addrspace(1)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(1)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_compare_exchange_uint64_local(i64 addrspace(3)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(3)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_compare_exchange_uint64(i64 addrspace(4)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(4)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_add_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_add_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_add_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_and_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_or_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_xor_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw max i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_min_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw min i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_and_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_or_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_xor_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw max i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_min_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw min i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_and_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_or_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_xor_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw max i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_min_unsigned(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw min i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_and_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_or_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_xor_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_max_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw max i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_min_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw min i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_and_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_or_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_xor_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_max_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw max i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_min_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw min i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_and_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_or_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_xor_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_max_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw max i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_min_int(i32 addrspace(4)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw min i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_and_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_or_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_xor_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_max_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw max i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_min_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw min i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_and_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_or_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_xor_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_max_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw max i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_min_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw min i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_and_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_or_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_xor_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_max_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw max i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @atomic_min_uint64(i64 addrspace(4)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw min i64 addrspace(4)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_dec_unsigned_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_dec_unsigned_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_dec_unsigned(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_inc_int_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_dec_int_global(i32 addrspace(1)* %x) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_inc_int_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_dec_int_local(i32 addrspace(3)* %x) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_inc_int(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @atomic_dec_int(i32 addrspace(4)* %x) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_and_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_or_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_xor_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_and_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_or_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_xor_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_add_int64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_sub_int64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw sub i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_and_int64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_or_int64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_xor_int64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_add_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_sub_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw sub i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_and_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_or_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_xor_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_and_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_or_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_xor_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_add_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_sub_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw sub i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_and_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw and i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_or_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw or i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_fetch_xor_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xor i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_add_int64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_sub_int64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw sub i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_and_int64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_or_int64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_xor_int64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_add_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw add i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_sub_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw sub i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_and_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw and i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_or_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw or i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_fetch_xor_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xor i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_int_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_exchange_int64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_exchange_uint64_local(i64 addrspace(3)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(3)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_int_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y) #2 {
entry:
  %ret = atomicrmw xchg i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !1
  ret i32 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_exchange_int64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_exchange_uint64_global(i64 addrspace(1)* %x, i64 %y) #2 {
entry:
  %ret = atomicrmw xchg i64 addrspace(1)* %x, i64 %y seq_cst, !mem.scope !1
  ret i64 %ret
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_int_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_unsigned_local(i32 addrspace(3)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(3)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_compare_exchange_int64_local(i64 addrspace(3)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(3)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_compare_exchange_uint64_local(i64 addrspace(3)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(3)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_int_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i32 @__hsail_atomic_compare_exchange_unsigned_global(i32 addrspace(1)* %x, i32 %y, i32 %z) #2 {
entry:
  %val_success = cmpxchg i32 addrspace(1)* %x, i32 %y, i32 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i32, i1 } %val_success, 0
  ret i32 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_compare_exchange_int64_global(i64 addrspace(1)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(1)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

; Function Attrs: alwaysinline nounwind
define linkonce_odr spir_func i64 @__hsail_atomic_compare_exchange_uint64_global(i64 addrspace(1)* %x, i64 %y, i64 %z) #2 {
entry:
  %val_success = cmpxchg i64 addrspace(1)* %x, i64 %y, i64 %z seq_cst seq_cst, !mem.scope !1
  %value_loaded = extractvalue { i64, i1 } %val_success, 0
  ret i64 %value_loaded
}

attributes #0 = { alwaysinline nounwind readnone }
attributes #1 = { alwaysinline nounwind readnone }
attributes #2 = { alwaysinline nounwind }
attributes #3 = { nounwind }

!0 = !{}
!1 = !{i32 5}
