; ModuleID = 'hsa_math.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

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
define linkonce_odr spir_func float @opencl_fmax(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z4fmaxff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fmaxff(float, float) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_fmin(float %x, float %y) #0 {
entry:
  %x.addr = alloca float, align 4
  %y.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  store float %y, float* %y.addr, align 4
  %0 = load float* %x.addr, align 4
  %1 = load float* %y.addr, align 4
  %call = call spir_func float @_Z4fminff(float %0, float %1) #1
  ret float %call
}

; Function Attrs: nounwind readnone
declare spir_func float @_Z4fminff(float, float) #1

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
  %call = call spir_func float @_Z4modffPU3AS4f(float %0, float addrspace(4)* %3)
  ret float %call
}

declare spir_func float @_Z4modffPU3AS4f(float, float addrspace(4)*)

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
  %call = call spir_func float @_Z4modffPU3AS4f(float %0, float addrspace(4)* %3)
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
  %call = call spir_func float @_Z4modffPU3AS4f(float %0, float addrspace(4)* %1)
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
  %call = call spir_func double @_Z4modfdPU3AS4d(double %0, double addrspace(4)* %3)
  ret double %call
}

declare spir_func double @_Z4modfdPU3AS4d(double, double addrspace(4)*)

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
  %call = call spir_func double @_Z4modfdPU3AS4d(double %0, double addrspace(4)* %3)
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
  %call = call spir_func double @_Z4modfdPU3AS4d(double %0, double addrspace(4)* %1)
  ret double %call
}

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
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %0, i32 addrspace(4)* %3)
  ret float %call
}

declare spir_func float @_Z5frexpfPU3AS4i(float, i32 addrspace(4)*)

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
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %0, i32 addrspace(4)* %3)
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
  %call = call spir_func float @_Z5frexpfPU3AS4i(float %0, i32 addrspace(4)* %1)
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
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %0, i32 addrspace(4)* %3)
  ret double %call
}

declare spir_func double @_Z5frexpdPU3AS4i(double, i32 addrspace(4)*)

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
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %0, i32 addrspace(4)* %3)
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
  %call = call spir_func double @_Z5frexpdPU3AS4i(double %0, i32 addrspace(4)* %1)
  ret double %call
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @opencl_min(i32 %x, i32 %y) #0 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32* %x.addr, align 4
  %1 = load i32* %y.addr, align 4
  %call = call spir_func i32 @_Z3minii(i32 %0, i32 %1) #1
  ret i32 %call
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z3minii(i32, i32) #1

; Function Attrs: nounwind
define linkonce_odr spir_func float @opencl_max(float %x, float %y) #0 {
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
define linkonce_odr spir_func i32 @atomic_add_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_unsigned(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_add_int(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_unsigned(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_int_global(i32 addrspace(1)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(1)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_int_local(i32 addrspace(3)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(3)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_max_int(i32 addrspace(4)* %x, i32 %y) #0 {
entry:
  %ret = atomicrmw max i32 addrspace(4)* %x, i32 %y seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned_global(i32 addrspace(1)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned_local(i32 addrspace(3)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_unsigned(i32 addrspace(4)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_int_global(i32 addrspace(1)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(1)* %x, i32 1 seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_int_local(i32 addrspace(3)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(3)* %x, i32 1 seq_cst, !mem.scope !4
  ret i32 %ret
}

; Function Attrs: nounwind
define linkonce_odr spir_func i32 @atomic_inc_int(i32 addrspace(4)* %x) #0 {
entry:
  %ret = atomicrmw add i32 addrspace(4)* %x, i32 1 seq_cst, !mem.scope !4
  ret i32 %ret
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}

!0 = metadata !{}
!1 = metadata !{}
!2 = metadata !{}
!3 = metadata !{}
!4 = metadata !{i32 5}
