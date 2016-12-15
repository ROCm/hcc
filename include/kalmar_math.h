//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
  #include <cmath>
#if __KALMAR_ACCELERATOR__ == 1
  extern "C" float __hc_acos(float x) restrict(amp);
  extern "C" double __hc_acos_double(double x) restrict(amp);

  extern "C" float __hc_acosh(float x) restrict(amp);
  extern "C" double __hc_acosh_double(double x) restrict(amp);

  extern "C" float __hc_asin(float x) restrict(amp);
  extern "C" double __hc_asin_double(double x) restrict(amp);

  extern "C" float __hc_asinh(float x) restrict(amp);
  extern "C" double __hc_asinh_double(double x) restrict(amp);

  extern "C" float __hc_atan(float x) restrict(amp);
  extern "C" double __hc_atan_double(double x) restrict(amp);

  extern "C" float __hc_atanh(float x) restrict(amp);
  extern "C" double __hc_atanh_double(double x) restrict(amp);

  extern "C" float __hc_atan2(float x, float y) restrict(amp);
  extern "C" double __hc_atan2_double(double x, double y) restrict(amp);

  extern "C" float __hc_cbrt(float x) restrict(amp);
  extern "C" double __hc_cbrt_double(double x) restrict(amp);

  extern "C" float __hc_ceil(float x) restrict(amp);
  extern "C" double __hc_ceil_double(double x) restrict(amp);

  extern "C" float __hc_copysign(float x, float y) restrict(amp);
  extern "C" double __hc_copysign_double(double x, double y) restrict(amp);

  extern "C" float __hc_cos(float x) restrict(amp);
  extern "C" double __hc_cos_double(double x) restrict(amp);

  extern "C" float __hc_cosh(float x) restrict(amp);
  extern "C" double __hc_cosh_double(double x) restrict(amp);

  extern "C" float __hc_cospi(float x) restrict(amp);
  extern "C" double __hc_cospi_double(double x) restrict(amp);

  extern "C" float __hc_erf(float x) restrict(amp);
  extern "C" double __hc_erf_double(double x) restrict(amp);

  extern "C" float __hc_erfc(float x) restrict(amp);
  extern "C" double __hc_erfc_double(double x) restrict(amp);

  /* FIXME missing erfinv */

  /* FIXME missing erfcinv */

  extern "C" float __hc_exp(float x) restrict(amp);
  extern "C" double __hc_exp_double(double x) restrict(amp);

  extern "C" float __hc_exp10(float x) restrict(amp);
  extern "C" double __hc_exp10_double(double x) restrict(amp);

  extern "C" float __hc_exp2(float x) restrict(amp);
  extern "C" double __hc_exp2_double(double x) restrict(amp);

  extern "C" float __hc_expm1(float x) restrict(amp);
  extern "C" double __hc_expm1_double(double x) restrict(amp);

  extern "C" float __hc_fabs(float x) restrict(amp);
  extern "C" double __hc_fabs_double(double x) restrict(amp);

  extern "C" float __hc_fdim(float x, float y) restrict(amp);
  extern "C" double __hc_fdim_double(double x, double y) restrict(amp);

  extern "C" float __hc_floor(float x) restrict(amp);
  extern "C" double __hc_floor_double(double x) restrict(amp);

  extern "C" float __hc_fma(float x, float y, float z) restrict(amp);
  extern "C" double __hc_fma_double(double x, double y, double z) restrict(amp);

  extern "C" float __hc_fmax(float x, float y) restrict(amp);
  extern "C" double __hc_fmax_double(double x, double y) restrict(amp);

  extern "C" float __hc_fmin(float x, float y) restrict(amp);
  extern "C" double __hc_fmin_double(double x, double y) restrict(amp);

  extern "C" float __hc_fmod(float x, float y) restrict(amp);
  extern "C" double __hc_fmod_double(double x, double y) restrict(amp);

  /* FIXME missing fpclassify */

  /* SPIR pass will deduce the address space of the pointer and emit correct
     call instruction */
  extern "C" float __hc_frexpf(float x, int *exp) restrict(amp);
  extern "C" double __hc_frexp(double x, int *exp) restrict(amp);

  extern "C" float __hc_hypot(float x, float y) restrict(amp);
  extern "C" double __hc_hypot_double(double x, double y) restrict(amp);

  extern "C" int __hc_ilogb(float x) restrict(amp);
  extern "C" int __hc_ilogb_double(double x) restrict(amp);

  extern "C" int __hc_isfinite(float x) restrict(amp);
  extern "C" int __hc_isfinite_double(double x) restrict(amp);

  extern "C" int __hc_isinf(float x) restrict(amp);
  extern "C" int __hc_isinf_double(double x) restrict(amp);

  extern "C" int __hc_isnan(float x) restrict(amp);
  extern "C" int __hc_isnan_double(double x) restrict(amp);

  extern "C" int __hc_isnormal(float x) restrict(amp);
  extern "C" int __hc_isnormal_double(double x) restrict(amp);

  extern "C" float __hc_ldexp(float x, int exp) restrict(amp);
  extern "C" double __hc_ldexp_double(double x, int exp) restrict(amp);

  /* SPIR pass will deduce the address space of the pointer and emit correct
     call instruction */
  extern "C" float __hc_lgammaf(float x, int *exp) restrict(amp);
  extern "C" double __hc_lgamma(double x, int *exp) restrict(amp);

  extern "C" float __hc_log(float x) restrict(amp);
  extern "C" double __hc_log_double(double x) restrict(amp);

  extern "C" float __hc_log10(float x) restrict(amp);
  extern "C" double __hc_log10_double(double x) restrict(amp);

  extern "C" float __hc_log2(float x) restrict(amp);
  extern "C" double __hc_log2_double(double x) restrict(amp);

  extern "C" float __hc_log1p(float x) restrict(amp);
  extern "C" double __hc_log1p_double(double x) restrict(amp);

  extern "C" float __hc_logb(float x) restrict(amp);
  extern "C" double __hc_logb_double(double x) restrict(amp);

  /* SPIR pass will deduce the address space of the pointer and emit correct
     call instruction */
  extern "C" float __hc_modff(float x, float *iptr) restrict(amp);
  extern "C" double __hc_modf(double x, double *iptr) restrict(amp);

  extern "C" float __hc_nan(int tagp) restrict(amp);
  extern "C" double __hc_nan_double(unsigned long tagp) restrict(amp);

  extern "C" float __hc_nearbyint(float x) restrict(amp);
  extern "C" double __hc_nearbyint_double(double x) restrict(amp);

  extern "C" float __hc_nextafter(float x, float y) restrict(amp);
  extern "C" double __hc_nextafter_double(double x, double y) restrict(amp);

  extern "C" float __hc_pow(float x, float y) restrict(amp);
  extern "C" double __hc_pow_double(double x, double y) restrict(amp);

  extern "C" float __hc_remainder(float x, float y) restrict(amp);
  extern "C" double __hc_remainder_double(double x, double y) restrict(amp);

  /* SPIR pass will deduce the address space of the pointer and emit correct
     call instruction */
  extern "C" float __hc_remquof(float x, float y, int *quo) restrict(amp);
  extern "C" double __hc_remquo(double x, double y, int *quo) restrict(amp);

  extern "C" float __hc_round(float x) restrict(amp);
  extern "C" double __hc_round_double(double x) restrict(amp);

  extern "C" float __hc_rsqrt(float x) restrict(amp);
  extern "C" double __hc_rsqrt_double(double x) restrict(amp);

  extern "C" float __hc_sinpi(float x) restrict(amp);
  extern "C" double __hc_sinpi_double(double x) restrict(amp);

  extern "C" int   __hc_signbit(float x) restrict(amp);
  extern "C" int   __hc_signbit_double(double x) restrict(amp);

  extern "C" float __hc_sin(float x) restrict(amp);
  extern "C" double __hc_sin_double(double x) restrict(amp);

  /* SPIR pass will deduce the address space of the pointer and emit correct
     call instruction */
  extern "C" void __hc_sincosf(float x, float *s, float *c) restrict(amp);
  extern "C" void __hc_sincos(double x, double *s, double *c) restrict(amp);

  extern "C" float __hc_sinh(float x) restrict(amp);
  extern "C" double __hc_sinh_double(double x) restrict(amp);

  extern "C" float __hc_sqrt(float x) restrict(amp);
  extern "C" double __hc_sqrt_double(double x) restrict(amp);

  extern "C" float __hc_tgamma(float x) restrict(amp);
  extern "C" double __hc_tgamma_double(double x) restrict(amp);

  extern "C" float __hc_tan(float x) restrict(amp);
  extern "C" double __hc_tan_double(double x) restrict(amp);

  extern "C" float __hc_tanh(float x) restrict(amp);
  extern "C" double __hc_tanh_double(double x) restrict(amp);

  extern "C" float __hc_tanpi(float x) restrict(amp);
  extern "C" double __hc_tanpi_double(double x) restrict(amp);

  extern "C" float __hc_trunc(float x) restrict(amp);
  extern "C" double __hc_trunc_double(double x) restrict(amp);


#endif

namespace Kalmar {
namespace fast_math {

  inline float host_acosf(float x) restrict(cpu) { return ::acosf(x); }
  inline float acosf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acosf(x);
    #endif
  }

  inline float host_acos(float x) restrict(cpu) { return ::acosf(x); }
  inline float acos(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acos(x);
    #endif
  }

  inline float host_asinf(float x) restrict(cpu) { return ::asinf(x); }
  inline float asinf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asinf(x);
    #endif
  }

  inline float host_asin(float x) restrict(cpu) { return ::asinf(x); }
  inline float asin(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asin(x);
    #endif
  }

  inline float host_atanf(float x) restrict(cpu) { return ::atanf(x); }
  inline float atanf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atanf(x);
    #endif
  }

  inline float host_atan(float x) restrict(cpu) { return ::atanf(x); }
  inline float atan(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atan(x);
    #endif
  }

  inline float host_atan2f(float x, float y) restrict(cpu) { return ::atan2f(x, y); }
  inline float atan2f(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(x, y);
    #else
      return host_atan2f(x, y);
    #endif
  }

  inline float host_atan2(float x, float y) restrict(cpu) { return ::atan2f(x, y); }
  inline float atan2(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(x, y);
    #else
      return host_atan2(x, y);
    #endif
  }

  inline float host_ceilf(float x) restrict(cpu) { return ::ceilf(x); }
  inline float ceilf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceilf(x);
    #endif
  }

  inline float host_ceil(float x) restrict(cpu) { return ::ceilf(x); }
  inline float ceil(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceil(x);
    #endif
  }

  inline float host_cosf(float x) restrict(cpu) { return ::cosf(x); }
  inline float cosf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos(x);
    #else
      return host_cosf(x);
    #endif
  }

  inline float host_cos(float x) restrict(cpu) { return ::cosf(x); }
  inline float cos(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos(x);
    #else
      return host_cos(x);
    #endif
  }

  inline float host_coshf(float x) restrict(cpu) { return ::coshf(x); }
  inline float coshf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_coshf(x);
    #endif
  }

  inline float host_cosh(float x) restrict(cpu) { return ::coshf(x); }
  inline float cosh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_cosh(x);
    #endif
  }

  inline float host_expf(float x) restrict(cpu) { return ::expf(x); }
  inline float expf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp(x);
    #else
      return host_expf(x);
    #endif
  }

  inline float host_exp(float x) restrict(cpu) { return ::expf(x); }
  inline float exp(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp(x);
    #else
      return host_exp(x);
    #endif
  }

  inline float host_exp2f(float x) restrict(cpu) { return ::exp2f(x); }
  inline float exp2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2(x);
    #else
      return host_exp2f(x);
    #endif
  }

  inline float host_exp2(float x) restrict(cpu) { return ::exp2f(x); }
  inline float exp2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2(x);
    #else
      return host_exp2(x);
    #endif
  }

  inline float host_fabsf(float x) restrict(cpu) { return ::fabsf(x); }
  inline float fabsf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabsf(x);
    #endif
  }

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  inline float host_fabs(float x) restrict(cpu,amp) { return std::fabs(x); }
#else
  inline float host_fabs(float x) restrict(cpu) { return ::fabsf(x); }
#endif
  inline float fabs(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabs(x);
    #endif
  }

  inline float host_floorf(float x) restrict(cpu) { return ::floorf(x); }
  inline float floorf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floorf(x);
    #endif
  }

  inline float host_floor(float x) restrict(cpu) { return ::floorf(x); }
  inline float floor(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floor(x);
    #endif
  }

  inline float host_fmaxf(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float fmaxf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmaxf(x, y);
    #endif
  }

  inline float host_fmax(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float fmax(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }

  inline float host_max(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float max(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_max(x, y);
    #endif
  }

  inline float host_fminf(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float fminf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fminf(x, y);
    #endif
  }

  inline float host_fmin(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float fmin(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }

  inline float host_min(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float min(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_min(x, y);
    #endif
  }

  inline float host_fmodf(float x, float y) restrict(cpu) { return ::fmodf(x, y); }
  inline float fmodf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmodf(x, y);
    #endif
  }

  inline float host_fmod(float x, float y) restrict(cpu) { return ::fmodf(x, y); }
  inline float fmod(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }

  inline float host_frexpf(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
  inline float frexpf(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexpf(x, exp);
    #else
      return host_frexpf(x, exp);
    #endif
  }

  inline float host_frexp(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
  inline float frexp(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexpf(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline int host_isfinite(float x) restrict(cpu) { return ::isfinite(x); }
  inline int isfinite(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isfinite(x);
    #else
      return host_isfinite(x);
    #endif
  }

  inline int host_isinf(float x) restrict(cpu) { return ::isinf(x); }
  inline int isinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isinf(x);
    #else
      return host_isinf(x);
    #endif
  }

  inline int host_isnan(float x) restrict(cpu) { return ::isnan(x); }
  inline int isnan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnan(x);
    #elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      return std::isnan(x);
    #else
      return host_isnan(x);
    #endif
  }

  inline float host_ldexpf(float x, int exp) restrict(cpu) { return ::ldexpf(x,exp); }
  inline float ldexpf(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexpf(x,exp);
    #endif
  }

  inline float host_ldexp(float x, int exp) restrict(cpu) { return ::ldexpf(x,exp); }
  inline float ldexp(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }

  inline float host_logf(float x) restrict(cpu) { return ::logf(x); }
  inline float logf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log(x);
    #else
      return host_logf(x);
    #endif
  }

  inline float host_log(float x) restrict(cpu) { return ::logf(x); }
  inline float log(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log(x);
    #else
      return host_log(x);
    #endif
  }

  inline float host_log10f(float x) restrict(cpu) { return ::log10f(x); }
  inline float log10f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10(x);
    #else
      return host_log10f(x);
    #endif
  }

  inline float host_log10(float x) restrict(cpu) { return ::log10f(x); }
  inline float log10(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10(x);
    #else
      return host_log10(x);
    #endif
  }

  inline float host_log2f(float x) restrict(cpu) { return ::log2f(x); }
  inline float log2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2(x);
    #else
      return host_log2f(x);
    #endif
  }

  inline float host_log2(float x) restrict(cpu) { return ::log2f(x); }
  inline float log2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2(x);
    #else
      return host_log2(x);
    #endif
  }

  inline float host_modff(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
  inline float modff(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modff(x, iptr);
    #else
      return host_modff(x, iptr);
    #endif
  }

  inline float host_modf(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
  inline float modf(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modff(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline float host_powf(float x, float y) restrict(cpu) { return ::powf(x, y); }
  inline float powf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_powf(x, y);
    #endif
  }

  inline float host_pow(float x, float y) restrict(cpu) { return ::powf(x, y); }
  inline float pow(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_pow(x, y);
    #endif
  }

  inline float host_roundf(float x) restrict(cpu) { return ::roundf(x); }
  inline float roundf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_roundf(x);
    #endif
  }

  inline float host_round(float x) restrict(cpu) { return ::roundf(x); }
  inline float round(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_round(x);
    #endif
  }

  inline float  host_rsqrtf(float x) restrict(cpu) { return 1.0f / (::sqrtf(x)); }
  inline float  rsqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt(x);
    #else
      return host_rsqrtf(x);
    #endif
  }

  inline float  host_rsqrt(float x) restrict(cpu) { return 1.0f / (::sqrtf(x)); }
  inline float  rsqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt(x);
    #else
      return host_rsqrt(x);
    #endif
  }

  inline int host_signbitf(float x) restrict(cpu) { return ::signbit(x); }
  inline int signbitf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbitf(x);
    #endif
  }

  inline int host_signbit(float x) restrict(cpu) { return ::signbit(x); }
  inline int signbit(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbit(x);
    #endif
  }

  inline float host_sinf(float x) restrict(cpu) { return ::sinf(x); }
  inline float sinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin(x);
    #else
      return host_sinf(x);
    #endif
  }

  inline float host_sin(float x) restrict(cpu) { return ::sinf(x); }
  inline float sin(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin(x);
    #else
      return host_sin(x);
    #endif
  }

  inline void host_sincosf(float x, float *s, float *c) restrict(cpu) { ::sincosf(x, s, c); }
  inline void sincosf(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      __hc_sincosf(x, s, c);
    #else
      host_sincosf(x, s, c);
    #endif
  }

  inline void host_sincos(float x, float *s, float *c) restrict(cpu) { ::sincosf(x, s, c); }
  inline void sincos(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      __hc_sincosf(x, s, c);
    #else
      host_sincos(x, s, c);
    #endif
  }

  inline float host_sinhf(float x) restrict(cpu) { return ::sinhf(x); }
  inline float sinhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinhf(x);
    #endif
  }

  inline float host_sinh(float x) restrict(cpu) { return ::sinhf(x); }
  inline float sinh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinh(x);
    #endif
  }

  inline float host_sqrtf(float x) restrict(cpu) { return ::sqrtf(x); }
  inline float sqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt(x);
    #else
      return host_sqrtf(x);
    #endif
  }

  inline float host_sqrt(float x) restrict(cpu) { return ::sqrtf(x); }
  inline float sqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt(x);
    #else
      return host_sqrt(x);
    #endif
  }

  inline float host_tanf(float x) restrict(cpu) { return ::tanf(x); }
  inline float tanf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan(x);
    #else
      return host_tanf(x);
    #endif
  }

  inline float host_tan(float x) restrict(cpu) { return ::tanf(x); }
  inline float tan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan(x);
    #else
      return host_tan(x);
    #endif
  }

  inline float host_tanhf(float x) restrict(cpu) { return ::tanhf(x); }
  inline float tanhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanhf(x);
    #endif
  }

  inline float host_tanh(float x) restrict(cpu) { return ::tanhf(x); }
  inline float tanh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }

  inline float host_truncf(float x) restrict(cpu) { return ::truncf(x); }
  inline float truncf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_truncf(x);
    #endif
  }

  inline float host_trunc(float x) restrict(cpu) { return ::truncf(x); }
  inline float trunc(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_trunc(x);
    #endif
  }

} // namesapce fast_math

  namespace precise_math {

  inline float host_acosf(float x) restrict(cpu) { return ::acosf(x); }
  inline float acosf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acosf(x);
    #endif
  }

  inline float host_acos(float x) restrict(cpu) { return ::acosf(x); }
  inline float acos(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acos(x);
    #endif
  }

  inline double host_acos(double x) restrict(cpu) { return ::acos(x); }
  inline double acos(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos_double(x);
    #else
      return host_acos(x);
    #endif
  }

  inline float host_acoshf(float x) restrict(cpu) { return ::acoshf(x); }
  inline float acoshf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acosh(x);
    #else
      return host_acoshf(x);
    #endif
  }

  inline float host_acosh(float x) restrict(cpu) { return ::acoshf(x); }
  inline float acosh(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acosh(x);
    #else
      return host_acosh(x);
    #endif
  }

  inline double host_acosh(double x) restrict(cpu) { return ::acosh(x); }
  inline double acosh(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acosh_double(x);
    #else
      return host_acosh(x);
    #endif
  }

  inline float host_asinf(float x) restrict(cpu) { return ::asinf(x); }
  inline float asinf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asinf(x);
    #endif
  }

  inline float host_asin(float x) restrict(cpu) { return ::asinf(x); }
  inline float asin(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asin(x);
    #endif
  }

  inline double host_asin(double x) restrict(cpu) { return ::asin(x); }
  inline double asin(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin_double(x);
    #else
      return host_asin(x);
    #endif
  }

  inline float host_asinhf(float x) restrict(cpu) { return ::asinhf(x); }
  inline float asinhf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asinh(x);
    #else
      return host_asinhf(x);
    #endif
  }

  inline float host_asinh(float x) restrict(cpu) { return ::asinhf(x); }
  inline float asinh(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asinh(x);
    #else
      return host_asinh(x);
    #endif
  }

  inline double host_asinh(double x) restrict(cpu) { return ::asinh(x); }
  inline double asinh(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asinh_double(x);
    #else
      return host_asinh(x);
    #endif
  }

  inline float host_atanf(float x) restrict(cpu) { return ::atanf(x); }
  inline float atanf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atanf(x);
    #endif
  }

  inline float host_atan(float x) restrict(cpu) { return ::atanf(x); }
  inline float atan(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atan(x);
    #endif
  }

  inline double host_atan(double x) restrict(cpu) { return ::atan(x); }
  inline double atan(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan_double(x);
    #else
      return host_atan(x);
    #endif
  }

  inline float host_atanhf(float x) restrict(cpu) { return ::atanhf(x); }
  inline float atanhf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atanh(x);
    #else
      return host_atanhf(x);
    #endif
  }

  inline float host_atanh(float x) restrict(cpu) { return ::atanhf(x); }
  inline float atanh(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atanh(x);
    #else
      return host_atanh(x);
    #endif
  }

  inline double host_atanh(double x) restrict(cpu) { return ::atanh(x); }
  inline double atanh(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atanh_double(x);
    #else
      return host_atanh(x);
    #endif
  }

  inline float host_atan2f(float x, float y) restrict(cpu) { return ::atan2f(x, y); }
  inline float atan2f(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(x, y);
    #else
      return host_atan2f(x, y);
    #endif
  }

  inline float host_atan2(float x, float y) restrict(cpu) { return ::atan2f(x, y); }
  inline float atan2(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(x, y);
    #else
      return host_atan2(x, y);
    #endif
  }

  inline double host_atan2(double x, double y) restrict(cpu) { return ::atan2(x, y); }
  inline double atan2(double x, double y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2_double(x, y);
    #else
      return host_atan2(x, y);
    #endif
  }

  inline float host_cbrtf(float x) restrict(cpu) { return ::cbrtf(x); }
  inline float cbrtf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cbrt(x);
    #else
      return host_cbrtf(x);
    #endif
  }

  inline float host_cbrt(float x) restrict(cpu) { return ::cbrtf(x); }
  inline float cbrt(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cbrt(x);
    #else
      return host_cbrt(x);
    #endif
  }

  inline double host_cbrt(double x) restrict(cpu) { return ::cbrt(x); }
  inline double cbrt(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cbrt_double(x);
    #else
      return host_cbrt(x);
    #endif
  }

  inline float host_ceilf(float x) restrict(cpu) { return ::ceilf(x); }
  inline float ceilf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceilf(x);
    #endif
  }

  inline float host_ceil(float x) restrict(cpu) { return ::ceilf(x); }
  inline float ceil(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceil(x);
    #endif
  }

  inline double host_ceil(double x) restrict(cpu) { return ::ceil(x); }
  inline double ceil(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil_double(x);
    #else
      return host_ceil(x);
    #endif
  }

  inline float host_copysignf(float x, float y) restrict(cpu) { return ::copysignf(x, y); }
  inline float copysignf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_copysign(x, y);
    #else
      return host_copysignf(x, y);
    #endif
  }

  inline float host_copysign(float x, float y) restrict(cpu) { return ::copysignf(x, y); }
  inline float copysign(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_copysign(x, y);
    #else
      return host_copysign(x, y);
    #endif
  }

  inline double host_copysign(double x, double y) restrict(cpu) { return ::copysign(x, y); }
  inline double copysign(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_copysign_double(x, y);
    #else
      return host_copysign(x, y);
    #endif
  }

  inline float host_cosf(float x) restrict(cpu) { return ::cosf(x); }
  inline float cosf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos(x);
    #else
      return host_cosf(x);
    #endif
  }

  inline float host_cos(float x) restrict(cpu) { return ::cosf(x); }
  inline float cos(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos(x);
    #else
      return host_cos(x);
    #endif
  }

  inline double host_cos(double x) restrict(cpu) { return ::cos(x); }
  inline double cos(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos_double(x);
    #else
      return host_cos(x);
    #endif
  }

  inline float host_coshf(float x) restrict(cpu) { return ::coshf(x); }
  inline float coshf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_coshf(x);
    #endif
  }

  inline float host_cosh(float x) restrict(cpu) { return ::coshf(x); }
  inline float cosh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_cosh(x);
    #endif
  }

  inline double host_cosh(double x) restrict(cpu) { return ::coshf(x); }
  inline double cosh(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh_double(x);
    #else
      return host_cosh(x);
    #endif
  }

  inline float host_cospif(float x) restrict(cpu) { return ::cosf((float)M_PI * x); }
  inline float cospif(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cospi(x);
    #else
      return host_cospif(x);
    #endif
  }

  inline float host_cospi(float x) restrict(cpu) { return ::cosf((float)M_PI * x); }
  inline float cospi(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cospi(x);
    #else
      return host_cospi(x);
    #endif
  }

  inline double host_cospi(double x) restrict(cpu) { return ::cos(M_PI * x); }
  inline double cospi(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cospi_double(x);
    #else
      return host_cospi(x);
    #endif
  }

  inline float host_erff(float x) restrict(cpu) { return ::erff(x); }
  inline float erff(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erf(x);
    #else
      return host_erff(x);
    #endif
  }

  inline float host_erf(float x) restrict(cpu) { return ::erff(x); }
  inline float erf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erf(x);
    #else
      return host_erf(x);
    #endif
  }

  inline double host_erf(double x) restrict(cpu) { return ::erf(x); }
  inline double erf(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erf_double(x);
    #else
      return host_erf(x);
    #endif
  }

  inline float host_erfcf(float x) restrict(cpu) { return ::erfcf(x); }
  inline float erfcf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfc(x);
    #else
      return host_erfcf(x);
    #endif
  }

  inline float host_erfc(float x) restrict(cpu) { return ::erfcf(x); }
  inline float erfc(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfc(x);
    #else
      return host_erfc(x);
    #endif
  }

  inline double host_erfc(double x) restrict(cpu) { return ::erfc(x); }
  inline double erfc(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfc_double(x);
    #else
      return host_erfc(x);
    #endif
  }

  /* FIXME missing erfinv */

  /* FIXME missing erfcinv */

  inline float host_expf(float x) restrict(cpu) { return ::expf(x); }
  inline float expf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp(x);
    #else
      return host_expf(x);
    #endif
  }

  inline float host_exp(float x) restrict(cpu) { return ::expf(x); }
  inline float exp(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp(x);
    #else
      return host_exp(x);
    #endif
  }

  inline double host_exp(double x) restrict(cpu) { return ::exp(x); }
  inline double exp(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp_double(x);
    #else
      return host_exp(x);
    #endif
  }

  inline float host_exp2f(float x) restrict(cpu) { return ::exp2f(x); }
  inline float exp2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2(x);
    #else
      return host_exp2f(x);
    #endif
  }

  inline float host_exp2(float x) restrict(cpu) { return ::exp2f(x); }
  inline float exp2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2(x);
    #else
      return host_exp2(x);
    #endif
  }

  inline double host_exp2(double x) restrict(cpu) { return ::exp2(x); }
  inline double exp2(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2_double(x);
    #else
      return host_exp2(x);
    #endif
  }

  inline float host_exp10f(float x) restrict(cpu) { return ::exp10f(x); }
  inline float exp10f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp10(x);
    #else
      return host_exp10f(x);
    #endif
  }

  inline float host_exp10(float x) restrict(cpu) { return ::exp10f(x); }
  inline float exp10(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp10(x);
    #else
      return host_exp10(x);
    #endif
  }

  inline double host_exp10(double x) restrict(cpu) { return ::exp10(x); }
  inline double exp10(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp10_double(x);
    #else
      return host_exp10(x);
    #endif
  }

  inline float host_expm1f(float x) restrict(cpu) { return ::expm1f(x); }
  inline float expm1f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_expm1(x);
    #else
      return host_expm1f(x);
    #endif
  }

  inline float host_expm1(float x) restrict(cpu) { return ::expm1f(x); }
  inline float expm1(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_expm1(x);
    #else
      return host_expm1(x);
    #endif
  }

  inline double host_expm1(double x) restrict(cpu) { return ::expm1(x); }
  inline double expm1(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_expm1_double(x);
    #else
      return host_expm1(x);
    #endif
  }

  inline float host_fabsf(float x) restrict(cpu) { return ::fabsf(x); }
  inline float fabsf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabsf(x);
    #endif
  }

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  inline float host_fabs(float x) restrict(cpu,amp) { return std::fabs(x); }
#else
  inline float host_fabs(float x) restrict(cpu) { return ::fabsf(x); }
#endif
  inline float fabs(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabs(x);
    #endif
  }

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  inline double host_fabs(double x) restrict(cpu,amp) { return std::fabs(x); }
#else
  inline double host_fabs(double x) restrict(cpu) { return ::fabs(x); }
#endif
  inline double fabs(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs_double(x);
    #else
      return host_fabs(x);
    #endif
  }

  inline float host_fdimf(float x, float y) restrict(cpu) { return ::fdimf(x, y); }
  inline float fdimf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fdim(x, y);
    #else
      return host_fdimf(x, y);
    #endif
  }

  inline float host_fdim(float x, float y) restrict(cpu) { return ::fdimf(x, y); }
  inline float fdim(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fdim(x, y);
    #else
      return host_fdim(x, y);
    #endif
  }

  inline double host_fdim(double x, double y) restrict(cpu) { return ::fdim(x, y); }
  inline double fdim(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fdim_double(x, y);
    #else
      return host_fdim(x, y);
    #endif
  }

  inline float host_floorf(float x) restrict(cpu) { return ::floorf(x); }
  inline float floorf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floorf(x);
    #endif
  }

  inline float host_floor(float x) restrict(cpu) { return ::floorf(x); }
  inline float floor(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floor(x);
    #endif
  }

  inline double host_floor(double x) restrict(cpu) { return ::floor(x); }
  inline double floor(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor_double(x);
    #else
      return host_floor(x);
    #endif
  }


  inline float host_fmaf(float x, float y, float z) restrict(cpu) { return ::fmaf(x, y , z); }
  inline float fmaf(float x, float y, float z) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fma(x, y , z);
    #else
      return host_fmaf(x, y , z);
    #endif
  }

  inline float host_fma(float x, float y, float z) restrict(cpu) { return ::fmaf(x, y , z); }
  inline float fma(float x, float y, float z) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fma(x, y , z);
    #else
      return host_fma(x, y , z);
    #endif
  }

  inline double host_fma(double x, double y, double z) restrict(cpu) { return ::fma(x, y , z); }
  inline double fma(double x, double y, double z) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fma_double(x, y , z);
    #else
      return host_fma(x, y , z);
    #endif
  }

  inline float host_fmaxf(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float fmaxf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmaxf(x, y);
    #endif
  }

  inline float host_fmax(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float fmax(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }

  inline double host_fmax(double x, double y) restrict(cpu) { return ::fmax(x, y); }
  inline double fmax(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax_double(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }

  inline float host_max(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float max(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_max(x, y);
    #endif
  }

  inline double host_max(double x, double y) restrict(cpu) { return ::fmax(x, y); }
  inline double max(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax_double(x, y);
    #else
      return host_max(x, y);
    #endif
  }

  inline float host_fminf(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float fminf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fminf(x, y);
    #endif
  }

  inline float host_fmin(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float fmin(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }

  inline double host_fmin(double x, double y) restrict(cpu) { return ::fmin(x, y); }
  inline double fmin(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin_double(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }

  inline float host_min(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float min(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_min(x, y);
    #endif
  }

  inline double host_min(double x, double y) restrict(cpu) { return ::fmin(x, y); }
  inline double min(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin_double(x, y);
    #else
      return host_min(x, y);
    #endif
  }

  inline float host_fmodf(float x, float y) restrict(cpu) { return ::fmodf(x, y); }
  inline float fmodf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmodf(x, y);
    #endif
  }

  inline float host_fmod(float x, float y) restrict(cpu) { return ::fmodf(x, y); }
  inline float fmod(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }

  inline double host_fmod(double x, double y) restrict(cpu) { return ::fmod(x, y); }
  inline double fmod(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod_double(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }

  /* FIXME missing fpclassify */

  inline float host_frexpf(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
  inline float frexpf(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexpf(x, exp);
    #else
      return host_frexpf(x, exp);
    #endif
  }

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  inline float host_frexp(float x, int *exp) restrict(cpu,amp) { return std::frexp(x, exp); }
#else
  inline float host_frexp(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
#endif
  inline float frexp(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline double host_frexp(double x, int *exp) restrict(cpu) { return ::frexp(x, exp); }
  inline double frexp(double x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline float host_hypotf(float x, float y) restrict(cpu) { return ::hypotf(x, y); }
  inline float hypotf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_hypot(x, y);
    #else
      return host_hypotf(x, y);
    #endif
  }

  inline float host_hypot(float x, float y) restrict(cpu) { return ::hypotf(x, y); }
  inline float hypot(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_hypot(x, y);
    #else
      return host_hypot(x, y);
    #endif
  }

  inline double host_hypot(double x, double y) restrict(cpu) { return ::hypot(x, y); }
  inline double hypot(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_hypot_double(x, y);
    #else
      return host_hypot(x, y);
    #endif
  }

  inline int host_ilogbf(float x) restrict(cpu) { return ::ilogbf(x); }
  inline int ilogbf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ilogb(x);
    #else
      return host_ilogbf(x);
    #endif
  }

  inline int host_ilogb(float x) restrict(cpu) { return ::ilogbf(x); }
  inline int ilogb(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ilogb(x);
    #else
      return host_ilogb(x);
    #endif
  }

  inline int host_ilogb(double x) restrict(cpu) { return ::ilogb(x); }
  inline int ilogb(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ilogb_double(x);
    #else
      return host_ilogb(x);
    #endif
  }

  inline int host_isfinite(float x) restrict(cpu) { return ::isfinite(x); }
  inline int isfinite(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isfinite(x);
    #else
      return host_isfinite(x);
    #endif
  }

  inline int host_isfinite(double x) restrict(cpu) { return ::isfinite(x); }
  inline int isfinite(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isfinite_double(x);
    #else
      return host_isfinite(x);
    #endif
  }

  inline int host_isinf(float x) restrict(cpu) { return ::isinf(x); }
  inline int isinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isinf(x);
    #else
      return host_isinf(x);
    #endif
  }

  inline int host_isinf(double x) restrict(cpu) { return ::isinf(x); }
  inline int isinf(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isinf_double(x);
    #else
      return host_isinf(x);
    #endif
  }

  inline int host_isnan(float x) restrict(cpu) { return ::isnan(x); }
  inline int isnan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnan(x);
    #else
      return host_isnan(x);
    #endif
  }

  inline int host_isnan(double x) restrict(cpu) { return ::isnan(x); }
  inline int isnan(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnan_double(x);
    #else
      return host_isnan(x);
    #endif
  }

  inline int host_isnormal(float x) restrict(cpu) { return ::isnormal(x); }
  inline int isnormal(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnormal(x);
    #else
      return host_isnormal(x);
    #endif
  }

  inline int host_isnormal(double x) restrict(cpu) { return ::isnormal(x); }
  inline int isnormal(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnormal_double(x);
    #else
      return host_isnormal(x);
    #endif
  }

  inline float host_ldexpf(float x, int exp) restrict(cpu) { return ::ldexpf(x,exp); }
  inline float ldexpf(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexpf(x,exp);
    #endif
  }

  inline float host_ldexp(float x, int exp) restrict(cpu) { return ::ldexpf(x,exp); }
  inline float ldexp(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }

  inline double host_ldexp(double x, int exp) restrict(cpu) { return ::ldexp(x,exp); }
  inline double ldexp(double x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp_double(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }

  inline float host_lgammaf(float x, int *sign) restrict(cpu) { return ::lgammaf_r(x, sign); }
  inline float lgammaf(float x, int *sign) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_lgammaf(x, sign);
    #else
      return host_lgammaf(x, sign);
    #endif
  }

  inline float host_lgamma(float x, int *sign) restrict(cpu) { return ::lgammaf_r(x, sign); }
  inline float lgamma(float x, int *sign) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_lgammaf(x, sign);
    #else
      return host_lgamma(x, sign);
    #endif
  }

  inline double host_lgamma(double x, int *sign) restrict(cpu) { return ::lgamma_r(x, sign); }
  inline double lgamma(double x, int *sign) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_lgamma(x, sign);
    #else
      return host_lgamma(x, sign);
    #endif
  }

  inline float host_logf(float x) restrict(cpu) { return ::logf(x); }
  inline float logf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log(x);
    #else
      return host_logf(x);
    #endif
  }

  inline float host_log(float x) restrict(cpu) { return ::logf(x); }
  inline float log(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log(x);
    #else
      return host_log(x);
    #endif
  }

  inline double host_log(double x) restrict(cpu) { return ::log(x); }
  inline double log(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log_double(x);
    #else
      return host_log(x);
    #endif
  }

  inline float host_log10f(float x) restrict(cpu) { return ::log10f(x); }
  inline float log10f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10(x);
    #else
      return host_log10f(x);
    #endif
  }

  inline float host_log10(float x) restrict(cpu) { return ::log10f(x); }
  inline float log10(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10(x);
    #else
      return host_log10(x);
    #endif
  }

  inline double host_log10(double x) restrict(cpu) { return ::log10(x); }
  inline double log10(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10_double(x);
    #else
      return host_log10(x);
    #endif
  }

  inline float host_log2f(float x) restrict(cpu) { return ::log2f(x); }
  inline float log2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2(x);
    #else
      return host_log2f(x);
    #endif
  }

  inline float host_log2(float x) restrict(cpu) { return ::log2f(x); }
  inline float log2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2(x);
    #else
      return host_log2(x);
    #endif
  }

  inline double host_log2(double x) restrict(cpu) { return ::log2(x); }
  inline double log2(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2_double(x);
    #else
      return host_log2(x);
    #endif
  }

  inline float host_log1pf(float x) restrict(cpu) { return ::log1pf(x); }
  inline float log1pf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log1p(x);
    #else
      return host_log1pf(x);
    #endif
  }

  inline float host_log1p(float x) restrict(cpu) { return ::log1pf(x); }
  inline float log1p(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log1p(x);
    #else
      return host_log1p(x);
    #endif
  }

  inline double host_log1p(double x) restrict(cpu) { return ::log1p(x); }
  inline double log1p(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log1p(x);
    #else
      return host_log1p(x);
    #endif
  }

  inline float host_logbf(float x) restrict(cpu) { return ::logbf(x); }
  inline float logbf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_logb(x);
    #else
      return host_logbf(x);
    #endif
  }

  inline float host_logb(float x) restrict(cpu) { return ::logbf(x); }
  inline float logb(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_logb(x);
    #else
      return host_logb(x);
    #endif
  }

  inline double host_logb(double x) restrict(cpu) { return ::logb(x); }
  inline double logb(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_logb_double(x);
    #else
      return host_logb(x);
    #endif
  }

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  inline float host_modff(float x, float *iptr) restrict(cpu,amp) { return std::modf(x, iptr); }
#else
  inline float host_modff(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
#endif
  inline float modff(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modff(x, iptr);
    #else
      return host_modff(x, iptr);
    #endif
  }

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
  inline float host_modf(float x, float *iptr) restrict(cpu,amp) { return std::modf(x, iptr); }
#else
  inline float host_modf(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
#endif
  inline float modf(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modff(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline double host_modf(double x, double *iptr) restrict(cpu) { return ::modf(x, iptr); }
  inline double modf(double x, double *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modf(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline float host_nanf(int tagp) restrict(cpu) { return ::nanf(reinterpret_cast<const char*>(&tagp)); }
  inline float nanf(int tagp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nan(tagp);
    #else
      return host_nanf(tagp);
    #endif
  }

  inline double host_nan(int tagp) restrict(cpu) { return ::nan(reinterpret_cast<const char*>(&tagp)); }
  inline double nan(int tagp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nan_double(static_cast<unsigned long>(tagp));
    #else
      return host_nan(tagp);
    #endif
  }

  inline float host_nearbyintf(float x) restrict(cpu) { return ::nearbyintf(x); }
  inline float nearbyintf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nearbyint(x);
    #else
      return host_nearbyintf(x);
    #endif
  }

  inline float host_nearbyint(float x) restrict(cpu) { return ::nearbyintf(x); }
  inline float nearbyint(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nearbyint(x);
    #else
      return host_nearbyint(x);
    #endif
  }

  inline double host_nearbyint(double x) restrict(cpu) { return ::nearbyint(x); }
  inline double nearbyint(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nearbyint_double(x);
    #else
      return host_nearbyint(x);
    #endif
  }

  inline float host_nextafterf(float x, float y) restrict(cpu) { return ::nextafterf(x, y); }
  inline float nextafterf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nextafter(x, y);
    #else
      return host_nextafterf(x, y);
    #endif
  }

  inline float host_nextafter(float x, float y) restrict(cpu) { return ::nextafterf(x, y); }
  inline float nextafter(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nextafter(x, y);
    #else
      return host_nextafter(x, y);
    #endif
  }

  inline double host_nextafter(double x, double y) restrict(cpu) { return ::nextafter(x, y); }
  inline double nextafter(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nextafter_double(x, y);
    #else
      return host_nextafter(x, y);
    #endif
  }

  inline float host_powf(float x, float y) restrict(cpu) { return ::powf(x, y); }
  inline float powf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_powf(x, y);
    #endif
  }

  inline float host_pow(float x, float y) restrict(cpu) { return ::powf(x, y); }
  inline float pow(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_pow(x, y);
    #endif
  }

  inline double host_pow(double x, double y) restrict(cpu) { return ::pow(x, y); }
  inline double pow(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow_double(x, y);
    #else
      return host_pow(x, y);
    #endif
  }

  inline float host_rcbrtf(float x) restrict(cpu) { return 1.0f / (::cbrtf(x)); }
  inline float rcbrtf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return 1.0f / __hc_cbrt(x);
    #else
      return host_rcbrtf(x);
    #endif
  }

  inline float host_rcbrt(float x) restrict(cpu) { return 1.0f / (::cbrtf(x)); }
  inline float rcbrt(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return 1.0f / __hc_cbrt(x);
    #else
      return host_rcbrt(x);
    #endif
  }

  inline double host_rcbrt(double x) restrict(cpu) { return 1.0 / (::cbrt(x)); }
  inline double rcbrt(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return 1.0 / __hc_cbrt_double(x);
    #else
      return host_rcbrt(x);
    #endif
  }

  inline float host_remainderf(float x, float y) restrict(cpu) { return ::remainderf(x, y); }
  inline float remainderf(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remainder(x, y);
    #else
      return host_remainderf(x, y);
    #endif
  }

  inline float host_remainder(float x, float y) restrict(cpu) { return ::remainderf(x, y);}
  inline float remainder(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remainder(x, y);
    #else
      return host_remainder(x, y);
    #endif
  }

  inline double host_remainder(double x, double y) restrict(cpu) { return ::remainder(x, y);}
  inline double remainder(double x, double y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remainder_double(x, y);
    #else
      return host_remainder(x, y);
    #endif
  }

  inline float host_remquof(float x, float y, int *quo) restrict(cpu) { return ::remquof(x, y, quo); }
  inline float remquof(float x, float y, int *quo) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remquof(x, y, quo);
    #else
      return host_remquof(x, y, quo);
    #endif
  }

  inline float host_remquo(float x, float y, int *quo) restrict(cpu) { return ::remquof(x, y, quo); }
  inline float remquo(float x, float y, int *quo) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remquof(x, y, quo);
    #else
      return host_remquo(x, y, quo);
    #endif
  }

  inline double host_remquo(double x, double y, int *quo) restrict(cpu) { return ::remquo(x, y, quo); }
  inline double remquo(double x, double y, int *quo) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remquo(x, y, quo);
    #else
      return host_remquo(x, y, quo);
    #endif
  }

  inline float host_roundf(float x) restrict(cpu) { return ::roundf(x); }
  inline float roundf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_roundf(x);
    #endif
  }

  inline float host_round(float x) restrict(cpu) { return ::roundf(x); }
  inline float round(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_round(x);
    #endif
  }

  inline double host_round(double x) restrict(cpu) { return ::round(x); }
  inline double round(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round_double(x);
    #else
      return host_round(x);
    #endif
  }

  inline float  host_rsqrtf(float x) restrict(cpu) { return 1.0f / (::sqrtf(x)); }
  inline float  rsqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt(x);
    #else
      return host_rsqrtf(x);
    #endif
  }

  inline float  host_rsqrt(float x) restrict(cpu) { return 1.0f / (::sqrtf(x)); }
  inline float  rsqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt(x);
    #else
      return host_rsqrt(x);
    #endif
  }

  inline double  host_rsqrt(double x) restrict(cpu) { return 1.0 / (::sqrt(x)); }
  inline double  rsqrt(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt_double(x);
    #else
      return host_rsqrt(x);
    #endif
  }

  inline float host_sinpif(float x) restrict(cpu) { return ::sinf((float)M_PI * x); }
  inline float sinpif(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinpi(x);
    #else
      return host_sinpif(x);
    #endif
  }

  inline float host_sinpi(float x) restrict(cpu) { return ::sinf((float)M_PI * x); }
  inline float sinpi(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinpi(x);
    #else
      return host_sinpi(x);
    #endif
  }

  inline double host_sinpi(double x) restrict(cpu) { return ::sin(M_PI * x); }
  inline double sinpi(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinpi_double(x);
    #else
      return host_sinpi(x);
    #endif
  }

  inline float host_scalbf(float x, float exp) restrict(cpu) { return ::scalbf(x, exp); }
  inline float scalbf(float x, float exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return x * powf(2.0f, exp);
    #else
      return host_scalbf(x, exp);
    #endif
  }

  inline float host_scalb(float x, float exp) restrict(cpu) { return ::scalbf(x, exp); }
  inline float scalb(float x, float exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return x * powf(2.0f, exp);
    #else
      return host_scalb(x, exp);
    #endif
  }

  inline double host_scalb(double x, double exp) restrict(cpu) { return ::scalb(x, exp); }
  inline double scalb(double x, double exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return x * pow(2.0, exp);
    #else
      return host_scalb(x, exp);
    #endif
  }

  inline float host_scalbnf(float x, int exp) restrict(cpu) { return ::scalbnf(x, exp); }
  inline float scalbnf(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x, exp);
    #else
      return host_scalbnf(x, exp);
    #endif
  }

  inline float host_scalbn(float x, int exp) restrict(cpu) { return ::scalbnf(x, exp); }
  inline float scalbn(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x, exp);
    #else
      return host_scalbn(x, exp);
    #endif
  }

  inline double host_scalbn(double x, int exp) restrict(cpu) { return ::scalbn(x, exp); }
  inline double scalbn(double x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp_double(x, exp);
    #else
      return host_scalbn(x, exp);
    #endif
  }

  inline int host_signbitf(float x) restrict(cpu) { return ::signbit(x); }
  inline int signbitf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbitf(x);
    #endif
  }

  inline int host_signbit(float x) restrict(cpu) { return ::signbit(x); }
  inline int signbit(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbit(x);
    #endif
  }

  inline int host_signbit(double x) restrict(cpu) { return ::signbit(x); }
  inline int signbit(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit_double(x);
    #else
      return host_signbit(x);
    #endif
  }

  inline float host_sinf(float x) restrict(cpu) { return ::sinf(x); }
  inline float sinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin(x);
    #else
      return host_sinf(x);
    #endif
  }

  inline float host_sin(float x) restrict(cpu) { return ::sinf(x); }
  inline float sin(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin(x);
    #else
      return host_sin(x);
    #endif
  }

  inline double host_sin(double x) restrict(cpu) { return ::sin(x); }
  inline double sin(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin_double(x);
    #else
      return host_sin(x);
    #endif
  }

  inline void host_sincosf(float x, float *s, float *c) restrict(cpu) { ::sincosf(x, s, c); }
  inline void sincosf(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      __hc_sincosf(x, s, c);
    #else
      host_sincosf(x, s, c);
    #endif
  }

  inline void host_sincos(float x, float *s, float *c) restrict(cpu) { ::sincosf(x, s, c); }
  inline void sincos(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      __hc_sincosf(x, s, c);
    #else
      host_sincos(x, s, c);
    #endif
  }

  inline void host_sincos(double x, double *s, double *c) restrict(cpu) { ::sincos(x, s, c); }
  inline void sincos(double x, double *s, double *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      __hc_sincos(x, s, c);
    #else
      host_sincos(x, s, c);
    #endif
  }

  inline float host_sinhf(float x) restrict(cpu) { return ::sinhf(x); }
  inline float sinhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinhf(x);
    #endif
  }

  inline float host_sinh(float x) restrict(cpu) { return ::sinhf(x); }
  inline float sinh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinh(x);
    #endif
  }

  inline double host_sinh(double x) restrict(cpu) { return ::sinh(x); }
  inline double sinh(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh_double(x);
    #else
      return host_sinh(x);
    #endif
  }

  inline float host_sqrtf(float x) restrict(cpu) { return ::sqrtf(x); }
  inline float sqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt(x);
    #else
      return host_sqrtf(x);
    #endif
  }

  inline float host_sqrt(float x) restrict(cpu) { return ::sqrtf(x); }
  inline float sqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt(x);
    #else
      return host_sqrt(x);
    #endif
  }

  inline double host_sqrt(double x) restrict(cpu) { return ::sqrt(x); }
  inline double sqrt(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt_double(x);
    #else
      return host_sqrt(x);
    #endif
  }

  inline float host_tgammaf(float x) restrict(cpu) { return ::tgammaf(x); }
  inline float tgammaf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tgamma(x);
    #else
      return host_tgammaf(x);
    #endif
  }

  inline float host_tgamma(float x) restrict(cpu) { return ::tgammaf(x); }
  inline float tgamma(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tgamma(x);
    #else
      return host_tgamma(x);
    #endif
  }

  inline double host_tgamma(double x) restrict(cpu) { return ::tgamma(x); }
  inline double tgamma(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tgamma_double(x);
    #else
      return host_tgamma(x);
    #endif
  }

  inline float host_tanf(float x) restrict(cpu) { return ::tanf(x); }
  inline float tanf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan(x);
    #else
      return host_tanf(x);
    #endif
  }

  inline float host_tan(float x) restrict(cpu) { return ::tanf(x); }
  inline float tan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan(x);
    #else
      return host_tan(x);
    #endif
  }

  inline double host_tan(double x) restrict(cpu) { return ::tan(x); }
  inline double tan(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan_double(x);
    #else
      return host_tan(x);
    #endif
  }

  inline float host_tanhf(float x) restrict(cpu) { return ::tanhf(x); }
  inline float tanhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanhf(x);
    #endif
  }

  inline float host_tanh(float x) restrict(cpu) { return ::tanhf(x); }
  inline float tanh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }

  inline double host_tanh(double x) restrict(cpu) { return ::tanh(x); }
  inline double tanh(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }

  inline float host_tanpif(float x) restrict(cpu) { return ::tanf(M_PI * x); }
  inline float tanpif(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanpi(x);
    #else
      return host_tanpif(x);
    #endif
  }

  inline float host_tanpi(float x) restrict(cpu) { return ::tanf(M_PI * x); }
  inline float tanpi(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanpi(x);
    #else
      return host_tanpif(x);
    #endif
  }

  inline double host_tanpi(double x) restrict(cpu) { return ::tan(M_PI * x); }
  inline double tanpi(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanpi_double(x);
    #else
      return host_tanpi(x);
    #endif
  }

  inline float host_truncf(float x) restrict(cpu) { return ::truncf(x); }
  inline float truncf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_truncf(x);
    #endif
  }

  inline float host_trunc(float x) restrict(cpu) { return ::truncf(x); }
  inline float trunc(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_trunc(x);
    #endif
  }

  inline double host_trunc(double x) restrict(cpu) { return ::trunc(x); }
  inline double trunc(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc_double(x);
    #else
      return host_trunc(x);
    #endif
  }

} // namespace precise_math

} // namespace Kalmar
