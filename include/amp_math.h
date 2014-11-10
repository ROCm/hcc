//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
  #include <cmath>
#ifdef __GPU__
  extern "C" float opencl_acos(float x) restrict(amp);
  extern "C" float opencl_acosh(float x) restrict(amp);
  extern "C" float opencl_asin(float x) restrict(amp);
  extern "C" float opencl_asinh(float x) restrict(amp);
  extern "C" float opencl_atan(float x) restrict(amp);
  extern "C" float opencl_atanh(float x) restrict(amp);
  extern "C" float opencl_atan2(float x, float y) restrict(amp);
  extern "C" float opencl_cos(float x) restrict(amp);
  extern "C" float opencl_cosh(float x) restrict(amp);
  extern "C" float opencl_cbrt(float x) restrict(amp);
  extern "C" float opencl_ceil(float x) restrict(amp);
  extern "C" float opencl_copysign(float x, float y) restrict(amp);
  extern "C" float opencl_cospi(float x) restrict(amp);
  extern "C" float opencl_erf(float x) restrict(amp);
  extern "C" float opencl_erfc(float x) restrict(amp);
  extern "C" float opencl_exp(float x) restrict(amp);
  extern "C" float opencl_exp10(float x) restrict(amp);
  extern "C" float opencl_exp2(float x) restrict(amp);
  extern "C" float opencl_expm1(float x) restrict(amp);
  extern "C" float opencl_fabs(float x) restrict(amp);
  extern "C" float opencl_fdim(float x, float y) restrict(amp);
  extern "C" float opencl_floor(float x) restrict(amp);
  extern "C" float opencl_fma(float x, float y, float z) restrict(amp);
  extern "C" float opencl_fmax(float x, float y) restrict(amp);
  extern "C" float opencl_fmin(float x, float y) restrict(amp);
  extern "C" float opencl_fmod(float x, float y) restrict(amp);
  extern "C" int opencl_isinf(float x) restrict(amp);
  extern "C" int opencl_isfinite(float x) restrict(amp);
  extern "C" int opencl_ilogb(float x) restrict(amp);
  extern "C" int opencl_isnan(float x) restrict(amp);
  extern "C" int opencl_isnormal(float x) restrict(amp);
  extern "C" float opencl_ldexp(float x, int exp) restrict(amp);
  extern "C" float opencl_log(float x) restrict(amp);
  extern "C" float opencl_log2(float x) restrict(amp);
  extern "C" float opencl_log10(float x) restrict(amp);
  extern "C" float opencl_log1p(float x) restrict(amp);
  extern "C" float opencl_logb(float x) restrict(amp);
  extern "C" float opencl_hypot(float x, float y) restrict(amp);
  extern "C" float opencl_nextafter(float x, float y) restrict(amp);
  extern "C" int opencl_min(int x, int y) restrict(amp);
  extern "C" float opencl_max(float x, float y) restrict(amp);
  extern "C" float opencl_pow(float x, float y) restrict(amp);
  extern "C" float opencl_round(float x) restrict(amp);
  extern "C" float opencl_remainder(float x, float y) restrict(amp);
  extern "C" float opencl_rsqrt(float x) restrict(amp);
  extern "C" float opencl_sin(float x) restrict(amp);
  extern "C" float opencl_sinh(float x) restrict(amp);
  extern "C" int   opencl_signbit(float x) restrict(amp);
  extern "C" float opencl_sinpi(float x) restrict(amp);
  extern "C" float opencl_sqrt(float x) restrict(amp);
  extern "C" float opencl_tan(float x) restrict(amp);
  extern "C" float opencl_tanh(float x) restrict(amp);
  extern "C" float opencl_tgamma(float x) restrict(amp);
  extern "C" float opencl_trunc(float x) restrict(amp);

  /* SPIR pass will deduce the address space of the pointer and emit correct
     call instruction */
  extern "C" float opencl_modff(float x, float *iptr) restrict(amp);
  extern "C" double opencl_modf(double x, double *iptr) restrict(amp);
  extern "C" float opencl_frexpf(float x, int *exp) restrict(amp);
  extern "C" double opencl_frexp(double x, int *exp) restrict(amp);

#endif

namespace Concurrency {
namespace fast_math {

  inline float host_asin(float x) restrict(cpu) { return ::asin(x); }
  inline float asin(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asin(x);
    #else
      return host_asin(x);
    #endif
  }
  inline float host_asinf(float x) restrict(cpu) { return ::asinf(x); }
  inline float asinf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asin(x);
    #else
      return host_asinf(x);
    #endif
  }
  inline float host_acos(float x) restrict(cpu) { return ::acos(x); }
  inline float acos(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acos(x);
    #else
      return host_acos(x);
    #endif
  }
  inline float host_acosf(float x) restrict(cpu) { return ::acosf(x); }
  inline float acosf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acos(x);
    #else
      return host_acosf(x);
    #endif
  }
  inline float host_atan(float x) restrict(cpu) { return ::atan(x); }
  inline float atan(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan(x);
    #else
      return host_atan(x);
    #endif
  }
  inline float host_atanf(float x) restrict(cpu) { return ::atanf(x); }
  inline float atanf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan(x);
    #else
      return host_atanf(x);
    #endif
  }
  inline float host_atan2(float x, float y) restrict(cpu) { return ::atan2(x, y); }
  inline float atan2(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return host_atan2(x, y);
    #endif
  }
  inline float host_atan2f(float x, float y) restrict(cpu) { return ::atan2f(x, y); }
  inline float atan2f(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return host_atan2f(x, y);
    #endif
  }
  inline float host_ceil(float x) restrict(cpu) { return ::ceil(x); }
  inline float ceil(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ceil(x);
    #else
      return host_ceil(x);
    #endif
  }
  inline float host_ceilf(float x) restrict(cpu) { return ::ceilf(x); }
  inline float ceilf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ceil(x);
    #else
      return host_ceilf(x);
    #endif
  }
  inline float host_cos(float x) restrict(cpu) { return ::cos(x); }
  inline float cos(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cos(x);
    #else
      return host_cos(x);
    #endif
  }
  inline float host_cosf(float x) restrict(cpu) { return ::cosf(x); }
  inline float cosf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cos(x);
    #else
      return host_cosf(x);
    #endif
  }
  inline float host_cosh(float x) restrict(cpu) { return ::cosh(x); }
  inline float cosh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cosh(x);
    #else
      return host_cosh(x);
    #endif
  }
  inline float host_coshf(float x) restrict(cpu) { return ::cosh(x); }
  inline float coshf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cosh(x);
    #else
      return host_cosh(x);
    #endif
  }
  inline float host_exp(float x) restrict(cpu) { return ::exp(x); }
  inline float exp(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp(x);
    #else
      return host_exp(x);
    #endif
  }
  inline float host_expf(float x) restrict(cpu) { return ::expf(x); }
  inline float expf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp(x);
    #else
      return host_expf(x);
    #endif
  }
  inline float host_exp2(float x) restrict(cpu) { return ::exp2(x); }
  inline float exp2(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp2(x);
    #else
      return host_exp2(x);
    #endif
  }
  inline float host_exp2f(float x) restrict(cpu) { return ::exp2f(x); }
  inline float exp2f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp2(x);
    #else
      return host_exp2f(x);
    #endif
  }
  inline float host_fabs(float x) restrict(cpu) { return ::fabs(x); }
  inline float fabs(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fabs(x);
    #else
      return host_fabs(x);
    #endif
  }
  inline float host_fabsf(float x) restrict(cpu) { return ::fabsf(x); }
  inline float fabsf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fabs(x);
    #else
      return host_fabsf(x);
    #endif
  }
  inline float host_floor(float x) restrict(cpu) { return ::floor(x); }
  inline float floor(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_floor(x);
    #else
      return host_floor(x);
    #endif
  }
  inline float host_floorf(float x) restrict(cpu) { return ::floorf(x); }
  inline float floorf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_floor(x);
    #else
      return host_floorf(x);
    #endif
  }
  inline float host_fmax(float x, float y) restrict(cpu) { return ::fmax(x, y); }
  inline float fmax(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }
  inline float host_fmaxf(float x, float y) restrict(cpu) { return ::fmaxf(x, y); }
  inline float fmaxf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return host_fmaxf(x, y);
    #endif
  }
  inline float host_fmin(float x, float y) restrict(cpu) { return ::fmin(x, y); }
  inline float fmin(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmin(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }
  inline float host_fminf(float x, float y) restrict(cpu) { return ::fminf(x, y); }
  inline float fminf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmin(x, y);
    #else
      return host_fminf(x, y);
    #endif
  }
  inline float host_fmod(float x, float y) restrict(cpu) { return ::fmod(x, y); }
  inline float fmod(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmod(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }
  inline float host_fmodf(float x, float y) restrict(cpu) { return ::fmodf(x, y); }
  inline float fmodf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmod(x, y);
    #else
      return host_fmodf(x, y);
    #endif
  }
  inline int host_isfinite(float x) restrict(cpu) { return ::isfinite(x); }
  inline int isfinite(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isfinite(x);
    #else
      return host_isfinite(x);
    #endif
  }
  inline int host_isinf(float x) restrict(cpu) { return ::isinf(x); }
  inline int isinf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isinf(x);
    #else
      return host_isinf(x);
    #endif
  }
  inline int host_isnan(float x) restrict(cpu) { return ::isnan(x); }
  inline int isnan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnan(x);
    #else
      return host_isnan(x);
    #endif
  }
  inline float host_ldexp(float x, int exp) restrict(cpu) { return ::ldexp(x,exp); }
  inline float ldexp(float x, int exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ldexp(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }
  inline float host_ldexpf(float x, int exp) restrict(cpu) { return ::ldexpf(x,exp); }
  inline float ldexpf(float x, int exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ldexp(x,exp);
    #else
      return host_ldexpf(x,exp);
    #endif
  }
  inline float host_log(float x) restrict(cpu) { return ::log(x); }
  inline float log(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log(x);
    #else
      return host_log(x);
    #endif
  }
  inline float host_logf(float x) restrict(cpu) { return ::logf(x); }
  inline float logf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log(x);
    #else
      return host_logf(x);
    #endif
  }
  inline float host_log2(float x) restrict(cpu) { return ::log2(x); }
  inline float log2(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log2(x);
    #else
      return host_log2(x);
    #endif
  }
  inline float host_log2f(float x) restrict(cpu) { return ::log2f(x); }
  inline float log2f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log2(x);
    #else
      return host_log2f(x);
    #endif
  }
  inline float host_log10(float x) restrict(cpu) { return ::log10(x); }
  inline float log10(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log10(x);
    #else
      return host_log10(x);
    #endif
  }
  inline float host_log10f(float x) restrict(cpu) { return ::log10f(x); }
  inline float log10f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log10(x);
    #else
      return host_log10f(x);
    #endif
  }
  inline float host_pow(float x, float y) restrict(cpu) { return ::pow(x, y); }
  inline float pow(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_pow(x, y);
    #else
      return host_pow(x, y);
    #endif
  }
  inline float host_powf(float x, float y) restrict(cpu) { return ::pow(x, y); }
  inline float powf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_pow(x, y);
    #else
      return host_pow(x, y);
    #endif
  }
  inline float host_round(float x) restrict(cpu) { return ::round(x); }
  inline float round(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_round(x);
    #else
      return host_round(x);
    #endif
  }
  inline float host_roundf(float x) restrict(cpu) { return ::round(x); }
  inline float roundf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_round(x);
    #else
      return host_round(x);
    #endif
  }
  inline float  host_rsqrt(float x) restrict(cpu) { return 1 / (::sqrt(x)); }
  inline float  rsqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_rsqrt(x);
    #else
      return host_rsqrt(x);
    #endif
  }
  inline float  host_rsqrtf(float x) restrict(cpu) { return 1 / (::sqrt(x)); }
  inline float  rsqrtf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_rsqrt(x);
    #else
      return host_rsqrtf(x);
    #endif
  }
  inline float host_sin(float x) restrict(cpu) { return ::sin(x); }
  inline float sin(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sin(x);
    #else
      return host_sin(x);
    #endif
  }
  inline float host_sinf(float x) restrict(cpu) { return ::sinf(x); }
  inline float sinf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sin(x);
    #else
      return host_sinf(x);
    #endif
  }
  inline int host_signbit(float x) restrict(cpu) { return ::signbit(x); }
  inline int signbit(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_signbit(x);
    #else
      return host_signbit(x);
    #endif
  }
  inline int host_signbitf(float x) restrict(cpu) { return ::signbit(x); }
  inline int signbitf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_signbit(x);
    #else
      return host_signbit(x);
    #endif
  }
  inline float host_sinh(float x) restrict(cpu) { return ::sinh(x); }
  inline float sinh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinh(x);
    #else
      return host_sinh(x);
    #endif
  }
  inline float host_sinhf(float x) restrict(cpu) { return ::sinhf(x); }
  inline float sinhf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinh(x);
    #else
      return host_sinhf(x);
    #endif
  }
  inline float host_sqrt(float x) restrict(cpu) { return ::sqrt(x); }
  inline float sqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sqrt(x);
    #else
      return host_sqrt(x);
    #endif
  }
  inline float host_sqrtf(float x) restrict(cpu) { return ::sqrtf(x); }
  inline float sqrtf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sqrt(x);
    #else
      return host_sqrtf(x);
    #endif
  }
  inline float host_tan(float x) restrict(cpu) { return ::tan(x); }
  inline float tan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tan(x);
    #else
      return host_tan(x);
    #endif
  }
  inline float host_tanf(float x) restrict(cpu) { return ::tanf(x); }
  inline float tanf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tan(x);
    #else
      return host_tanf(x);
    #endif
  }
  inline float host_tanh(float x) restrict(cpu) { return ::tanh(x); }
  inline float tanh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }
  inline float host_tanhf(float x) restrict(cpu) { return ::tanhf(x); }
  inline float tanhf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tanh(x);
    #else
      return host_tanhf(x);
    #endif
  }
  inline float host_trunc(float x) restrict(cpu) { return ::trunc(x); }
  inline float trunc(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_trunc(x);
    #else
      return host_trunc(x);
    #endif
  }

  inline float host_truncf(float x) restrict(cpu) { return ::truncf(x); }
  inline float truncf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_trunc(x);
    #else
      return host_truncf(x);
    #endif
  }

  inline float host_modff(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
  inline float modff(float x, float *iptr) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_modff(x, iptr);
    #else
      return host_modff(x, iptr);
    #endif
  }

  inline float host_modf(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
  inline float modf(float x, float *iptr) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_modff(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline float host_frexpf(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
  inline float frexpf(float x, int *exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_frexpf(x, exp);
    #else
      return host_frexpf(x, exp);
    #endif
  }

  inline float host_frexp(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
  inline float frexp(float x, int *exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_frexpf(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }
} // namesapce fast_math
  inline int host_min(int x, int y) restrict(cpu) {
    using std::min;
    return min(x, y);
  }
  inline int min(int x, int y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_min(x, y);
    #else
      return host_min(x, y);
    #endif
  }
  inline int host_max(float x, float y) restrict(cpu) {
    using std::max;
    return max(x, y);
  }
  inline float max(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_max(x, y);
    #else
      return host_max(x, y);
    #endif
  }

  namespace precise_math {

  inline float host_acosh(float x) restrict(cpu) { return ::acosh(x); }
  inline float acosh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acosh(x);
    #else
      return host_acosh(x);
    #endif
  }
  inline float host_acoshf(float x) restrict(cpu) { return ::acoshf(x); }
  inline float acoshf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acosh(x);
    #else
      return host_acoshf(x);
    #endif
  }
  inline float host_asinh(float x) restrict(cpu) { return ::asinh(x); }
  inline float asinh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asinh(x);
    #else
      return host_asinh(x);
    #endif
  }
  inline float host_asinhf(float x) restrict(cpu) { return ::asinhf(x); }
  inline float asinhf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asinh(x);
    #else
      return host_asinhf(x);
    #endif
  }
  inline float host_atanh(float x) restrict(cpu) { return ::atanh(x); }
  inline float atanh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atanh(x);
    #else
      return host_atanh(x);
    #endif
  }
  inline float host_atanhf(float x) restrict(cpu) { return ::atanhf(x); }
  inline float atanhf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atanh(x);
    #else
      return host_atanhf(x);
    #endif
  }
  inline float host_atan2(float x, float y) restrict(cpu) { return ::atan2(x, y); }
  inline float atan2(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return host_atan2(x, y);
    #endif
  }
  inline float host_atan2f(float x, float y) restrict(cpu) { return ::atan2f(x, y); }
  inline float atan2f(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return host_atan2f(x, y);
    #endif
  }
  inline float host_cbrt(float x) restrict(cpu) { return ::cbrt(x); }
  inline float cbrt(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_cbrt(x);
    #else
      return host_cbrt(x);
    #endif
  }
  inline float host_cbrtf(float x) restrict(cpu) { return ::cbrtf(x); }
  inline float cbrtf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_cbrt(x);
    #else
      return host_cbrtf(x);
    #endif
  }
  inline float host_copysign(float x, float y) restrict(cpu) { return ::copysign(x, y); }
  inline float copysign(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_copysign(x, y);
    #else
      return host_copysign(x, y);
    #endif
  }
  inline float host_copysignf(float x, float y) restrict(cpu) { return ::copysignf(x, y); }
  inline float copysignf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_copysign(x, y);
    #else
      return host_copysignf(x, y);
    #endif
  }
  inline float host_cospi(float x) restrict(cpu) { return ::cos(M_PI * x); }
  inline float cospi(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cospi(x);
    #else
      return host_cospi(x);
    #endif
  }
  inline float host_cospif(float x) restrict(cpu) { return ::cos(M_PI * x); }
  inline float cospif(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cospi(x);
    #else
      return host_cospif(x);
    #endif
  }
  inline float host_erf(float x) restrict(cpu) { return ::erf(x); }
  inline float erf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erf(x);
    #else
      return host_erf(x);
    #endif
  }
  inline float host_erff(float x) restrict(cpu) { return ::erff(x); }
  inline float erff(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erf(x);
    #else
      return host_erff(x);
    #endif
  }
  inline float host_erfc(float x) restrict(cpu) { return ::erfc(x); }
  inline float erfc(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erfc(x);
    #else
      return host_erfc(x);
    #endif
  }
  inline float host_erfcf(float x) restrict(cpu) { return ::erfcf(x); }
  inline float erfcf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erfc(x);
    #else
      return host_erfcf(x);
    #endif
  }
  inline float host_exp10(float x) restrict(cpu) { return ::exp10(x); }
  inline float exp10(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp10(x);
    #else
      return host_exp10(x);
    #endif
  }
  inline float host_exp10f(float x) restrict(cpu) { return ::exp10f(x); }
  inline float exp10f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp10(x);
    #else
      return host_exp10f(x);
    #endif
  }
  inline float host_expm1(float x) restrict(cpu) { return ::expm1(x); }
  inline float expm1(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_expm1(x);
    #else
      return host_expm1(x);
    #endif
  }
  inline float host_expm1f(float x) restrict(cpu) { return ::expm1f(x); }
  inline float expm1f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_expm1(x);
    #else
      return host_expm1f(x);
    #endif
  }
  inline float host_fdim(float x, float y) restrict(cpu) { return ::fdim(x, y); }
  inline float fdim(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fdim(x, y);
    #else
      return host_fdim(x, y);
    #endif
  }
  inline float host_fdimf(float x, float y) restrict(cpu) { return ::fdimf(x, y); }
  inline float fdimf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fdim(x, y);
    #else
      return host_fdimf(x, y);
    #endif
  }
  inline float host_fma(float x, float y, float z) restrict(cpu) { return ::fma(x, y , z); }
  inline float fma(float x, float y, float z) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fma(x, y , z);
    #else
      return host_fma(x, y , z);
    #endif
  }
  inline float host_fmaf(float x, float y, float z) restrict(cpu) { return ::fmaf(x, y , z); }
  inline float fmaf(float x, float y, float z) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fma(x, y , z);
    #else
      return host_fmaf(x, y , z);
    #endif
  }
  inline float host_fmax(float x, float y) restrict(cpu) { return ::fmax(x, y); }
  inline float fmax(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }
  inline int host_ilogb(float x) restrict(cpu) { return ::ilogb(x); }
  inline int ilogb(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ilogb(x);
    #else
      return host_ilogb(x);
    #endif
  }
  inline int host_ilogbf(float x) restrict(cpu) { return ::ilogbf(x); }
  inline int ilogbf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ilogb(x);
    #else
      return host_ilogbf(x);
    #endif
  }
  inline int host_isnan(float x) restrict(cpu) { return ::isnan(x); }
  inline int isnan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnan(x);
    #else
      return host_isnan(x);
    #endif
  }
  inline int host_isnormal(float x) restrict(cpu) { return ::isnormal(x); }
  inline int isnormal(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnormal(x);
    #else
      return host_isnormal(x);
    #endif
  }
  inline float host_log1p(float x) restrict(cpu) { return ::log1p(x); }
  inline float log1p(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log1p(x);
    #else
      return host_log1p(x);
    #endif
  }
  inline float host_log1pf(float x) restrict(cpu) { return ::log1pf(x); }
  inline float log1pf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log1p(x);
    #else
      return host_log1pf(x);
    #endif
  }
  inline float host_logb(float x) restrict(cpu) { return ::logb(x); }
  inline float logb(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_logb(x);
    #else
      return host_logb(x);
    #endif
  }
  inline float host_logbf(float x) restrict(cpu) { return ::logbf(x); }
  inline float logbf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_logb(x);
    #else
      return host_logbf(x);
    #endif
  }
  inline float host_nextafter(float x, float y) restrict(cpu) { return ::nextafter(x, y); }
  inline float nextafter(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_nextafter(x, y);
    #else
      return host_nextafter(x, y);
    #endif
  }
  inline float host_hypotf(float x, float y) restrict(cpu) { return ::hypotf(x, y); }
  inline float hypotf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_hypot(x, y);
    #else
      return host_hypotf(x, y);
    #endif
  }
  inline float host_rcbrt(float x) restrict(cpu) { return 1 / (::cbrt(x)); }
  inline float rcbrt(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return 1 / opencl_cbrt(x);
    #else
      return host_rcbrt(x);
    #endif
  }
  inline float host_rcbrtf(float x) restrict(cpu) { return 1 / (::cbrt(x)); }
  inline float rcbrtf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return 1 / opencl_cbrt(x);
    #else
      return host_rcbrtf(x);
    #endif
  }
  inline float host_remainder(double x, double y) restrict(cpu) { return ::remainder(x, y);}
  inline float remainder(double x, double y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_remainder(x, y);
    #else
      return host_remainder(x, y);
    #endif
  }
  inline float host_remainderf(double x, double y) restrict(cpu) { return ::remainder(x, y); }
  inline float remainderf(double x, double y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_remainder(x, y);
    #else
      return host_remainder(x, y);
    #endif
  }
  inline float host_tgamma(float x) restrict(cpu) { return ::tgamma(x); }
  inline float tgamma(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tgamma(x);
    #else
      return host_tgamma(x);
    #endif
  }
  inline float host_tgammaf(float x) restrict(cpu) { return ::tgammaf(x); }
  inline float tgammaf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tgamma(x);
    #else
      return host_tgammaf(x);
    #endif
  }
  inline float host_scalbn(float x, int exp) restrict(cpu) { return ::scalbn(x, exp); }
  inline float scalbn(float x, int exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ldexp(x, exp);
    #else
      return host_scalbn(x, exp);
    #endif
  }
  inline float host_scalbnf(float x, int exp) restrict(cpu) { return ::scalbn(x, exp); }
  inline float scalbnf(float x, int exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ldexp(x, exp);
    #else
      return host_scalbn(x, exp);
    #endif
  }
  inline float host_sinpi(float x) restrict(cpu) { return ::sin(M_PI * x); }
  inline float sinpi(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinpi(x);
    #else
      return host_sinpi(x);
    #endif
  }
  inline float host_sinpif(float x) restrict(cpu) { return ::sin(M_PI * x); }
  inline float sinpif(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinpi(x);
    #else
      return host_sinpif(x);
    #endif
  }

  inline float host_modff(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
  inline float modff(float x, float *iptr) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_modff(x, iptr);
    #else
      return host_modff(x, iptr);
    #endif
  }

  inline float host_modf(float x, float *iptr) restrict(cpu) { return ::modff(x, iptr); }
  inline float modf(float x, float *iptr) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_modff(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline double host_modf(double x, double *iptr) restrict(cpu) { return ::modf(x, iptr); }
  inline double modf(double x, double *iptr) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_modf(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline float host_frexpf(float x, int *exp) restrict(cpu) { return ::frexpf(x, exp); }
  inline float frexpf(float x, int *exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_frexpf(x, exp);
    #else
      return host_frexpf(x, exp);
    #endif
  }

  inline float host_frexp(float x, int *exp) restrict(cpu) { return ::frexp(x, exp); }
  inline float frexp(float x, int *exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_frexp(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline double host_frexp(double x, int *exp) restrict(cpu) { return ::frexp(x, exp); }
  inline double frexp(double x, int *exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_frexp(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  using fast_math::acos;
  using fast_math::acosf;

  using fast_math::asin;
  using fast_math::asinf;

  using fast_math::atan;
  using fast_math::atanf;

  using fast_math::ceil;
  using fast_math::ceilf;

  using fast_math::cos;
  using fast_math::cosf;

  using fast_math::cosh;
  using fast_math::coshf;

  using fast_math::exp;
  using fast_math::expf;

  using fast_math::exp2;
  using fast_math::exp2f;

  using fast_math::fmin;
  using fast_math::fminf;

  using fast_math::fmod;
  using fast_math::fmodf;

  using fast_math::floor;
  using fast_math::floorf;

  using fast_math::fabs;
  using fast_math::fabsf;

  using fast_math::ldexp;
  using fast_math::ldexpf;

  using fast_math::log;
  using fast_math::logf;

  using fast_math::log10;
  using fast_math::log10f;

  using fast_math::log2;
  using fast_math::log2f;

  using fast_math::isinf;

  using fast_math::isfinite;

  using fast_math::pow;
  using fast_math::powf;

  using fast_math::round;
  using fast_math::roundf;

  using fast_math::signbit;
  using fast_math::signbitf;

  using fast_math::sin;
  using fast_math::sinf;

  using fast_math::sinh;
  using fast_math::sinhf;

  using fast_math::sqrt;
  using fast_math::sqrtf;

  using fast_math::rsqrt;
  using fast_math::rsqrtf;

  using fast_math::tan;
  using fast_math::tanf;

  using fast_math::tanh;
  using fast_math::tanhf;

  using fast_math::trunc;
  using fast_math::truncf;

} // namespace precise_math

} // namespace Concurrency
