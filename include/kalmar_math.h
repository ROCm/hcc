//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <stdexcept>

[[noreturn]]
inline
void unimplemented_math_fn(const std::string& fn)
{
    throw std::runtime_error{fn + "is unimplemented for host."};
}

[[noreturn]]
inline
float erfcinv(float) { unimplemented_math_fn(__PRETTY_FUNCTION__); }

[[noreturn]]
inline
double erfcinv(double) { unimplemented_math_fn(__PRETTY_FUNCTION__); }

[[noreturn]]
inline
float erfinv(float) { unimplemented_math_fn(__PRETTY_FUNCTION__); }

[[noreturn]]
inline
double erfinv(double) { unimplemented_math_fn(__PRETTY_FUNCTION__); }

[[noreturn]]
inline
__fp16 modf(__fp16, __fp16*) { unimplemented_math_fn(__PRETTY_FUNCTION__); }

template<typename T>
[[noreturn]]
inline
void sincos(T, T*, T*) { unimplemented_math_fn(__PRETTY_FUNCTION__); }

extern "C" __fp16 __hc_acos_half(__fp16 x) restrict(amp);
extern "C" float __hc_acos(float x) restrict(amp);
extern "C" double __hc_acos_double(double x) restrict(amp);

extern "C" __fp16 __hc_acosh_half(__fp16 x) restrict(amp);
extern "C" float __hc_acosh(float x) restrict(amp);
extern "C" double __hc_acosh_double(double x) restrict(amp);

extern "C" __fp16 __hc_asin_half(__fp16 x) restrict(amp);
extern "C" float __hc_asin(float x) restrict(amp);
extern "C" double __hc_asin_double(double x) restrict(amp);

extern "C" __fp16 __hc_asinh_half(__fp16 x) restrict(amp);
extern "C" float __hc_asinh(float x) restrict(amp);
extern "C" double __hc_asinh_double(double x) restrict(amp);

extern "C" __fp16 __hc_atan_half(__fp16 x) restrict(amp);
extern "C" float __hc_atan(float x) restrict(amp);
extern "C" double __hc_atan_double(double x) restrict(amp);

extern "C" __fp16 __hc_atanh_half(__fp16 x) restrict(amp);
extern "C" float __hc_atanh(float x) restrict(amp);
extern "C" double __hc_atanh_double(double x) restrict(amp);

extern "C" __fp16 __hc_atan2_half(__fp16 y, __fp16 x) restrict(amp);
extern "C" float __hc_atan2(float y, float x) restrict(amp);
extern "C" double __hc_atan2_double(double y, double x) restrict(amp);

extern "C" __fp16 __hc_cbrt_half(__fp16 x) restrict(amp);
extern "C" float __hc_cbrt(float x) restrict(amp);
extern "C" double __hc_cbrt_double(double x) restrict(amp);

extern "C" __fp16 __hc_ceil_half(__fp16 x) restrict(amp);
extern "C" float __hc_ceil(float x) restrict(amp);
extern "C" double __hc_ceil_double(double x) restrict(amp);

extern "C" __fp16 __hc_copysign_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_copysign(float x, float y) restrict(amp);
extern "C" double __hc_copysign_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_cos_half(__fp16 x) restrict(amp);
extern "C" float __hc_cos(float x) restrict(amp);
extern "C" float __hc_cos_native(float x) restrict(amp);
extern "C" double __hc_cos_double(double x) restrict(amp);

extern "C" __fp16 __hc_cosh_half(__fp16 x) restrict(amp);
extern "C" float __hc_cosh(float x) restrict(amp);
extern "C" double __hc_cosh_double(double x) restrict(amp);

extern "C" __fp16 __hc_cospi_half(__fp16 x) restrict(amp);
extern "C" float __hc_cospi(float x) restrict(amp);
extern "C" double __hc_cospi_double(double x) restrict(amp);

extern "C" __fp16 __hc_erf_half(__fp16 x) restrict(amp);
extern "C" float __hc_erf(float x) restrict(amp);
extern "C" double __hc_erf_double(double x) restrict(amp);

extern "C" __fp16 __hc_erfc_half(__fp16 x) restrict(amp);
extern "C" float __hc_erfc(float x) restrict(amp);
extern "C" double __hc_erfc_double(double x) restrict(amp);

extern "C" __fp16 __hc_erfcinv_half(__fp16 x) restrict(amp);
extern "C" float __hc_erfcinv(float x) restrict(amp);
extern "C" double __hc_erfcinv_double(double x) restrict(amp);

extern "C" __fp16 __hc_erfinv_half(__fp16 x) restrict(amp);
extern "C" float __hc_erfinv(float x) restrict(amp);
extern "C" double __hc_erfinv_double(double x) restrict(amp);

extern "C" __fp16 __hc_exp_half(__fp16 x) restrict(amp);
extern "C" float __hc_exp(float x) restrict(amp);
extern "C" float __hc_exp_native(float x) restrict(amp);
extern "C" double __hc_exp_double(double x) restrict(amp);

extern "C" __fp16 __hc_exp10_half(__fp16 x) restrict(amp);
extern "C" float __hc_exp10(float x) restrict(amp);
extern "C" float __hc_exp10_native(float x) restrict(amp);
extern "C" double __hc_exp10_double(double x) restrict(amp);

extern "C" __fp16 __hc_exp2_half(__fp16 x) restrict(amp);
extern "C" float __hc_exp2(float x) restrict(amp);
extern "C" float __hc_exp2_native(float x) restrict(amp);
extern "C" double __hc_exp2_double(double x) restrict(amp);

extern "C" __fp16 __hc_expm1_half(__fp16 x) restrict(amp);
extern "C" float __hc_expm1(float x) restrict(amp);
extern "C" double __hc_expm1_double(double x) restrict(amp);

extern "C" __fp16 __hc_fabs_half(__fp16 x) restrict(amp);
extern "C" float __hc_fabs(float x) restrict(amp);
extern "C" double __hc_fabs_double(double x) restrict(amp);

extern "C" __fp16 __hc_fdim_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_fdim(float x, float y) restrict(amp);
extern "C" double __hc_fdim_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_floor_half(__fp16 x) restrict(amp);
extern "C" float __hc_floor(float x) restrict(amp);
extern "C" double __hc_floor_double(double x) restrict(amp);

extern "C" __fp16 __hc_fma_half(__fp16 x, __fp16 y, __fp16 z) restrict(amp);
extern "C" float __hc_fma(float x, float y, float z) restrict(amp);
extern "C" double __hc_fma_double(double x, double y, double z) restrict(amp);

extern "C" __fp16 __hc_fmax_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_fmax(float x, float y) restrict(amp);
extern "C" double __hc_fmax_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_fmin_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_fmin(float x, float y) restrict(amp);
extern "C" double __hc_fmin_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_fmod_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_fmod(float x, float y) restrict(amp);
extern "C" double __hc_fmod_double(double x, double y) restrict(amp);

extern "C" int __hc_fpclassify_half(__fp16 x) restrict(amp);
extern "C" int __hc_fpclassify(float x) restrict(amp);
extern "C" int __hc_fpclassify_double(double x) restrict(amp);

extern "C" __fp16 __hc_frexp_half(__fp16 x, int *exp) restrict(amp);
extern "C" float __hc_frexp(float x, int *exp) restrict(amp);
extern "C" double __hc_frexp_double(double x, int *exp) restrict(amp);

extern "C" __fp16 __hc_hypot_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_hypot(float x, float y) restrict(amp);
extern "C" double __hc_hypot_double(double x, double y) restrict(amp);

extern "C" int __hc_ilogb_half(__fp16 x) restrict(amp);
extern "C" int __hc_ilogb(float x) restrict(amp);
extern "C" int __hc_ilogb_double(double x) restrict(amp);

extern "C" int __hc_isfinite_half(__fp16 x) restrict(amp);
extern "C" int __hc_isfinite(float x) restrict(amp);
extern "C" int __hc_isfinite_double(double x) restrict(amp);

extern "C" int __hc_isinf_half(__fp16 x) restrict(amp);
extern "C" int __hc_isinf(float x) restrict(amp);
extern "C" int __hc_isinf_double(double x) restrict(amp);

extern "C" int __hc_isnan_half(__fp16 x) restrict(amp);
extern "C" int __hc_isnan(float x) restrict(amp);
extern "C" int __hc_isnan_double(double x) restrict(amp);

extern "C" int __hc_isnormal_half(__fp16 x) restrict(amp);
extern "C" int __hc_isnormal(float x) restrict(amp);
extern "C" int __hc_isnormal_double(double x) restrict(amp);

extern "C" __fp16 __hc_ldexp_half(__fp16 x, int exp) restrict(amp);
extern "C" float __hc_ldexp(float x, int exp) restrict(amp);
extern "C" double __hc_ldexp_double(double x, int exp) restrict(amp);

extern "C" __fp16 __hc_lgamma_half(__fp16 x) restrict(amp);
extern "C" float __hc_lgamma(float x) restrict(amp);
extern "C" double __hc_lgamma_double(double x) restrict(amp);

extern "C" __fp16 __hc_log_half(__fp16 x) restrict(amp);
extern "C" float __hc_log(float x) restrict(amp);
extern "C" float __hc_log_native(float x) restrict(amp);
extern "C" double __hc_log_double(double x) restrict(amp);

extern "C" __fp16 __hc_log10_half(__fp16 x) restrict(amp);
extern "C" float __hc_log10(float x) restrict(amp);
extern "C" float __hc_log10_native(float x) restrict(amp);
extern "C" double __hc_log10_double(double x) restrict(amp);

extern "C" __fp16 __hc_log2_half(__fp16 x) restrict(amp);
extern "C" float __hc_log2(float x) restrict(amp);
extern "C" float __hc_log2_native(float x) restrict(amp);
extern "C" double __hc_log2_double(double x) restrict(amp);

extern "C" __fp16 __hc_log1p_half(__fp16 x) restrict(amp);
extern "C" float __hc_log1p(float x) restrict(amp);
extern "C" double __hc_log1p_double(double x) restrict(amp);

extern "C" __fp16 __hc_logb_half(__fp16 x) restrict(amp);
extern "C" float __hc_logb(float x) restrict(amp);
extern "C" double __hc_logb_double(double x) restrict(amp);

extern "C" __fp16 __hc_modf_half(__fp16 x, __fp16 *iptr) restrict(amp);
extern "C" float __hc_modf(float x, float *iptr) restrict(amp);
extern "C" double __hc_modf_double(double x, double *iptr) restrict(amp);

extern "C" __fp16 __hc_nan_half(int tagp) restrict(amp);
extern "C" float __hc_nan(int tagp) restrict(amp);
extern "C" double __hc_nan_double(unsigned long tagp) restrict(amp);

extern "C" __fp16 __hc_nearbyint_half(__fp16 x) restrict(amp);
extern "C" float __hc_nearbyint(float x) restrict(amp);
extern "C" double __hc_nearbyint_double(double x) restrict(amp);

extern "C" __fp16 __hc_nextafter_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_nextafter(float x, float y) restrict(amp);
extern "C" double __hc_nextafter_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_pow_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_pow(float x, float y) restrict(amp);
extern "C" double __hc_pow_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_rcbrt_half(__fp16 x) restrict(amp);
extern "C" float __hc_rcbrt(float x) restrict(amp);
extern "C" double __hc_rcbrt_double(double x) restrict(amp);

extern "C" __fp16 __hc_remainder_half(__fp16 x, __fp16 y) restrict(amp);
extern "C" float __hc_remainder(float x, float y) restrict(amp);
extern "C" double __hc_remainder_double(double x, double y) restrict(amp);

extern "C" __fp16 __hc_remquo_half(__fp16 x, __fp16 y, int *quo) restrict(amp);
extern "C" float __hc_remquo(float x, float y, int *quo) restrict(amp);
extern "C" double __hc_remquo_double(double x, double y, int *quo) restrict(amp);

extern "C" __fp16 __hc_round_half(__fp16 x) restrict(amp);
extern "C" float __hc_round(float x) restrict(amp);
extern "C" double __hc_round_double(double x) restrict(amp);

extern "C" __fp16 __hc_rsqrt_half(__fp16 x) restrict(amp);
extern "C" float __hc_rsqrt(float x) restrict(amp);
extern "C" float __hc_rsqrt_native(float x) restrict(amp);
extern "C" double __hc_rsqrt_double(double x) restrict(amp);

extern "C" __fp16 __hc_scalb_half(__fp16 x, __fp16 exp) restrict(amp);
extern "C" float __hc_scalb(float x, float exp) restrict(amp);
extern "C" double __hc_scalb_double(double x, double exp) restrict(amp);

extern "C" __fp16 __hc_scalbn_half(__fp16 x, int exp) restrict(amp);
extern "C" float __hc_scalbn(float x, int exp) restrict(amp);
extern "C" double __hc_scalbn_double(double x, int exp) restrict(amp);

extern "C" __fp16 __hc_sinpi_half(__fp16 x) restrict(amp);
extern "C" float __hc_sinpi(float x) restrict(amp);
extern "C" double __hc_sinpi_double(double x) restrict(amp);

extern "C" int __hc_signbit_half(__fp16 x) restrict(amp);
extern "C" int __hc_signbit(float x) restrict(amp);
extern "C" int __hc_signbit_double(double x) restrict(amp);

extern "C" __fp16 __hc_sin_half(__fp16 x) restrict(amp);
extern "C" float __hc_sin(float x) restrict(amp);
extern "C" float __hc_sin_native(float x) restrict(amp);
extern "C" double __hc_sin_double(double x) restrict(amp);

extern "C" __fp16 __hc_sincos_half(__fp16 x, __fp16 *c) restrict(amp);
extern "C" float __hc_sincos(float x, float *c) restrict(amp);
extern "C" double __hc_sincos_double(double x, double *c) restrict(amp);

extern "C" __fp16 __hc_sinh_half(__fp16 x) restrict(amp);
extern "C" float __hc_sinh(float x) restrict(amp);
extern "C" double __hc_sinh_double(double x) restrict(amp);

extern "C" __fp16 __hc_sqrt_half(__fp16 x) restrict(amp);
extern "C" float __hc_sqrt(float x) restrict(amp);
extern "C" float __hc_sqrt_native(float x) restrict(amp);
extern "C" double __hc_sqrt_double(double x) restrict(amp);

extern "C" __fp16 __hc_tgamma_half(__fp16 x) restrict(amp);
extern "C" float __hc_tgamma(float x) restrict(amp);
extern "C" double __hc_tgamma_double(double x) restrict(amp);

extern "C" __fp16 __hc_tan_half(__fp16 x) restrict(amp);
extern "C" float __hc_tan(float x) restrict(amp);
extern "C" float __hc_tan_native(float x) restrict(amp);
extern "C" double __hc_tan_double(double x) restrict(amp);

extern "C" __fp16 __hc_tanh_half(__fp16 x) restrict(amp);
extern "C" float __hc_tanh(float x) restrict(amp);
extern "C" double __hc_tanh_double(double x) restrict(amp);

extern "C" __fp16 __hc_tanpi_half(__fp16 x) restrict(amp);
extern "C" float __hc_tanpi(float x) restrict(amp);
extern "C" double __hc_tanpi_double(double x) restrict(amp);

extern "C" __fp16 __hc_trunc_half(__fp16 x) restrict(amp);
extern "C" float __hc_trunc(float x) restrict(amp);
extern "C" double __hc_trunc_double(double x) restrict(amp);

namespace Kalmar {
namespace fast_math {
  inline float host_acosf(float x) restrict(cpu) { return std::acos(x); }
  inline float acosf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acosf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 acos(__fp16 x) restrict(amp) { return __hc_acos_half(x); }

  inline float host_acos(float x) restrict(cpu) { return std::acos(x); }
  inline float acos(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acos(x);
    #endif
  }

  inline float host_asinf(float x) restrict(cpu) { return std::asin(x); }
  inline float asinf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asinf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 asin(__fp16 x) restrict(amp) { return __hc_asin_half(x); }

  inline float host_asin(float x) restrict(cpu) { return std::asin(x); }
  inline float asin(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asin(x);
    #endif
  }

  inline float host_atanf(float x) restrict(cpu) { return std::atan(x); }
  inline float atanf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atanf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 atan(__fp16 x) restrict(amp) { return __hc_atan_half(x); }

  inline float host_atan(float x) restrict(cpu) { return std::atan(x); }
  inline float atan(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atan(x);
    #endif
  }

  inline float host_atan2f(float y, float x) restrict(cpu) {
    return std::atan2(y, x);
  }
  inline float atan2f(float y, float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(y, x);
    #else
      return host_atan2f(y, x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 atan2(__fp16 y, __fp16 x) restrict(amp) {
    return __hc_atan2_half(y, x);
  }

  inline float host_atan2(float y, float x) restrict(cpu) {
    return std::atan2(y, x);
  }
  inline float atan2(float y, float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(y, x);
    #else
      return host_atan2(y, x);
    #endif
  }

  inline float host_ceilf(float x) restrict(cpu) { return std::ceil(x); }
  inline float ceilf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceilf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 ceil(__fp16 x) restrict(amp) { return __hc_ceil_half(x); }

  inline float host_ceil(float x) restrict(cpu) { return std::ceil(x); }
  inline float ceil(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceil(x);
    #endif
  }

  inline float host_cosf(float x) restrict(cpu) { return std::cos(x); }
  inline float cosf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos_native(x);
    #else
      return host_cosf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 cos(__fp16 x) restrict(amp) { return __hc_cos_half(x); }

  inline float host_cos(float x) restrict(cpu) { return std::cos(x); }
  inline float cos(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos_native(x);
    #else
      return host_cos(x);
    #endif
  }

  inline float host_coshf(float x) restrict(cpu) { return std::cosh(x); }
  inline float coshf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_coshf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 cosh(__fp16 x) restrict(amp) { return __hc_cosh_half(x); }

  inline float host_cosh(float x) restrict(cpu) { return std::cosh(x); }
  inline float cosh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_cosh(x);
    #endif
  }

  inline float host_expf(float x) restrict(cpu) { return std::exp(x); }
  inline float expf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp_native(x);
    #else
      return host_expf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 exp(__fp16 x) restrict(amp) { return __hc_exp_half(x); }

  inline float host_exp(float x) restrict(cpu) { return std::exp(x); }
  inline float exp(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp_native(x);
    #else
      return host_exp(x);
    #endif
  }

  inline float host_exp2f(float x) restrict(cpu) { return std::exp2(x); }
  inline float exp2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2_native(x);
    #else
      return host_exp2f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 exp2(__fp16 x) restrict(amp) { return __hc_exp2_half(x); }

  inline float host_exp2(float x) restrict(cpu) { return std::exp2(x); }
  inline float exp2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2_native(x);
    #else
      return host_exp2(x);
    #endif
  }

  inline float host_fabsf(float x) restrict(cpu) { return std::fabs(x); }
  inline float fabsf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabsf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fabs(__fp16 x) restrict(amp) { return __hc_fabs_half(x); }

  inline float host_fabs(float x) restrict(cpu) { return std::fabs(x); }
  inline float fabs(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabs(x);
    #endif
  }

  inline float host_floorf(float x) restrict(cpu) { return std::floor(x); }
  inline float floorf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floorf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 floor(__fp16 x) restrict(amp) { return __hc_floor_half(x); }

  inline float host_floor(float x) restrict(cpu) { return std::floor(x); }
  inline float floor(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floor(x);
    #endif
  }

  inline float host_fmaxf(float x, float y) restrict(cpu) {
    return std::fmax(x, y);
  }
  inline float fmaxf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmaxf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fmax(__fp16 x, __fp16 y) restrict(amp) { return __hc_fmax_half(x, y); }

  inline float host_fmax(float x, float y) restrict(cpu) {
    return std::fmax(x, y);
}
  inline float fmax(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }

  inline float host_fminf(float x, float y) restrict(cpu) {
    return std::fmin(x, y);
  }
  inline float fminf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fminf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fmin(__fp16 x, __fp16 y) restrict(amp) { return __hc_fmin_half(x, y); }

  inline float host_fmin(float x, float y) restrict(cpu) {
    return std::fmin(x, y);
  }
  inline float fmin(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }

  inline float host_fmodf(float x, float y) restrict(cpu) {
    return std::fmod(x, y);
  }
  inline float fmodf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmodf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fmod(__fp16 x, __fp16 y) restrict(amp) { return __hc_fmod_half(x, y); }

  inline float host_fmod(float x, float y) restrict(cpu) {
    return std::fmod(x, y);
  }
  inline float fmod(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }

  inline float host_frexpf(float x, int *exp) restrict(cpu) {
    return std::frexp(x, exp);
  }
  inline float frexpf(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp(x, exp);
    #else
      return host_frexpf(x, exp);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 frexp(__fp16 x, int *exp) restrict(amp) {
    return __hc_frexp_half(x, exp);
  }

  inline float host_frexp(float x, int *exp) restrict(cpu) {
    return std::frexp(x, exp);
  }
  inline float frexp(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline
  __attribute__((used))
  int isfinite(__fp16 x) restrict(amp) { return __hc_isfinite_half(x); }

  inline int host_isfinite(float x) restrict(cpu) { return std::isfinite(x); }
  inline int isfinite(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isfinite(x);
    #else
      return host_isfinite(x);
    #endif
  }

  inline
  __attribute__((used))
  int isinf(__fp16 x) restrict(amp) { return __hc_isinf_half(x); }

  inline int host_isinf(float x) restrict(cpu) { return std::isinf(x); }
  inline int isinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isinf(x);
    #else
      return host_isinf(x);
    #endif
  }

  inline
  __attribute__((used))
  int isnan(__fp16 x) restrict(amp) { return __hc_isnan_half(x); }

  inline int host_isnan(float x) restrict(cpu) { return std::isnan(x); }
  inline int isnan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnan(x);
    #elif __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
      return std::isnan(x);
    #else
      return host_isnan(x);
    #endif
  }

  inline float host_ldexpf(float x, int exp) restrict(cpu) {
    return std::ldexp(x,exp);
  }
  inline float ldexpf(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexpf(x,exp);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 ldexp(__fp16 x, int exp) restrict(amp) {
    return __hc_ldexp_half(x,exp);
  }

  inline float host_ldexp(float x, int exp) restrict(cpu) {
    return std::ldexp(x,exp);
  }
  inline float ldexp(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }

  inline float host_logf(float x) restrict(cpu) { return std::log(x); }
  inline float logf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log_native(x);
    #else
      return host_logf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log(__fp16 x) restrict(amp) { return __hc_log_half(x); }

  inline float host_log(float x) restrict(cpu) { return std::log(x); }
  inline float log(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log_native(x);
    #else
      return host_log(x);
    #endif
  }

  inline float host_log10f(float x) restrict(cpu) { return std::log10(x); }
  inline float log10f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10_native(x);
    #else
      return host_log10f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log10(__fp16 x) restrict(amp) { return __hc_log10_half(x); }

  inline float host_log10(float x) restrict(cpu) { return std::log10(x); }
  inline float log10(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10_native(x);
    #else
      return host_log10(x);
    #endif
  }

  inline float host_log2f(float x) restrict(cpu) { return std::log2(x); }
  inline float log2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2_native(x);
    #else
      return host_log2f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log2(__fp16 x) restrict(amp) { return __hc_log2_half(x); }

  inline float host_log2(float x) restrict(cpu) { return std::log2(x); }
  inline float log2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2_native(x);
    #else
      return host_log2(x);
    #endif
  }

  inline float host_modff(float x, float *iptr) restrict(cpu) {
    return std::modf(x, iptr);
  }
  inline float modff(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modf(x, iptr);
    #else
      return host_modff(x, iptr);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 modf(__fp16 x, __fp16 *iptr) restrict(amp) {
    return __hc_modf_half(x, iptr);
  }

  inline float host_modf(float x, float *iptr) restrict(cpu) {
    return std::modf(x, iptr);
  }
  inline float modf(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modf(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline float host_powf(float x, float y) restrict(cpu) {
    return std::pow(x, y);
  }
  inline float powf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_powf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 pow(__fp16 x, __fp16 y) restrict(amp) { return __hc_pow_half(x, y); }

  inline float host_pow(float x, float y) restrict(cpu) {
    return std::pow(x, y);
  }
  inline float pow(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_pow(x, y);
    #endif
  }

  inline float host_roundf(float x) restrict(cpu) { return std::round(x); }
  inline float roundf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_roundf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 round(__fp16 x) restrict(amp) { return __hc_round_half(x); }

  inline float host_round(float x) restrict(cpu) { return std::round(x); }
  inline float round(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_round(x);
    #endif
  }

  inline float host_rsqrtf(float x) restrict(cpu) {
    return 1.0f / (std::sqrt(x));
  }
  inline float rsqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt_native(x);
    #else
      return host_rsqrtf(x);
    #endif
  }

  inline
  __fp16 rsqrt(__fp16 x) restrict(amp) { return __hc_rsqrt_half(x); }

  inline float host_rsqrt(float x) restrict(cpu) {
    return 1.0f / (std::sqrt(x));
  }
  inline float rsqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt_native(x);
    #else
      return host_rsqrt(x);
    #endif
  }

  inline int host_signbitf(float x) restrict(cpu) { return std::signbit(x); }
  inline int signbitf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbitf(x);
    #endif
  }

  inline
  __attribute__((used)) int signbit(__fp16 x) restrict(amp) {
    return __hc_signbit_half(x);
  }

  inline int host_signbit(float x) restrict(cpu) { return std::signbit(x); }
  inline int signbit(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbit(x);
    #endif
  }

  inline float host_sinf(float x) restrict(cpu) { return std::sin(x); }
  inline float sinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin_native(x);
    #else
      return host_sinf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sin(__fp16 x) restrict(amp) { return __hc_sin_half(x); }

  inline float host_sin(float x) restrict(cpu) { return std::sin(x); }
  inline float sin(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin_native(x);
    #else
      return host_sin(x);
    #endif
  }

  inline void host_sincosf(float x, float *s, float *c) restrict(cpu) {
    ::sincosf(x, s, c);
  }
  inline void sincosf(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      *s = __hc_sincos(x, c);
    #else
      host_sincosf(x, s, c);
    #endif
  }

  inline
  __attribute__((used))
  void sincos(__fp16 x, __fp16 *s, __fp16 *c) restrict(amp) {
    *s = __hc_sincos_half(x, c);
  }

  inline void host_sincos(float x, float *s, float *c) restrict(cpu) {
    ::sincosf(x, s, c);
  }
  inline void sincos(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      *s = __hc_sincos(x, c);
    #else
      host_sincos(x, s, c);
    #endif
  }

  inline float host_sinhf(float x) restrict(cpu) { return std::sinh(x); }
  inline float sinhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinhf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sinh(__fp16 x) restrict(amp) { return __hc_sinh_half(x); }

  inline float host_sinh(float x) restrict(cpu) { return std::sinh(x); }
  inline float sinh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinh(x);
    #endif
  }

  inline float host_sqrtf(float x) restrict(cpu) { return std::sqrt(x); }
  inline float sqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt_native(x);
    #else
      return host_sqrtf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sqrt(__fp16 x) restrict(amp) { return __hc_sqrt_half(x); }

  inline float host_sqrt(float x) restrict(cpu) { return std::sqrt(x); }
  inline float sqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt_native(x);
    #else
      return host_sqrt(x);
    #endif
  }

  inline float host_tanf(float x) restrict(cpu) { return std::tan(x); }
  inline float tanf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan_native(x);
    #else
      return host_tanf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 tan(__fp16 x) restrict(amp) { return __hc_tan_half(x); }

  inline float host_tan(float x) restrict(cpu) { return std::tan(x); }
  inline float tan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan_native(x);
    #else
      return host_tan(x);
    #endif
  }

  inline float host_tanhf(float x) restrict(cpu) { return std::tanh(x); }
  inline float tanhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanhf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 tanh(__fp16 x) restrict(amp) { return __hc_tanh_half(x); }

  inline float host_tanh(float x) restrict(cpu) { return std::tanh(x); }
  inline float tanh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }

  inline float host_truncf(float x) restrict(cpu) { return std::trunc(x); }
  inline float truncf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_truncf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 trunc(__fp16 x) restrict(amp) { return __hc_trunc_half(x); }

  inline float host_trunc(float x) restrict(cpu) { return std::trunc(x); }
  inline float trunc(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_trunc(x);
    #endif
  }
} // namespace fast_math

  namespace precise_math {
  inline float host_acosf(float x) restrict(cpu) { return ::acos(x); }
  inline float acosf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acosf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 acos(__fp16 x) restrict(amp) { return __hc_acos_half(x); }

  inline float host_acos(float x) restrict(cpu) { return std::acos(x); }
  inline float acos(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos(x);
    #else
      return host_acos(x);
    #endif
  }

  inline double host_acos(double x) restrict(cpu) { return std::acos(x); }
  inline double acos(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acos_double(x);
    #else
      return host_acos(x);
    #endif
  }

  inline float host_acoshf(float x) restrict(cpu) { return std::acosh(x); }
  inline float acoshf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acosh(x);
    #else
      return host_acoshf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 acosh(__fp16 x) restrict(amp) { return __hc_acosh_half(x); }

  inline float host_acosh(float x) restrict(cpu) { return std::acosh(x); }
  inline float acosh(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acosh(x);
    #else
      return host_acosh(x);
    #endif
  }

  inline double host_acosh(double x) restrict(cpu) { return std::acosh(x); }
  inline double acosh(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_acosh_double(x);
    #else
      return host_acosh(x);
    #endif
  }

  inline float host_asinf(float x) restrict(cpu) { return std::asin(x); }
  inline float asinf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asinf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 asin(__fp16 x) restrict(amp) { return __hc_asin_half(x); }

  inline float host_asin(float x) restrict(cpu) { return std::asin(x); }
  inline float asin(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin(x);
    #else
      return host_asin(x);
    #endif
  }

  inline double host_asin(double x) restrict(cpu) { return std::asin(x); }
  inline double asin(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asin_double(x);
    #else
      return host_asin(x);
    #endif
  }

  inline float host_asinhf(float x) restrict(cpu) { return std::asinh(x); }
  inline float asinhf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asinh(x);
    #else
      return host_asinhf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 asinh(__fp16 x) restrict(amp) { return __hc_asinh_half(x); }

  inline float host_asinh(float x) restrict(cpu) { return std::asinh(x); }
  inline float asinh(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asinh(x);
    #else
      return host_asinh(x);
    #endif
  }

  inline double host_asinh(double x) restrict(cpu) { return std::asinh(x); }
  inline double asinh(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_asinh_double(x);
    #else
      return host_asinh(x);
    #endif
  }

  inline float host_atanf(float x) restrict(cpu) { return std::atan(x); }
  inline float atanf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atanf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 atan(__fp16 x) restrict(amp) { return __hc_atan_half(x); }

  inline float host_atan(float x) restrict(cpu) { return std::atan(x); }
  inline float atan(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan(x);
    #else
      return host_atan(x);
    #endif
  }

  inline double host_atan(double x) restrict(cpu) { return std::atan(x); }
  inline double atan(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan_double(x);
    #else
      return host_atan(x);
    #endif
  }

  inline float host_atanhf(float x) restrict(cpu) { return std::atanh(x); }
  inline float atanhf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atanh(x);
    #else
      return host_atanhf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 atanh(__fp16 x) restrict(amp) { return __hc_atanh_half(x); }

  inline float host_atanh(float x) restrict(cpu) { return std::atanh(x); }
  inline float atanh(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atanh(x);
    #else
      return host_atanh(x);
    #endif
  }

  inline double host_atanh(double x) restrict(cpu) { return std::atanh(x); }
  inline double atanh(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atanh_double(x);
    #else
      return host_atanh(x);
    #endif
  }

  inline float host_atan2f(float y, float x) restrict(cpu) {
    return std::atan2(y, x);
  }
  inline float atan2f(float y, float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(y, x);
    #else
      return host_atan2f(y, x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 atan2(__fp16 x, __fp16 y) restrict(amp) {
    return __hc_atan2_half(x, y);
  }

  inline float host_atan2(float y, float x) restrict(cpu) {
    return std::atan2(y, x);
  }
  inline float atan2(float y, float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2(y, x);
    #else
      return host_atan2(y, x);
    #endif
  }

  inline double host_atan2(double y, double x) restrict(cpu) {
    return std::atan2(y, x);
  }
  inline double atan2(double y, double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_atan2_double(y, x);
    #else
      return host_atan2(y, x);
    #endif
  }

  inline float host_cbrtf(float x) restrict(cpu) { return std::cbrt(x); }
  inline float cbrtf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cbrt(x);
    #else
      return host_cbrtf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 cbrt(__fp16 x) restrict(amp) { return __hc_cbrt_half(x); }

  inline float host_cbrt(float x) restrict(cpu) { return std::cbrt(x); }
  inline float cbrt(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cbrt(x);
    #else
      return host_cbrt(x);
    #endif
  }

  inline double host_cbrt(double x) restrict(cpu) { return std::cbrt(x); }
  inline double cbrt(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cbrt_double(x);
    #else
      return host_cbrt(x);
    #endif
  }

  inline float host_ceilf(float x) restrict(cpu) { return std::ceil(x); }
  inline float ceilf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceilf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 ceil(__fp16 x) restrict(amp) { return __hc_ceil_half(x); }

  inline float host_ceil(float x) restrict(cpu) { return std::ceil(x); }
  inline float ceil(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil(x);
    #else
      return host_ceil(x);
    #endif
  }

  inline double host_ceil(double x) restrict(cpu) { return std::ceil(x); }
  inline double ceil(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ceil_double(x);
    #else
      return host_ceil(x);
    #endif
  }

  inline float host_copysignf(float x, float y) restrict(cpu) {
    return std::copysign(x, y);
  }
  inline float copysignf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_copysign(x, y);
    #else
      return host_copysignf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 copysign(__fp16 x, __fp16 y) restrict(amp) {
    return __hc_copysign_half(x, y);
  }

  inline float host_copysign(float x, float y) restrict(cpu) {
    return std::copysign(x, y);
  }
  inline float copysign(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_copysign(x, y);
    #else
      return host_copysign(x, y);
    #endif
  }

  inline double host_copysign(double x, double y) restrict(cpu) {
    return std::copysign(x, y);
  }
  inline double copysign(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_copysign_double(x, y);
    #else
      return host_copysign(x, y);
    #endif
  }

  inline float host_cosf(float x) restrict(cpu) { return std::cos(x); }
  inline float cosf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos(x);
    #else
      return host_cosf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 cos(__fp16 x) restrict(amp) { return __hc_cos_half(x); }

  inline float host_cos(float x) restrict(cpu) { return std::cos(x); }
  inline float cos(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos(x);
    #else
      return host_cos(x);
    #endif
  }

  inline double host_cos(double x) restrict(cpu) { return std::cos(x); }
  inline double cos(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cos_double(x);
    #else
      return host_cos(x);
    #endif
  }

  inline float host_coshf(float x) restrict(cpu) { return std::cosh(x); }
  inline float coshf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_coshf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 cosh(__fp16 x) restrict(amp) { return __hc_cosh_half(x); }

  inline float host_cosh(float x) restrict(cpu) { return std::cosh(x); }
  inline float cosh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh(x);
    #else
      return host_cosh(x);
    #endif
  }

  inline double host_cosh(double x) restrict(cpu) { return std::cosh(x); }
  inline double cosh(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cosh_double(x);
    #else
      return host_cosh(x);
    #endif
  }

  inline float host_cospif(float x) restrict(cpu) {
    return std::cos((float)M_PI * x);
  }
  inline float cospif(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cospi(x);
    #else
      return host_cospif(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 cospi(__fp16 x) restrict(amp) { return __hc_cospi_half(x); }

  inline float host_cospi(float x) restrict(cpu) {
    return std::cos((float)M_PI * x);
  }
  inline float cospi(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cospi(x);
    #else
      return host_cospi(x);
    #endif
  }

  inline double host_cospi(double x) restrict(cpu) {
    return std::cos(M_PI * x);
  }
  inline double cospi(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_cospi_double(x);
    #else
      return host_cospi(x);
    #endif
  }

  inline float host_erff(float x) restrict(cpu) { return std::erf(x); }
  inline float erff(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erf(x);
    #else
      return host_erff(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 erf(__fp16 x) restrict(amp) { return __hc_erf_half(x); }

  inline float host_erf(float x) restrict(cpu) { return std::erf(x); }
  inline float erf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erf(x);
    #else
      return host_erf(x);
    #endif
  }

  inline double host_erf(double x) restrict(cpu) { return std::erf(x); }
  inline double erf(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erf_double(x);
    #else
      return host_erf(x);
    #endif
  }

  inline float host_erfcf(float x) restrict(cpu) { return std::erfcf(x); }
  inline float erfcf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfc(x);
    #else
      return host_erfcf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 erfc(__fp16 x) restrict(amp) { return __hc_erfc_half(x); }

  inline float host_erfc(float x) restrict(cpu) { return std::erfcf(x); }
  inline float erfc(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfc(x);
    #else
      return host_erfc(x);
    #endif
  }

  inline double host_erfc(double x) restrict(cpu) { return std::erfc(x); }
  inline double erfc(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfc_double(x);
    #else
      return host_erfc(x);
    #endif
  }

  inline float erfcinvf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfcinv(x);
    #else
      return ::erfcinv(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 erfcinv(__fp16 x) restrict(amp) { return __hc_erfcinv_half(x); }

  inline float erfcinv(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfcinv(x);
    #else
      return ::erfcinv(x);
    #endif
  }

  inline double erfcinv(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfcinv_double(x);
    #else
      return ::erfcinv(x);
    #endif
  }

  inline float erfinvf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfinv(x);
    #else
      return ::erfinv(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 erfinv(__fp16 x) restrict(amp) { return __hc_erfinv_half(x); }

  inline float erfinv(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfinv(x);
    #else
      return ::erfinv(x);
    #endif
  }

  inline double erfinv(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_erfinv_double(x);
    #else
      return ::erfinv(x);
    #endif
  }

  inline float host_expf(float x) restrict(cpu) { return std::exp(x); }
  inline float expf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp(x);
    #else
      return host_expf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 exp(__fp16 x) restrict(amp) { return __hc_exp_half(x); }

  inline float host_exp(float x) restrict(cpu) { return std::exp(x); }
  inline float exp(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp(x);
    #else
      return host_exp(x);
    #endif
  }

  inline double host_exp(double x) restrict(cpu) { return std::exp(x); }
  inline double exp(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp_double(x);
    #else
      return host_exp(x);
    #endif
  }

  inline float host_exp2f(float x) restrict(cpu) { return std::exp2(x); }
  inline float exp2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2(x);
    #else
      return host_exp2f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 exp2(__fp16 x) restrict(amp) { return __hc_exp2_half(x); }

  inline float host_exp2(float x) restrict(cpu) { return std::exp2(x); }
  inline float exp2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2(x);
    #else
      return host_exp2(x);
    #endif
  }

  inline double host_exp2(double x) restrict(cpu) { return std::exp2(x); }
  inline double exp2(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp2_double(x);
    #else
      return host_exp2(x);
    #endif
  }

  inline float host_exp10f(float x) restrict(cpu) { return std::pow(10.f, x); }
  inline float exp10f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp10(x);
    #else
      return host_exp10f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 exp10(__fp16 x) restrict(amp) { return __hc_exp10_half(x); }

  inline float host_exp10(float x) restrict(cpu) { return std::pow(10.f, x); }
  inline float exp10(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp10(x);
    #else
      return host_exp10(x);
    #endif
  }

  inline double host_exp10(double x) restrict(cpu) { return std::pow(10., x); }
  inline double exp10(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_exp10_double(x);
    #else
      return host_exp10(x);
    #endif
  }

  inline float host_expm1f(float x) restrict(cpu) { return std::expm1(x); }
  inline float expm1f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_expm1(x);
    #else
      return host_expm1f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 expm1(__fp16 x) restrict(amp) { return __hc_expm1_half(x); }

  inline float host_expm1(float x) restrict(cpu) { return std::expm1(x); }
  inline float expm1(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_expm1(x);
    #else
      return host_expm1(x);
    #endif
  }

  inline double host_expm1(double x) restrict(cpu) { return std::expm1(x); }
  inline double expm1(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_expm1_double(x);
    #else
      return host_expm1(x);
    #endif
  }

  inline float host_fabsf(float x) restrict(cpu) { return std::fabs(x); }
  inline float fabsf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabsf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fabs(__fp16 x) restrict(amp) { return __hc_fabs_half(x); }

  inline float host_fabs(float x) restrict(cpu) { return std::fabs(x); }
  inline float fabs(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs(x);
    #else
      return host_fabs(x);
    #endif
  }

  inline double host_fabs(double x) restrict(cpu) { return std::fabs(x); }
  inline double fabs(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fabs_double(x);
    #else
      return host_fabs(x);
    #endif
  }

  inline float host_fdimf(float x, float y) restrict(cpu) {
    return std::fdim(x, y);
  }
  inline float fdimf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fdim(x, y);
    #else
      return host_fdimf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fdim(__fp16 x, __fp16 y) restrict(amp) { return __hc_fdim_half(x, y); }

  inline float host_fdim(float x, float y) restrict(cpu) {
    return std::fdim(x, y);
  }
  inline float fdim(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fdim(x, y);
    #else
      return host_fdim(x, y);
    #endif
  }

  inline double host_fdim(double x, double y) restrict(cpu) {
    return std::fdim(x, y);
  }
  inline double fdim(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fdim_double(x, y);
    #else
      return host_fdim(x, y);
    #endif
  }

  inline float host_floorf(float x) restrict(cpu) { return std::floor(x); }
  inline float floorf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floorf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 floor(__fp16 x) restrict(amp) { return __hc_floor_half(x); }

  inline float host_floor(float x) restrict(cpu) { return std::floor(x); }
  inline float floor(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor(x);
    #else
      return host_floor(x);
    #endif
  }

  inline double host_floor(double x) restrict(cpu) { return std::floor(x); }
  inline double floor(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_floor_double(x);
    #else
      return host_floor(x);
    #endif
  }


  inline float host_fmaf(float x, float y, float z) restrict(cpu) {
    return std::fmaf(x, y, z);
  }
  inline float fmaf(float x, float y, float z) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fma(x, y , z);
    #else
      return host_fmaf(x, y , z);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fma(__fp16 x, __fp16 y, __fp16 z) restrict(amp) {
    return __hc_fma_half(x, y, z);
  }

  inline float host_fma(float x, float y, float z) restrict(cpu) {
    return std::fmaf(x, y, z);
  }
  inline float fma(float x, float y, float z) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fma(x, y , z);
    #else
      return host_fma(x, y , z);
    #endif
  }

  inline double host_fma(double x, double y, double z) restrict(cpu) {
    return std::fma(x, y, z);
  }
  inline double fma(double x, double y, double z) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fma_double(x, y , z);
    #else
      return host_fma(x, y , z);
    #endif
  }

  inline float host_fmaxf(float x, float y) restrict(cpu) {
    return std::fmax(x, y);
  }
  inline float fmaxf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmaxf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fmax(__fp16 x, __fp16 y) restrict(amp) { return __hc_fmax_half(x, y); }

  inline float host_fmax(float x, float y) restrict(cpu) {
    return std::fmax(x, y);
  }
  inline float fmax(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }

  inline double host_fmax(double x, double y) restrict(cpu) {
    return std::fmax(x, y);
  }
  inline double fmax(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmax_double(x, y);
    #else
      return host_fmax(x, y);
    #endif
  }

  inline float host_fminf(float x, float y) restrict(cpu) {
    return std::fmin(x, y);
  }
  inline float fminf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fminf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fmin(__fp16 x, __fp16 y) restrict(amp) { return __hc_fmin_half(x, y); }

  inline float host_fmin(float x, float y) restrict(cpu) {
    return std::fmin(x, y);
  }
  inline float fmin(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }

  inline double host_fmin(double x, double y) restrict(cpu) {
    return std::fmin(x, y);
  }
  inline double fmin(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmin_double(x, y);
    #else
      return host_fmin(x, y);
    #endif
  }

  inline float host_fmodf(float x, float y) restrict(cpu) {
    return std::fmod(x, y);
  }
  inline float fmodf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmodf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 fmod(__fp16 x, __fp16 y) restrict(amp) { return __hc_fmod_half(x, y); }

  inline float host_fmod(float x, float y) restrict(cpu) {
    return std::fmod(x, y);
  }
  inline float fmod(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }

  inline double host_fmod(double x, double y) restrict(cpu) {
    return std::fmod(x, y);
  }
  inline double fmod(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_fmod_double(x, y);
    #else
      return host_fmod(x, y);
    #endif
  }

  inline
  __attribute__((used))
  int fpclassify(__fp16 x) restrict(amp) { return __hc_fpclassify_half(x); }

  inline
  int fpclassify(float x) restrict(cpu) { return std::fpclassify(x); }

  inline
  __attribute__((used))
  int fpclassify(float x) restrict(amp) { return __hc_fpclassify(x); }

  inline
  int fpclassify(double x) restrict(cpu) { return std::fpclassify(x); }

  inline
  __attribute__((used))
  int fpclassify(double x) restrict(amp) { return __hc_fpclassify_double(x); }

  inline float host_frexpf(float x, int *exp) restrict(cpu) {
    return std::frexp(x, exp);
  }
  inline float frexpf(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp(x, exp);
    #else
      return host_frexpf(x, exp);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 frexp(__fp16 x, int* e) restrict(amp) { return __hc_frexp_half(x, e); }

  inline float host_frexp(float x, int *exp) restrict(cpu) {
    return std::frexp(x, exp);
  }
  inline float frexp(float x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline double host_frexp(double x, int *exp) restrict(cpu) {
    return std::frexp(x, exp);
  }
  inline double frexp(double x, int *exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_frexp_double(x, exp);
    #else
      return host_frexp(x, exp);
    #endif
  }

  inline float host_hypotf(float x, float y) restrict(cpu) {
    return std::hypot(x, y);
  }
  inline float hypotf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_hypot(x, y);
    #else
      return host_hypotf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 hypot(__fp16 x, __fp16 y) restrict(amp) {
    return __hc_hypot_half(x, y);
  }

  inline float host_hypot(float x, float y) restrict(cpu) {
    return std::hypot(x, y);
  }
  inline float hypot(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_hypot(x, y);
    #else
      return host_hypot(x, y);
    #endif
  }

  inline double host_hypot(double x, double y) restrict(cpu) {
    return std::hypot(x, y);
  }
  inline double hypot(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_hypot_double(x, y);
    #else
      return host_hypot(x, y);
    #endif
  }

  inline int host_ilogbf(float x) restrict(cpu) { return std::ilogb(x); }
  inline int ilogbf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ilogb(x);
    #else
      return host_ilogbf(x);
    #endif
  }

  inline
  __attribute__((used))
  int ilogb(__fp16 x) restrict(amp) { return __hc_ilogb_half(x); }

  inline int host_ilogb(float x) restrict(cpu) { return std::ilogb(x); }
  inline int ilogb(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ilogb(x);
    #else
      return host_ilogb(x);
    #endif
  }

  inline int host_ilogb(double x) restrict(cpu) { return std::ilogb(x); }
  inline int ilogb(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ilogb_double(x);
    #else
      return host_ilogb(x);
    #endif
  }

  inline
  __attribute__((used))
  int isfinite(__fp16 x) restrict(amp) { return __hc_isfinite_half(x); }

  inline int host_isfinite(float x) restrict(cpu) { return std::isfinite(x); }
  inline int isfinite(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isfinite(x);
    #else
      return host_isfinite(x);
    #endif
  }

  inline int host_isfinite(double x) restrict(cpu) { return std::isfinite(x); }
  inline int isfinite(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isfinite_double(x);
    #else
      return host_isfinite(x);
    #endif
  }

  inline
  __attribute__((used))
  int isinf(__fp16 x) restrict(amp) { return __hc_isinf_half(x); }

  inline int host_isinf(float x) restrict(cpu) { return std::isinf(x); }
  inline int isinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isinf(x);
    #else
      return host_isinf(x);
    #endif
  }

  inline int host_isinf(double x) restrict(cpu) { return std::isinf(x); }
  inline int isinf(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isinf_double(x);
    #else
      return host_isinf(x);
    #endif
  }

  inline
  __attribute__((used))
  int isnan(__fp16 x) restrict(amp) { return __hc_isnan_half(x); }

  inline int host_isnan(float x) restrict(cpu) { return std::isnan(x); }
  inline int isnan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnan(x);
    #else
      return host_isnan(x);
    #endif
  }

  inline int host_isnan(double x) restrict(cpu) { return std::isnan(x); }
  inline int isnan(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnan_double(x);
    #else
      return host_isnan(x);
    #endif
  }

  inline
  __attribute__((used))
  int isnormal(__fp16 x) restrict(amp) { return __hc_isnormal_half(x); }

  inline int host_isnormal(float x) restrict(cpu) { return std::isnormal(x); }
  inline int isnormal(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnormal(x);
    #else
      return host_isnormal(x);
    #endif
  }

  inline int host_isnormal(double x) restrict(cpu) { return std::isnormal(x); }
  inline int isnormal(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_isnormal_double(x);
    #else
      return host_isnormal(x);
    #endif
  }

  inline float host_ldexpf(float x, int exp) restrict(cpu) {
    return std::ldexp(x,exp);
  }
  inline float ldexpf(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexpf(x,exp);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 ldexp(__fp16 x, int e) restrict(amp) { return __hc_ldexp_half(x, e); }

  inline float host_ldexp(float x, int exp) restrict(cpu) {
    return std::ldexp(x,exp);
  }
  inline float ldexp(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }

  inline double host_ldexp(double x, int exp) restrict(cpu) {
    return std::ldexp(x,exp);
  }
  inline double ldexp(double x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_ldexp_double(x,exp);
    #else
      return host_ldexp(x,exp);
    #endif
  }

  inline float host_lgammaf(float x) restrict(cpu) { return std::lgamma(x); }
  inline float lgammaf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_lgamma(x);
    #else
      return host_lgammaf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 lgamma(__fp16 x) restrict(amp) { return __hc_lgamma_half(x); }

  inline float host_lgamma(float x) restrict(cpu) { return std::lgamma(x); }
  inline float lgamma(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_lgamma(x);
    #else
      return host_lgamma(x);
    #endif
  }

  inline double host_lgamma(double x) restrict(cpu) { return std::lgamma(x); }
  inline double lgamma(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_lgamma_double(x);
    #else
      return host_lgamma(x);
    #endif
  }

  inline float host_logf(float x) restrict(cpu) { return std::log(x); }
  inline float logf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log(x);
    #else
      return host_logf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log(__fp16 x) restrict(amp) { return __hc_log_half(x); }

  inline float host_log(float x) restrict(cpu) { return std::log(x); }
  inline float log(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log(x);
    #else
      return host_log(x);
    #endif
  }

  inline double host_log(double x) restrict(cpu) { return std::log(x); }
  inline double log(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log_double(x);
    #else
      return host_log(x);
    #endif
  }

  inline float host_log10f(float x) restrict(cpu) { return std::log10(x); }
  inline float log10f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10(x);
    #else
      return host_log10f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log10(__fp16 x) restrict(amp) { return __hc_log10_half(x); }

  inline float host_log10(float x) restrict(cpu) { return std::log10(x); }
  inline float log10(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10(x);
    #else
      return host_log10(x);
    #endif
  }

  inline double host_log10(double x) restrict(cpu) { return std::log10(x); }
  inline double log10(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log10_double(x);
    #else
      return host_log10(x);
    #endif
  }

  inline float host_log2f(float x) restrict(cpu) { return std::log2(x); }
  inline float log2f(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2(x);
    #else
      return host_log2f(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log2(__fp16 x) restrict(amp) { return __hc_log2_half(x); }

  inline float host_log2(float x) restrict(cpu) { return std::log2(x); }
  inline float log2(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2(x);
    #else
      return host_log2(x);
    #endif
  }

  inline double host_log2(double x) restrict(cpu) { return std::log2(x); }
  inline double log2(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log2_double(x);
    #else
      return host_log2(x);
    #endif
  }

  inline float host_log1pf(float x) restrict(cpu) { return std::log1p(x); }
  inline float log1pf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log1p(x);
    #else
      return host_log1pf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 log1p(__fp16 x) restrict(amp) { return __hc_log1p_half(x); }

  inline float host_log1p(float x) restrict(cpu) { return std::log1p(x); }
  inline float log1p(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log1p(x);
    #else
      return host_log1p(x);
    #endif
  }

  inline double host_log1p(double x) restrict(cpu) { return std::log1p(x); }
  inline double log1p(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_log1p(x);
    #else
      return host_log1p(x);
    #endif
  }

  inline float host_logbf(float x) restrict(cpu) { return std::logb(x); }
  inline float logbf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_logb(x);
    #else
      return host_logbf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 logb(__fp16 x) restrict(amp) { return __hc_logb_half(x); }

  inline float host_logb(float x) restrict(cpu) { return std::logb(x); }
  inline float logb(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_logb(x);
    #else
      return host_logb(x);
    #endif
  }

  inline double host_logb(double x) restrict(cpu) { return std::logb(x); }
  inline double logb(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_logb_double(x);
    #else
      return host_logb(x);
    #endif
  }

  inline float host_modff(float x, float *iptr) restrict(cpu) {
    return std::modf(x, iptr);
  }
  inline float modff(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modf(x, iptr);
    #else
      return host_modff(x, iptr);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 modf(__fp16 x, __fp16* p) restrict(amp) {
    return __hc_modf_half(x, p);
  }

  inline float host_modf(float x, float *iptr) restrict(cpu) {
    return std::modf(x, iptr);
  }
  inline float modf(float x, float *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modf(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }

  inline double host_modf(double x, double *iptr) restrict(cpu) {
    return std::modf(x, iptr);
  }
  inline double modf(double x, double *iptr) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_modf_double(x, iptr);
    #else
      return host_modf(x, iptr);
    #endif
  }


  inline
  __attribute__((used))
  __fp16 nanh(int x) restrict(amp) { return __hc_nan_half(x); }

  inline float host_nanf(int tagp) restrict(cpu) {
    return std::nan(reinterpret_cast<const char*>(&tagp));
  }
  inline float nanf(int tagp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nan(tagp);
    #else
      return host_nanf(tagp);
    #endif
  }

  inline double host_nan(int tagp) restrict(cpu) {
    return std::nan(reinterpret_cast<const char*>(&tagp));
  }
  inline double nan(int tagp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nan_double(static_cast<unsigned long>(tagp));
    #else
      return host_nan(tagp);
    #endif
  }

  inline float host_nearbyintf(float x) restrict(cpu) {
    return std::nearbyint(x);
  }
  inline float nearbyintf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nearbyint(x);
    #else
      return host_nearbyintf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 nearbyint(__fp16 x) restrict(amp) { return __hc_nearbyint_half(x); }

  inline float host_nearbyint(float x) restrict(cpu) {
    return std::nearbyint(x);
  }
  inline float nearbyint(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nearbyint(x);
    #else
      return host_nearbyint(x);
    #endif
  }

  inline double host_nearbyint(double x) restrict(cpu) {
    return std::nearbyint(x);
  }
  inline double nearbyint(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nearbyint_double(x);
    #else
      return host_nearbyint(x);
    #endif
  }

  inline float host_nextafterf(float x, float y) restrict(cpu) {
    return std::nextafterf(x, y);
  }
  inline float nextafterf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nextafter(x, y);
    #else
      return host_nextafterf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 nextafter(__fp16 x, __fp16 y) restrict(amp) {
    return __hc_nextafter_half(x, y);
  }

  inline float host_nextafter(float x, float y) restrict(cpu) {
    return std::nextafter(x, y);
  }
  inline float nextafter(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nextafter(x, y);
    #else
      return host_nextafter(x, y);
    #endif
  }

  inline double host_nextafter(double x, double y) restrict(cpu) {
    return std::nextafter(x, y);
  }
  inline double nextafter(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_nextafter_double(x, y);
    #else
      return host_nextafter(x, y);
    #endif
  }

  inline float host_powf(float x, float y) restrict(cpu) {
    return std::pow(x, y);
  }
  inline float powf(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_powf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 pow(__fp16 x, __fp16 y) restrict(amp) { return __hc_pow_half(x, y); }

  inline float host_pow(float x, float y) restrict(cpu) {
    return std::pow(x, y);
  }
  inline float pow(float x, float y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow(x, y);
    #else
      return host_pow(x, y);
    #endif
  }

  inline double host_pow(double x, double y) restrict(cpu) {
    return std::pow(x, y);
  }
  inline double pow(double x, double y) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_pow_double(x, y);
    #else
      return host_pow(x, y);
    #endif
  }

  inline float host_rcbrtf(float x) restrict(cpu) {
    return 1.0f / std::cbrt(x);
  }
  inline float rcbrtf(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rcbrt(x);
    #else
      return host_rcbrtf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 rcbrt(__fp16 x) restrict(amp) { return __hc_rcbrt_half(x); }

  inline float host_rcbrt(float x) restrict(cpu) {
    return 1.0f / std::cbrt(x);
  }
  inline float rcbrt(float x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rcbrt(x);
    #else
      return host_rcbrt(x);
    #endif
  }

  inline double host_rcbrt(double x) restrict(cpu) {
    return 1.0 / std::cbrt(x);
  }
  inline double rcbrt(double x) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rcbrt_double(x);
    #else
      return host_rcbrt(x);
    #endif
  }

  inline float host_remainderf(float x, float y) restrict(cpu) {
    return std::remainder(x, y);
  }
  inline float remainderf(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remainder(x, y);
    #else
      return host_remainderf(x, y);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 remainder(__fp16 x, __fp16 y) restrict(amp) {
    return __hc_remainder_half(x, y);
  }

  inline float host_remainder(float x, float y) restrict(cpu) {
    return std::remainder(x, y);
  }
  inline float remainder(float x, float y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remainder(x, y);
    #else
      return host_remainder(x, y);
    #endif
  }

  inline double host_remainder(double x, double y) restrict(cpu) {
    return std::remainder(x, y);
  }
  inline double remainder(double x, double y) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remainder_double(x, y);
    #else
      return host_remainder(x, y);
    #endif
  }

  inline float host_remquof(float x, float y, int *quo) restrict(cpu) {
    return std::remquo(x, y, quo);
  }
  inline float remquof(float x, float y, int *quo) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remquo(x, y, quo);
    #else
      return host_remquof(x, y, quo);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 remquo(__fp16 x, __fp16 y, int* q) restrict(amp) {
    return __hc_remquo_half(x, y, q);
  }

  inline float host_remquo(float x, float y, int *quo) restrict(cpu) {
    return std::remquo(x, y, quo);
  }
  inline float remquo(float x, float y, int *quo) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remquo(x, y, quo);
    #else
      return host_remquo(x, y, quo);
    #endif
  }

  inline double host_remquo(double x, double y, int *quo) restrict(cpu) {
    return std::remquo(x, y, quo);
  }
  inline double remquo(double x, double y, int *quo) restrict(amp,cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_remquo_double(x, y, quo);
    #else
      return host_remquo(x, y, quo);
    #endif
  }

  inline float host_roundf(float x) restrict(cpu) { return std::round(x); }
  inline float roundf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_roundf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 round(__fp16 x) restrict(amp) { return __hc_round_half(x); }

  inline float host_round(float x) restrict(cpu) { return std::round(x); }
  inline float round(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round(x);
    #else
      return host_round(x);
    #endif
  }

  inline double host_round(double x) restrict(cpu) { return std::round(x); }
  inline double round(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_round_double(x);
    #else
      return host_round(x);
    #endif
  }

  inline float host_rsqrtf(float x) restrict(cpu) {
    return 1.0f / std::sqrt(x);
  }
  inline float rsqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt(x);
    #else
      return host_rsqrtf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 rsqrt(__fp16 x) restrict(amp) { return __hc_rsqrt_half(x); }

  inline float host_rsqrt(float x) restrict(cpu) { return 1.0f / std::sqrt(x); }
  inline float rsqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt(x);
    #else
      return host_rsqrt(x);
    #endif
  }

  inline double host_rsqrt(double x) restrict(cpu) {
    return 1.0 / std::sqrt(x);
  }
  inline double rsqrt(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_rsqrt_double(x);
    #else
      return host_rsqrt(x);
    #endif
  }

  inline float host_sinpif(float x) restrict(cpu) {
    return std::sin((float)M_PI * x);
  }
  inline float sinpif(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinpi(x);
    #else
      return host_sinpif(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sinpi(__fp16 x) restrict(amp) { return __hc_sinpi_half(x); }

  inline float host_sinpi(float x) restrict(cpu) {
    return std::sin((float)M_PI * x);
  }
  inline float sinpi(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinpi(x);
    #else
      return host_sinpi(x);
    #endif
  }

  inline double host_sinpi(double x) restrict(cpu) {
    return std::sin(M_PI * x);
  }
  inline double sinpi(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinpi_double(x);
    #else
      return host_sinpi(x);
    #endif
  }

  inline float host_scalbf(float x, float exp) restrict(cpu) {
    return std::scalbn(x, exp);
  }
  inline float scalbf(float x, float exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_scalb(x, exp);
    #else
      return host_scalbf(x, exp);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 scalb(__fp16 x, __fp16 y) restrict(amp) {
    return __hc_scalb_half(x, y);
  }

  inline float host_scalb(float x, float exp) restrict(cpu) {
    return std::scalbn(x, exp);
  }
  inline float scalb(float x, float exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_scalb(x, exp);
    #else
      return host_scalb(x, exp);
    #endif
  }

  inline double host_scalb(double x, double exp) restrict(cpu) {
    return std::scalbn(x, exp);
  }
  inline double scalb(double x, double exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_scalb_double(x, exp);
    #else
      return host_scalb(x, exp);
    #endif
  }

  inline float host_scalbnf(float x, int exp) restrict(cpu) {
    return std::scalbn(x, exp);
  }
  inline float scalbnf(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_scalbn(x, exp);
    #else
      return host_scalbnf(x, exp);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 scalbn(__fp16 x, int e) restrict(amp) {
    return __hc_scalbn_half(x, e);
  }

  inline float host_scalbn(float x, int exp) restrict(cpu) {
    return std::scalbn(x, exp);
  }
  inline float scalbn(float x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_scalbn(x, exp);
    #else
      return host_scalbn(x, exp);
    #endif
  }

  inline double host_scalbn(double x, int exp) restrict(cpu) {
    return std::scalbn(x, exp);
  }
  inline double scalbn(double x, int exp) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_scalbn_double(x, exp);
    #else
      return host_scalbn(x, exp);
    #endif
  }

  inline int host_signbitf(float x) restrict(cpu) { return std::signbit(x); }
  inline int signbitf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbitf(x);
    #endif
  }

  inline
  __attribute__((used))
  int signbit(__fp16 x) restrict(amp) { return __hc_signbit_half(x); }

  inline int host_signbit(float x) restrict(cpu) { return std::signbit(x); }
  inline int signbit(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit(x);
    #else
      return host_signbit(x);
    #endif
  }

  inline int host_signbit(double x) restrict(cpu) { return std::signbit(x); }
  inline int signbit(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_signbit_double(x);
    #else
      return host_signbit(x);
    #endif
  }

  inline float host_sinf(float x) restrict(cpu) { return std::sin(x); }
  inline float sinf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin(x);
    #else
      return host_sinf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sin(__fp16 x) restrict(amp) { return __hc_sin_half(x); }

  inline float host_sin(float x) restrict(cpu) { return std::sin(x); }
  inline float sin(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin(x);
    #else
      return host_sin(x);
    #endif
  }

  inline double host_sin(double x) restrict(cpu) { return std::sin(x); }
  inline double sin(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sin_double(x);
    #else
      return host_sin(x);
    #endif
  }

  inline void host_sincosf(float x, float *s, float *c) restrict(cpu) {
    ::sincosf(x, s, c);
  }
  inline void sincosf(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      *s = __hc_sincos(x, c);
    #else
      host_sincosf(x, s, c);
    #endif
  }

  inline
  __attribute__((used))
  void sincos(__fp16 x, __fp16* s, __fp16* c) restrict(amp) {
    *s = __hc_sincos_half(x, c);
  }

  inline void host_sincos(float x, float *s, float *c) restrict(cpu) {
    ::sincosf(x, s, c);
  }
  inline void sincos(float x, float *s, float *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      *s = __hc_sincos(x, c);
    #else
      host_sincos(x, s, c);
    #endif
  }

  inline void host_sincos(double x, double *s, double *c) restrict(cpu) {
    ::sincos(x, s, c);
  }
  inline void sincos(double x, double *s, double *c) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      *s = __hc_sincos_double(x, c);
    #else
      host_sincos(x, s, c);
    #endif
  }

  inline float host_sinhf(float x) restrict(cpu) { return std::sinh(x); }
  inline float sinhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinhf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sinh(__fp16 x) restrict(amp) { return __hc_sinh_half(x); }

  inline float host_sinh(float x) restrict(cpu) { return std::sinh(x); }
  inline float sinh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh(x);
    #else
      return host_sinh(x);
    #endif
  }

  inline double host_sinh(double x) restrict(cpu) { return std::sinh(x); }
  inline double sinh(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sinh_double(x);
    #else
      return host_sinh(x);
    #endif
  }

  inline float host_sqrtf(float x) restrict(cpu) { return std::sqrt(x); }
  inline float sqrtf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt(x);
    #else
      return host_sqrtf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 sqrt(__fp16 x) restrict(amp) { return __hc_sqrt_half(x); }

  inline float host_sqrt(float x) restrict(cpu) { return std::sqrt(x); }
  inline float sqrt(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt(x);
    #else
      return host_sqrt(x);
    #endif
  }

  inline double host_sqrt(double x) restrict(cpu) { return std::sqrt(x); }
  inline double sqrt(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_sqrt_double(x);
    #else
      return host_sqrt(x);
    #endif
  }

  inline float host_tgammaf(float x) restrict(cpu) { return std::tgamma(x); }
  inline float tgammaf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tgamma(x);
    #else
      return host_tgammaf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 tgamma(__fp16 x) restrict(amp) { return __hc_tgamma_half(x); }

  inline float host_tgamma(float x) restrict(cpu) { return std::tgamma(x); }
  inline float tgamma(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tgamma(x);
    #else
      return host_tgamma(x);
    #endif
  }

  inline double host_tgamma(double x) restrict(cpu) { return std::tgamma(x); }
  inline double tgamma(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tgamma_double(x);
    #else
      return host_tgamma(x);
    #endif
  }

  inline float host_tanf(float x) restrict(cpu) { return std::tan(x); }
  inline float tanf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan(x);
    #else
      return host_tanf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 tan(__fp16 x) restrict(amp) { return __hc_tan_half(x); }

  inline float host_tan(float x) restrict(cpu) { return std::tan(x); }
  inline float tan(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan(x);
    #else
      return host_tan(x);
    #endif
  }

  inline double host_tan(double x) restrict(cpu) { return std::tan(x); }
  inline double tan(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tan_double(x);
    #else
      return host_tan(x);
    #endif
  }

  inline float host_tanhf(float x) restrict(cpu) { return std::tanh(x); }
  inline float tanhf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanhf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 tanh(__fp16 x) restrict(amp) { return __hc_tanh_half(x); }

  inline float host_tanh(float x) restrict(cpu) { return std::tanh(x); }
  inline float tanh(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }

  inline double host_tanh(double x) restrict(cpu) { return std::tanh(x); }
  inline double tanh(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanh(x);
    #else
      return host_tanh(x);
    #endif
  }

  inline float host_tanpif(float x) restrict(cpu) { return std::tan(M_PI * x); }
  inline float tanpif(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanpi(x);
    #else
      return host_tanpif(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 tanpi(__fp16 x) restrict(amp) { return __hc_tanpi_half(x); }

  inline float host_tanpi(float x) restrict(cpu) { return std::tan(M_PI * x); }
  inline float tanpi(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanpi(x);
    #else
      return host_tanpif(x);
    #endif
  }

  inline double host_tanpi(double x) restrict(cpu) { return std::tan(M_PI * x); }
  inline double tanpi(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_tanpi_double(x);
    #else
      return host_tanpi(x);
    #endif
  }

  inline float host_truncf(float x) restrict(cpu) { return std::trunc(x); }
  inline float truncf(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_truncf(x);
    #endif
  }

  inline
  __attribute__((used))
  __fp16 trunc(__fp16 x) restrict(amp) { return __hc_trunc_half(x); }

  inline float host_trunc(float x) restrict(cpu) { return std::trunc(x); }
  inline float trunc(float x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc(x);
    #else
      return host_trunc(x);
    #endif
  }

  inline double host_trunc(double x) restrict(cpu) { return std::trunc(x); }
  inline double trunc(double x) restrict(amp, cpu) {
    #if __KALMAR_ACCELERATOR__ == 1
      return __hc_trunc_double(x);
    #else
      return host_trunc(x);
    #endif
  }

} // namespace precise_math

} // namespace Kalmar
