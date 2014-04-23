#pragma once
  #include <cmath>
#ifdef __GPU__
  extern "C" float opencl_acos(float x);
  extern "C" float opencl_acosh(float x);
  extern "C" float opencl_asin(float x);
  extern "C" float opencl_asinh(float x);
  extern "C" float opencl_atan(float x);
  extern "C" float opencl_atanh(float x);
  extern "C" float opencl_atan2(float x, float y);
  extern "C" float opencl_cos(float x);
  extern "C" float opencl_cosh(float x);
  extern "C" float opencl_cbrt(float x);
  extern "C" float opencl_ceil(float x);
  extern "C" float opencl_copysign(float x, float y);
  extern "C" float opencl_exp(float x);
  extern "C" float opencl_exp2(float x);
  extern "C" float opencl_fabs(float x);
  extern "C" float opencl_fdim(float x, float y);
  extern "C" float opencl_floor(float x);
  extern "C" float opencl_fma(float x, float y, float z);
  extern "C" float opencl_fmax(float x, float y);
  extern "C" float opencl_fmin(float x, float y);
  extern "C" float opencl_fmod(float x, float y);
  extern "C" float opencl_hypot(float x, float y);
  extern "C" float opencl_ilogb(float x);
  extern "C" int opencl_isnan(float x);
  extern "C" float opencl_ldexp(float x, int exp);
  extern "C" float opencl_log(float x);
  extern "C" float opencl_log2(float x);
  extern "C" float opencl_log10(float x);
  extern "C" float opencl_log1p(float x);
  extern "C" float opencl_logb(float x);
  extern "C" int opencl_min(int x, int y);
  extern "C" float opencl_max(float x, float y);
  extern "C" float opencl_pow(float x, float y);
  extern "C" float opencl_round(float x);
  extern "C" float opencl_remainder(float x, float y);
  extern "C" float opencl_sin(float x);
  extern "C" float opencl_sinh(float x);
  extern "C" float opencl_sqrt(float x);
  extern "C" float opencl_tan(float x);
  extern "C" float opencl_tanh(float x);
  extern "C" float opencl_tgamma(float x);
  extern "C" float opencl_trunc(float x);
#endif

namespace Concurrency {
namespace fast_math {
  float asin(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asin(x);
    #else
      return ::asin(x);
    #endif
  }

  float acos(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acos(x);
    #else
      return ::acos(x);
    #endif
  }

  float atan(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan(x);
    #else
      return ::atan(x);
    #endif
  }

  float atan2(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return ::atan2(x, y);
    #endif
  }

  float ceil(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ceil(x);
    #else
      return ::ceil(x);
    #endif
  }

  float cos(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cos(x);
    #else
      return ::cos(x);
    #endif
  }

  float cosh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cosh(x);
    #else
      return ::cosh(x);
    #endif
  }

  float exp(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp(x);
    #else
      return ::exp(x);
    #endif
  }

  float expf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp(x);
    #else
      return ::expf(x);
    #endif
  }

  float exp2(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp2(x);
    #else
      return ::exp2(x);
    #endif
  }

  float fabs(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fabs(x);
    #else
      return ::fabs(x);
    #endif
  }

  float fabsf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fabs(x);
    #else
      return ::fabsf(x);
    #endif
  }

  float floor(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_floor(x);
    #else
      return ::floor(x);
    #endif
  }

  float fmax(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return ::fmax(x, y);
    #endif
  }

  float fmin(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmin(x, y);
    #else
      return ::fmin(x, y);
    #endif
  }

  float fmod(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmod(x, y);
    #else
      return ::fmod(x, y);
    #endif
  }

  int isnan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnan(x);
    #else
      return ::isnan(x);
    #endif
  }

  float ldexp(float x, int exp) {
    #ifdef __GPU__
      return opencl_ldexp(x,exp);
    #else
      return ::ldexp(x,exp);
    #endif
  }

  float log(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log(x);
    #else
      return ::log(x);
    #endif
  }

  float logf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log(x);
    #else
      return ::logf(x);
    #endif
  }

  float log2(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log2(x);
    #else
      return ::log2(x);
    #endif
  }

  float log10(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log10(x);
    #else
      return ::log10(x);
    #endif
  }

  float pow(float x, float y) {
    #ifdef __GPU__
      return opencl_pow(x, y);
    #else
      return ::pow(x, y);
    #endif
  }

  float round(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_round(x);
    #else
      return ::round(x);
    #endif
  }

  float sin(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sin(x);
    #else
      return ::sinf(x);
    #endif
  }

  float sinh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinh(x);
    #else
      return ::sinh(x);
    #endif
  }

  float sqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sqrt(x);
    #else
      return ::sqrt(x);
    #endif
  }

  float sqrtf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sqrt(x);
    #else
      return ::sqrt(x);
    #endif
  }

  float tan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tan(x);
    #else
      return ::tan(x);
    #endif
  }

  float tanh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tanh(x);
    #else
      return ::tanh(x);
    #endif
  }

  float trunc(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_trunc(x);
    #else
      return ::trunc(x);
    #endif
  }

} // namesapce fast_math

  int min(int x, int y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_min(x, y);
    #else
      using std::min;
      return min(x, y);
    #endif
  }

  float max(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_max(x, y);
    #else
      using std::max;
      return max(x, y);
    #endif
  }

  namespace precise_math {

  float acos(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acos(x);
    #else
      return ::acos(x);
    #endif
  }

  float acosh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acosh(x);
    #else
      return ::acosh(x);
    #endif
  }

  float asin(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asin(x);
    #else
      return ::asin(x);
    #endif
  }

  float asinh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asinh(x);
    #else
      return ::asinh(x);
    #endif
  }

  float atan(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan(x);
    #else
      return ::atan(x);
    #endif
  }

  float atanh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atanh(x);
    #else
      return ::atanh(x);
    #endif
  }

  float atan2(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return ::atan2(x, y);
    #endif
  }

  float fdim(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fdim(x, y);
    #else
      return ::fdim(x, y);
    #endif
  }

  float cbrt(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_cbrt(x);
    #else
      return ::cbrt(x);
    #endif
  }

  float ceil(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ceil(x);
    #else
      return ::ceil(x);
    #endif
  }

  float cos(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cos(x);
    #else
      return ::cos(x);
    #endif
  }

  float cosh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cosh(x);
    #else
      return ::cosh(x);
    #endif
  }

  float copysign(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_copysign(x, y);
    #else
      return ::copysign(x, y);
    #endif
  }

  float exp(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp(x);
    #else
      return ::exp(x);
    #endif
  }

  float exp2(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp2(x);
    #else
      return ::exp2(x);
    #endif
  }

  float fma(float x, float y, float z) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fma(x, y , z);
    #else
      return ::fma(x, y , z);
    #endif
  }

  float fmax(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return ::fmax(x, y);
    #endif
  }

  float fmin(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmin(x, y);
    #else
      return ::fmin(x, y);
    #endif
  }

  float fmod(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmod(x, y);
    #else
      return ::fmod(x, y);
    #endif
  }

  float floor(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_floor(x);
    #else
      return ::floor(x);
    #endif
  }

  float fabs(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fabs(x);
    #else
      return ::fabs(x);
    #endif
  }

  float ilogb(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ilogb(x);
    #else
      return ::ilogb(x);
    #endif
  }

  int isnan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnan(x);
    #else
      return ::isnan(x);
    #endif
  }

  float ldexp(float x, int exp) {
    #ifdef __GPU__
      return opencl_ldexp(x,exp);
    #else
      return ::ldexp(x,exp);
    #endif
  }

  float log(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log(x);
    #else
      return ::log(x);
    #endif
  }

  float log10(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log10(x);
    #else
      return ::log10(x);
    #endif
  }

  float log1p(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log1p(x);
    #else
      return ::log1p(x);
    #endif
  }

  float log2(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log2(x);
    #else
      return ::log2(x);
    #endif
  }

  float logb(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_logb(x);
    #else
      return ::logb(x);
    #endif
  }

  float pow(float x, float y) {
    #ifdef __GPU__
      return opencl_pow(x, y);
    #else
      return ::pow(x, y);
    #endif
  }

  float remainder(double x, double y) {
    #ifdef __GPU__
      return opencl_remainder(x, y);
    #else
      return ::remainder(x, y);
    #endif
  }

  float round(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_round(x);
    #else
      return ::round(x);
    #endif
  }

  float sin(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sin(x);
    #else
      return ::sinf(x);
    #endif
  }

  float sinh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinh(x);
    #else
      return ::sinh(x);
    #endif
  }

  float sqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sqrt(x);
    #else
      return ::sqrt(x);
    #endif
  }

  float tan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tan(x);
    #else
      return ::tan(x);
    #endif
  }

  float tanh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tanh(x);
    #else
      return ::tanh(x);
    #endif
  }

  float tgamma(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tgamma(x);
    #else
      return ::tgamma(x);
    #endif
  }

  float trunc(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_trunc(x);
    #else
      return ::trunc(x);
    #endif
  }

} // namespace precise_math

} // namespace Concurrency
