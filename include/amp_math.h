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
  extern "C" float opencl_cospi(float x);
  extern "C" float opencl_erf(float x);
  extern "C" float opencl_erfc(float x);
  extern "C" float opencl_exp(float x);
  extern "C" float opencl_exp10(float x);
  extern "C" float opencl_exp2(float x);
  extern "C" float opencl_expm1(float x);
  extern "C" float opencl_fabs(float x);
  extern "C" float opencl_fdim(float x, float y);
  extern "C" float opencl_floor(float x);
  extern "C" float opencl_fma(float x, float y, float z);
  extern "C" float opencl_fmax(float x, float y);
  extern "C" float opencl_fmin(float x, float y);
  extern "C" float opencl_fmod(float x, float y);
  extern "C" int opencl_isinf(float x);
  extern "C" int opencl_isfinite(float x);
  extern "C" int opencl_ilogb(float x);
  extern "C" int opencl_isnan(float x);
  extern "C" int opencl_isnormal(float x);
  extern "C" float opencl_ldexp(float x, int exp);
  extern "C" float opencl_log(float x);
  extern "C" float opencl_log2(float x);
  extern "C" float opencl_log10(float x);
  extern "C" float opencl_log1p(float x);
  extern "C" float opencl_logb(float x);
  extern "C" float opencl_hypot(float x, float y);
  extern "C" float opencl_nextafter(float x, float y);
  extern "C" int opencl_min(int x, int y);
  extern "C" float opencl_max(float x, float y);
  extern "C" float opencl_pow(float x, float y);
  extern "C" float opencl_round(float x);
  extern "C" float opencl_remainder(float x, float y);
  extern "C" float opencl_rsqrt(float x);
  extern "C" float opencl_sin(float x);
  extern "C" float opencl_sinh(float x);
  extern "C" int   opencl_signbit(float x);
  extern "C" float opencl_sinpi(float x);
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

  float asinf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asin(x);
    #else
      return ::asinf(x);
    #endif
  }

  float acos(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acos(x);
    #else
      return ::acos(x);
    #endif
  }

  float acosf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acos(x);
    #else
      return ::acosf(x);
    #endif
  }

  float atan(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan(x);
    #else
      return ::atan(x);
    #endif
  }

  float atanf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan(x);
    #else
      return ::atanf(x);
    #endif
  }

  float atan2(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return ::atan2(x, y);
    #endif
  }

  float atan2f(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return ::atan2f(x, y);
    #endif
  }

  float ceil(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ceil(x);
    #else
      return ::ceil(x);
    #endif
  }

  float ceilf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ceil(x);
    #else
      return ::ceilf(x);
    #endif
  }

  float cos(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cos(x);
    #else
      return ::cos(x);
    #endif
  }

  float cosf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cos(x);
    #else
      return ::cosf(x);
    #endif
  }

  float cosh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cosh(x);
    #else
      return ::cosh(x);
    #endif
  }

  float coshf(float x) restrict(amp, cpu) {
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

  float exp2f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp2(x);
    #else
      return ::exp2f(x);
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

  float floorf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_floor(x);
    #else
      return ::floorf(x);
    #endif
  }

  float fmax(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return ::fmax(x, y);
    #endif
  }

  float fmaxf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return ::fmaxf(x, y);
    #endif
  }

  float fmin(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmin(x, y);
    #else
      return ::fmin(x, y);
    #endif
  }

  float fminf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmin(x, y);
    #else
      return ::fminf(x, y);
    #endif
  }

  float fmod(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmod(x, y);
    #else
      return ::fmod(x, y);
    #endif
  }

  float fmodf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmod(x, y);
    #else
      return ::fmodf(x, y);
    #endif
  }

  int isfinite(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isfinite(x);
    #else
      return ::isfinite(x);
    #endif
  }

  int isinf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isinf(x);
    #else
      return ::isinf(x);
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

  float ldexpf(float x, int exp) {
    #ifdef __GPU__
      return opencl_ldexp(x,exp);
    #else
      return ::ldexpf(x,exp);
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

  float log2f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log2(x);
    #else
      return ::log2f(x);
    #endif
  }

  float log10(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log10(x);
    #else
      return ::log10(x);
    #endif
  }

  float log10f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log10(x);
    #else
      return ::log10f(x);
    #endif
  }

  float pow(float x, float y) {
    #ifdef __GPU__
      return opencl_pow(x, y);
    #else
      return ::pow(x, y);
    #endif
  }

  float powf(float x, float y) {
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

  float roundf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_round(x);
    #else
      return ::round(x);
    #endif
  }

  float  rsqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_rsqrt(x);
    #else
      return 1 / (::sqrt(x));
    #endif
  }

  float  rsqrtf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_rsqrt(x);
    #else
      return 1 / (::sqrt(x));
    #endif
  }

  float sin(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sin(x);
    #else
      return ::sin(x);
    #endif
  }

  float sinf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sin(x);
    #else
      return ::sinf(x);
    #endif
  }

  int signbit(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_signbit(x);
    #else
      return ::signbit(x);
    #endif
  }

  int signbitf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_signbit(x);
    #else
      return ::signbit(x);
    #endif
  }

  float sinh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinh(x);
    #else
      return ::sinh(x);
    #endif
  }

  float sinhf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinh(x);
    #else
      return ::sinhf(x);
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
      return ::sqrtf(x);
    #endif
  }

  float tan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tan(x);
    #else
      return ::tan(x);
    #endif
  }

  float tanf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tan(x);
    #else
      return ::tanf(x);
    #endif
  }

  float tanh(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tanh(x);
    #else
      return ::tanh(x);
    #endif
  }

  float tanhf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tanh(x);
    #else
      return ::tanhf(x);
    #endif
  }

  float trunc(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_trunc(x);
    #else
      return ::trunc(x);
    #endif
  }

  float truncf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_trunc(x);
    #else
      return ::truncf(x);
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

  float acosh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acosh(x);
    #else
      return ::acosh(x);
    #endif
  }

  float acoshf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_acosh(x);
    #else
      return ::acoshf(x);
    #endif
  }

  float asinh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asinh(x);
    #else
      return ::asinh(x);
    #endif
  }

  float asinhf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_asinh(x);
    #else
      return ::asinhf(x);
    #endif
  }

  float atanh(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atanh(x);
    #else
      return ::atanh(x);
    #endif
  }

  float atanhf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atanh(x);
    #else
      return ::atanhf(x);
    #endif
  }

  float atan2(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return ::atan2(x, y);
    #endif
  }

  float atan2f(float x, float y) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_atan2(x, y);
    #else
      return ::atan2f(x, y);
    #endif
  }

  float cbrt(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_cbrt(x);
    #else
      return ::cbrt(x);
    #endif
  }

  float cbrtf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return opencl_cbrt(x);
    #else
      return ::cbrtf(x);
    #endif
  }

  float copysign(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_copysign(x, y);
    #else
      return ::copysign(x, y);
    #endif
  }

  float copysignf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_copysign(x, y);
    #else
      return ::copysignf(x, y);
    #endif
  }

  float cospi(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cospi(x);
    #else
      return ::cos(M_PI * x);
    #endif
  }

  float cospif(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_cospi(x);
    #else
      return ::cos(M_PI * x);
    #endif
  }

  float erf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erf(x);
    #else
      return ::erf(x);
    #endif
  }

  float erff(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erf(x);
    #else
      return ::erff(x);
    #endif
  }

  float erfc(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erfc(x);
    #else
      return ::erfc(x);
    #endif
  }

  float erfcf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_erfc(x);
    #else
      return ::erfcf(x);
    #endif
  }

  float exp10(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp10(x);
    #else
      return ::exp10(x);
    #endif
  }

  float exp10f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_exp10(x);
    #else
      return ::exp10f(x);
    #endif
  }

  float expm1(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_expm1(x);
    #else
      return ::expm1(x);
    #endif
  }

  float expm1f(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_expm1(x);
    #else
      return ::expm1f(x);
    #endif
  }

  float fdim(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fdim(x, y);
    #else
      return ::fdim(x, y);
    #endif
  }

  float fdimf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fdim(x, y);
    #else
      return ::fdimf(x, y);
    #endif
  }

  float fma(float x, float y, float z) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fma(x, y , z);
    #else
      return ::fma(x, y , z);
    #endif
  }

  float fmaf(float x, float y, float z) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fma(x, y , z);
    #else
      return ::fmaf(x, y , z);
    #endif
  }

  float fmax(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fmax(x, y);
    #else
      return ::fmax(x, y);
    #endif
  }

  int ilogb(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ilogb(x);
    #else
      return ::ilogb(x);
    #endif
  }

  int ilogbf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ilogb(x);
    #else
      return ::ilogbf(x);
    #endif
  }

  int isnan(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnan(x);
    #else
      return ::isnan(x);
    #endif
  }

  int isnormal(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_isnormal(x);
    #else
      return ::isnormal(x);
    #endif
  }

  float log1p(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log1p(x);
    #else
      return ::log1p(x);
    #endif
  }

  float log1pf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_log1p(x);
    #else
      return ::log1pf(x);
    #endif
  }

  float logb(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_logb(x);
    #else
      return ::logb(x);
    #endif
  }

  float logbf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_logb(x);
    #else
      return ::logbf(x);
    #endif
  }

  float nextafter(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_nextafter(x, y);
    #else
      return ::nextafter(x, y);
    #endif
  }

  float hypotf(float x, float y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_hypot(x, y);
    #else
      return ::hypotf(x, y);
    #endif
  }

  float rcbrt(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return 1 / opencl_cbrt(x);
    #else
      return 1 / (::cbrt(x));
    #endif
  }

  float rcbrtf(float x) restrict(amp,cpu) {
    #ifdef __GPU__
      return 1 / opencl_cbrt(x);
    #else
      return 1 / (::cbrt(x));
    #endif
  }

  float remainder(double x, double y) {
    #ifdef __GPU__
      return opencl_remainder(x, y);
    #else
      return ::remainder(x, y);
    #endif
  }

  float remainderf(double x, double y) {
    #ifdef __GPU__
      return opencl_remainder(x, y);
    #else
      return ::remainder(x, y);
    #endif
  }

  float tgamma(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tgamma(x);
    #else
      return ::tgamma(x);
    #endif
  }

  float tgammaf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_tgamma(x);
    #else
      return ::tgammaf(x);
    #endif
  }

  float scalbn(float x, int exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ldexp(x, exp);
    #else
      return ::scalbn(x, exp);
    #endif
  }

  float scalbnf(float x, int exp) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_ldexp(x, exp);
    #else
      return ::scalbn(x, exp);
    #endif
  }

  float sinpi(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinpi(x);
    #else
      return ::sin(M_PI * x);
    #endif
  }

  float sinpif(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_sinpi(x);
    #else
      return ::sin(M_PI * x);
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
