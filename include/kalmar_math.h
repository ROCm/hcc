//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <stdexcept>

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
extern "C" __fp16 __hc_cos_native_half(__fp16 x) restrict(amp);
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
extern "C" double __hc_exp_double(double x) restrict(amp);

extern "C" __fp16 __hc_exp10_half(__fp16 x) restrict(amp);
extern "C" float __hc_exp10(float x) restrict(amp);
extern "C" double __hc_exp10_double(double x) restrict(amp);

extern "C" __fp16 __hc_exp2_native_half(__fp16 x) restrict(amp);
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

extern "C" __fp16 __hc_ldexp_half(__fp16 x, std::int16_t exp) [[hc]];
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
extern "C" double __hc_log10_double(double x) restrict(amp);

extern "C" __fp16 __hc_log2_half(__fp16 x) restrict(amp);
extern "C" __fp16 __hc_log2_native_half(__fp16 x) restrict(amp);
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

// TODO: rcp is implementation only, it does not have a public interface.
extern "C" __fp16 __hc_rcp_native_half(__fp16 x) restrict(amp);
extern "C" float __hc_rcp_native(float x) restrict(amp);

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
extern "C" __fp16 __hc_rsqrt_native_half(__fp16 x) restrict(amp);
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
extern "C" __fp16 __hc_sin_native_half(__fp16 x) restrict(amp);
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
extern "C" __fp16 __hc_sqrt_native_half(__fp16 x) restrict(amp);
extern "C" float __hc_sqrt(float x) restrict(amp);
extern "C" float __hc_sqrt_native(float x) restrict(amp);
extern "C" double __hc_sqrt_double(double x) restrict(amp);

extern "C" __fp16 __hc_tgamma_half(__fp16 x) restrict(amp);
extern "C" float __hc_tgamma(float x) restrict(amp);
extern "C" double __hc_tgamma_double(double x) restrict(amp);

extern "C" __fp16 __hc_tan_half(__fp16 x) restrict(amp);
extern "C" float __hc_tan(float x) restrict(amp);
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

#define HCC_MATH_LIB_FN inline __attribute__((used, hc))
namespace Kalmar
{
    namespace fast_math
    {
        using std::acos;
        using ::acosf;
        using std::asin;
        using ::asinf;
        using std::atan;
        using ::atanf;
        using std::atan2;
        using ::atan2f;
        using std::ceil;
        using ::ceilf;
        using std::cos;
        using ::cosf;
        using std::cosh;
        using ::coshf;
        using std::exp;
        using ::exp10;
        using std::exp2;
        using ::exp10f;
        using ::exp2f;
        using ::expf;
        using std::fabs;
        using ::fabsf;
        using std::floor;
        using ::floorf;
        using std::fmax;
        using ::fmaxf;
        using std::fmin;
        using ::fminf;
        using std::fmod;
        using ::fmodf;
        using std::frexp;
        using ::frexpf;
        using std::isfinite;
        using std::isinf;
        using std::isnan;
        using std::isnormal;
        using std::ldexp;
        using ::ldexpf;
        using std::log;
        using ::logf;
        using std::log10;
        using ::log10f;
        using std::log2;
        using ::log2f;
        using std::modf;
        using ::modff;
        using std::pow;
        using ::powf;
        using std::round;
        using ::roundf;
        using std::signbit;
        using std::sin;
        using ::sinf;
        using std::sinh;
        using ::sinhf;
        using std::sqrt;
        using ::sqrtf;
        using std::tan;
        using ::tanf;
        using std::tanh;
        using ::tanhf;
        using std::trunc;
        using ::truncf;

        HCC_MATH_LIB_FN
        float acosf(float x) { return __hc_acos(x); }

        HCC_MATH_LIB_FN
        __fp16 acos(__fp16 x) { return __hc_acos_half(x); }

        HCC_MATH_LIB_FN
        float acos(float x) { return fast_math::acosf(x); }

        HCC_MATH_LIB_FN
        float asinf(float x) { return __hc_asin(x); }

        HCC_MATH_LIB_FN
        __fp16 asin(__fp16 x) { return __hc_asin_half(x); }

        HCC_MATH_LIB_FN
        float asin(float x) { return fast_math::asinf(x); }

        HCC_MATH_LIB_FN
        float atanf(float x) { return __hc_atan(x); }

        HCC_MATH_LIB_FN
        __fp16 atan(__fp16 x) { return __hc_atan_half(x); }

        HCC_MATH_LIB_FN
        float atan(float x) { return fast_math::atanf(x); }

        HCC_MATH_LIB_FN
        float atan2f(float y, float x) { return __hc_atan2(y, x); }

        HCC_MATH_LIB_FN
        __fp16 atan2(__fp16 y, __fp16 x) { return __hc_atan2_half(y, x); }

        HCC_MATH_LIB_FN
        float atan2(float y, float x) { return fast_math::atan2f(y, x); }

        HCC_MATH_LIB_FN
        float ceilf(float x) { return __hc_ceil(x); }

        HCC_MATH_LIB_FN
        __fp16 ceil(__fp16 x) { return __hc_ceil_half(x); }

        HCC_MATH_LIB_FN
        float ceil(float x) { return fast_math::ceilf(x); }

        HCC_MATH_LIB_FN
        float cosf(float x) { return __hc_cos_native(x); }

        HCC_MATH_LIB_FN
        __fp16 cos(__fp16 x) { return __hc_cos_native_half(x); }

        HCC_MATH_LIB_FN
        float cos(float x) { return fast_math::cosf(x); }

        HCC_MATH_LIB_FN
        float coshf(float x) { return __hc_cosh(x); }

        HCC_MATH_LIB_FN
        __fp16 cosh(__fp16 x) { return __hc_cosh_half(x); }

        HCC_MATH_LIB_FN
        float cosh(float x) { return fast_math::coshf(x); }

        HCC_MATH_LIB_FN
        float expf(float x) { return __hc_exp2_native(M_LOG2E * x); }

        HCC_MATH_LIB_FN
        __fp16 exp(__fp16 x) { return __hc_exp2_native_half(M_LOG2E * x); }

        HCC_MATH_LIB_FN
        float exp(float x) { return fast_math::expf(x); }

        HCC_MATH_LIB_FN
        float exp2f(float x) { return __hc_exp2_native(x); }

        HCC_MATH_LIB_FN
        __fp16 exp2(__fp16 x) { return __hc_exp2_native_half(x); }

        HCC_MATH_LIB_FN
        float exp2(float x) { return fast_math::exp2f(x); }

        HCC_MATH_LIB_FN
        float fabsf(float x) { return __hc_fabs(x); }

        HCC_MATH_LIB_FN
        __fp16 fabs(__fp16 x) { return __hc_fabs_half(x); }

        HCC_MATH_LIB_FN
        float fabs(float x) { return fast_math::fabsf(x); }

        HCC_MATH_LIB_FN
        float floorf(float x) { return __hc_floor(x); }

        HCC_MATH_LIB_FN
        __fp16 floor(__fp16 x) { return __hc_floor_half(x); }

        HCC_MATH_LIB_FN
        float floor(float x) { return fast_math::floorf(x); }

        HCC_MATH_LIB_FN
        float fmaxf(float x, float y) { return __hc_fmax(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fmax(__fp16 x, __fp16 y) { return __hc_fmax_half(x, y); }

        HCC_MATH_LIB_FN
        float fmax(float x, float y) { return fast_math::fmaxf(x, y); }

        HCC_MATH_LIB_FN
        float fminf(float x, float y) { return __hc_fmin(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fmin(__fp16 x, __fp16 y) { return __hc_fmin_half(x, y); }

        HCC_MATH_LIB_FN
        float fmin(float x, float y) { return fast_math::fminf(x, y); }

        HCC_MATH_LIB_FN
        float fmodf(float x, float y) { return __hc_fmod(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fmod(__fp16 x, __fp16 y) { return __hc_fmod_half(x, y); }

        HCC_MATH_LIB_FN
        float fmod(float x, float y) { return fast_math::fmodf(x, y); }

        HCC_MATH_LIB_FN
        float frexpf(float x, int *exp) { return __hc_frexp(x, exp); }

        HCC_MATH_LIB_FN
        __fp16 frexp(__fp16 x, int *exp) { return __hc_frexp_half(x, exp); }

        HCC_MATH_LIB_FN
        float frexp(float x, int *exp) { return fast_math::frexpf(x, exp); }

        HCC_MATH_LIB_FN
        int isfinite(__fp16 x) { return __hc_isfinite_half(x); }

        HCC_MATH_LIB_FN
        int isfinite(float x) { return __hc_isfinite(x); }

        HCC_MATH_LIB_FN
        int isinf(__fp16 x) { return __hc_isinf_half(x); }

        HCC_MATH_LIB_FN
        int isinf(float x) { return __hc_isinf(x); }

        HCC_MATH_LIB_FN
        int isnan(__fp16 x) { return __hc_isnan_half(x); }

        HCC_MATH_LIB_FN
        int isnan(float x) { return __hc_isnan(x); }

        HCC_MATH_LIB_FN
        float ldexpf(float x, int exp) { return __hc_ldexp(x,exp); }

        HCC_MATH_LIB_FN
        __fp16 ldexp(__fp16 x, std::uint16_t exp)
        {
            return __hc_ldexp_half(x, exp);
        }

        HCC_MATH_LIB_FN
        float ldexp(float x, int exp) { return fast_math::ldexpf(x, exp); }

        namespace
        {   // TODO: this is temporary, lifted straight out of irif.h.
            // Namespace is merely for documentation.
            #define M_LOG2_10_F 0x1.a934f0p+1f
            // Value of 1 / log2(10)
            #define M_RLOG2_10_F 0x1.344136p-2f
            // Value of 1 / M_LOG2E_F = 1 / log2(e)
            #define M_RLOG2_E_F 0x1.62e430p-1f
        }

        HCC_MATH_LIB_FN
        float logf(float x) { return __hc_log2_native(x) * M_RLOG2_E_F; }

        HCC_MATH_LIB_FN
        __fp16 log(__fp16 x)
        {
            return __hc_log2_native_half(x) * static_cast<__fp16>(M_RLOG2_E_F);
        }

        HCC_MATH_LIB_FN
        float log(float x) { return fast_math::logf(x); }

        HCC_MATH_LIB_FN
        float log10f(float x) { return __hc_log2_native(x) * M_RLOG2_10_F; }

        HCC_MATH_LIB_FN
        __fp16 log10(__fp16 x)
        {
            return __hc_log2_native_half(x) * static_cast<__fp16>(M_RLOG2_10_F);
        }

        HCC_MATH_LIB_FN
        float log10(float x) { return fast_math::log10f(x); }

        HCC_MATH_LIB_FN
        float log2f(float x) { return __hc_log2_native(x); }

        HCC_MATH_LIB_FN
        __fp16 log2(__fp16 x) { return __hc_log2_native_half(x); }

        HCC_MATH_LIB_FN
        float log2(float x) { return fast_math::log2f(x); }

        HCC_MATH_LIB_FN
        float modff(float x, float *iptr) { return __hc_modf(x, iptr); }

        HCC_MATH_LIB_FN
        __fp16 modf(__fp16 x, __fp16 *iptr) { return __hc_modf_half(x, iptr); }


        HCC_MATH_LIB_FN
        float modf(float x, float *iptr) { return fast_math::modff(x, iptr); }

        HCC_MATH_LIB_FN
        float powf(float x, float y) { return __hc_pow(x, y); }

        HCC_MATH_LIB_FN
        __fp16 pow(__fp16 x, __fp16 y) { return __hc_pow_half(x, y); }

        HCC_MATH_LIB_FN
        float pow(float x, float y) { return fast_math::powf(x, y); }

        HCC_MATH_LIB_FN
        float roundf(float x) { return __hc_round(x); }

        HCC_MATH_LIB_FN
        __fp16 round(__fp16 x) { return __hc_round_half(x); }

        HCC_MATH_LIB_FN
        float round(float x) { return fast_math::roundf(x); }

        HCC_MATH_LIB_FN
        float rsqrtf(float x) { return __hc_rsqrt_native(x); }

        HCC_MATH_LIB_FN
        __fp16 rsqrt(__fp16 x) { return __hc_rsqrt_native_half(x); }

        HCC_MATH_LIB_FN
        float rsqrt(float x) { return fast_math::rsqrtf(x); }

        HCC_MATH_LIB_FN
        int signbitf(float x) { return __hc_signbit(x); }

        HCC_MATH_LIB_FN
        int signbit(__fp16 x) { return __hc_signbit_half(x); }

        HCC_MATH_LIB_FN
        int signbit(float x) { return fast_math::signbitf(x); }

        HCC_MATH_LIB_FN
        float sinf(float x) { return __hc_sin_native(x); }

        HCC_MATH_LIB_FN
        __fp16 sin(__fp16 x) { return __hc_sin_native_half(x); }

        HCC_MATH_LIB_FN
        float sin(float x) { return fast_math::sinf(x); }

        HCC_MATH_LIB_FN
        void sincosf(float x, float *s, float *c) { *s = __hc_sincos(x, c); }

        HCC_MATH_LIB_FN
        void sincos(__fp16 x, __fp16 *s, __fp16 *c)
        {
            *s = __hc_sincos_half(x, c);
        }

        HCC_MATH_LIB_FN
        void sincos(float x, float *s, float *c)
        {
            fast_math::sincosf(x, s, c);
        }

        HCC_MATH_LIB_FN
        float sinhf(float x) { return __hc_sinh(x); }

        HCC_MATH_LIB_FN
        __fp16 sinh(__fp16 x) { return __hc_sinh_half(x); }

        HCC_MATH_LIB_FN
        float sinh(float x) { return fast_math::sinhf(x); }

        HCC_MATH_LIB_FN
        float sqrtf(float x) { return __hc_sqrt_native(x); }

        HCC_MATH_LIB_FN
        __fp16 sqrt(__fp16 x) { return __hc_sqrt_native_half(x); }

        HCC_MATH_LIB_FN
        float sqrt(float x) { return fast_math::sqrtf(x); }

        HCC_MATH_LIB_FN
        float tanf(float x) { return __hc_tan(x); }

        HCC_MATH_LIB_FN
        __fp16 tan(__fp16 x)
        {
            return __hc_sin_native_half(x) *
                __hc_rcp_native_half(__hc_cos_native_half(x));
        }

        HCC_MATH_LIB_FN
        float tan(float x) { return fast_math::tanf(x); }

        HCC_MATH_LIB_FN
        float tanhf(float x) { return __hc_tanh(x); }

        HCC_MATH_LIB_FN
        __fp16 tanh(__fp16 x) { return __hc_tanh_half(x); }

        HCC_MATH_LIB_FN
        float tanh(float x) { return fast_math::tanhf(x); }

        HCC_MATH_LIB_FN
        float truncf(float x) { return __hc_trunc(x); }

        HCC_MATH_LIB_FN
        __fp16 trunc(__fp16 x) { return __hc_trunc_half(x); }

        HCC_MATH_LIB_FN
        float trunc(float x) { return fast_math::truncf(x); }
    } // namespace fast_math

    namespace precise_math
    {
        using std::acos;
        using std::acosh;
        using ::acoshf;
        using ::acosf;
        using std::asin;
        using std::asinh;
        using ::asinhf;
        using ::asinf;
        using std::atan;
        using std::atan2;
        using ::atan2f;
        using std::atanh;
        using ::atanhf;
        using ::atanf;
        using std::cbrt;
        using ::cbrtf;
        using std::ceil;
        using ::ceilf;
        using std::copysign;
        using ::copysignf;
        using std::cos;
        using std::cosh;
        using ::coshf;
        using ::cosf;
        using std::erf;
        using std::erfc;
        using ::erfcf;
        using ::erff;
        using std::exp;
        using ::exp10;
        using ::exp10f;
        using std::exp2;
        using ::exp2f;
        using ::expf;
        using std::expm1;
        using ::expm1f;
        using std::fabs;
        using ::fabsf;
        using std::fdim;
        using ::fdimf;
        using std::floor;
        using ::floorf;
        using std::fma;
        using ::fmaf;
        using std::fmax;
        using ::fmaxf;
        using std::fmin;
        using ::fminf;
        using std::fmod;
        using ::fmodf;
        using std::frexp;
        using ::frexpf;
        using std::hypot;
        using ::hypotf;
        using std::ilogb;
        using ::ilogbf;
        using std::isfinite;
        using std::isinf;
        using std::isnan;
        using std::isnormal;
        using std::ldexp;
        using ::ldexpf;
        using std::log;
        using std::log10;
        using std::log1p;
        using std::log2;
        using std::logb;
        using ::log10f;
        using ::log1pf;
        using ::log2f;
        using ::logbf;
        using ::logf;
        using std::modf;
        using ::modff;
        using std::nearbyint;
        using ::nearbyintf;
        using std::nextafter;
        using ::nextafterf;
        using std::pow;
        using ::powf;
        using std::remainder;
        using ::remainderf;
        using std::remquo;
        using ::remquof;
        using std::round;
        using ::roundf;
        using std::scalbn;
        using ::scalbnf;
        using std::signbit;
        using std::sin;
        using std::sinh;
        using ::sinhf;
        using ::sinf;
        using std::sqrt;
        using ::sqrtf;
        using std::tan;
        using std::tanh;
        using ::tanhf;
        using ::tanf;
        using std::tgamma;
        using ::tgammaf;
        using std::trunc;
        using ::truncf;

        HCC_MATH_LIB_FN
        float acosf(float x) { return __hc_acos(x); }

        HCC_MATH_LIB_FN
        __fp16 acos(__fp16 x) { return __hc_acos_half(x); }

        HCC_MATH_LIB_FN
        float acos(float x) { return precise_math::acosf(x); }

        HCC_MATH_LIB_FN
        double acos(double x) { return __hc_acos_double(x); }

        HCC_MATH_LIB_FN
        float acoshf(float x) { return __hc_acosh(x); }

        HCC_MATH_LIB_FN
        __fp16 acosh(__fp16 x) { return __hc_acosh_half(x); }

        HCC_MATH_LIB_FN
        float acosh(float x) { return precise_math::acoshf(x); }

        HCC_MATH_LIB_FN
        double acosh(double x) { return __hc_acosh_double(x); }

        HCC_MATH_LIB_FN
        float asinf(float x) { return __hc_asin(x); }

        HCC_MATH_LIB_FN
        __fp16 asin(__fp16 x) { return __hc_asin_half(x); }

        HCC_MATH_LIB_FN
        float asin(float x) { return precise_math::asinf(x); }

        HCC_MATH_LIB_FN
        double asin(double x) { return __hc_asin_double(x); }

        HCC_MATH_LIB_FN
        float asinhf(float x) { return __hc_asinh(x); }

        HCC_MATH_LIB_FN
        __fp16 asinh(__fp16 x) { return __hc_asinh_half(x); }

        HCC_MATH_LIB_FN
        float asinh(float x) { return precise_math::asinhf(x); }

        HCC_MATH_LIB_FN
        double asinh(double x) { return __hc_asinh_double(x); }

        HCC_MATH_LIB_FN
        float atanf(float x) { return __hc_atan(x); }

        HCC_MATH_LIB_FN
        __fp16 atan(__fp16 x) { return __hc_atan_half(x); }

        HCC_MATH_LIB_FN
        float atan(float x) { return precise_math::atanf(x); }

        HCC_MATH_LIB_FN
        double atan(double x) { return __hc_atan_double(x); }

        HCC_MATH_LIB_FN
        float atanhf(float x) { return __hc_atanh(x); }

        HCC_MATH_LIB_FN
        __fp16 atanh(__fp16 x) { return __hc_atanh_half(x); }

        HCC_MATH_LIB_FN
        float atanh(float x) { return precise_math::atanhf(x); }

        HCC_MATH_LIB_FN
        double atanh(double x) { return __hc_atanh_double(x); }

        HCC_MATH_LIB_FN
        float atan2f(float y, float x) { return __hc_atan2(y, x); }

        HCC_MATH_LIB_FN
        __fp16 atan2(__fp16 x, __fp16 y) { return __hc_atan2_half(x, y); }

        HCC_MATH_LIB_FN
        float atan2(float y, float x) { return precise_math::atan2f(y, x); }

        HCC_MATH_LIB_FN
        double atan2(double y, double x) { return __hc_atan2_double(y, x); }

        HCC_MATH_LIB_FN
        float cbrtf(float x) { return __hc_cbrt(x); }

        HCC_MATH_LIB_FN
        __fp16 cbrt(__fp16 x) { return __hc_cbrt_half(x); }

        HCC_MATH_LIB_FN
        float cbrt(float x) { return precise_math::cbrtf(x); }

        HCC_MATH_LIB_FN
        double cbrt(double x) { return __hc_cbrt_double(x); }

        HCC_MATH_LIB_FN
        float ceilf(float x) { return __hc_ceil(x); }

        HCC_MATH_LIB_FN
        __fp16 ceil(__fp16 x) { return __hc_ceil_half(x); }

        HCC_MATH_LIB_FN
        float ceil(float x) { return precise_math::ceilf(x); }

        HCC_MATH_LIB_FN
        double ceil(double x) { return __hc_ceil_double(x); }

        HCC_MATH_LIB_FN
        float copysignf(float x, float y) { return __hc_copysign(x, y); }

        HCC_MATH_LIB_FN
        __fp16 copysign(__fp16 x, __fp16 y) { return __hc_copysign_half(x, y); }

        HCC_MATH_LIB_FN
        float copysign(float x, float y)
        {
            return precise_math::copysignf(x, y);
        }

        HCC_MATH_LIB_FN
        double copysign(double x, double y)
        {
            return __hc_copysign_double(x, y);
        }

        HCC_MATH_LIB_FN
        float cosf(float x) { return __hc_cos(x); }

        HCC_MATH_LIB_FN
        __fp16 cos(__fp16 x) { return __hc_cos_half(x); }

        HCC_MATH_LIB_FN
        float cos(float x) { return precise_math::cosf(x); }

        HCC_MATH_LIB_FN
        double cos(double x) { return __hc_cos_double(x); }

        HCC_MATH_LIB_FN
        float coshf(float x) { return __hc_cosh(x); }

        HCC_MATH_LIB_FN
        __fp16 cosh(__fp16 x) { return __hc_cosh_half(x); }

        HCC_MATH_LIB_FN
        float cosh(float x) { return precise_math::coshf(x); }

        HCC_MATH_LIB_FN
        double cosh(double x) { return __hc_cosh_double(x); }

        HCC_MATH_LIB_FN
        float cospif(float x) { return __hc_cospi(x); }

        HCC_MATH_LIB_FN
        __fp16 cospi(__fp16 x) { return __hc_cospi_half(x); }

        HCC_MATH_LIB_FN
        float cospi(float x) { return precise_math::cospif(x); }

        HCC_MATH_LIB_FN
        double cospi(double x) { return __hc_cospi_double(x); }

        HCC_MATH_LIB_FN
        float erff(float x) { return __hc_erf(x); }

        HCC_MATH_LIB_FN
        __fp16 erf(__fp16 x) { return __hc_erf_half(x); }

        HCC_MATH_LIB_FN
        float erf(float x) { return precise_math::erff(x); }

        HCC_MATH_LIB_FN
        double erf(double x) { return __hc_erf_double(x); }

        HCC_MATH_LIB_FN
        float erfcf(float x) { return __hc_erfc(x); }

        HCC_MATH_LIB_FN
        __fp16 erfc(__fp16 x) { return __hc_erfc_half(x); }

        HCC_MATH_LIB_FN
        float erfc(float x) { return precise_math::erfcf(x); }

        HCC_MATH_LIB_FN
        double erfc(double x) { return __hc_erfc_double(x); }

        HCC_MATH_LIB_FN
        float erfcinvf(float x) { return __hc_erfcinv(x); }

        HCC_MATH_LIB_FN
        __fp16 erfcinv(__fp16 x) { return __hc_erfcinv_half(x); }

        HCC_MATH_LIB_FN
        float erfcinv(float x) { return precise_math::erfcinvf(x); }

        HCC_MATH_LIB_FN
        double erfcinv(double x) { return __hc_erfcinv_double(x); }

        HCC_MATH_LIB_FN
        float erfinvf(float x) { return __hc_erfinv(x); }

        HCC_MATH_LIB_FN
        __fp16 erfinv(__fp16 x) { return __hc_erfinv_half(x); }

        HCC_MATH_LIB_FN
        float erfinv(float x) { return precise_math::erfinvf(x); }

        HCC_MATH_LIB_FN
        double erfinv(double x) { return __hc_erfinv_double(x); }

        HCC_MATH_LIB_FN
        float expf(float x) { return __hc_exp(x); }

        HCC_MATH_LIB_FN
        __fp16 exp(__fp16 x) { return __hc_exp_half(x); }

        HCC_MATH_LIB_FN
        float exp(float x) { return precise_math::expf(x); }

        HCC_MATH_LIB_FN
        double exp(double x) { return __hc_exp_double(x); }

        HCC_MATH_LIB_FN
        float exp2f(float x) { return __hc_exp2(x); }

        HCC_MATH_LIB_FN
        __fp16 exp2(__fp16 x) { return __hc_exp2_half(x); }

        HCC_MATH_LIB_FN
        float exp2(float x) { return precise_math::exp2f(x); }

        HCC_MATH_LIB_FN
        double exp2(double x) { return __hc_exp2_double(x); }

        HCC_MATH_LIB_FN
        float exp10f(float x) { return __hc_exp10(x); }

        HCC_MATH_LIB_FN
        __fp16 exp10(__fp16 x) { return __hc_exp10_half(x); }

        HCC_MATH_LIB_FN
        float exp10(float x) { return precise_math::exp10f(x); }

        HCC_MATH_LIB_FN
        double exp10(double x) { return __hc_exp10_double(x); }

        HCC_MATH_LIB_FN
        float expm1f(float x) { return __hc_expm1(x); }

        HCC_MATH_LIB_FN
        __fp16 expm1(__fp16 x) { return __hc_expm1_half(x); }

        HCC_MATH_LIB_FN
        float expm1(float x) { return precise_math::expm1f(x); }

        HCC_MATH_LIB_FN
        double expm1(double x) { return __hc_expm1_double(x); }

        HCC_MATH_LIB_FN
        float fabsf(float x) { return __hc_fabs(x); }

        HCC_MATH_LIB_FN
        __fp16 fabs(__fp16 x) { return __hc_fabs_half(x); }

        HCC_MATH_LIB_FN
        float fabs(float x) { return precise_math::fabsf(x); }

        HCC_MATH_LIB_FN
        double fabs(double x) { return __hc_fabs_double(x); }

        HCC_MATH_LIB_FN
        float fdimf(float x, float y) { return __hc_fdim(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fdim(__fp16 x, __fp16 y) { return __hc_fdim_half(x, y); }

        HCC_MATH_LIB_FN
        float fdim(float x, float y) { return precise_math::fdimf(x, y); }

        HCC_MATH_LIB_FN
        double fdim(double x, double y) { return __hc_fdim_double(x, y); }

        HCC_MATH_LIB_FN
        float floorf(float x) { return __hc_floor(x); }

        HCC_MATH_LIB_FN
        __fp16 floor(__fp16 x) { return __hc_floor_half(x); }

        HCC_MATH_LIB_FN
        float floor(float x) { return precise_math::floorf(x); }

        HCC_MATH_LIB_FN
        double floor(double x) { return __hc_floor_double(x); }

        HCC_MATH_LIB_FN
        float fmaf(float x, float y, float z) { return __hc_fma(x, y, z); }

        HCC_MATH_LIB_FN
        __fp16 fma(__fp16 x, __fp16 y, __fp16 z)
        {
            return __hc_fma_half(x, y, z);
        }

        HCC_MATH_LIB_FN
        float fma(float x, float y, float z)
        {
            return precise_math::fmaf(x, y, z);
        }

        HCC_MATH_LIB_FN
        double fma(double x, double y, double z)
        {
            return __hc_fma_double(x, y, z);
        }

        HCC_MATH_LIB_FN
        float fmaxf(float x, float y) { return __hc_fmax(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fmax(__fp16 x, __fp16 y) { return __hc_fmax_half(x, y); }

        HCC_MATH_LIB_FN
        float fmax(float x, float y) { return precise_math::fmaxf(x, y); }

        HCC_MATH_LIB_FN
        double fmax(double x, double y) { return __hc_fmax_double(x, y); }

        HCC_MATH_LIB_FN
        float fminf(float x, float y) { return __hc_fmin(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fmin(__fp16 x, __fp16 y) { return __hc_fmin_half(x, y); }

        HCC_MATH_LIB_FN
        float fmin(float x, float y) { return precise_math::fminf(x, y); }

        HCC_MATH_LIB_FN
        double fmin(double x, double y) { return __hc_fmin_double(x, y); }

        HCC_MATH_LIB_FN
        float fmodf(float x, float y) { return __hc_fmod(x, y); }

        HCC_MATH_LIB_FN
        __fp16 fmod(__fp16 x, __fp16 y) { return __hc_fmod_half(x, y); }

        HCC_MATH_LIB_FN
        float fmod(float x, float y) { return precise_math::fmodf(x, y); }

        HCC_MATH_LIB_FN
        double fmod(double x, double y) { return __hc_fmod_double(x, y); }

        HCC_MATH_LIB_FN
        int fpclassify(__fp16 x) { return __hc_fpclassify_half(x); }

        HCC_MATH_LIB_FN
        int fpclassify(float x) { return __hc_fpclassify(x); }

        HCC_MATH_LIB_FN
        int fpclassify(double x) { return __hc_fpclassify_double(x); }

        HCC_MATH_LIB_FN
        float frexpf(float x, int *exp) { return __hc_frexp(x, exp); }

        HCC_MATH_LIB_FN
        __fp16 frexp(__fp16 x, int* exp) { return __hc_frexp_half(x, exp); }

        HCC_MATH_LIB_FN
        float frexp(float x, int *exp) { return precise_math::frexpf(x, exp); }

        HCC_MATH_LIB_FN
        double frexp(double x, int *exp) { return __hc_frexp_double(x, exp); }

        HCC_MATH_LIB_FN
        float hypotf(float x, float y) { return __hc_hypot(x, y); }

        HCC_MATH_LIB_FN
        __fp16 hypot(__fp16 x, __fp16 y) { return __hc_hypot_half(x, y); }

        HCC_MATH_LIB_FN
        float hypot(float x, float y) { return precise_math::hypotf(x, y); }

        HCC_MATH_LIB_FN
        double hypot(double x, double y) { return __hc_hypot_double(x, y); }

        HCC_MATH_LIB_FN
        int ilogbf(float x) { return __hc_ilogb(x); }

        HCC_MATH_LIB_FN
        int ilogb(__fp16 x) { return __hc_ilogb_half(x); }

        HCC_MATH_LIB_FN
        int ilogb(float x) { return precise_math::ilogbf(x); }

        HCC_MATH_LIB_FN
        int ilogb(double x) { return __hc_ilogb_double(x); }

        HCC_MATH_LIB_FN
        int isfinite(__fp16 x) { return __hc_isfinite_half(x); }

        HCC_MATH_LIB_FN
        int isfinite(float x) { return __hc_isfinite(x); }

        HCC_MATH_LIB_FN
        int isfinite(double x) { return __hc_isfinite_double(x); }

        HCC_MATH_LIB_FN
        int isinf(__fp16 x) { return __hc_isinf_half(x); }

        HCC_MATH_LIB_FN
        int isinf(float x) { return __hc_isinf(x); }

        HCC_MATH_LIB_FN
        int isinf(double x) { return __hc_isinf_double(x); }

        HCC_MATH_LIB_FN
        int isnan(__fp16 x) { return __hc_isnan_half(x); }

        HCC_MATH_LIB_FN
        int isnan(float x) { return __hc_isnan(x); }

        HCC_MATH_LIB_FN
        int isnan(double x) { return __hc_isnan_double(x); }

        HCC_MATH_LIB_FN
        int isnormal(__fp16 x) { return __hc_isnormal_half(x); }

        HCC_MATH_LIB_FN
        int isnormal(float x) { return __hc_isnormal(x); }

        HCC_MATH_LIB_FN
        int isnormal(double x) { return __hc_isnormal_double(x); }

        HCC_MATH_LIB_FN
        float ldexpf(float x, int exp) { return __hc_ldexp(x, exp); }

        HCC_MATH_LIB_FN
        __fp16 ldexp(__fp16 x, std::int16_t e) { return __hc_ldexp_half(x, e); }

        HCC_MATH_LIB_FN
        float ldexp(float x, int exp) { return precise_math::ldexpf(x, exp); }

        HCC_MATH_LIB_FN
        double ldexp(double x, int exp) { return __hc_ldexp_double(x,exp); }

        HCC_MATH_LIB_FN
        float lgammaf(float x) { return __hc_lgamma(x); }

        HCC_MATH_LIB_FN
        __fp16 lgamma(__fp16 x) { return __hc_lgamma_half(x); }

        HCC_MATH_LIB_FN
        float lgamma(float x) { return precise_math::lgammaf(x); }

        HCC_MATH_LIB_FN
        double lgamma(double x) { return __hc_lgamma_double(x); }

        HCC_MATH_LIB_FN
        float logf(float x) { return __hc_log(x); }

        HCC_MATH_LIB_FN
        __fp16 log(__fp16 x) { return __hc_log_half(x); }

        HCC_MATH_LIB_FN
        float log(float x) { return precise_math::logf(x); }

        HCC_MATH_LIB_FN
        double log(double x) { return __hc_log_double(x); }

        HCC_MATH_LIB_FN
        float log10f(float x) { return __hc_log10(x); }

        HCC_MATH_LIB_FN
        __fp16 log10(__fp16 x) { return __hc_log10_half(x); }

        HCC_MATH_LIB_FN
        float log10(float x) { return precise_math::log10f(x); }

        HCC_MATH_LIB_FN
        double log10(double x) { return __hc_log10_double(x); }

        HCC_MATH_LIB_FN
        float log2f(float x) { return __hc_log2(x); }

        HCC_MATH_LIB_FN
        __fp16 log2(__fp16 x) { return __hc_log2_half(x); }

        HCC_MATH_LIB_FN
        float log2(float x) { return precise_math::log2f(x); }

        HCC_MATH_LIB_FN
        double log2(double x) { return __hc_log2_double(x); }

        HCC_MATH_LIB_FN
        float log1pf(float x) { return __hc_log1p(x); }

        HCC_MATH_LIB_FN
        __fp16 log1p(__fp16 x) { return __hc_log1p_half(x); }

        HCC_MATH_LIB_FN
        float log1p(float x) { return precise_math::log1pf(x); }

        HCC_MATH_LIB_FN
        double log1p(double x) { return __hc_log1p(x); }

        HCC_MATH_LIB_FN
        float logbf(float x) { return __hc_logb(x); }

        HCC_MATH_LIB_FN
        __fp16 logb(__fp16 x) { return __hc_logb_half(x); }

        HCC_MATH_LIB_FN
        float logb(float x) { return precise_math::logbf(x); }

        HCC_MATH_LIB_FN
        double logb(double x) { return __hc_logb_double(x); }

        HCC_MATH_LIB_FN
        float modff(float x, float *iptr) { return __hc_modf(x, iptr); }

        HCC_MATH_LIB_FN
        __fp16 modf(__fp16 x, __fp16* p) { return __hc_modf_half(x, p); }

        HCC_MATH_LIB_FN
        float modf(float x, float* p) { return precise_math::modff(x, p); }

        HCC_MATH_LIB_FN
        double modf(double x, double* p) { return __hc_modf_double(x, p); }

        HCC_MATH_LIB_FN
        __fp16 nanh(int x) { return __hc_nan_half(x); }

        HCC_MATH_LIB_FN
        float nanf(int tagp) { return __hc_nan(tagp); }

        HCC_MATH_LIB_FN
        double nan(int tagp)
        {
            return __hc_nan_double(static_cast<unsigned long>(tagp));
        }

        HCC_MATH_LIB_FN
        float nearbyintf(float x) { return __hc_nearbyint(x); }

        HCC_MATH_LIB_FN
        __fp16 nearbyint(__fp16 x) { return __hc_nearbyint_half(x); }

        HCC_MATH_LIB_FN
        float nearbyint(float x) { return precise_math::nearbyintf(x); }

        HCC_MATH_LIB_FN
        double nearbyint(double x) { return __hc_nearbyint_double(x); }

        HCC_MATH_LIB_FN
        float nextafterf(float x, float y) { return __hc_nextafter(x, y); }

        HCC_MATH_LIB_FN
        __fp16 nextafter(__fp16 x, __fp16 y)
        {
            return __hc_nextafter_half(x, y);
        }

        HCC_MATH_LIB_FN
        float nextafter(float x, float y)
        {
            return precise_math::nextafterf(x, y);
        }

        HCC_MATH_LIB_FN
        double nextafter(double x, double y)
        {
            return __hc_nextafter_double(x, y);
        }

        HCC_MATH_LIB_FN
        float powf(float x, float y) { return __hc_pow(x, y); }

        HCC_MATH_LIB_FN
        __fp16 pow(__fp16 x, __fp16 y) { return __hc_pow_half(x, y); }

        HCC_MATH_LIB_FN
        float pow(float x, float y) { return precise_math::powf(x, y); }

        HCC_MATH_LIB_FN
        double pow(double x, double y) { return __hc_pow_double(x, y); }

        HCC_MATH_LIB_FN
        float rcbrtf(float x) { return __hc_rcbrt(x); }

        HCC_MATH_LIB_FN
        __fp16 rcbrt(__fp16 x) { return __hc_rcbrt_half(x); }

        HCC_MATH_LIB_FN
        float rcbrt(float x) { return precise_math::rcbrtf(x); }

        HCC_MATH_LIB_FN
        double rcbrt(double x) { return __hc_rcbrt_double(x); }

        HCC_MATH_LIB_FN
        float remainderf(float x, float y) { return __hc_remainder(x, y); }

        HCC_MATH_LIB_FN
        __fp16 remainder(__fp16 x, __fp16 y)
        {
            return __hc_remainder_half(x, y);
        }

        HCC_MATH_LIB_FN
        float remainder(float x, float y)
        {
            return precise_math::remainderf(x, y);
        }

        HCC_MATH_LIB_FN
        double remainder(double x, double y)
        {
            return __hc_remainder_double(x, y);
        }

        HCC_MATH_LIB_FN
        float remquof(float x, float y, int *quo)
        {
            return __hc_remquo(x, y, quo);
        }

        HCC_MATH_LIB_FN
        __fp16 remquo(__fp16 x, __fp16 y, int* q)
        {
            return __hc_remquo_half(x, y, q);
        }

        HCC_MATH_LIB_FN
        float remquo(float x, float y, int *quo)
        {
            return precise_math::remquof(x, y, quo);
        }

        HCC_MATH_LIB_FN
        double remquo(double x, double y, int *quo)
        {
            return __hc_remquo_double(x, y, quo);
        }

        HCC_MATH_LIB_FN
        float roundf(float x) { return __hc_round(x); }

        HCC_MATH_LIB_FN
        __fp16 round(__fp16 x) { return __hc_round_half(x); }

        HCC_MATH_LIB_FN
        float round(float x) { return precise_math::roundf(x); }

        HCC_MATH_LIB_FN
        double round(double x) { return __hc_round_double(x); }

        HCC_MATH_LIB_FN
        float rsqrtf(float x) { return __hc_rsqrt(x); }

        HCC_MATH_LIB_FN
        __fp16 rsqrt(__fp16 x) { return __hc_rsqrt_half(x); }

        HCC_MATH_LIB_FN
        float rsqrt(float x) { return precise_math::rsqrtf(x); }

        HCC_MATH_LIB_FN
        double rsqrt(double x) { return __hc_rsqrt_double(x); }

        HCC_MATH_LIB_FN
        float sinpif(float x) { return __hc_sinpi(x); }

        HCC_MATH_LIB_FN
        __fp16 sinpi(__fp16 x) { return __hc_sinpi_half(x); }

        HCC_MATH_LIB_FN
        float sinpi(float x) { return precise_math::sinpif(x); }

        HCC_MATH_LIB_FN
        double sinpi(double x) { return __hc_sinpi_double(x); }

        HCC_MATH_LIB_FN
        float scalbf(float x, float exp) { return __hc_scalb(x, exp); }

        HCC_MATH_LIB_FN
        __fp16 scalb(__fp16 x, __fp16 y) { return __hc_scalb_half(x, y); }

        HCC_MATH_LIB_FN
        float scalb(float x, float exp) { return precise_math::scalbf(x, exp); }

        HCC_MATH_LIB_FN
        double scalb(double x, double exp) { return __hc_scalb_double(x, exp); }

        HCC_MATH_LIB_FN
        float scalbnf(float x, int exp) { return __hc_scalbn(x, exp); }

        HCC_MATH_LIB_FN
        __fp16 scalbn(__fp16 x, int e) { return __hc_scalbn_half(x, e); }

        HCC_MATH_LIB_FN
        float scalbn(float x, int exp) { return precise_math::scalbnf(x, exp); }

        HCC_MATH_LIB_FN
        double scalbn(double x, int exp) { return __hc_scalbn_double(x, exp); }

        HCC_MATH_LIB_FN
        int signbitf(float x) { return __hc_signbit(x); }

        HCC_MATH_LIB_FN
        int signbit(__fp16 x) { return __hc_signbit_half(x); }

        HCC_MATH_LIB_FN
        int signbit(float x) { return precise_math::signbitf(x); }

        HCC_MATH_LIB_FN
        int signbit(double x) { return __hc_signbit_double(x); }

        HCC_MATH_LIB_FN
        float sinf(float x) { return __hc_sin(x); }

        HCC_MATH_LIB_FN
        __fp16 sin(__fp16 x) { return __hc_sin_half(x); }

        HCC_MATH_LIB_FN
        float sin(float x) { return precise_math::sinf(x); }

        HCC_MATH_LIB_FN
        double sin(double x) { return __hc_sin_double(x); }

        HCC_MATH_LIB_FN
        void sincosf(float x, float *s, float *c) { *s = __hc_sincos(x, c); }

        HCC_MATH_LIB_FN
        void sincos(__fp16 x, __fp16* s, __fp16* c)
        {
            *s = __hc_sincos_half(x, c);
        }

        HCC_MATH_LIB_FN
        void sincos(float x, float *s, float *c)
        {
            precise_math::sincosf(x, s, c);
        }

        HCC_MATH_LIB_FN
        void sincos(double x, double *s, double *c)
        {
            *s = __hc_sincos_double(x, c);
        }

        HCC_MATH_LIB_FN
        float sinhf(float x) { return __hc_sinh(x); }

        HCC_MATH_LIB_FN
        __fp16 sinh(__fp16 x) { return __hc_sinh_half(x); }

        HCC_MATH_LIB_FN
        float sinh(float x) { return precise_math::sinhf(x); }

        HCC_MATH_LIB_FN
        double sinh(double x) { return __hc_sinh_double(x); }

        HCC_MATH_LIB_FN
        float sqrtf(float x) { return __hc_sqrt(x); }

        HCC_MATH_LIB_FN
        __fp16 sqrt(__fp16 x) { return __hc_sqrt_half(x); }

        HCC_MATH_LIB_FN
        float sqrt(float x) { return precise_math::sqrtf(x); }

        HCC_MATH_LIB_FN
        double sqrt(double x) { return __hc_sqrt_double(x); }

        HCC_MATH_LIB_FN
        float tgammaf(float x) { return __hc_tgamma(x); }

        HCC_MATH_LIB_FN
        __fp16 tgamma(__fp16 x) { return __hc_tgamma_half(x); }

        HCC_MATH_LIB_FN
        float tgamma(float x) { return precise_math::tgammaf(x); }

        HCC_MATH_LIB_FN
        double tgamma(double x) { return __hc_tgamma_double(x); }

        HCC_MATH_LIB_FN
        float tanf(float x) { return __hc_tan(x); }

        HCC_MATH_LIB_FN
        __fp16 tan(__fp16 x) { return __hc_tan_half(x); }

        HCC_MATH_LIB_FN
        float tan(float x) { return precise_math::tanf(x); }

        HCC_MATH_LIB_FN
        double tan(double x) { return __hc_tan_double(x); }

        HCC_MATH_LIB_FN
        float tanhf(float x) { return __hc_tanh(x); }

        HCC_MATH_LIB_FN
        __fp16 tanh(__fp16 x) { return __hc_tanh_half(x); }

        HCC_MATH_LIB_FN
        float tanh(float x) { return precise_math::tanhf(x); }

        HCC_MATH_LIB_FN
        double tanh(double x) { return __hc_tanh(x); }

        HCC_MATH_LIB_FN
        float tanpif(float x) { return __hc_tanpi(x); }

        HCC_MATH_LIB_FN
        __fp16 tanpi(__fp16 x) { return __hc_tanpi_half(x); }

        HCC_MATH_LIB_FN
        float tanpi(float x) { return precise_math::tanpif(x); }

        HCC_MATH_LIB_FN
        double tanpi(double x) { return __hc_tanpi_double(x); }

        HCC_MATH_LIB_FN
        float truncf(float x) { return __hc_trunc(x); }

        HCC_MATH_LIB_FN
        __fp16 trunc(__fp16 x) { return __hc_trunc_half(x); }

        HCC_MATH_LIB_FN
        float trunc(float x) { return precise_math::truncf(x); }

        HCC_MATH_LIB_FN
        double trunc(double x) { return __hc_trunc_double(x); }
    } // namespace precise_math
} // namespace Kalmar
