//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <stdexcept>

extern "C" _Float16 __ocml_acos_f16(_Float16 x) [[hc]];
extern "C" float __ocml_acos_f32(float x) [[hc]];
extern "C" double __ocml_acos_f64(double x) [[hc]];

extern "C" _Float16 __ocml_acosh_f16(_Float16 x) [[hc]];
extern "C" float __ocml_acosh_f32(float x) [[hc]];
extern "C" double __ocml_acosh_f64(double x) [[hc]];

extern "C" _Float16 __ocml_asin_f16(_Float16 x) [[hc]];
extern "C" float __ocml_asin_f32(float x) [[hc]];
extern "C" double __ocml_asin_f64(double x) [[hc]];

extern "C" _Float16 __ocml_asinh_f16(_Float16 x) [[hc]];
extern "C" float __ocml_asinh_f32(float x) [[hc]];
extern "C" double __ocml_asinh_f64(double x) [[hc]];

extern "C" _Float16 __ocml_atan_f16(_Float16 x) __attribute__((const, hc));
extern "C" float __ocml_atan_f32(float x) __attribute__((const, hc));
extern "C" double __ocml_atan_f64(double x) __attribute__((const, hc));

extern "C" _Float16 __ocml_atanh_f16(_Float16 x) __attribute__((const, hc));
extern "C" float __ocml_atanh_f32(float x) __attribute__((const, hc));
extern "C" double __ocml_atanh_f64(double x) __attribute__((const, hc));

extern "C" _Float16 __ocml_atan2_f16(_Float16 y, _Float16 x) __attribute__((const, hc));
extern "C" float __ocml_atan2_f32(float y, float x) __attribute__((const, hc));
extern "C" double __ocml_atan2_f64(double y, double x) __attribute__((const, hc));

extern "C" _Float16 __ocml_cbrt_f16(_Float16 x) [[hc]];
extern "C" float __ocml_cbrt_f32(float x) [[hc]];
extern "C" double __ocml_cbrt_f64(double x) [[hc]];

extern "C" _Float16 __ocml_ceil_f16(_Float16 x) [[hc]];
extern "C" float __ocml_ceil_f32(float x) [[hc]];
extern "C" double __ocml_ceil_f64(double x) [[hc]];

extern "C" _Float16 __ocml_copysign_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_copysign_f32(float x, float y) [[hc]];
extern "C" double __ocml_copysign_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_cos_f16(_Float16 x) [[hc]];
extern "C" _Float16 __ocml_native_cos_f16(_Float16 x) [[hc]];
extern "C" float __ocml_cos_f32(float x) [[hc]];
extern "C" float __ocml_native_cos_f32(float x) [[hc]];
extern "C" double __ocml_cos_f64(double x) [[hc]];

extern "C" _Float16 __ocml_cosh_f16(_Float16 x) [[hc]];
extern "C" float __ocml_cosh_f32(float x) [[hc]];
extern "C" double __ocml_cosh_f64(double x) [[hc]];

extern "C" _Float16 __ocml_cospi_f16(_Float16 x) [[hc]];
extern "C" float __ocml_cospi_f32(float x) [[hc]];
extern "C" double __ocml_cospi_f64(double x) [[hc]];

extern "C" _Float16 __ocml_erf_f16(_Float16 x) [[hc]];
extern "C" float __ocml_erf_f32(float x) [[hc]];
extern "C" double __ocml_erf_f64(double x) [[hc]];

extern "C" _Float16 __ocml_erfc_f16(_Float16 x) [[hc]];
extern "C" float __ocml_erfc_f32(float x) [[hc]];
extern "C" double __ocml_erfc_f64(double x) [[hc]];

extern "C" _Float16 __ocml_erfcinv_f16(_Float16 x) [[hc]];
extern "C" float __ocml_erfcinv_f32(float x) [[hc]];
extern "C" double __ocml_erfcinv_f64(double x) [[hc]];

extern "C" _Float16 __ocml_erfinv_f16(_Float16 x) [[hc]];
extern "C" float __ocml_erfinv_f32(float x) [[hc]];
extern "C" double __ocml_erfinv_f64(double x) [[hc]];

extern "C" _Float16 __ocml_exp_f16(_Float16 x) [[hc]];
extern "C" float __ocml_exp_f32(float x) [[hc]];
extern "C" double __ocml_exp_f64(double x) [[hc]];

extern "C" _Float16 __ocml_exp10_f16(_Float16 x) [[hc]];
extern "C" float __ocml_exp10_f32(float x) [[hc]];
extern "C" double __ocml_exp10_f64(double x) [[hc]];

extern "C" _Float16 __ocml_native_exp2_f16(_Float16 x) [[hc]];
extern "C" _Float16 __ocml_exp2_f16(_Float16 x) [[hc]];
extern "C" float __ocml_exp2_f32(float x) [[hc]];
extern "C" float __ocml_native_exp2_f32(float x) [[hc]];
extern "C" double __ocml_exp2_f64(double x) [[hc]];

extern "C" _Float16 __ocml_expm1_f16(_Float16 x) [[hc]];
extern "C" float __ocml_expm1_f32(float x) [[hc]];
extern "C" double __ocml_expm1_f64(double x) [[hc]];

extern "C" _Float16 __ocml_fabs_f16(_Float16 x) [[hc]];
extern "C" float __ocml_fabs_f32(float x) [[hc]];
extern "C" double __ocml_fabs_f64(double x) [[hc]];

extern "C" _Float16 __ocml_fdim_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_fdim_f32(float x, float y) [[hc]];
extern "C" double __ocml_fdim_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_floor_f16(_Float16 x) [[hc]];
extern "C" float __ocml_floor_f32(float x) [[hc]];
extern "C" double __ocml_floor_f64(double x) [[hc]];

extern "C" _Float16 __ocml_fma_f16(_Float16 x, _Float16 y, _Float16 z) [[hc]];
extern "C" float __ocml_fma_f32(float x, float y, float z) [[hc]];
extern "C" double __ocml_fma_f64(double x, double y, double z) [[hc]];

extern "C" _Float16 __ocml_fmax_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_fmax_f32(float x, float y) [[hc]];
extern "C" double __ocml_fmax_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_fmin_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_fmin_f32(float x, float y) [[hc]];
extern "C" double __ocml_fmin_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_fmod_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_fmod_f32(float x, float y) [[hc]];
extern "C" double __ocml_fmod_f64(double x, double y) [[hc]];

extern "C" int __ocml_fpclassify_f16(_Float16 x) [[hc]];
extern "C" int __ocml_fpclassify_f32(float x) [[hc]];
extern "C" int __ocml_fpclassify_f64(double x) [[hc]];

extern "C" _Float16 __ocml_frexp_f16(_Float16 x, __attribute__((address_space(5))) int *exp) [[hc]];
extern "C" float __ocml_frexp_f32(float x, __attribute__((address_space(5))) int *exp) [[hc]];
extern "C" double __ocml_frexp_f64(double x, __attribute__((address_space(5))) int *exp) [[hc]];

extern "C" _Float16 __ocml_hypot_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_hypot_f32(float x, float y) [[hc]];
extern "C" double __ocml_hypot_f64(double x, double y) [[hc]];

extern "C" int __ocml_ilogb_f16(_Float16 x) [[hc]];
extern "C" int __ocml_ilogb_f32(float x) [[hc]];
extern "C" int __ocml_ilogb_f64(double x) [[hc]];

extern "C" int __ocml_isfinite_f16(_Float16 x) [[hc]];
extern "C" int __ocml_isfinite_f32(float x) [[hc]];
extern "C" int __ocml_isfinite_f64(double x) [[hc]];

extern "C" int __ocml_isinf_f16(_Float16 x) [[hc]];
extern "C" int __ocml_isinf_f32(float x) [[hc]];
extern "C" int __ocml_isinf_f64(double x) [[hc]];

extern "C" int __ocml_isnan_f16(_Float16 x) [[hc]];
extern "C" int __ocml_isnan_f32(float x) [[hc]];
extern "C" int __ocml_isnan_f64(double x) [[hc]];

extern "C" int __ocml_isnormal_f16(_Float16 x) [[hc]];
extern "C" int __ocml_isnormal_f32(float x) [[hc]];
extern "C" int __ocml_isnormal_f64(double x) [[hc]];

extern "C" _Float16 __ocml_ldexp_f16(_Float16 x, std::int16_t exp) [[hc]];
extern "C" float __ocml_ldexp_f32(float x, int exp) [[hc]];
extern "C" double __ocml_ldexp_f64(double x, int exp) [[hc]];

extern "C" _Float16 __ocml_lgamma_f16(_Float16 x) [[hc]];
extern "C" float __ocml_lgamma_f32(float x) [[hc]];
extern "C" double __ocml_lgamma_f64(double x) [[hc]];

extern "C" _Float16 __ocml_log_f16(_Float16 x) [[hc]];
extern "C" float __ocml_log_f32(float x) [[hc]];
extern "C" double __ocml_log_f64(double x) [[hc]];

extern "C" _Float16 __ocml_log10_f16(_Float16 x) [[hc]];
extern "C" float __ocml_log10_f32(float x) [[hc]];
extern "C" double __ocml_log10_f64(double x) [[hc]];

extern "C" _Float16 __ocml_log2_f16(_Float16 x) [[hc]];
extern "C" _Float16 __ocml_native_log2_f16(_Float16 x) [[hc]];
extern "C" float __ocml_log2_f32(float x) [[hc]];
extern "C" float __ocml_native_log2_f32(float x) [[hc]];
extern "C" double __ocml_log2_f64(double x) [[hc]];

extern "C" _Float16 __ocml_log1p_f16(_Float16 x) [[hc]];
extern "C" float __ocml_log1p_f32(float x) [[hc]];
extern "C" double __ocml_log1p_f64(double x) [[hc]];

extern "C" _Float16 __ocml_logb_f16(_Float16 x) [[hc]];
extern "C" float __ocml_logb_f32(float x) [[hc]];
extern "C" double __ocml_logb_f64(double x) [[hc]];

extern "C" _Float16 __ocml_modf_f16(_Float16 x, __attribute__((address_space(5))) _Float16 *iptr) [[hc]];
extern "C" float __ocml_modf_f32(float x, __attribute__((address_space(5))) float *iptr) [[hc]];
extern "C" double __ocml_modf_f64(double x, __attribute__((address_space(5))) double *iptr) [[hc]];

extern "C" _Float16 __ocml_nan_f16(int tagp) [[hc]];
extern "C" float __ocml_nan_f32(int tagp) [[hc]];
extern "C" double __ocml_nan_f64(unsigned long tagp) [[hc]];

extern "C" _Float16 __ocml_nearbyint_f16(_Float16 x) [[hc]];
extern "C" float __ocml_nearbyint_f32(float x) [[hc]];
extern "C" double __ocml_nearbyint_f64(double x) [[hc]];

extern "C" _Float16 __ocml_nextafter_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_nextafter_f32(float x, float y) [[hc]];
extern "C" double __ocml_nextafter_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_pow_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_pow_f32(float x, float y) [[hc]];
extern "C" double __ocml_pow_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_rcbrt_f16(_Float16 x) [[hc]];
extern "C" float __ocml_rcbrt_f32(float x) [[hc]];
extern "C" double __ocml_rcbrt_f64(double x) [[hc]];

// TODO: rcp is implementation only, it does not have a public interface.
extern "C" _Float16 __hc_rcp_native_f16(_Float16 x) [[hc]];
extern "C" float __hc_rcp_native(float x) [[hc]];

extern "C" _Float16 __ocml_remainder_f16(_Float16 x, _Float16 y) [[hc]];
extern "C" float __ocml_remainder_f32(float x, float y) [[hc]];
extern "C" double __ocml_remainder_f64(double x, double y) [[hc]];

extern "C" _Float16 __ocml_remquo_f16(_Float16 x, _Float16 y, __attribute__((address_space(5))) int *quo) [[hc]];
extern "C" float __ocml_remquo_f32(float x, float y, __attribute__((address_space(5))) int *quo) [[hc]];
extern "C" double __ocml_remquo_f64(double x, double y, __attribute__((address_space(5))) int *quo) [[hc]];

extern "C" _Float16 __ocml_round_f16(_Float16 x) [[hc]];
extern "C" float __ocml_round_f32(float x) [[hc]];
extern "C" double __ocml_round_f64(double x) [[hc]];

extern "C" _Float16 __ocml_rsqrt_f16(_Float16 x) [[hc]];
extern "C" _Float16 __ocml_native_rsqrt_f16(_Float16 x) [[hc]];
extern "C" float __ocml_rsqrt_f32(float x) [[hc]];
extern "C" float __ocml_native_rsqrt_f32(float x) [[hc]];
extern "C" double __ocml_rsqrt_f64(double x) [[hc]];

extern "C" _Float16 __ocml_scalb_f16(_Float16 x, _Float16 exp) [[hc]];
extern "C" float __ocml_scalb_f32(float x, float exp) [[hc]];
extern "C" double __ocml_scalb_f64(double x, double exp) [[hc]];

extern "C" _Float16 __ocml_scalbn_f16(_Float16 x, int exp) [[hc]];
extern "C" float __ocml_scalbn_f32(float x, int exp) [[hc]];
extern "C" double __ocml_scalbn_f64(double x, int exp) [[hc]];

extern "C" _Float16 __ocml_sinpi_f16(_Float16 x) [[hc]];
extern "C" float __ocml_sinpi_f32(float x) [[hc]];
extern "C" double __ocml_sinpi_f64(double x) [[hc]];

extern "C" int __ocml_signbit_f16(_Float16 x) [[hc]];
extern "C" int __ocml_signbit_f32(float x) [[hc]];
extern "C" int __ocml_signbit_f64(double x) [[hc]];

extern "C" _Float16 __ocml_sin_f16(_Float16 x) [[hc]];
extern "C" _Float16 __ocml_native_sin_f16(_Float16 x) [[hc]];
extern "C" float __ocml_sin_f32(float x) [[hc]];
extern "C" float __ocml_native_sin_f32(float x) [[hc]];
extern "C" double __ocml_sin_f64(double x) [[hc]];

extern "C" _Float16 __ocml_sincos_f16(_Float16 x, __attribute__((address_space(5))) _Float16 *c) [[hc]];
extern "C" float __ocml_sincos_f32(float x, __attribute__((address_space(5))) float *c) [[hc]];
extern "C" double __ocml_sincos_f64(double x, __attribute__((address_space(5))) double *c) [[hc]];

extern "C" _Float16 __ocml_sinh_f16(_Float16 x) [[hc]];
extern "C" float __ocml_sinh_f32(float x) [[hc]];
extern "C" double __ocml_sinh_f64(double x) [[hc]];

extern "C" _Float16 __ocml_sqrt_f16(_Float16 x) [[hc]];
extern "C" _Float16 __ocml_native_sqrt_f16(_Float16 x) [[hc]];
extern "C" float __ocml_sqrt_f32(float x) [[hc]];
extern "C" float __ocml_native_sqrt_f32(float x) [[hc]];
extern "C" double __ocml_sqrt_f64(double x) [[hc]];

extern "C" _Float16 __ocml_tgamma_f16(_Float16 x) [[hc]];
extern "C" float __ocml_tgamma_f32(float x) [[hc]];
extern "C" double __ocml_tgamma_f64(double x) [[hc]];

extern "C" _Float16 __ocml_tan_f16(_Float16 x) [[hc]];
extern "C" float __ocml_tan_f32(float x) [[hc]];
extern "C" double __ocml_tan_f64(double x) [[hc]];

extern "C" _Float16 __ocml_tanh_f16(_Float16 x) [[hc]];
extern "C" float __ocml_tanh_f32(float x) [[hc]];
extern "C" double __ocml_tanh_f64(double x) [[hc]];

extern "C" _Float16 __ocml_tanpi_f16(_Float16 x) [[hc]];
extern "C" float __ocml_tanpi_f32(float x) [[hc]];
extern "C" double __ocml_tanpi_f64(double x) [[hc]];

extern "C" _Float16 __ocml_trunc_f16(_Float16 x) [[hc]];
extern "C" float __ocml_trunc_f32(float x) [[hc]];
extern "C" double __ocml_trunc_f64(double x) [[hc]];

#define HCC_MATH_LIB_FN inline __attribute__((hc))
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
        float acosf(float x) { return __ocml_acos_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 acos(_Float16 x) { return __ocml_acos_f16(x); }

        HCC_MATH_LIB_FN
        float acos(float x) { return fast_math::acosf(x); }

        HCC_MATH_LIB_FN
        float asinf(float x) { return __ocml_asin_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 asin(_Float16 x) { return __ocml_asin_f16(x); }

        HCC_MATH_LIB_FN
        float asin(float x) { return fast_math::asinf(x); }

        HCC_MATH_LIB_FN
        float atanf(float x) { return __ocml_atan_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 atan(_Float16 x) { return __ocml_atan_f16(x); }

        HCC_MATH_LIB_FN
        float atan(float x) { return fast_math::atanf(x); }

        HCC_MATH_LIB_FN
        float atan2f(float y, float x) { return __ocml_atan2_f32(y, x); }

        HCC_MATH_LIB_FN
        _Float16 atan2(_Float16 y, _Float16 x) { return __ocml_atan2_f16(y, x); }

        HCC_MATH_LIB_FN
        float atan2(float y, float x) { return fast_math::atan2f(y, x); }

        HCC_MATH_LIB_FN
        float ceilf(float x) { return __ocml_ceil_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 ceil(_Float16 x) { return __ocml_ceil_f16(x); }

        HCC_MATH_LIB_FN
        float ceil(float x) { return fast_math::ceilf(x); }

        HCC_MATH_LIB_FN
        float cosf(float x) { return __ocml_native_cos_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 cos(_Float16 x) { return __ocml_native_cos_f16(x); }

        HCC_MATH_LIB_FN
        float cos(float x) { return fast_math::cosf(x); }

        HCC_MATH_LIB_FN
        float coshf(float x) { return __ocml_cosh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 cosh(_Float16 x) { return __ocml_cosh_f16(x); }

        HCC_MATH_LIB_FN
        float cosh(float x) { return fast_math::coshf(x); }

        HCC_MATH_LIB_FN
        float expf(float x) { return __ocml_native_exp2_f32(M_LOG2E * x); }

        HCC_MATH_LIB_FN
        _Float16 exp(_Float16 x) { return __ocml_native_exp2_f16(M_LOG2E * x); }

        HCC_MATH_LIB_FN
        float exp(float x) { return fast_math::expf(x); }

        HCC_MATH_LIB_FN
        float exp2f(float x) { return __ocml_native_exp2_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 exp2(_Float16 x) { return __ocml_native_exp2_f16(x); }

        HCC_MATH_LIB_FN
        float exp2(float x) { return fast_math::exp2f(x); }

        HCC_MATH_LIB_FN
        float fabsf(float x) { return __ocml_fabs_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 fabs(_Float16 x) { return __ocml_fabs_f16(x); }

        HCC_MATH_LIB_FN
        float fabs(float x) { return fast_math::fabsf(x); }

        HCC_MATH_LIB_FN
        float floorf(float x) { return __ocml_floor_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 floor(_Float16 x) { return __ocml_floor_f16(x); }

        HCC_MATH_LIB_FN
        float floor(float x) { return fast_math::floorf(x); }

        HCC_MATH_LIB_FN
        float fmaxf(float x, float y) { return __ocml_fmax_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fmax(_Float16 x, _Float16 y) { return __ocml_fmax_f16(x, y); }

        HCC_MATH_LIB_FN
        float fmax(float x, float y) { return fast_math::fmaxf(x, y); }

        HCC_MATH_LIB_FN
        float fminf(float x, float y) { return __ocml_fmin_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fmin(_Float16 x, _Float16 y) { return __ocml_fmin_f16(x, y); }

        HCC_MATH_LIB_FN
        float fmin(float x, float y) { return fast_math::fminf(x, y); }

        HCC_MATH_LIB_FN
        float fmodf(float x, float y) { return __ocml_fmod_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fmod(_Float16 x, _Float16 y) { return __ocml_fmod_f16(x, y); }

        HCC_MATH_LIB_FN
        float fmod(float x, float y) { return fast_math::fmodf(x, y); }

        HCC_MATH_LIB_FN
        float frexpf(float x, int *exp) {
        	int e;
        	float ret = __ocml_frexp_f32(x, (__attribute__((address_space(5))) int*) &e);
        	*exp = e; return ret;
        }

        HCC_MATH_LIB_FN
        _Float16 frexp(_Float16 x, int *exp) {
        	int e;
        	_Float16 ret = __ocml_frexp_f16(x, (__attribute__((address_space(5))) int*) &e);
            *exp = e; return ret;
        }

        HCC_MATH_LIB_FN
        float frexp(float x, int *exp) { return fast_math::frexpf(x, exp); }

        HCC_MATH_LIB_FN
        int isfinite(_Float16 x) { return __ocml_isfinite_f16(x); }

        HCC_MATH_LIB_FN
        int isfinite(float x) { return __ocml_isfinite_f32(x); }

        HCC_MATH_LIB_FN
        int isinf(_Float16 x) { return __ocml_isinf_f16(x); }

        HCC_MATH_LIB_FN
        int isinf(float x) { return __ocml_isinf_f32(x); }

        HCC_MATH_LIB_FN
        int isnan(_Float16 x) { return __ocml_isnan_f16(x); }

        HCC_MATH_LIB_FN
        int isnan(float x) { return __ocml_isnan_f32(x); }

        HCC_MATH_LIB_FN
        float ldexpf(float x, int exp) { return __ocml_ldexp_f32(x,exp); }

        HCC_MATH_LIB_FN
        _Float16 ldexp(_Float16 x, std::uint16_t exp)
        {
            return __ocml_ldexp_f16(x, exp);
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
        float logf(float x) { return __ocml_native_log2_f32(x) * M_RLOG2_E_F; }

        HCC_MATH_LIB_FN
        _Float16 log(_Float16 x)
        {
            return __ocml_native_log2_f16(x) * static_cast<_Float16>(M_RLOG2_E_F);
        }

        HCC_MATH_LIB_FN
        float log(float x) { return fast_math::logf(x); }

        HCC_MATH_LIB_FN
        float log10f(float x) { return __ocml_native_log2_f32(x) * M_RLOG2_10_F; }

        HCC_MATH_LIB_FN
        _Float16 log10(_Float16 x)
        {
            return __ocml_native_log2_f16(x) * static_cast<_Float16>(M_RLOG2_10_F);
        }

        HCC_MATH_LIB_FN
        float log10(float x) { return fast_math::log10f(x); }

        HCC_MATH_LIB_FN
        float log2f(float x) { return __ocml_native_log2_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 log2(_Float16 x) { return __ocml_native_log2_f16(x); }

        HCC_MATH_LIB_FN
        float log2(float x) { return fast_math::log2f(x); }

        HCC_MATH_LIB_FN
        float modff(float x, float *iptr) {
        	float i;  float ret = __ocml_modf_f32(x, (__attribute__((address_space(5))) float*)&i);
        	*iptr = i; return ret;
        }

        HCC_MATH_LIB_FN
        _Float16 modf(_Float16 x, _Float16 *iptr) {
        	_Float16 i; _Float16 ret = __ocml_modf_f16(x, (__attribute__((address_space(5))) _Float16*) &i);
        	*iptr = i; return ret;
        }

        HCC_MATH_LIB_FN
        float modf(float x, float *iptr) { return fast_math::modff(x, iptr); }

        HCC_MATH_LIB_FN
        float powf(float x, float y) { return __ocml_pow_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 pow(_Float16 x, _Float16 y) { return __ocml_pow_f16(x, y); }

        HCC_MATH_LIB_FN
        float pow(float x, float y) { return fast_math::powf(x, y); }

        HCC_MATH_LIB_FN
        float roundf(float x) { return __ocml_round_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 round(_Float16 x) { return __ocml_round_f16(x); }

        HCC_MATH_LIB_FN
        float round(float x) { return fast_math::roundf(x); }

        HCC_MATH_LIB_FN
        float rsqrtf(float x) { return __ocml_native_rsqrt_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 rsqrt(_Float16 x) { return __ocml_native_rsqrt_f16(x); }

        HCC_MATH_LIB_FN
        float rsqrt(float x) { return fast_math::rsqrtf(x); }

        HCC_MATH_LIB_FN
        int signbitf(float x) { return __ocml_signbit_f32(x); }

        HCC_MATH_LIB_FN
        int signbit(_Float16 x) { return __ocml_signbit_f16(x); }

        HCC_MATH_LIB_FN
        int signbit(float x) { return fast_math::signbitf(x); }

        HCC_MATH_LIB_FN
        float sinf(float x) { return __ocml_native_sin_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sin(_Float16 x) { return __ocml_native_sin_f16(x); }

        HCC_MATH_LIB_FN
        float sin(float x) { return fast_math::sinf(x); }

        HCC_MATH_LIB_FN
        void sincosf(float x, float *s, float *c) {
        	float lc;
        	*s = __ocml_sincos_f32(x, (__attribute__((address_space(5))) float*)&lc);
        	*c=lc;
        }

        HCC_MATH_LIB_FN
        void sincos(_Float16 x, _Float16 *s, _Float16 *c)
        {
        	_Float16 lc;
            *s = __ocml_sincos_f16(x, (__attribute__((address_space(5))) _Float16*) &lc);
            *c = lc;
        }

        HCC_MATH_LIB_FN
        void sincos(float x, float *s, float *c)
        {
            fast_math::sincosf(x, s, c);
        }

        HCC_MATH_LIB_FN
        float sinhf(float x) { return __ocml_sinh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sinh(_Float16 x) { return __ocml_sinh_f16(x); }

        HCC_MATH_LIB_FN
        float sinh(float x) { return fast_math::sinhf(x); }

        HCC_MATH_LIB_FN
        float sqrtf(float x) { return __ocml_native_sqrt_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sqrt(_Float16 x) { return __ocml_native_sqrt_f16(x); }

        HCC_MATH_LIB_FN
        float sqrt(float x) { return fast_math::sqrtf(x); }

        HCC_MATH_LIB_FN
        float tanf(float x) { return __ocml_tan_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 tan(_Float16 x)
        {
            return __ocml_native_sin_f16(x) *
                __hc_rcp_native_f16(__ocml_native_cos_f16(x));
        }

        HCC_MATH_LIB_FN
        float tan(float x) { return fast_math::tanf(x); }

        HCC_MATH_LIB_FN
        float tanhf(float x) { return __ocml_tanh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 tanh(_Float16 x) { return __ocml_tanh_f16(x); }

        HCC_MATH_LIB_FN
        float tanh(float x) { return fast_math::tanhf(x); }

        HCC_MATH_LIB_FN
        float truncf(float x) { return __ocml_trunc_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 trunc(_Float16 x) { return __ocml_trunc_f16(x); }

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
        float acosf(float x) { return __ocml_acos_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 acos(_Float16 x) { return __ocml_acos_f16(x); }

        HCC_MATH_LIB_FN
        float acos(float x) { return __ocml_acos_f32(x); }

        HCC_MATH_LIB_FN
        double acos(double x) { return __ocml_acos_f64(x); }

        HCC_MATH_LIB_FN
        float acoshf(float x) { return __ocml_acosh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 acosh(_Float16 x) { return __ocml_acosh_f16(x); }

        HCC_MATH_LIB_FN
        float acosh(float x) { return __ocml_acosh_f32(x); }

        HCC_MATH_LIB_FN
        double acosh(double x) { return __ocml_acosh_f64(x); }

        HCC_MATH_LIB_FN
        float asinf(float x) { return __ocml_asin_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 asin(_Float16 x) { return __ocml_asin_f16(x); }

        HCC_MATH_LIB_FN
        float asin(float x) { return __ocml_asin_f32(x); }

        HCC_MATH_LIB_FN
        double asin(double x) { return __ocml_asin_f64(x); }

        HCC_MATH_LIB_FN
        float asinhf(float x) { return __ocml_asinh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 asinh(_Float16 x) { return __ocml_asinh_f16(x); }

        HCC_MATH_LIB_FN
        float asinh(float x) { return __ocml_asinh_f32(x); }

        HCC_MATH_LIB_FN
        double asinh(double x) { return __ocml_asinh_f64(x); }

        HCC_MATH_LIB_FN
        float atanf(float x) { return __ocml_atan_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 atan(_Float16 x) { return __ocml_atan_f16(x); }

        HCC_MATH_LIB_FN
        float atan(float x) { return __ocml_atan_f32(x); }

        HCC_MATH_LIB_FN
        double atan(double x) { return __ocml_atan_f64(x); }

        HCC_MATH_LIB_FN
        float atanhf(float x) { return __ocml_atanh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 atanh(_Float16 x) { return __ocml_atanh_f16(x); }

        HCC_MATH_LIB_FN
        float atanh(float x) { return __ocml_atanh_f32(x); }

        HCC_MATH_LIB_FN
        double atanh(double x) { return __ocml_atanh_f64(x); }

        HCC_MATH_LIB_FN
        float atan2f(float y, float x) { return __ocml_atan2_f32(y, x); }

        HCC_MATH_LIB_FN
        _Float16 atan2(_Float16 x, _Float16 y) { return __ocml_atan2_f16(x, y); }

        HCC_MATH_LIB_FN
        float atan2(float y, float x) { return __ocml_atan2_f32(y, x); }

        HCC_MATH_LIB_FN
        double atan2(double y, double x) { return __ocml_atan2_f64(y, x); }

        HCC_MATH_LIB_FN
        float cbrtf(float x) { return __ocml_cbrt_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 cbrt(_Float16 x) { return __ocml_cbrt_f16(x); }

        HCC_MATH_LIB_FN
        float cbrt(float x) { return __ocml_cbrt_f32(x); }

        HCC_MATH_LIB_FN
        double cbrt(double x) { return __ocml_cbrt_f64(x); }

        HCC_MATH_LIB_FN
        float ceilf(float x) { return __ocml_ceil_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 ceil(_Float16 x) { return __ocml_ceil_f16(x); }

        HCC_MATH_LIB_FN
        float ceil(float x) { return __ocml_ceil_f32(x); }

        HCC_MATH_LIB_FN
        double ceil(double x) { return __ocml_ceil_f64(x); }

        HCC_MATH_LIB_FN
        float copysignf(float x, float y) { return __ocml_copysign_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 copysign(_Float16 x, _Float16 y) { return __ocml_copysign_f16(x, y); }

        HCC_MATH_LIB_FN
        float copysign(float x, float y)
        {
            return __ocml_copysign_f32(x, y);
        }

        HCC_MATH_LIB_FN
        double copysign(double x, double y)
        {
            return __ocml_copysign_f64(x, y);
        }

        HCC_MATH_LIB_FN
        float cosf(float x) { return __ocml_cos_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 cos(_Float16 x) { return __ocml_cos_f16(x); }

        HCC_MATH_LIB_FN
        float cos(float x) { return __ocml_cos_f32(x); }

        HCC_MATH_LIB_FN
        double cos(double x) { return __ocml_cos_f64(x); }

        HCC_MATH_LIB_FN
        float coshf(float x) { return __ocml_cosh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 cosh(_Float16 x) { return __ocml_cosh_f16(x); }

        HCC_MATH_LIB_FN
        float cosh(float x) { return __ocml_cosh_f32(x); }

        HCC_MATH_LIB_FN
        double cosh(double x) { return __ocml_cosh_f64(x); }

        HCC_MATH_LIB_FN
        float cospif(float x) { return __ocml_cospi_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 cospi(_Float16 x) { return __ocml_cospi_f16(x); }

        HCC_MATH_LIB_FN
        float cospi(float x) { return __ocml_cospi_f32(x); }

        HCC_MATH_LIB_FN
        double cospi(double x) { return __ocml_cospi_f64(x); }

        HCC_MATH_LIB_FN
        float erff(float x) { return __ocml_erf_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 erf(_Float16 x) { return __ocml_erf_f16(x); }

        HCC_MATH_LIB_FN
        float erf(float x) { return __ocml_erf_f32(x); }

        HCC_MATH_LIB_FN
        double erf(double x) { return __ocml_erf_f64(x); }

        HCC_MATH_LIB_FN
        float erfcf(float x) { return __ocml_erfc_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 erfc(_Float16 x) { return __ocml_erfc_f16(x); }

        HCC_MATH_LIB_FN
        float erfc(float x) { return __ocml_erfc_f32(x); }

        HCC_MATH_LIB_FN
        double erfc(double x) { return __ocml_erfc_f64(x); }

        HCC_MATH_LIB_FN
        float erfcinvf(float x) { return __ocml_erfcinv_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 erfcinv(_Float16 x) { return __ocml_erfcinv_f16(x); }

        HCC_MATH_LIB_FN
        float erfcinv(float x) { return __ocml_erfcinv_f32(x); }

        HCC_MATH_LIB_FN
        double erfcinv(double x) { return __ocml_erfcinv_f64(x); }

        HCC_MATH_LIB_FN
        float erfinvf(float x) { return __ocml_erfinv_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 erfinv(_Float16 x) { return __ocml_erfinv_f16(x); }

        HCC_MATH_LIB_FN
        float erfinv(float x) { return __ocml_erfinv_f32(x); }

        HCC_MATH_LIB_FN
        double erfinv(double x) { return __ocml_erfinv_f64(x); }

        HCC_MATH_LIB_FN
        float expf(float x) { return __ocml_exp_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 exp(_Float16 x) { return __ocml_exp_f16(x); }

        HCC_MATH_LIB_FN
        float exp(float x) { return __ocml_exp_f32(x); }

        HCC_MATH_LIB_FN
        double exp(double x) { return __ocml_exp_f64(x); }

        HCC_MATH_LIB_FN
        float exp2f(float x) { return __ocml_exp2_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 exp2(_Float16 x) { return __ocml_exp2_f16(x); }

        HCC_MATH_LIB_FN
        float exp2(float x) { return __ocml_exp2_f32(x); }

        HCC_MATH_LIB_FN
        double exp2(double x) { return __ocml_exp2_f64(x); }

        HCC_MATH_LIB_FN
        float exp10f(float x) { return __ocml_exp10_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 exp10(_Float16 x) { return __ocml_exp10_f16(x); }

        HCC_MATH_LIB_FN
        float exp10(float x) { return __ocml_exp10_f32(x); }

        HCC_MATH_LIB_FN
        double exp10(double x) { return __ocml_exp10_f64(x); }

        HCC_MATH_LIB_FN
        float expm1f(float x) { return __ocml_expm1_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 expm1(_Float16 x) { return __ocml_expm1_f16(x); }

        HCC_MATH_LIB_FN
        float expm1(float x) { return __ocml_expm1_f32(x); }

        HCC_MATH_LIB_FN
        double expm1(double x) { return __ocml_expm1_f64(x); }

        HCC_MATH_LIB_FN
        float fabsf(float x) { return __ocml_fabs_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 fabs(_Float16 x) { return __ocml_fabs_f16(x); }

        HCC_MATH_LIB_FN
        float fabs(float x) { return __ocml_fabs_f32(x); }

        HCC_MATH_LIB_FN
        double fabs(double x) { return __ocml_fabs_f64(x); }

        HCC_MATH_LIB_FN
        float fdimf(float x, float y) { return __ocml_fdim_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fdim(_Float16 x, _Float16 y) { return __ocml_fdim_f16(x, y); }

        HCC_MATH_LIB_FN
        float fdim(float x, float y) { return __ocml_fdim_f32(x, y); }

        HCC_MATH_LIB_FN
        double fdim(double x, double y) { return __ocml_fdim_f64(x, y); }

        HCC_MATH_LIB_FN
        float floorf(float x) { return __ocml_floor_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 floor(_Float16 x) { return __ocml_floor_f16(x); }

        HCC_MATH_LIB_FN
        float floor(float x) { return __ocml_floor_f32(x); }

        HCC_MATH_LIB_FN
        double floor(double x) { return __ocml_floor_f64(x); }

        HCC_MATH_LIB_FN
        float fmaf(float x, float y, float z) { return __ocml_fma_f32(x, y, z); }

        HCC_MATH_LIB_FN
        _Float16 fma(_Float16 x, _Float16 y, _Float16 z)
        {
            return __ocml_fma_f16(x, y, z);
        }

        HCC_MATH_LIB_FN
        float fma(float x, float y, float z)
        {
            return __ocml_fma_f32(x, y, z);
        }

        HCC_MATH_LIB_FN
        double fma(double x, double y, double z)
        {
            return __ocml_fma_f64(x, y, z);
        }

        HCC_MATH_LIB_FN
        float fmaxf(float x, float y) { return __ocml_fmax_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fmax(_Float16 x, _Float16 y) { return __ocml_fmax_f16(x, y); }

        HCC_MATH_LIB_FN
        float fmax(float x, float y) { return __ocml_fmax_f32(x, y); }

        HCC_MATH_LIB_FN
        double fmax(double x, double y) { return __ocml_fmax_f64(x, y); }

        HCC_MATH_LIB_FN
        float fminf(float x, float y) { return __ocml_fmin_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fmin(_Float16 x, _Float16 y) { return __ocml_fmin_f16(x, y); }

        HCC_MATH_LIB_FN
        float fmin(float x, float y) { return __ocml_fmin_f32(x, y); }

        HCC_MATH_LIB_FN
        double fmin(double x, double y) { return __ocml_fmin_f64(x, y); }

        HCC_MATH_LIB_FN
        float fmodf(float x, float y) { return __ocml_fmod_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 fmod(_Float16 x, _Float16 y) { return __ocml_fmod_f16(x, y); }

        HCC_MATH_LIB_FN
        float fmod(float x, float y) { return __ocml_fmod_f32(x, y); }

        HCC_MATH_LIB_FN
        double fmod(double x, double y) { return __ocml_fmod_f64(x, y); }

        HCC_MATH_LIB_FN
        int fpclassify(_Float16 x) { return __ocml_fpclassify_f16(x); }

        HCC_MATH_LIB_FN
        int fpclassify(float x) { return __ocml_fpclassify_f32(x); }

        HCC_MATH_LIB_FN
        int fpclassify(double x) { return __ocml_fpclassify_f64(x); }

        HCC_MATH_LIB_FN
        float frexpf(float x, int *exp) {
        	int e; float ret =__ocml_frexp_f32(x, (__attribute__((address_space(5))) int*) &e);
            *exp = e;  return ret;
        }

        HCC_MATH_LIB_FN
        _Float16 frexp(_Float16 x, int* exp) {
        	int e; _Float16 ret = __ocml_frexp_f16(x, (__attribute__((address_space(5))) int*) &e);
        	*exp = e; return ret;
        }

        HCC_MATH_LIB_FN
        float frexp(float x, int *exp) { return precise_math::frexpf(x, exp); }

        HCC_MATH_LIB_FN
        double frexp(double x, int *exp) { return precise_math::frexpf(x, exp); }

        HCC_MATH_LIB_FN
        float hypotf(float x, float y) { return __ocml_hypot_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 hypot(_Float16 x, _Float16 y) { return __ocml_hypot_f16(x, y); }

        HCC_MATH_LIB_FN
        float hypot(float x, float y) { return __ocml_hypot_f32(x, y); }

        HCC_MATH_LIB_FN
        double hypot(double x, double y) { return __ocml_hypot_f64(x, y); }

        HCC_MATH_LIB_FN
        int ilogbf(float x) { return __ocml_ilogb_f32(x); }

        HCC_MATH_LIB_FN
        int ilogb(_Float16 x) { return __ocml_ilogb_f16(x); }

        HCC_MATH_LIB_FN
        int ilogb(float x) { return __ocml_ilogb_f32(x); }

        HCC_MATH_LIB_FN
        int ilogb(double x) { return __ocml_ilogb_f64(x); }

        HCC_MATH_LIB_FN
        int isfinite(_Float16 x) { return __ocml_isfinite_f16(x); }

        HCC_MATH_LIB_FN
        int isfinite(float x) { return __ocml_isfinite_f32(x); }

        HCC_MATH_LIB_FN
        int isfinite(double x) { return __ocml_isfinite_f64(x); }

        HCC_MATH_LIB_FN
        int isinf(_Float16 x) { return __ocml_isinf_f16(x); }

        HCC_MATH_LIB_FN
        int isinf(float x) { return __ocml_isinf_f32(x); }

        HCC_MATH_LIB_FN
        int isinf(double x) { return __ocml_isinf_f64(x); }

        HCC_MATH_LIB_FN
        int isnan(_Float16 x) { return __ocml_isnan_f16(x); }

        HCC_MATH_LIB_FN
        int isnan(float x) { return __ocml_isnan_f32(x); }

        HCC_MATH_LIB_FN
        int isnan(double x) { return __ocml_isnan_f64(x); }

        HCC_MATH_LIB_FN
        int isnormal(_Float16 x) { return __ocml_isnormal_f16(x); }

        HCC_MATH_LIB_FN
        int isnormal(float x) { return __ocml_isnormal_f32(x); }

        HCC_MATH_LIB_FN
        int isnormal(double x) { return __ocml_isnormal_f64(x); }

        HCC_MATH_LIB_FN
        float ldexpf(float x, int exp) { return __ocml_ldexp_f32(x, exp); }

        HCC_MATH_LIB_FN
        _Float16 ldexp(_Float16 x, std::int16_t e) { return __ocml_ldexp_f16(x, e); }

        HCC_MATH_LIB_FN
        float ldexp(float x, int exp) { return __ocml_ldexp_f32(x, exp); }

        HCC_MATH_LIB_FN
        double ldexp(double x, int exp) { return __ocml_ldexp_f64(x,exp); }

        HCC_MATH_LIB_FN
        float lgammaf(float x) { return __ocml_lgamma_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 lgamma(_Float16 x) { return __ocml_lgamma_f16(x); }

        HCC_MATH_LIB_FN
        float lgamma(float x) { return __ocml_lgamma_f32(x); }

        HCC_MATH_LIB_FN
        double lgamma(double x) { return __ocml_lgamma_f64(x); }

        HCC_MATH_LIB_FN
        float logf(float x) { return __ocml_log_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 log(_Float16 x) { return __ocml_log_f16(x); }

        HCC_MATH_LIB_FN
        float log(float x) { return __ocml_log_f32(x); }

        HCC_MATH_LIB_FN
        double log(double x) { return __ocml_log_f64(x); }

        HCC_MATH_LIB_FN
        float log10f(float x) { return __ocml_log10_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 log10(_Float16 x) { return __ocml_log10_f16(x); }

        HCC_MATH_LIB_FN
        float log10(float x) { return __ocml_log10_f32(x); }

        HCC_MATH_LIB_FN
        double log10(double x) { return __ocml_log10_f64(x); }

        HCC_MATH_LIB_FN
        float log2f(float x) { return __ocml_log2_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 log2(_Float16 x) { return __ocml_log2_f16(x); }

        HCC_MATH_LIB_FN
        float log2(float x) { return __ocml_log2_f32(x); }

        HCC_MATH_LIB_FN
        double log2(double x) { return __ocml_log2_f64(x); }

        HCC_MATH_LIB_FN
        float log1pf(float x) { return __ocml_log1p_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 log1p(_Float16 x) { return __ocml_log1p_f16(x); }

        HCC_MATH_LIB_FN
        float log1p(float x) { return __ocml_log1p_f32(x); }

        HCC_MATH_LIB_FN
        double log1p(double x) { return __ocml_log1p_f32(x); }

        HCC_MATH_LIB_FN
        float logbf(float x) { return __ocml_logb_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 logb(_Float16 x) { return __ocml_logb_f16(x); }

        HCC_MATH_LIB_FN
        float logb(float x) { return __ocml_logb_f32(x); }

        HCC_MATH_LIB_FN
        double logb(double x) { return __ocml_logb_f64(x); }

        HCC_MATH_LIB_FN
        float modff(float x, float *iptr) {
        	float i;  float ret = __ocml_modf_f32(x, (__attribute__((address_space(5))) float*)&i);
        	*iptr = i; return ret;
        }

        HCC_MATH_LIB_FN
        _Float16 modf(_Float16 x, _Float16* p) {
        	_Float16 lp; _Float16 ret = __ocml_modf_f16(x, (__attribute__((address_space(5))) _Float16*) &lp);
        	*p = lp; return ret;
        }

        HCC_MATH_LIB_FN
        float modf(float x, float* p) { return precise_math::modff(x, p); }

        HCC_MATH_LIB_FN
        double modf(double x, double* p) {
        	double lp; double ret = __ocml_modf_f64(x, (__attribute__((address_space(5))) double*) &lp);
        	*p = lp; return ret;
        }

        HCC_MATH_LIB_FN
        _Float16 nanh(int x) { return __ocml_nan_f16(x); }

        HCC_MATH_LIB_FN
        float nanf(int tagp) { return __ocml_nan_f32(tagp); }

        HCC_MATH_LIB_FN
        double nan(int tagp)
        {
            return __ocml_nan_f64(static_cast<unsigned long>(tagp));
        }

        HCC_MATH_LIB_FN
        float nearbyintf(float x) { return __ocml_nearbyint_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 nearbyint(_Float16 x) { return __ocml_nearbyint_f16(x); }

        HCC_MATH_LIB_FN
        float nearbyint(float x) { return __ocml_nearbyint_f32(x); }

        HCC_MATH_LIB_FN
        double nearbyint(double x) { return __ocml_nearbyint_f64(x); }

        HCC_MATH_LIB_FN
        float nextafterf(float x, float y) { return __ocml_nextafter_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 nextafter(_Float16 x, _Float16 y)
        {
            return __ocml_nextafter_f16(x, y);
        }

        HCC_MATH_LIB_FN
        float nextafter(float x, float y)
        {
            return __ocml_nextafter_f32(x, y);
        }

        HCC_MATH_LIB_FN
        double nextafter(double x, double y)
        {
            return __ocml_nextafter_f64(x, y);
        }

        HCC_MATH_LIB_FN
        float powf(float x, float y) { return __ocml_pow_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 pow(_Float16 x, _Float16 y) { return __ocml_pow_f16(x, y); }

        HCC_MATH_LIB_FN
        float pow(float x, float y) { return __ocml_pow_f32(x, y); }

        HCC_MATH_LIB_FN
        double pow(double x, double y) { return __ocml_pow_f64(x, y); }

        HCC_MATH_LIB_FN
        float rcbrtf(float x) { return __ocml_rcbrt_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 rcbrt(_Float16 x) { return __ocml_rcbrt_f16(x); }

        HCC_MATH_LIB_FN
        float rcbrt(float x) { return __ocml_rcbrt_f32(x); }

        HCC_MATH_LIB_FN
        double rcbrt(double x) { return __ocml_rcbrt_f64(x); }

        HCC_MATH_LIB_FN
        float remainderf(float x, float y) { return __ocml_remainder_f32(x, y); }

        HCC_MATH_LIB_FN
        _Float16 remainder(_Float16 x, _Float16 y)
        {
            return __ocml_remainder_f16(x, y);
        }

        HCC_MATH_LIB_FN
        float remainder(float x, float y)
        {
            return __ocml_remainder_f32(x, y);
        }

        HCC_MATH_LIB_FN
        double remainder(double x, double y)
        {
            return __ocml_remainder_f64(x, y);
        }

        HCC_MATH_LIB_FN
        float remquof(float x, float y, int *quo)
        {
        	int lq; float ret = __ocml_remquo_f32(x, y, (__attribute__((address_space(5))) int*) &lq);
        	*quo = lq; return ret;
        }

        HCC_MATH_LIB_FN
        _Float16 remquo(_Float16 x, _Float16 y, int* q)
        {
        	int lq; _Float16 ret = __ocml_remquo_f16(x, y, (__attribute__((address_space(5))) int*) &lq);
        	*q = lq; return ret;
        }

        HCC_MATH_LIB_FN
        float remquo(float x, float y, int *quo) { return precise_math::remquof(x, y, quo); }

        HCC_MATH_LIB_FN
        double remquo(double x, double y, int *quo)
        {
        	int lq; double ret = __ocml_remquo_f64(x, y, (__attribute__((address_space(5))) int*) &lq);
        	*quo = lq; return ret;
        }

        HCC_MATH_LIB_FN
        float roundf(float x) { return __ocml_round_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 round(_Float16 x) { return __ocml_round_f16(x); }

        HCC_MATH_LIB_FN
        float round(float x) { return __ocml_round_f32(x); }

        HCC_MATH_LIB_FN
        double round(double x) { return __ocml_round_f64(x); }

        HCC_MATH_LIB_FN
        float rsqrtf(float x) { return __ocml_rsqrt_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 rsqrt(_Float16 x) { return __ocml_rsqrt_f16(x); }

        HCC_MATH_LIB_FN
        float rsqrt(float x) { return __ocml_rsqrt_f32(x); }

        HCC_MATH_LIB_FN
        double rsqrt(double x) { return __ocml_rsqrt_f64(x); }

        HCC_MATH_LIB_FN
        float sinpif(float x) { return __ocml_sinpi_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sinpi(_Float16 x) { return __ocml_sinpi_f16(x); }

        HCC_MATH_LIB_FN
        float sinpi(float x) { return __ocml_sinpi_f32(x); }

        HCC_MATH_LIB_FN
        double sinpi(double x) { return __ocml_sinpi_f64(x); }

        HCC_MATH_LIB_FN
        float scalbf(float x, float exp) { return __ocml_scalb_f32(x, exp); }

        HCC_MATH_LIB_FN
        _Float16 scalb(_Float16 x, _Float16 y) { return __ocml_scalb_f16(x, y); }

        HCC_MATH_LIB_FN
        float scalb(float x, float exp) { return __ocml_scalb_f32(x, exp); }

        HCC_MATH_LIB_FN
        double scalb(double x, double exp) { return __ocml_scalb_f64(x, exp); }

        HCC_MATH_LIB_FN
        float scalbnf(float x, int exp) { return __ocml_scalbn_f32(x, exp); }

        HCC_MATH_LIB_FN
        _Float16 scalbn(_Float16 x, int e) { return __ocml_scalbn_f16(x, e); }

        HCC_MATH_LIB_FN
        float scalbn(float x, int exp) { return __ocml_scalbn_f32(x, exp); }

        HCC_MATH_LIB_FN
        double scalbn(double x, int exp) { return __ocml_scalbn_f64(x, exp); }

        HCC_MATH_LIB_FN
        int signbitf(float x) { return __ocml_signbit_f32(x); }

        HCC_MATH_LIB_FN
        int signbit(_Float16 x) { return __ocml_signbit_f16(x); }

        HCC_MATH_LIB_FN
        int signbit(float x) { return __ocml_signbit_f32(x); }

        HCC_MATH_LIB_FN
        int signbit(double x) { return __ocml_signbit_f64(x); }

        HCC_MATH_LIB_FN
        float sinf(float x) { return __ocml_sin_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sin(_Float16 x) { return __ocml_sin_f16(x); }

        HCC_MATH_LIB_FN
        float sin(float x) { return __ocml_sin_f32(x); }

        HCC_MATH_LIB_FN
        double sin(double x) { return __ocml_sin_f64(x); }

        HCC_MATH_LIB_FN
        void sincosf(float x, float *s, float *c) {
        	float lc; *s = __ocml_sincos_f32(x, (__attribute__((address_space(5))) float*) &lc);
        	*c = lc;
        }

        HCC_MATH_LIB_FN
        void sincos(_Float16 x, _Float16* s, _Float16* c)
        {
            _Float16 lc; *s = __ocml_sincos_f16(x, (__attribute__((address_space(5))) _Float16*) &lc);
            *c = lc;
        }

        HCC_MATH_LIB_FN
        void sincos(float x, float *s, float *c) { precise_math::sincosf(x, s, c); }
       
        HCC_MATH_LIB_FN
        void sincos(double x, double *s, double *c)
        {
        	double lc; *s = __ocml_sincos_f64(x, (__attribute__((address_space(5))) double*) &lc);
        	*c = lc;
        }

        HCC_MATH_LIB_FN
        float sinhf(float x) { return __ocml_sinh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sinh(_Float16 x) { return __ocml_sinh_f16(x); }

        HCC_MATH_LIB_FN
        float sinh(float x) { return __ocml_sinh_f32(x); }

        HCC_MATH_LIB_FN
        double sinh(double x) { return __ocml_sinh_f64(x); }

        HCC_MATH_LIB_FN
        float sqrtf(float x) { return __ocml_sqrt_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 sqrt(_Float16 x) { return __ocml_sqrt_f16(x); }

        HCC_MATH_LIB_FN
        float sqrt(float x) { return __ocml_sqrt_f32(x); }

        HCC_MATH_LIB_FN
        double sqrt(double x) { return __ocml_sqrt_f64(x); }

        HCC_MATH_LIB_FN
        float tgammaf(float x) { return __ocml_tgamma_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 tgamma(_Float16 x) { return __ocml_tgamma_f16(x); }

        HCC_MATH_LIB_FN
        float tgamma(float x) { return __ocml_tgamma_f32(x); }

        HCC_MATH_LIB_FN
        double tgamma(double x) { return __ocml_tgamma_f64(x); }

        HCC_MATH_LIB_FN
        float tanf(float x) { return __ocml_tan_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 tan(_Float16 x) { return __ocml_tan_f16(x); }

        HCC_MATH_LIB_FN
        float tan(float x) { return __ocml_tan_f32(x); }

        HCC_MATH_LIB_FN
        double tan(double x) { return __ocml_tan_f64(x); }

        HCC_MATH_LIB_FN
        float tanhf(float x) { return __ocml_tanh_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 tanh(_Float16 x) { return __ocml_tanh_f16(x); }

        HCC_MATH_LIB_FN
        float tanh(float x) { return __ocml_tanh_f32(x); }

        HCC_MATH_LIB_FN
        double tanh(double x) { return __ocml_tanh_f32(x); }

        HCC_MATH_LIB_FN
        float tanpif(float x) { return __ocml_tanpi_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 tanpi(_Float16 x) { return __ocml_tanpi_f16(x); }

        HCC_MATH_LIB_FN
        float tanpi(float x) { return __ocml_tanpi_f32(x); }

        HCC_MATH_LIB_FN
        double tanpi(double x) { return __ocml_tanpi_f64(x); }

        HCC_MATH_LIB_FN
        float truncf(float x) { return __ocml_trunc_f32(x); }

        HCC_MATH_LIB_FN
        _Float16 trunc(_Float16 x) { return __ocml_trunc_f16(x); }

        HCC_MATH_LIB_FN
        float trunc(float x) { return __ocml_trunc_f32(x); }

        HCC_MATH_LIB_FN
        double trunc(double x) { return __ocml_trunc_f64(x); }
    } // namespace precise_math
} // namespace Kalmar
