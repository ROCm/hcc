
/**
 * work-item related functions
 */


ulong hc_get_grid_size(uint n) {
  return (long)get_global_size(n);
}

ulong hc_get_workitem_absolute_id(uint n) {
  return (long)get_global_id(n);
}

ulong hc_get_workitem_id(uint n) {
  return (long)get_local_id(n);
}


ulong hc_get_group_size(uint n) {
  return (long)get_local_size(n);
}


ulong hc_get_num_groups(uint n) {
  return (long)get_num_groups(n);
}

ulong hc_get_group_id(uint n) {
  return (long)get_group_id(n);
}

void hc_barrier(uint n) {
  return barrier((int)n);
}

ulong amp_get_global_size(uint n) {
  return (long)get_global_size(n);
}

ulong amp_get_global_id(uint n) {
  return (long)get_global_id(n);
}


ulong amp_get_local_size(uint n) {
  return (long)get_local_size(n);
}

ulong amp_get_local_id(uint n) {
  return (long)get_local_id(n);
}


ulong amp_get_num_groups(uint n) {
  return (long)get_num_groups(n);
}

ulong amp_get_group_id(uint n) {
  return (long)get_group_id(n);
}

void amp_barrier(uint n) {
  return barrier((int)n);
}


/**
 * math functions
 */
float opencl_acos(float x) {
  return acos(x);
}

double opencl_acos_double(double x) {
  return acos(x);
}

float opencl_acosh(float x) {
  return acosh(x);
}

double opencl_acosh_double(double x) {
  return acosh(x);
}

float opencl_asin(float x) {
  return asin(x);
}

double opencl_asin_double(double x) {
  return asin(x);
}

float opencl_asinh(float x) {
  return asinh(x);
}

double opencl_asinh_double(double x) {
  return asinh(x);
}

float opencl_atan(float x) {
  return atan(x);
}

double opencl_atan_double(double x) {
  return atan(x);
}

float opencl_atanh(float x) {
  return atanh(x);
}

double opencl_atanh_double(double x) {
  return atanh(x);
}

float opencl_atan2(float x, float y) {
  return atan2(x, y);
}

double opencl_atan2_double(double x, double y) {
  return atan2(x, y);
}

float opencl_cbrt(float x) {
  return cbrt(x);
}

double opencl_cbrt_double(double x) {
  return cbrt(x);
}

float opencl_ceil(float x) {
  return ceil(x);
}

double opencl_ceil_double(double x) {
  return ceil(x);
}

float opencl_copysign(float x, float y) {
  return copysign(x, y);
}

double opencl_copysign_double(double x, double y) {
  return copysign(x, y);
}

float opencl_cos(float x) {
  return cos(x);
}

double opencl_cos_double(double x) {
  return cos(x);
}

float opencl_cosh(float x) {
  return cosh(x);
}

double opencl_cosh_double(double x) {
  return cosh(x);
}

float opencl_cospi(float x) {
  return cospi(x);
}

double opencl_cospi_double(double x) {
  return cospi(x);
}

float opencl_erf(float x) {
  return erf(x);
}

double opencl_erf_double(double x) {
  return erf(x);
}

float opencl_erfc(float x) {
  return erfc(x);
}

double opencl_erfc_double(double x) {
  return erfc(x);
}

/* FIXME: missing erfinv */

/* FIXME: missing erfcinv */

float opencl_exp(float x) {
  return exp(x);
}

double opencl_exp_double(double x) {
  return exp(x);
}

float opencl_exp2(float x) {
  return exp2(x);
}

double opencl_exp2_double(double x) {
  return exp2(x);
}

float opencl_exp10(float x) {
  return exp10(x);
}

double opencl_exp10_double(double x) {
  return exp10(x);
}

float opencl_expm1(float x) {
  return expm1(x);
}

double opencl_expm1_double(double x) {
  return expm1(x);
}

float opencl_fabs(float x) {
  return fabs(x);
}

double opencl_fabs_double(double x) {
  return fabs(x);
}

float opencl_fdim(float x, float y) {
  return fdim(x, y);
}

double opencl_fdim_double(double x, double y) {
  return fdim(x, y);
}

float opencl_floor(float x) {
  return floor(x);
}

double opencl_floor_double(double x) {
  return floor(x);
}

float opencl_fma(float x, float y, float z) {
  return fma(x, y, z);
}

double opencl_fma_double(double x, double y, double z) {
  return fma(x, y, z);
}

float opencl_fmax(float x, float y) {
  return fmax(x, y);
}

double opencl_fmax_double(double x, double y) {
  return fmax(x, y);
}

float opencl_fmin(float x, float y) {
  return fmin(x, y);
}

double opencl_fmin_double(double x, double y) {
  return fmin(x, y);
}

float opencl_fmod(float x, float y) {
  return fmod(x, y);
}

double opencl_fmod_double(double x, double y) {
  return fmod(x, y);
}

/* FIXME: missing fpclassify */

float opencl_frexpf_global(float x, __global int *exp) {
  return frexp(x, exp);
}

float opencl_frexpf_local(float x, __local int *exp) {
  return frexp(x, exp);
}

float opencl_frexpf(float x, int *exp) {
  return frexp(x, exp);
}

double opencl_frexp_global(double x, __global int *exp) {
  return frexp(x, exp);
}

double opencl_frexp_local(double x, __local int *exp) {
  return frexp(x, exp);
}

double opencl_frexp(double x, int *exp) {
  return frexp(x, exp);
}

float opencl_hypot(float x, float y) {
  return hypot(x, y);
}

double opencl_hypot_double(double x, double y) {
  return hypot(x, y);
}

int opencl_ilogb(float x) {
  return ilogb(x);
}

int opencl_ilogb_double(double x) {
  return ilogb(x);
}

int opencl_isfinite(float x) {
  return isfinite(x);
}

int opencl_isfinite_double(double x) {
  return isfinite(x);
}

int opencl_isinf(float x) {
  return isinf(x);
}

int opencl_isinf_double(double x) {
  return isinf(x);
}

int opencl_isnan(float x) {
  return isnan(x);
}

int opencl_isnan_double(double x) {
  return isnan(x);
}

int opencl_isnormal(float x) {
  return isnormal(x);
}

double opencl_isnormal_double(double x) {
  return isnormal(x);
}

float opencl_lgammaf_global(float x, __global int *exp) {
  return lgamma_r(x, exp);
}

float opencl_lgammaf_local(float x, __local int *exp) {
  return lgamma_r(x, exp);
}

float opencl_lgammaf(float x, int *exp) {
  return lgamma_r(x, exp);
}

double opencl_lgamma_global(double x, __global int *exp) {
  return lgamma_r(x, exp);
}

double opencl_lgamma_local(double x, __local int *exp) {
  return lgamma_r(x, exp);
}

double opencl_lgamma(double x, int *exp) {
  return lgamma_r(x, exp);
}

float opencl_log(float x) {
  return log(x);
}

double opencl_log_double(double x) {
  return log(x);
}

float opencl_log10(float x) {
  return log10(x);
}

double opencl_log10_double(double x) {
  return log10(x);
}

float opencl_log2(float x) {
  return log2(x);
}

double opencl_log2_double(double x) {
  return log2(x);
}

float opencl_log1p(float x) {
  return log1p(x);
}

double opencl_log1p_double(double x) {
  return log1p(x);
}

float opencl_logb(float x) {
  return logb(x);
}

double opencl_logb_double(double x) {
  return logb(x);
}

float opencl_modff_global(float x, __global float *iptr) {
  return modf(x, iptr);
}

float opencl_modff_local(float x, __local float *iptr) {
  return modf(x, iptr);
}

float opencl_modff(float x, float *iptr) {
  return modf(x, iptr);
}

double opencl_modf_global(double x, __global double *iptr) {
  return modf(x, iptr);
}

double opencl_modf_local(double x, __local double *iptr) {
  return modf(x, iptr);
}

double opencl_modf(double x, double *iptr) {
  return modf(x, iptr);
}

float opencl_nan(int tagp) {
  return nan((uint)tagp);
}

double opencl_nan_double(int tagp) {
  return nan((ulong)tagp);
}

float opencl_nearbyint(float x) {
  return rint(x);
}

double opencl_nearbyint_double(double x) {
  return rint(x);
}

float opencl_nextafter(float x, float y) {
  return nextafter(x, y);
}

double opencl_nextafter_double(double x, double y) {
  return nextafter(x, y);
}

float opencl_pow(float x, float y) {
  return pow(x, y);
}

double opencl_pow_double(double x, double y) {
  return pow(x, y);
}

float opencl_remainder(float x, float y) {
  return remainder(x, y);
}

double opencl_remainder_double(double x, double y) {
  return remainder(x, y);
}

float opencl_remquof_global(float x, float y, __global int *quo) {
  return remquo(x, y, quo);
}

float opencl_remquof_local(float x, float y, __local int *quo) {
  return remquo(x, y, quo);
}

float opencl_remquof(float x, float y, int *quo) {
  return remquo(x, y, quo);
}

double opencl_remquo_global(double x, double y, __global int *quo) {
  return remquo(x, y, quo);
}

double opencl_remquo_local(double x, double y, __local int *quo) {
  return remquo(x, y, quo);
}

double opencl_remquo(double x, double y, int *quo) {
  return remquo(x, y, quo);
}

float opencl_round(float x) {
  return round(x);
}

double opencl_round_double(double x) {
  return round(x);
}

float opencl_rsqrt(float x) {
  return rsqrt(x);
}

double opencl_rsqrt_double(double x) {
  return rsqrt(x);
}

float opencl_sinpi(float x) {
  return sinpi(x);
}

double opencl_sinpi_double(double x) {
  return sinpi(x);
}

/* used by scalb and scalbn */
float opencl_ldexp(float x, int exp) {
  return ldexp(x, exp);
}

/* used by scalb and scalbn */
double opencl_ldexp_double(double x, int exp) {
  return ldexp(x, exp);
}

int opencl_signbit(float x) {
  return signbit(x);
}

int opencl_signbit_double(double x) {
  return signbit(x);
}

float opencl_sin(float x) {
  return sin(x);
}

double opencl_sin_double(double x) {
  return sin(x);
}

void opencl_sincosf_global(float x, __global float *s, __global float *c) {
  *s = sincos(x, c);
}

void opencl_sincosf_local(float x, __local float *s, __local float *c) {
  *s = sincos(x, c);
}

void opencl_sincosf(float x, float *s, float *c) {
  *s = sincos(x, c);
}

void opencl_sincos_global(double x, __global double *s, __global double *c) {
  *s = sincos(x, c);
}

void opencl_sincos_local(double x, __local double *s, __local double *c) {
  *s = sincos(x, c);
}

void opencl_sincos(double x, double *s, double *c) {
  *s = sincos(x, c);
}

float opencl_sinh(float x) {
  return sinh(x);
}

double opencl_sinh_double(double x) {
  return sinh(x);
}

float opencl_sqrt(float x) {
  return sqrt(x);
}

double opencl_sqrt_double(double x) {
  return sqrt(x);
}

float opencl_tgamma(float x) {
  return tgamma(x);
}

double opencl_tgamma_double(double x) {
  return tgamma(x);
}

float opencl_tan(float x) {
  return tan(x);
}

double opencl_tan_double(double x) {
  return tan(x);
}

float opencl_tanh(float x) {
  return tanh(x);
}

double opencl_tanh_double(double x) {
  return tanh(x);
}

float opencl_tanpi(float x) {
  return tanpi(x);
}

double opencl_tanpi_double(double x) {
  return tanpi(x);
}

float opencl_trunc(float x) {
  return trunc(x);
}

double opencl_trunc_double(double x) {
  return trunc(x);
}


/**
 * atomic functions
 */
unsigned atomic_add_unsigned_global(volatile __global unsigned *x, unsigned y) {
  return atomic_add(x, y);
}
unsigned atomic_add_unsigned_local(volatile __local unsigned *x, unsigned y) {
  return atomic_add(x, y);
}
unsigned atomic_add_unsigned(volatile unsigned *x, unsigned y) {
  unsigned old = *x;
  *x = old + y;
  return old;
}

int atomic_add_int_global(volatile __global int *x, int y) {
  return atomic_add(x, y);
}
int atomic_add_int_local(volatile __local int *x, int y) {
  return atomic_add(x, y);
}
int atomic_add_int(volatile int *x, int y) {
  int old = *x;
  *x = old + y;
  return old;
}

float atomic_add_float_global(volatile global float *x, const float y) {
  union {
    unsigned int i;
    float f;
  } now, old;
  do {
    old.f = *x;
    now.f = old.f + y;
  } while (atomic_cmpxchg((volatile global unsigned int *)x, old.i, now.i) != old.i);
  return now.f;
}
float atomic_add_float_local(volatile local float *x, const float y) {
  union {
    unsigned int i;
    float f;
  } now, old;

  do {
    old.f = *x;
    now.f = old.f + y;
  } while (atomic_cmpxchg((volatile local unsigned int *)x, old.i, now.i) != old.i);
  return now.f;
}
float atomic_add_float(volatile float *x, float y) {
  float old = *x;
  *x = old + y;
  return *x;
}

unsigned atomic_max_unsigned_global(volatile __global unsigned *x, unsigned y) {
  return atomic_max(x, y);
}
unsigned atomic_max_unsigned_local(volatile __local unsigned *x, unsigned y) {
  return atomic_max(x, y);
}
unsigned atomic_max_unsigned(volatile unsigned *x, unsigned y) {
  *x = max(*x, y);
  return *x;
}

int atomic_max_int_global(volatile __global int *x, int y) {
  return atomic_max(x, y);
}
int atomic_max_int_local(volatile __local int *x, int y) {
  return atomic_max(x, y);
}
int atomic_max_int(volatile int *x, int y) {
  *x = max(*x, y);
  return *x;
}

unsigned atomic_inc_unsigned_global(volatile __global unsigned *x) {
  return atomic_inc(x);
}
unsigned atomic_inc_unsigned_local(volatile __local unsigned *x) {
  return atomic_inc(x);
}
unsigned atomic_inc_unsigned(volatile unsigned *x) {
  unsigned old = *x;
  *x = old + 1;
  return old;
}

int atomic_inc_int_global(volatile __global int *x) {
  return atomic_inc(x);
}
int atomic_inc_int_local(volatile __local int *x) {
  return atomic_inc(x);
}
int atomic_inc_int(volatile int *x) {
  int old = *x;
  *x = old + 1;
  return old;
}


/**
 * memory functions
 */
static unsigned char * memcpy(unsigned char *dst,  __global unsigned char *src, unsigned int len) {
  for (int i = 0; i < len; ++i) {
    dst[i] = src[i];
  }
  return dst;
}
