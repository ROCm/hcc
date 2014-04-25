float opencl_asin(float x) {
  return asin(x);
}

float opencl_asinh(float x) {
  return asinh(x);
}

float opencl_acos(float x) {
  return acos(x);
}

float opencl_acosh(float x) {
  return acosh(x);
}

float opencl_atan(float x) {
  return atan(x);
}

float opencl_atanh(float x) {
  return atanh(x);
}

float opencl_atan2(float x, float y) {
  return atan2(x, y);
}

float opencl_cos(float x) {
  return cos(x);
}

float opencl_cosh(float x) {
  return cosh(x);
}

float opencl_cbrt(float x) {
  return cbrt(x);
}

float opencl_ceil(float x) {
  return ceil(x);
}

float opencl_copysign(float x, float y) {
  return copysign(x, y);
}

float opencl_fdim(float x, float y) {
  return fdim(x, y);
}

float opencl_fma(float x, float y, float z) {
  return fma(x, y, z);
}

float opencl_erf(float x) {
  return erf(x);
}


float opencl_erfc(float x) {
  return erfc(x);
}

float opencl_exp(float x) {
  return exp(x);
}

float opencl_exp10(float x) {
  return exp10(x);
}

float opencl_exp2(float x) {
  return exp2(x);
}

float opencl_expm1(float x) {
  return expm1(x);
}
float opencl_fabs(float x) {
  return fabs(x);
}

float opencl_floor(float x) {
  return floor(x);
}

float opencl_fmax(float x, float y) {
  return fmax(x, y);
}

float opencl_fmin(float x, float y) {
  return fmin(x, y);
}

float opencl_fmod(float x, float y) {
  return fmod(x, y);
}

float opencl_hypot(float x, float y) {
  return hypot(x, y);
}

float opencl_ldexp(float x, int exp) {
  return ldexp(x, exp);
}

float opencl_log(float x) {
  return log(x);
}

float opencl_log2(float x) {
  return log2(x);
}

float opencl_log10(float x) {
  return log10(x);
}

float opencl_log1p(float x) {
  return log1p(x);
}

float opencl_logb(float x) {
  return logb(x);
}


float opencl_nextafter(float x, float y) {
  return nextafter(x, y);
}

int opencl_ilogb(float x) {
  return ilogb(x);
}

float opencl_pow(float x, float y) {
  return pow(x, y);
}

float opencl_remainder(float x, float y) {
  return remainder(x, y);
}

float opencl_round(float x) {
  return round(x);
}

float opencl_sin(float x) {
  return sin(x);
}

float opencl_sinh(float x) {
  return sinh(x);
}

float opencl_sinpi(float x) {
  return sinpi(x);
}

float opencl_sqrt(float x) {
  return sqrt(x);
}

float opencl_tan(float x) {
  return tan(x);
}

float opencl_tanh(float x) {
  return tanh(x);
}

float opencl_tgamma(float x) {
  return tgamma(x);
}
float opencl_trunc(float x) {
  return trunc(x);
}

int opencl_min(int x, int y) {
  return min(x, y);
}

float opencl_max(float x, float y) {
  return max(x, y);
}

int opencl_isnan(float x) {
  return isnan(x);
}

unsigned atomic_add_global(volatile __global unsigned *x, unsigned y) {
  return atomic_add(x, y);
}
unsigned atomic_add_local(volatile __local unsigned *x, unsigned y) {
  return atomic_add(x, y);
}
unsigned atomic_max_global(volatile __global unsigned *x, unsigned y) {
  return atomic_max(x, y);
}
unsigned atomic_max_local(volatile __local unsigned *x, unsigned y) {
  return atomic_max(x, y);
}
unsigned atomic_inc_global(volatile __global unsigned *x) {
  return atomic_inc(x);
}
unsigned atomic_inc_local(volatile __local unsigned *x) {
  return atomic_inc(x);
}

