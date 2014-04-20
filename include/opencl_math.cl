float opencl_fastmath_cos(float x) {
  return cos(x);
}

float opencl_fastmath_exp(float x) {
  return exp(x);
}

float opencl_fastmath_fabs(float x) {
  return fabs(x);
}

float opencl_fastmath_log(float x) {
  return log(x);
}

float opencl_fastmath_sin(float x) {
  return sin(x);
}

float opencl_fastmath_sqrt(float x) {
  return sqrt(x);
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

