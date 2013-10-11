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

unsigned atomic_add_global(volatile __global unsigned *x, unsigned y) {
  return atomic_add(x, y);
}
unsigned atomic_add_local(volatile __local unsigned *x, unsigned y) {
  return atomic_add(x, y);
}
