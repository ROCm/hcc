#pragma once
  #include <cmath>
#ifdef __GPU__
  extern "C" float opencl_fastmath_cos(float x);
  extern "C" float opencl_fastmath_exp(float x);
  extern "C" float opencl_fastmath_fabs(float x);
  extern "C" float opencl_fastmath_log(float x);
  extern "C" float opencl_fastmath_sin(float x);
  extern "C" float opencl_fastmath_sqrt(float x);
  extern "C" int opencl_min(int x, int y);
#endif

namespace Concurrency {
namespace fast_math {
  float cos(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_cos(x);
    #else
      return ::cosf(x);
    #endif
  };

  float exp(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_exp(x);
    #else
      return ::expf(x);
    #endif
  };

  float expf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_exp(x);
    #else
      return ::expf(x);
    #endif
  };

  float fabs(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_fabs(x);
    #else
      return ::fabsf(x);
    #endif
  };

  float fabsf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_fabs(x);
    #else
      return ::fabsf(x);
    #endif
  };

  float logf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_log(x);
    #else
      return ::logf(x);
    #endif
  };

  float sin(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_sin(x);
    #else
      return ::sinf(x);
    #endif
  };

  float sqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_sqrt(x);
    #else
      return ::sqrtf(x);
    #endif
  };

  float sqrtf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_fastmath_sqrt(x);
    #else
      return ::sqrtf(x);
    #endif
  };
} // namesapce fast_math

  int min(int x, int y) restrict(amp, cpu) {
    #ifdef __GPU__
      return opencl_min(x, y);
    #else
      using std::min;
      return min(x, y);
    #endif
  }
} // namespace Concurrency
