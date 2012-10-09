#pragma once

#ifdef __GPU__
  extern "C" float cos(float x);
  extern "C" float exp(float x);
  extern "C" float fabs(float x);
  extern "C" float log(float x);
  extern "C" float sin(float x);
  extern "C" float sqrt(float x);
#else
  #include <cmath>
#endif

namespace Concurrency {
namespace fast_math {
  float cos(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::cos(x);
    #else
      return ::cosf(x);
    #endif
  };

  float exp(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::exp(x);
    #else
      return ::expf(x);
    #endif
  };

  float expf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::exp(x);
    #else
      return ::expf(x);
    #endif
  };

  float fabs(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::fabs(x);
    #else
      return ::fabsf(x);
    #endif
  };

  float fabsf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::fabs(x);
    #else
      return ::fabsf(x);
    #endif
  };

  float logf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::log(x);
    #else
      return ::logf(x);
    #endif
  };

  float sin(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::sin(x);
    #else
      return ::sinf(x);
    #endif
  };

  float sqrt(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::sqrt(x);
    #else
      return ::sqrtf(x);
    #endif
  };

  float sqrtf(float x) restrict(amp, cpu) {
    #ifdef __GPU__
      return ::sqrt(x);
    #else
      return ::sqrtf(x);
    #endif
  };
} // namesapce fast_math
  using std::min;
} // namespace Concurrency
