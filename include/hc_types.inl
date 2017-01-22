
#pragma once

#ifndef __CPU_GPU__

#if __HCC_AMP__
#define __CPU_GPU__   restrict(cpu,amp)
#else
#define __CPU_GPU__   [[cpu,hc]]
#endif

#endif

//namespace Kalmar {


template <bool isSigned> class __amp_norm_template;

typedef __amp_norm_template<true>  __amp_norm;
typedef __amp_norm_template<false> __amp_unorm;

//#define AMP_UNORM_MIN((__amp_unorm) 1.0f)

template <bool isSigned>
class __amp_norm_template {

public:

  typedef __amp_norm_template<isSigned> norm_type;

  explicit __amp_norm_template(float v) __CPU_GPU__ {
    data = set(v);
  }
  explicit __amp_norm_template(unsigned int v) __CPU_GPU__ {
    data = set((float)v);
  }
  explicit __amp_norm_template(int v) __CPU_GPU__ {
    data = set((float)v);
  }
  explicit __amp_norm_template(double v) __CPU_GPU__ {
    data = set((float)v);
  }
  __amp_norm_template(const norm_type& other) __CPU_GPU__ {
    data = other.data;
  }
  explicit __amp_norm_template(const __amp_norm_template<!isSigned>& other) __CPU_GPU__ {
    data = set(other.data);
  }

  void set(float f) {
    data = clamp(other.data);
  }

  norm_type& operator=(const norm_type& other) __CPU_GPU__ {
    data = other.data;
    return *this;
  }

  operator float() const __CPU__GPU { return data; }

  norm_type& operator+=(const norm_type& other) __CPU_GPU__ {  
    data = set(data + other.data);
    return *this;
  }

  operator

  constexpr static float min = isSigned?norm_min:unorm_min;
  constexpr static float max = isSigned?norm_max:unorm_max;

private:
  float data;
  
  static constexpr float unorm_min  = 0.0f;
  static constexpr float unorm_max  = 1.0f;

  static constexpr float norm_min  = -1.0f;
  static constexpr float norm_max  = 1.0f;

  float clamp(float v) {
    return v>max?max:(v<min?min:v);
  }
};



//} // namespace Kalmar
