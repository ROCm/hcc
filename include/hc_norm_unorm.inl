
#pragma once

#include <type_traits>

#ifndef __CPU_GPU__

#if __HCC_AMP__
#define __CPU_GPU__   restrict(cpu,amp)
#else
#define __CPU_GPU__   [[cpu,hc]]
#endif

#endif

template <bool isSigned> class __amp_norm_template;

typedef __amp_norm_template<true>  __amp_norm;
typedef __amp_norm_template<false> __amp_unorm;

typedef __amp_norm   norm;
typedef __amp_unorm unorm;

template <bool isSigned>
class __amp_norm_template {

public:

  typedef __amp_norm_template<isSigned> norm_type;

  __amp_norm_template() __CPU_GPU__ : data(0.0f) { }

  explicit __amp_norm_template(float v) __CPU_GPU__ {
    set(v);
  }
  explicit __amp_norm_template(unsigned int v) __CPU_GPU__ {
    set((float)v);
  }
  explicit __amp_norm_template(int v) __CPU_GPU__ {
    set((float)v);
  }
  explicit __amp_norm_template(double v) __CPU_GPU__ {
    set((float)v);
  }
  __amp_norm_template(const norm_type& other) __CPU_GPU__ {
    data = other.data;
  }

  explicit __amp_norm_template(const __amp_norm_template<!isSigned>& other) __CPU_GPU__ {
    set((float)other);
  }

  float get() __CPU_GPU__ {
    return data;
  }

  void set(float f) __CPU_GPU__ {
    data = clamp(f);
  }

  norm_type& operator=(const norm_type& other) __CPU_GPU__ {
    data = other.data;
    return *this;
  }

  norm_type& operator=(const float& other) __CPU_GPU__ {
    set(other);
    return *this;
  }

  operator float() const __CPU_GPU__ { return data; }

  norm_type& operator+=(const norm_type& other) __CPU_GPU__ {  
    set(data + other.data);
    return *this;
  }

  norm_type& operator-=(const norm_type& other) __CPU_GPU__ {  
    set(data - other.data);
    return *this;
  }

  norm_type& operator*=(const norm_type& other) __CPU_GPU__ {  
    set(data * other.data);
    return *this;
  }

  norm_type& operator/=(const norm_type& other) __CPU_GPU__ {  
    set(data / other.data);
    return *this;
  }
  
  norm_type& operator++() __CPU_GPU__ {
    set(data + 1.0f);
    return *this;
  }

  norm_type operator++(int) __CPU_GPU__ {
    norm_type r(*this);
    operator++();
    return r;
  }
  
  norm_type& operator--() __CPU_GPU__ {
    set(data - 1.0f);
    return *this;
  }

  norm_type operator--(int) __CPU_GPU__ {
    norm_type r(*this);
    operator--();
    return r;
  }

  template <typename T = norm_type
            , class = typename std::enable_if<T::isSigned,norm_type>::type >
  T operator-() __CPU_GPU__ {
    T r(-data);
    return r;
  }

  static constexpr float min = isSigned?-1.0f:0.0f;
  static constexpr float max = isSigned? 1.0f:1.0f;

private:
  float data;

  float clamp(float v) __CPU_GPU__ {
    return v>max?max:(v<min?min:v);
  }
};

template <bool isSigned>
__amp_norm_template<isSigned> operator+(const __amp_norm_template<isSigned>& lhs
                                        , const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return __amp_norm_template<isSigned>((float)lhs + (float)rhs);
}
 
template <bool isSigned>
__amp_norm_template<isSigned> operator-(const __amp_norm_template<isSigned>& lhs
                                        , const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return __amp_norm_template<isSigned>((float)lhs - (float)rhs);
}

template <bool isSigned>
__amp_norm_template<isSigned> operator*(const __amp_norm_template<isSigned>& lhs
                                        , const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return __amp_norm_template<isSigned>((float)lhs * (float)rhs);
}

template <bool isSigned>
__amp_norm_template<isSigned> operator/(const __amp_norm_template<isSigned>& lhs
                                        , const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return __amp_norm_template<isSigned>((float)lhs / (float)rhs);
}

template <bool isSigned>
bool operator==(const __amp_norm_template<isSigned>& lhs
               ,const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return ((float)lhs == (float)rhs);
}

template <bool isSigned>
bool operator!=(const __amp_norm_template<isSigned>& lhs
               ,const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return ((float)lhs != (float)rhs);
}

template <bool isSigned>
bool operator>(const __amp_norm_template<isSigned>& lhs
               ,const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return ((float)lhs > (float)rhs);
}

template <bool isSigned>
bool operator<(const __amp_norm_template<isSigned>& lhs
               ,const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return ((float)lhs < (float)rhs);
}

template <bool isSigned>
bool operator>=(const __amp_norm_template<isSigned>& lhs
               ,const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return ((float)lhs >= (float)rhs);
}

template <bool isSigned>
bool operator<=(const __amp_norm_template<isSigned>& lhs
               ,const __amp_norm_template<isSigned>& rhs) __CPU_GPU__ {
  return ((float)lhs <= (float)rhs);
}

#define UNORM_MIN  ((unorm)0.0f)
#define UNORM_MAX  ((unorm)1.0f)
#define UNORM_ZERO ((norm)0.0f)
#define NORM_ZERO  ((norm)0.0f)
#define NORM_MIN   ((norm)-1.0f)
#define NORM_MAX   ((norm)1.0f)

