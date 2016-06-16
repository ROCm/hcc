
#pragma once


#define DECLARE_VECTOR_TYPE_INTERNAL(VECTOR_TYPE,SCALAR_TYPE,NUM_ELEMENT) typedef SCALAR_TYPE VECTOR_TYPE __attribute__((ext_vector_type(NUM_ELEMENT)))

DECLARE_VECTOR_TYPE_INTERNAL(__uchar2,  unsigned char,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__uchar3,  unsigned char,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__uchar4,  unsigned char,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__uchar8,  unsigned char,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__uchar16, unsigned char, 16);

DECLARE_VECTOR_TYPE_INTERNAL(__char2,  char,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__char3,  char,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__char4,  char,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__char8,  char,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__char16, char, 16);


DECLARE_VECTOR_TYPE_INTERNAL(__ushort2,  unsigned short,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__ushort3,  unsigned short,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__ushort4,  unsigned short,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__ushort8,  unsigned short,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__ushort16, unsigned short, 16);

DECLARE_VECTOR_TYPE_INTERNAL(__short2,  short,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__short3,  short,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__short4,  short,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__short8,  short,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__short16, short, 16);


DECLARE_VECTOR_TYPE_INTERNAL(__uint2,  unsigned int,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__uint3,  unsigned int,  3);
DECLARE_VECTOR_TYPE_INTERNAL(__uint4,  unsigned int,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__uint8,  unsigned int,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__uint16, unsigned int, 16);

DECLARE_VECTOR_TYPE_INTERNAL(__int2,  int,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__int3,  int,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__int4,  int,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__int8,  int,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__int16, int, 16);


DECLARE_VECTOR_TYPE_INTERNAL(__ulonglong2,  unsigned long long,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__ulonglong3,  unsigned long long,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__ulonglong4,  unsigned long long,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__ulonglong8,  unsigned long long,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__ulonglong16, unsigned long long, 16);

DECLARE_VECTOR_TYPE_INTERNAL(__longlong2,  long long,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__longlong3,  long long,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__longlong4,  long long,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__longlong8,  long long,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__longlong16, long long, 16);


DECLARE_VECTOR_TYPE_INTERNAL(__float2,  float,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__float3,  float,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__float4,  float,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__float8,  float,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__float16, float, 16);


DECLARE_VECTOR_TYPE_INTERNAL(__double2,  double,  2);
DECLARE_VECTOR_TYPE_INTERNAL(__double3,  double,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__double4,  double,  4);
DECLARE_VECTOR_TYPE_INTERNAL(__double8,  double,  8);
DECLARE_VECTOR_TYPE_INTERNAL(__double16, double, 16);


#ifndef __CPU_GPU__

#if __HCC_AMP__
#define __CPU_GPU__   restrict(cpu,amp)
#else
#define __CPU_GPU__   [[cpu,hc]]
#endif

#endif


template <typename SCALAR_TYPE, typename VECTOR_TYPE>
class __vector_2 {

public:
  typedef SCALAR_TYPE value_type;
  typedef __vector_2<SCALAR_TYPE,VECTOR_TYPE> __scalartype_N;

  static const int size = 2;

  __vector_2() __CPU_GPU__ { data = { static_cast<SCALAR_TYPE>(0), static_cast<SCALAR_TYPE>(0)}; }
 
  // component-wise constructor
  __vector_2(SCALAR_TYPE v1, SCALAR_TYPE v2) __CPU_GPU__ { data = { v1, v2 }; }

  
  //__vector_2(SCALAR_TYPE value) __CPU_GPU__ { data = { static_cast<SCALAR_TYPE>(value), static_cast<SCALAR_TYPE>(value)}; }

  __vector_2(VECTOR_TYPE value) __CPU_GPU__ : data(value) {  }


  // conversion constructor from other short vector types
  template <typename ST, typename VT>
  explicit __vector_2(const  __vector_2<ST,VT>& other)  __CPU_GPU__ { data = { static_cast<SCALAR_TYPE>(other.get_x()),
                                                                               static_cast<SCALAR_TYPE>(other.get_y()) }; }
 
  SCALAR_TYPE get_x() const __CPU_GPU__ { return data.x; }
  SCALAR_TYPE get_y() const __CPU_GPU__ { return data.y; }
  VECTOR_TYPE get_vector() const __CPU_GPU__ { return data; }

  void set_x(SCALAR_TYPE v) __CPU_GPU__ { data.x = v; }
  void set_y(SCALAR_TYPE v)  __CPU_GPU__ { data.y = v; }
  void set_vector(VECTOR_TYPE v)  __CPU_GPU__ { data = v; }

  __scalartype_N& operator=(const __scalartype_N& rhs) __CPU_GPU__ { 
    data = rhs.data;
    return *this;
  }

  __scalartype_N& operator++() __CPU_GPU__ { data++; }
  __scalartype_N operator++(int) __CPU_GPU__ { 
    __scalartype_N r(*this);
    operator++();
    return r;
  }
  __scalartype_N& operator--() __CPU_GPU__ { data--; }
  __scalartype_N operator--(int) __CPU_GPU__ { 
    __scalartype_N r(*this);
    operator--();
    return r;
  }

  __scalartype_N  operator+(const __scalartype_N& rhs) __CPU_GPU__ {
    __scalartype_N r;   
    r.data = this->data+rhs.data;
    return r;
  }
  __scalartype_N& operator+=(const __scalartype_N& rhs) __CPU_GPU__ { 
    data += rhs.data;
    return *this;
  }

  __scalartype_N& operator-=(const __scalartype_N& rhs) __CPU_GPU__ { 
    data -= rhs.data;
    return *this;
  }
 
  __scalartype_N& operator*=(const __scalartype_N& rhs) __CPU_GPU__ { 
    data *= rhs.data;
    return *this;
  }
 
  __scalartype_N& operator/=(const __scalartype_N& rhs) __CPU_GPU__ { 
    data /= rhs.data;
    return *this;
  }



  // operator- template enabled only for short vector of signed integral types or floats 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_signed<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator-() __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = -this->data;
    return r;
  }

  // template to detect applying operator- to short vector of unsigned integral types and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_signed<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator-() __CPU_GPU__ { 
    static_assert(std::is_signed<SCALAR_TYPE>::value, "operator- can only support short vector of signed integral or floating-point types.");
    return __int_scalartype_N();
  }


  // operator~ template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator~() __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = ~this->data;
    return r;
  }

  // template to detect applying operator% to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator~() __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator~ can only support short vector of integral types.");
    return __int_scalartype_N();
  }



  // operator% template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator%(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = data%lhs.data;
    return r;
  }

  // template to detect applying operator% to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator%(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator% can only support short vector of integral types.");
    return __int_scalartype_N();
  }

  template<typename __int_scalartype_N = __scalartype_N>
  __int_scalartype_N& operator%=(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this%lhs;
    return *this;
  }



  // operator^ template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator^(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = data^lhs.data;
    return r;
  }

  // template to detect applying operator^ to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator^(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator^ can only support integral short vector.");
    return __int_scalartype_N();
  }

  template<typename __int_scalartype_N = __scalartype_N>
  __int_scalartype_N& operator^=(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this^lhs;
    return *this;
  }



  // operator| template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator|(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = data|lhs.data;
    return r;
  }

  // template to detect applying operator| to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator|(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator| can only support integral short vector.");
    return __int_scalartype_N();
  }

  template<typename __int_scalartype_N = __scalartype_N>
  __int_scalartype_N& operator|=(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this|lhs;
    return *this;
  }



  // operator& template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator&(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = data&lhs.data;
    return r;
  }

  // template to detect applying operator& to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator&(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator& can only support integral short vector.");
    return __int_scalartype_N();
  }

  template<typename __int_scalartype_N = __scalartype_N>
  __int_scalartype_N& operator&=(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this&lhs;
    return *this;
  }


  // operator>> template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator>>(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = data>>lhs.data;
    return r;
  }

  // template to detect applying operator>> to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator>>(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator>> can only support integral short vector.");
    return __int_scalartype_N();
  }

  template<typename __int_scalartype_N = __scalartype_N>
  __int_scalartype_N& operator>>=(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this>>lhs;
    return *this;
  }


  // operator<< template enabled only for integral short vector 
  template<typename __int_scalartype_N = __scalartype_N
           ,class = typename std::enable_if<std::is_integral<typename __int_scalartype_N::value_type>::value>::type >
  __int_scalartype_N operator<<(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    __int_scalartype_N r;
    r.data = data<<lhs.data;
    return r;
  }

  // template to detect applying operator<< to float short vector and to generate an error
  template<typename __int_scalartype_N = __scalartype_N>
  typename std::enable_if<!std::is_integral<typename __int_scalartype_N::value_type>::value
                          , __int_scalartype_N >::type
  operator<<(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<SCALAR_TYPE>::value, "operator<< can only support integral short vector.");
    return __int_scalartype_N();
  }

  template<typename __int_scalartype_N = __scalartype_N>
  __int_scalartype_N& operator<<=(const __int_scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this<<lhs;
    return *this;
  }



  // boolean operator
  bool operator==(const __scalartype_N& rhs) __CPU_GPU__ { return (data.x == rhs.data.x 
                                                                   && data.y == rhs.data.y); }
  bool operator!=(const __scalartype_N& rhs) __CPU_GPU__ { return !(*this==rhs); }
  
private:
  VECTOR_TYPE data;

};


template <typename SCALAR_TYPE, typename VECTOR_TYPE>
__vector_2<SCALAR_TYPE,VECTOR_TYPE> operator+(const __vector_2<SCALAR_TYPE,VECTOR_TYPE>& lhs
                                              , const __vector_2<SCALAR_TYPE,VECTOR_TYPE>& rhs) __CPU_GPU__ {
  __vector_2<SCALAR_TYPE,VECTOR_TYPE> r(lhs.get_vector() + rhs.get_vector());
  return r;
}





#define DECLARE_VECTOR_TYPE_CLASS(SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX, CLASS_PREFIX) \
typedef __vector_2<SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX ## 2>   CLASS_PREFIX ## 2;

DECLARE_VECTOR_TYPE_CLASS(unsigned char, __uchar, uchar);
DECLARE_VECTOR_TYPE_CLASS(char, __char, char);
DECLARE_VECTOR_TYPE_CLASS(unsigned short, __ushort, ushort);
DECLARE_VECTOR_TYPE_CLASS(short, __short, short);
DECLARE_VECTOR_TYPE_CLASS(unsigned int, __uint, uint);
DECLARE_VECTOR_TYPE_CLASS(int, __int, int);
DECLARE_VECTOR_TYPE_CLASS(unsigned long long, __ulonglong, ulong);
DECLARE_VECTOR_TYPE_CLASS(long long, __longlong, long);
DECLARE_VECTOR_TYPE_CLASS(float, __float, float);
DECLARE_VECTOR_TYPE_CLASS(double, __double, double);



