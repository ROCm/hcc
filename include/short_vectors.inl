
#pragma once

#include <type_traits>
#include "hc_types.inl"

#ifndef __CPU_GPU__

#if __HCC_AMP__
#define __CPU_GPU__   restrict(cpu,amp)
#else
#define __CPU_GPU__   [[cpu,hc]]
#endif

#endif

template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
class __vector;

#define DECLARE_VECTOR_TYPE_CLASS(SCALAR_TYPE, CLASS_PREFIX) \
typedef __vector<SCALAR_TYPE, 1>    CLASS_PREFIX ## 1; \
typedef __vector<SCALAR_TYPE, 2>    CLASS_PREFIX ## 2; \
typedef __vector<SCALAR_TYPE, 4>    CLASS_PREFIX ## 3; \
typedef __vector<SCALAR_TYPE, 4>    CLASS_PREFIX ## 4; \
typedef __vector<SCALAR_TYPE, 8>    CLASS_PREFIX ## 8; \
typedef __vector<SCALAR_TYPE, 16>   CLASS_PREFIX ## 16; 

DECLARE_VECTOR_TYPE_CLASS(unsigned char, uchar);
DECLARE_VECTOR_TYPE_CLASS(char, char);
DECLARE_VECTOR_TYPE_CLASS(unsigned short, ushort);
DECLARE_VECTOR_TYPE_CLASS(short, short);
DECLARE_VECTOR_TYPE_CLASS(unsigned int, uint);
DECLARE_VECTOR_TYPE_CLASS(int, int);
DECLARE_VECTOR_TYPE_CLASS(unsigned long long, ulong);
DECLARE_VECTOR_TYPE_CLASS(long long, long);
DECLARE_VECTOR_TYPE_CLASS(float, float);
DECLARE_VECTOR_TYPE_CLASS(double, double);

typedef uchar1 uchar_1;
typedef uchar2 uchar_2;
typedef uchar3 uchar_3;
typedef uchar4 uchar_4;
typedef uchar8 uchar_8;
typedef uchar16 uchar_16;

typedef char1 char_1;
typedef char2 char_2;
typedef char3 char_3;
typedef char4 char_4;
typedef char8 char_8;
typedef char16 char_16;

typedef ushort1 ushort_1;
typedef ushort2 ushort_2;
typedef ushort3 ushort_3;
typedef ushort4 ushort_4;
typedef ushort8 ushort_8;
typedef ushort16 ushort_16;

typedef short1 short_1;
typedef short2 short_2;
typedef short3 short_3;
typedef short4 short_4;
typedef short8 short_8;
typedef short16 short_16;

typedef uint1 uint_1;
typedef uint2 uint_2;
typedef uint3 uint_3;
typedef uint4 uint_4;
typedef uint8 uint_8;
typedef uint16 uint_16;

typedef int1 int_1;
typedef int2 int_2;
typedef int3 int_3;
typedef int4 int_4;
typedef int8 int_8;
typedef int16 int_16;

typedef ulong1 ulong_1;
typedef ulong2 ulong_2;
typedef ulong3 ulong_3;
typedef ulong4 ulong_4;
typedef ulong8 ulong_8;
typedef ulong16 ulong_16;

typedef long1 long_1;
typedef long2 long_2;
typedef long3 long_3;
typedef long4 long_4;
typedef long8 long_8;
typedef long16 long_16;

typedef float1 float_1;
typedef float2 float_2;
typedef float3 float_3;
typedef float4 float_4;
typedef float8 float_8;
typedef float16 float_16;

typedef double1 double_1;
typedef double2 double_2;
typedef double3 double_3;
typedef double4 double_4;
typedef double8 double_8;
typedef double16 double_16;

template<typename SCALAR_TYPE, int SIZE> struct short_vector {
  short_vector() {
    typedef __vector<SCALAR_TYPE,SIZE> type;
  }
};

template <typename>
struct short_vector_traits;

template <typename SCALAR_TYPE, int SIZE>
struct short_vector_traits<__vector<SCALAR_TYPE, SIZE>> {
  typedef typename __vector<SCALAR_TYPE, SIZE>::value_type value_type;
  static int const size = __vector<SCALAR_TYPE, SIZE>::size;
};

template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
class __vector {

  static_assert((std::is_integral<SCALAR_TYPE>::value
                || std::is_floating_point<SCALAR_TYPE>::value)
                , "short_vector of this data type is not supported");

  static_assert((VECTOR_LENGTH==1 || VECTOR_LENGTH==2 || VECTOR_LENGTH==3 
                || VECTOR_LENGTH==4 || VECTOR_LENGTH==8 || VECTOR_LENGTH==16)
                  , "short_vector of this size is not supported");

public:
  typedef SCALAR_TYPE value_type;

  // Number of components in a vector
  // NOTE: size != size in memory`
  static const int size = VECTOR_LENGTH;

  // Actual vector size in memory
  // NOTE: A 3-component vector has the same size as a 4-component vector
  static constexpr int data_vector_length = VECTOR_LENGTH==3?4:VECTOR_LENGTH;

  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(data_vector_length)));

  typedef __vector<value_type,size> __scalartype_N;

private:
  typedef value_type v1_type_internal  __attribute__((ext_vector_type(1)));
  typedef value_type v2_type_internal  __attribute__((ext_vector_type(2)));
  typedef value_type v3_type_internal  __attribute__((ext_vector_type(4)));
  typedef value_type v4_type_internal  __attribute__((ext_vector_type(4)));
  typedef value_type v8_type_internal  __attribute__((ext_vector_type(8)));
  typedef value_type v16_type_internal  __attribute__((ext_vector_type(16)));


public:

  __vector() __CPU_GPU__ { data = static_cast<vector_value_type>(static_cast<value_type>(0)); }; 

  // the vector type overloaded constructor below already covers this scalar case
  //__vector(value_type value) __CPU_GPU__ { data = { static_cast<value_type>(value), static_cast<value_type>(value)}; }
  __vector(vector_value_type value) __CPU_GPU__ : data(value) {}

  __vector(const __scalartype_N& other) __CPU_GPU__ : data(other.data) { }


  // component-wise constructor
  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==2,value_type>::type > 
  __vector(value_type v1,value_type v2) __CPU_GPU__ {
    data = {v1,v2}; 
  }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==3,value_type>::type > 
  __vector(value_type v1,value_type v2,value_type v3) __CPU_GPU__ {
    data = {v1,v2,v3,static_cast<value_type>(0)}; 
  }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==4,value_type>::type > 
  __vector(value_type v1,value_type v2, value_type v3, value_type v4) __CPU_GPU__ {
    data = {v1,v2,v3,v4}; 
  }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==8,value_type>::type > 
  __vector(value_type v1,value_type v2, value_type v3, value_type v4
          ,value_type v5,value_type v6, value_type v7, value_type v8) __CPU_GPU__ {
    data = {v1,v2,v3,v4,v5,v6,v7,v8}; 
  }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==16,value_type>::type > 
  __vector(value_type v1,value_type v2, value_type v3, value_type v4
          ,value_type v5,value_type v6, value_type v7, value_type v8
          ,value_type v9,value_type v10, value_type v11, value_type v12
          ,value_type v13,value_type v14, value_type v15, value_type v16) __CPU_GPU__ {
    data = {v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16}; 
  }

  
  // conversion constructor from other short vector types
  template <typename ST>
  explicit __vector(const  __vector<ST,1>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,2>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                            ,static_cast<value_type>(other.get_s1()) }; }

  template < typename ST>
  explicit __vector(const  __vector<ST,3>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(0) };           }

  template <typename ST>
  explicit __vector(const  __vector<ST,4>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(other.get_s3()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,8>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(other.get_s3()) 
                                                                             ,static_cast<value_type>(other.get_s4())
                                                                             ,static_cast<value_type>(other.get_s5())
                                                                             ,static_cast<value_type>(other.get_s6()) 
                                                                             ,static_cast<value_type>(other.get_s7()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,16>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(other.get_s3()) 
                                                                             ,static_cast<value_type>(other.get_s4())
                                                                             ,static_cast<value_type>(other.get_s5())
                                                                             ,static_cast<value_type>(other.get_s6()) 
                                                                             ,static_cast<value_type>(other.get_s7()) 
                                                                             ,static_cast<value_type>(other.get_s8())
                                                                             ,static_cast<value_type>(other.get_s9())
                                                                             ,static_cast<value_type>(other.get_sA()) 
                                                                             ,static_cast<value_type>(other.get_sB()) 
                                                                             ,static_cast<value_type>(other.get_sC())
                                                                             ,static_cast<value_type>(other.get_sD())
                                                                             ,static_cast<value_type>(other.get_sE()) 
                                                                             ,static_cast<value_type>(other.get_sF()) }; }



  // one-component accessors

#define DECLARE_VECTOR_ONE_COMPONENT_GET_SET(N,MIN_V_SIZE) \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  value_type get_s ##N() const __CPU_GPU__ { return data.s ##N; } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_s ##N(value_type v) const __CPU_GPU__ { data.s ##N = v; }

  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(0,1)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(1,2)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(2,3)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(3,4)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(4,8)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(5,8)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(6,8)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(7,8)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(8,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(9,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(A,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(B,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(C,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(D,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(E,16)
  DECLARE_VECTOR_ONE_COMPONENT_GET_SET(F,16)

  value_type get_x() const __CPU_GPU__ { return get_s0(); }

  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=2,value_type>::type >
  value_type get_y() const __CPU_GPU__ { return get_s1(); }

  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=3,value_type>::type >
  value_type get_z() const __CPU_GPU__ { return get_s2(); }

  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=4,value_type>::type >
  value_type get_w() const __CPU_GPU__ { return get_s3(); }

  void set_x(value_type v) __CPU_GPU__ { set_s0(v); }

  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=2,value_type>::type >
  void set_y(value_type v) __CPU_GPU__ { set_s1(v); }

  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=3,value_type>::type >
  void set_z(value_type v) __CPU_GPU__ { set_s2(v); }

  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=4,value_type>::type >
  void set_w(value_type v) __CPU_GPU__ { set_s3(v); }


  // two-component accessors

#define DECLARE_VECTOR_TWO_COMPONENT_GET_SET(C0,C1,MIN_V_SIZE) \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  __vector<value_type, 2> get_ ##C0 ##C1 () { return create_vector2(data.C0 ## C1); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  __vector<value_type, 2> get_ ##C1 ##C0 () { return create_vector2(data.C1 ## C0); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C0 ##C1 (const __vector<value_type, 2>& v) { data.C0 ## C1 = v.get_vector();  } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C1 ##C0 (const __vector<value_type, 2>& v) { data.C1 ## C0 = v.get_vector();  } 

  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,y,2)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,z,3)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,w,4)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,z,3)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,w,4)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(w,z,4)


  // three-component accessors
#define DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C1,C2,MIN_V_SIZE) \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  __vector<value_type, 3> get_ ##C0 ##C1 ## C2 () { return create_vector3(data.C0 ## C1 ## C2); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C0 ##C1 ##C2 (const __vector<value_type, 3>& v) { data.C0 ## C1 ## C2 = v.get_vector(); }  

#define DECLARE_VECTOR_THREE_COMPONENT_GET_SET(C0,C1,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C1,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C2,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C1,C0,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C1,C2,C0,MIN_V_SIZE) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C2,C0,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C2,C1,C0,MIN_V_SIZE) 

  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,y,z,3)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,y,w,4)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,z,w,4)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(y,z,w,4) 


  // four-component accessors

#define DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C1,C2,C3,MIN_V_SIZE) \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  __vector<value_type, 4> get_ ##C0 ##C1 ## C2 ## C3 () { return create_vector4(data.C0 ## C1 ## C2 ## C3); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C0 ##C1 ##C2 ##C3 (const __vector<value_type, 4>& v) { data.C0 ## C1 ## C2 ## C3 = v.get_vector(); }  

#define DECLARE_VECTOR_FOUR_COMPONENT_GET_SET(C0,C1,C2,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C1,C2,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C1,C3,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C2,C1,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C2,C3,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C3,C1,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C3,C2,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C0,C2,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C0,C3,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C2,C0,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C2,C3,C0,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C3,C0,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C3,C2,C0,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C0,C1,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C0,C3,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C1,C0,C3,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C1,C3,C0,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C3,C0,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C3,C1,C0,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C0,C1,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C0,C2,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C1,C0,C2,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C1,C2,C0,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C2,C0,C1,MIN_V_SIZE) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C2,C1,C0,MIN_V_SIZE) 

  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET(x,y,z,w,4);


  vector_value_type get_vector() const __CPU_GPU__ { return data; }
  void set_vector(vector_value_type v)  __CPU_GPU__ { data = v; }

  __scalartype_N& operator=(const __scalartype_N& rhs) __CPU_GPU__ { 
    data = rhs.data;
    return *this;
  }

  __scalartype_N& operator++() __CPU_GPU__ { 
     data += static_cast<vector_value_type>(static_cast<value_type>(1)); 
     return *this; 
  }
  __scalartype_N operator++(int) __CPU_GPU__ { 
    __scalartype_N r(*this);
    operator++();
    return r;
  }
  __scalartype_N& operator--() __CPU_GPU__ { 
    data -= static_cast<vector_value_type>(static_cast<value_type>(1)); 
    return *this;
  }
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

  __scalartype_N operator-() __CPU_GPU__ {
    static_assert(std::is_signed<value_type>::value, "operator- can only support short vector of signed integral or floating-point types.");
    __scalartype_N r;
    r.data = -data;
    return r;
  }

  __scalartype_N operator~() __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator~ can only support short vector of integral types.");
    __scalartype_N r;
    r.data = ~data;
    return r;
  }

  __scalartype_N operator%(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator% can only support short vector of integral types.");
    __scalartype_N r;
    r.data = data%lhs.data;
    return r;
  }
  __scalartype_N& operator%=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this%lhs;
    return *this;
  }

  __scalartype_N operator^(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator^ can only support integral short vector.");
    __scalartype_N r;
    r.data = data^lhs.data;
    return r;
  }
  __scalartype_N& operator^=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this^lhs;
    return *this;
  }

  __scalartype_N operator|(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator| can only support integral short vector.");
    __scalartype_N r;
    r.data = data|lhs.data;
    return r;
  }
  __scalartype_N& operator|=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this|lhs;
    return *this;
  }

  __scalartype_N operator&(const __scalartype_N& lhs) __CPU_GPU__ { 
   static_assert(std::is_integral<value_type>::value, "operator& can only support integral short vector.");
    __scalartype_N r;
    r.data = data&lhs.data;
    return r;
  }
  __scalartype_N& operator&=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this&lhs;
    return *this;
  }

  __scalartype_N operator>>(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator>> can only support integral short vector.");
    __scalartype_N r;
    r.data = data>>lhs.data;
    return r;
  }
  __scalartype_N& operator>>=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this>>lhs;
    return *this;
  }

  __scalartype_N operator<<(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator<< can only support integral short vector.");
    __scalartype_N r;
    r.data = data<<lhs.data;
    return r;
  }
  __scalartype_N& operator<<=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this<<lhs;
    return *this;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==1,value_type>::type >
  bool operator==(const __vector<value_type, 1>& rhs) __CPU_GPU__ { 
    return (data.x == rhs.data.x); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==2,value_type>::type >
  bool operator==(const __vector<value_type, 2>& rhs) __CPU_GPU__ { 
    return (data.x == rhs.data.x 
         && data.y == rhs.data.y); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==4,value_type>::type >
  bool operator==(const __vector<value_type, 4>& rhs) __CPU_GPU__ { 
    return   ((data.s0 == rhs.data.s0) && (data.s1 == rhs.data.s1))
              && ((data.s2 == rhs.data.s2) && (data.s3 == rhs.data.s3));

  }


  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==8,value_type>::type >
  bool operator==(const __vector<value_type, 8>& rhs) __CPU_GPU__ {
    return    (((data.s0 == rhs.data.s0) && (data.s1 == rhs.data.s1))
              && ((data.s2 == rhs.data.s2) && (data.s3 == rhs.data.s3)))
            &&  
              (((data.s4 == rhs.data.s4) && (data.s5 == rhs.data.s5))
              && ((data.s6 == rhs.data.s6) && (data.s7 == rhs.data.s7)))
              ;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==16,value_type>::type >
  bool operator==(const __vector<value_type, 16>& rhs) __CPU_GPU__ {

    return (   (((data.s0 == rhs.data.s0) && (data.s1 == rhs.data.s1))
              && ((data.s2 == rhs.data.s2) && (data.s3 == rhs.data.s3)))
            &&  
              (((data.s4 == rhs.data.s4) && (data.s5 == rhs.data.s5))
              && ((data.s6 == rhs.data.s6) && (data.s7 == rhs.data.s7)))
           )
           &&
           (  (((data.s8 == rhs.data.s8) && (data.s9 == rhs.data.s9))
              && ((data.sA == rhs.data.sA) && (data.sB == rhs.data.sB)))
            &&  
              (((data.sC == rhs.data.sC) && (data.sD == rhs.data.sD))
              && ((data.sE == rhs.data.sE) && (data.sF == rhs.data.sF)))
           )
           ;
  }

  bool operator!=(const __scalartype_N& rhs) __CPU_GPU__ { return !(*this==rhs); }

private:
  vector_value_type data;

  __vector<value_type,2> create_vector2(v2_type_internal v) __CPU_GPU__ {
    return __vector<value_type,2>(v);
  }

  __vector<value_type,3> create_vector3(v3_type_internal v) __CPU_GPU__ {
    return __vector<value_type,3>(v);
  }

  __vector<value_type,4> create_vector4(v4_type_internal v) __CPU_GPU__ {
    return __vector<value_type,4>(v);
  }
};


template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_LENGTH> operator+(const __vector<SCALAR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() + rhs.get_vector());
  return r;
}


template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_LENGTH> operator-(const __vector<SCALAR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() - rhs.get_vector());
  return r;
}

template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_LENGTH> operator*(const __vector<SCALAR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() * rhs.get_vector());
  return r;
}

template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_LENGTH> operator/(const __vector<SCALAR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() / rhs.get_vector());
  return r;
}

