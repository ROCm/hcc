
#pragma once

#include "hc_types.inl"

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

template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
class __vector;


#define DECLARE_VECTOR_TYPE_CLASS(SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX, CLASS_PREFIX) \
typedef __vector<SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX ## 2, 2>   CLASS_PREFIX ## 2; \
typedef __vector<SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX ## 3, 4>   CLASS_PREFIX ## 3; \
typedef __vector<SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX ## 4, 4>   CLASS_PREFIX ## 4; \
typedef __vector<SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX ## 8, 8>   CLASS_PREFIX ## 8; \
typedef __vector<SCALAR_TYPE, INTERNAL_VECTOR_TYPE_PREFIX ## 16, 16>   CLASS_PREFIX ## 16; 

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


template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
class __vector {

public:

  static const int size = VECTOR_LENGTH;

  typedef SCALAR_TYPE value_type;
  typedef VECTOR_TYPE vector_value_type;

  typedef __vector<value_type,vector_value_type,size> __scalartype_N;


private:
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
  template < typename ST, typename VT>
  explicit __vector(const  __vector<ST,VT,2>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                            ,static_cast<value_type>(other.get_s1()) }; }

  template < typename ST, typename VT>
  explicit __vector(const  __vector<ST,VT,3>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(0) };           }

  template <typename ST, typename VT>
  explicit __vector(const  __vector<ST,VT,4>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(other.get_s3()) }; }

  template <typename ST, typename VT>
  explicit __vector(const  __vector<ST,VT,8>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
                                                                             ,static_cast<value_type>(other.get_s1())
                                                                             ,static_cast<value_type>(other.get_s2()) 
                                                                             ,static_cast<value_type>(other.get_s3()) 
                                                                             ,static_cast<value_type>(other.get_s4())
                                                                             ,static_cast<value_type>(other.get_s5())
                                                                             ,static_cast<value_type>(other.get_s6()) 
                                                                             ,static_cast<value_type>(other.get_s7()) }; }

  template <typename ST, typename VT>
  explicit __vector(const  __vector<ST,VT,16>& other)  __CPU_GPU__ { data = { static_cast<value_type>(other.get_s0())
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
  value_type get_s ##N() const __CPU_GPU__ {   \
    static_assert(size>=MIN_V_SIZE , "invalid vector component"); \
    return data.s ##N; } \
  void set_s ##N(value_type v) const __CPU_GPU__ { \
    static_assert(size>=MIN_V_SIZE , "invalid vector component"); \
    data.s ##N = v; }

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
  value_type get_y() const __CPU_GPU__ { return get_s1(); }
  value_type get_z() const __CPU_GPU__ { return get_s2(); }
  value_type get_w() const __CPU_GPU__ { return get_s3(); }

  void set_x(value_type v) __CPU_GPU__ { set_s0(v); }
  void set_y(value_type v) __CPU_GPU__ { set_s1(v); }
  void set_z(value_type v) __CPU_GPU__ { set_s2(v); }
  void set_w(value_type v) __CPU_GPU__ { set_s3(v); }


  // two-component accessors

#define DECLARE_VECTOR_TWO_COMPONENT_GET_SET(C0,C1) \
  __vector<value_type, v2_type_internal, 2> get_ ##C0 ##C1 () { return create_vector2(data.C0 ## C1); } \
  __vector<value_type, v2_type_internal, 2> get_ ##C1 ##C0 () { return create_vector2(data.C1 ## C0); } \
  void set_ ##C0 ##C1 (const __vector<value_type, v2_type_internal, 2>& v) { data.C0 ## C1 = v.get_vector();  } \
  void set_ ##C1 ##C0 (const __vector<value_type, v2_type_internal, 2>& v) { data.C1 ## C0 = v.get_vector();  } 

  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,y)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,z)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,w)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,z)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,w)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(w,z)


  // three-component accessors
#define DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C1,C2) \
  __vector<value_type, v3_type_internal, 3> get_ ##C0 ##C1 ## C2 () { return create_vector3(data.C0 ## C1 ## C2); } \
  void set_ ##C0 ##C1 ##C2 (const __vector<value_type, v3_type_internal, 3>& v) { data.C0 ## C1 ## C2 = v.get_vector().xyz; }  

#define DECLARE_VECTOR_THREE_COMPONENT_GET_SET(C0,C1,C2) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C1,C2) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C2,C1) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C1,C0,C2) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C1,C2,C0) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C2,C0,C1) \
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C2,C1,C0) 

  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,y,z)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,y,w)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,z,w)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(y,z,w) 


  // four-component accessors

#define DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C1,C2,C3) \
  __vector<value_type, v4_type_internal, 4> get_ ##C0 ##C1 ## C2 ## C3 () { return create_vector4(data.C0 ## C1 ## C2 ## C3); } \
  void set_ ##C0 ##C1 ##C2 ##C3 (const __vector<value_type, v4_type_internal, 4>& v) { data.C0 ## C1 ## C2 ## C3 = v.get_vector(); }  

#define DECLARE_VECTOR_FOUR_COMPONENT_GET_SET(C0,C1,C2,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C1,C2,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C1,C3,C2) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C2,C1,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C2,C3,C1) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C3,C1,C2) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C0,C3,C2,C1) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C0,C2,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C0,C3,C2) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C2,C0,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C2,C3,C0) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C3,C0,C2) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C1,C3,C2,C0) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C0,C1,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C0,C3,C1) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C1,C0,C3) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C1,C3,C0) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C3,C0,C1) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C2,C3,C1,C0) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C0,C1,C2) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C0,C2,C1) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C1,C0,C2) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C1,C2,C0) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C2,C0,C1) \
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET_PAIR(C3,C2,C1,C0) 

  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET(x,y,z,w);


  vector_value_type get_vector() const __CPU_GPU__ { return data; }
  void set_vector(vector_value_type v)  __CPU_GPU__ { data = v; }

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
            , class = typename std::enable_if<T::size==2,value_type>::type >
  bool operator==(const __vector<value_type, vector_value_type, 2>& rhs) __CPU_GPU__ { 
    return (data.x == rhs.data.x 
         && data.y == rhs.data.y); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==4,value_type>::type >
  bool operator==(const __vector<value_type, vector_value_type, 4>& rhs) __CPU_GPU__ { 
    return   ((data.s0 == rhs.data.s0) && (data.s1 == rhs.data.s1))
              && ((data.s2 == rhs.data.s2) && (data.s3 == rhs.data.s3));

  }


  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==8,value_type>::type >
  bool operator==(const __vector<value_type, vector_value_type, 8>& rhs) __CPU_GPU__ {
    return    (((data.s0 == rhs.data.s0) && (data.s1 == rhs.data.s1))
              && ((data.s2 == rhs.data.s2) && (data.s3 == rhs.data.s3)))
            &&  
              (((data.s4 == rhs.data.s4) && (data.s5 == rhs.data.s5))
              && ((data.s6 == rhs.data.s6) && (data.s7 == rhs.data.s7)))
              ;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==16,value_type>::type >
  bool operator==(const __vector<value_type, vector_value_type, 16>& rhs) __CPU_GPU__ {

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

  __vector<value_type,v2_type_internal,2> create_vector2(v2_type_internal v) {
    return __vector<value_type,v2_type_internal,2>(v);
  }

  __vector<value_type,v3_type_internal,3> create_vector3(v3_type_internal v) {
    return __vector<value_type,v3_type_internal,3>(v);
  }

  __vector<value_type,v4_type_internal,4> create_vector4(v4_type_internal v) {
    return __vector<value_type,v4_type_internal,4>(v);
  }
};


template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> operator+(const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() + rhs.get_vector());
  return r;
}


template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> operator-(const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() - rhs.get_vector());
  return r;
}

template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> operator*(const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() * rhs.get_vector());
  return r;
}

template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
__vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> operator/(const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& lhs
                                                          , const __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE,VECTOR_TYPE,VECTOR_LENGTH> r(lhs.get_vector() / rhs.get_vector());
  return r;
}


