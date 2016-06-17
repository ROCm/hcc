
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

/*
template <typename SCALAR_TYPE_ORG, typename VECTOR_TYPE_ORG, unsigned int VECTOR_LENGTH_ORG>
class __vector;
*/


template <typename SCALAR_TYPE, typename VECTOR_TYPE, unsigned int VECTOR_LENGTH>
class __vector {

public:

  static const int size = VECTOR_LENGTH;

  typedef SCALAR_TYPE value_type;
  typedef VECTOR_TYPE vector_value_type;

  typedef __vector<value_type,vector_value_type,size> __scalartype_N;

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


  value_type get_s0() const __CPU_GPU__ {
    return data.s0;
  }

  value_type get_s1() const __CPU_GPU__ {
    static_assert(size>=2, "invalid component for vector with a length less than 2");
    return data.s1;
  }

  value_type get_s2() const __CPU_GPU__ {
    static_assert(size>=3, "invalid component for vector with a length less than 3");
    return data.s2;
  }

  value_type get_s3() const __CPU_GPU__ {
    static_assert(size>=4, "invalid component for vector with a length less than 4");
    return data.s3;
  }

  value_type get_s4() const __CPU_GPU__ {
    static_assert(size>=8, "invalid component for vector with a length less than 8");
    return data.s4;
  }

  value_type get_s5() const __CPU_GPU__ {
    static_assert(size>=8, "invalid component for vector with a length less than 8");
    return data.s5;
  }

  value_type get_s6() const __CPU_GPU__ {
    static_assert(size>=8, "invalid component for vector with a length less than 8");
    return data.s6;
  }

  value_type get_s7() const __CPU_GPU__ {
    static_assert(size>=8, "invalid component for vector with a length less than 8");
    return data.s7;
  }

  value_type get_s8() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.s8;
  }

  value_type get_s9() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.s9;
  }

  value_type get_sA() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.sA;
  }

  value_type get_sB() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.sB;
  }

  value_type get_sC() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.sC;
  }

  value_type get_sD() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.sD;
  }

  value_type get_sE() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.sE;
  }

  value_type get_sF() const __CPU_GPU__ {
    static_assert(size>=16, "invalid component for vector with a length less than 16");
    return data.sF;
  }

  value_type get_x() const __CPU_GPU__ { return get_s0(); }
  value_type get_y() const __CPU_GPU__ { return get_s1(); }
  value_type get_z() const __CPU_GPU__ { return get_s2(); }
  value_type get_w() const __CPU_GPU__ { return get_s3(); }

  vector_value_type get_vector() const __CPU_GPU__ { return data; }

  void set_x(value_type v) __CPU_GPU__ { data.x = v; }
  void set_y(value_type v)  __CPU_GPU__ { data.y = v; }
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



  // boolean operator
  bool operator==(const __scalartype_N& rhs) __CPU_GPU__ { return (data.x == rhs.data.x 
                                                                   && data.y == rhs.data.y); }
  bool operator!=(const __scalartype_N& rhs) __CPU_GPU__ { return !(*this==rhs); }
  

private:
  vector_value_type data;

};

template <typename SCALAR_TYPE_P, typename VECTOR_TYPE_P, unsigned int VECTOR_LENGTH_P>
__vector<SCALAR_TYPE_P,VECTOR_TYPE_P,VECTOR_LENGTH_P> operator+(const __vector<SCALAR_TYPE_P,VECTOR_TYPE_P,VECTOR_LENGTH_P>& lhs
                                                          , const __vector<SCALAR_TYPE_P,VECTOR_TYPE_P,VECTOR_LENGTH_P>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE_P,VECTOR_TYPE_P,VECTOR_LENGTH_P> r(lhs.get_vector() + rhs.get_vector());
  return r;
}


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


