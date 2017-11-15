
#pragma once

#include "hc_norm_unorm.inl"

#ifndef __CPU_GPU__

#if __HCC_AMP__
#define __CPU_GPU__   restrict(cpu,amp)
#else
#define __CPU_GPU__   [[cpu,hc]]
#endif

#endif

template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
class __vector;

// NOTE: A single-component vector (short vector with 1 component) in the hc namespace
// is implemented with the __vector class with 1 component.
// However, for C++AMP (Concurrency namespace), a single-component vector is mapped to a
// scalar according to the C++AMP specification 
#if !__HCC_AMP__

#define DECLARE_VECTOR_TYPE_CLASS(SCALAR_TYPE, CLASS_PREFIX) \
typedef __vector<SCALAR_TYPE, 1>    CLASS_PREFIX ## 1; \
typedef __vector<SCALAR_TYPE, 2>    CLASS_PREFIX ## 2; \
typedef __vector<SCALAR_TYPE, 3>    CLASS_PREFIX ## 3; \
typedef __vector<SCALAR_TYPE, 4>    CLASS_PREFIX ## 4; \
typedef __vector<SCALAR_TYPE, 8>    CLASS_PREFIX ## 8; \
typedef __vector<SCALAR_TYPE, 16>   CLASS_PREFIX ## 16; 

#else

#define DECLARE_VECTOR_TYPE_CLASS(SCALAR_TYPE, CLASS_PREFIX) \
typedef SCALAR_TYPE    CLASS_PREFIX ## 1; \
typedef __vector<SCALAR_TYPE, 2>    CLASS_PREFIX ## 2; \
typedef __vector<SCALAR_TYPE, 3>    CLASS_PREFIX ## 3; \
typedef __vector<SCALAR_TYPE, 4>    CLASS_PREFIX ## 4; \
typedef __vector<SCALAR_TYPE, 8>    CLASS_PREFIX ## 8; \
typedef __vector<SCALAR_TYPE, 16>   CLASS_PREFIX ## 16; 

#endif

DECLARE_VECTOR_TYPE_CLASS(unsigned char, uchar);
DECLARE_VECTOR_TYPE_CLASS(char, char);
DECLARE_VECTOR_TYPE_CLASS(unsigned short, ushort);
DECLARE_VECTOR_TYPE_CLASS(short, short);
DECLARE_VECTOR_TYPE_CLASS(unsigned int, uint);
DECLARE_VECTOR_TYPE_CLASS(int, int);
DECLARE_VECTOR_TYPE_CLASS(unsigned long, ulong);
DECLARE_VECTOR_TYPE_CLASS(long, long);
DECLARE_VECTOR_TYPE_CLASS(unsigned long long, ulonglong);
DECLARE_VECTOR_TYPE_CLASS(long long, longlong);
#if !__HCC_AMP__
DECLARE_VECTOR_TYPE_CLASS(hc::half, half);
#endif
DECLARE_VECTOR_TYPE_CLASS(float, float);
DECLARE_VECTOR_TYPE_CLASS(double, double);
DECLARE_VECTOR_TYPE_CLASS(norm, norm);
DECLARE_VECTOR_TYPE_CLASS(unorm, unorm);

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

typedef ulonglong1 ulonglong_1;
typedef ulonglong2 ulonglong_2;
typedef ulonglong3 ulonglong_3;
typedef ulonglong4 ulonglong_4;
typedef ulonglong8 ulonglong_8;
typedef ulonglong16 ulonglong_16;

typedef longlong1 longlong_1;
typedef longlong2 longlong_2;
typedef longlong3 longlong_3;
typedef longlong4 longlong_4;
typedef longlong8 longlong_8;
typedef longlong16 longlong_16;

#if !__HCC_AMP__
typedef half1 half_1;
typedef half2 half_2;
typedef half3 half_3;
typedef half4 half_4;
typedef half8 half_8;
typedef half16 half_16;
#endif

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

typedef norm1 norm_1;
typedef norm2 norm_2;
typedef norm3 norm_3;
typedef norm4 norm_4;
typedef norm8 norm_8;
typedef norm16 norm_16;

typedef unorm1 unorm_1;
typedef unorm2 unorm_2;
typedef unorm3 unorm_3;
typedef unorm4 unorm_4;
typedef unorm8 unorm_8;
typedef unorm16 unorm_16;

template<typename SCALAR_TYPE, int SIZE> 
struct short_vector {
#if !__HCC_AMP__
  typedef typename __vector<SCALAR_TYPE,SIZE>::type type;
#else
  typedef typename std::conditional<SIZE==1
                                  , SCALAR_TYPE
                                  , __vector<SCALAR_TYPE,SIZE>>::type type;
#endif
};


// short_vector_traits for single component vector
template <typename SCALAR_TYPE>
struct short_vector_traits {
  static_assert((std::is_integral<SCALAR_TYPE>::value
                || std::is_floating_point<SCALAR_TYPE>::value
                || std::is_same<SCALAR_TYPE, norm>::value
                || std::is_same<SCALAR_TYPE, unorm>::value
#if !__HCC_AMP__
                || std::is_same<SCALAR_TYPE,hc::half>::value
#endif
                )
                , "short_vector of this data type is not supported");
  typedef SCALAR_TYPE value_type;
  static int const size = 1;
};

// short_vector_traits for non-single component vetor
template <typename SCALAR_TYPE, int SIZE>
struct short_vector_traits<__vector<SCALAR_TYPE, SIZE>> {
  typedef typename __vector<SCALAR_TYPE, SIZE>::value_type value_type;
  static int const size = __vector<SCALAR_TYPE, SIZE>::size;
};



template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
class __vector_data_container {
  static_assert((VECTOR_LENGTH==1 || VECTOR_LENGTH==2 || VECTOR_LENGTH==3 
                || VECTOR_LENGTH==4 || VECTOR_LENGTH==8 || VECTOR_LENGTH==16)
                , "short_vector of this size is not supported");
};


template <typename SCALAR_TYPE>
class __vector_data_container<SCALAR_TYPE,1> {

public:

  static const unsigned int size = 1;
  typedef SCALAR_TYPE value_type; 
  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(size)));

  union {
    vector_value_type data;
    SCALAR_TYPE           ar[size];
    struct { SCALAR_TYPE  x; };
  };

  __vector_data_container() __CPU_GPU__ { 
    data = static_cast<SCALAR_TYPE>(0); 
  }

  __vector_data_container(vector_value_type v) __CPU_GPU__ { 
    data = v; 
  }

  __attribute__((annotate("user_deserialize")))
  __vector_data_container(const SCALAR_TYPE x) __CPU_GPU__ {
    data = { x };
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    for (auto &component : ar) {
      s.Append(sizeof(SCALAR_TYPE), &component);
    }
  }
};


template <typename SCALAR_TYPE>
class __vector_data_container<SCALAR_TYPE,2> {

public:

  static const unsigned int size = 2;
  typedef SCALAR_TYPE value_type; 
  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(size)));

  union {
    vector_value_type data;
    SCALAR_TYPE           ar[size];
    struct { SCALAR_TYPE  x,y; };
  };

  __vector_data_container() __CPU_GPU__ { 
    data = static_cast<SCALAR_TYPE>(0); 
  }

  __vector_data_container(vector_value_type v) __CPU_GPU__ { 
    data = v; 
  }

  __attribute__((annotate("user_deserialize")))
  __vector_data_container(const SCALAR_TYPE x, const SCALAR_TYPE y) __CPU_GPU__ {
    data = { x, y };
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    for (auto &component : ar) {
      s.Append(sizeof(SCALAR_TYPE), &component);
    }
  }
};


template <typename SCALAR_TYPE>
class __vector_data_container<SCALAR_TYPE,3> {

public:

  static const unsigned int size = 3;
  typedef SCALAR_TYPE value_type; 
  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(size)));

  union {
    vector_value_type data;
    SCALAR_TYPE           ar[size];
    struct { SCALAR_TYPE  x,y,z; };
  };

  __vector_data_container() __CPU_GPU__ { 
    data = static_cast<SCALAR_TYPE>(0); 
  }

  __vector_data_container(vector_value_type v) __CPU_GPU__ { 
    data = v; 
  }

  __attribute__((annotate("user_deserialize")))
  __vector_data_container(const SCALAR_TYPE x, const SCALAR_TYPE y, const SCALAR_TYPE z) __CPU_GPU__ {
    data = { x, y, z };
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    for (auto &component : ar) {
      s.Append(sizeof(SCALAR_TYPE), &component);
    }
  }
};


template <typename SCALAR_TYPE>
class __vector_data_container<SCALAR_TYPE,4> {

public:

  static const unsigned int size = 4;
  typedef SCALAR_TYPE value_type; 
  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(size)));

  union {
    vector_value_type data;
    SCALAR_TYPE           ar[size];
    struct { SCALAR_TYPE  x,y,z,w; };
  };

  __vector_data_container() __CPU_GPU__ { 
    data = static_cast<SCALAR_TYPE>(0); 
  }

  __vector_data_container(vector_value_type v) __CPU_GPU__ { 
    data = v; 
  }

  __attribute__((annotate("user_deserialize")))
  __vector_data_container(const SCALAR_TYPE x, const SCALAR_TYPE y, const SCALAR_TYPE z, const SCALAR_TYPE w) __CPU_GPU__ {
    data = { x,y,z,w };
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    for (auto &component : ar) {
      s.Append(sizeof(SCALAR_TYPE), &component);
    }
  }
};


template <typename SCALAR_TYPE>
class __vector_data_container<SCALAR_TYPE,8> {

public:

  static const unsigned int size = 8;
  typedef SCALAR_TYPE value_type; 
  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(size)));

  union {
    vector_value_type data;
    SCALAR_TYPE           ar[size];
    struct { SCALAR_TYPE  x,y,z,w,s4,s5,s6,s7; };
  };

  __vector_data_container() __CPU_GPU__ { 
    data = static_cast<SCALAR_TYPE>(0); 
  }

  __vector_data_container(vector_value_type v) __CPU_GPU__ { 
    data = v; 
  }

  __attribute__((annotate("user_deserialize")))
  __vector_data_container(const SCALAR_TYPE x, const SCALAR_TYPE y, const SCALAR_TYPE z, const SCALAR_TYPE w
     , const SCALAR_TYPE s4, const SCALAR_TYPE s5, const SCALAR_TYPE s6, const SCALAR_TYPE s7) __CPU_GPU__ {
    data = { x,y,z,w,s4,s5,s6,s7 };
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    for (auto &component : ar) {
      s.Append(sizeof(SCALAR_TYPE), &component);
    }
  }
};


template <typename SCALAR_TYPE>
class __vector_data_container<SCALAR_TYPE,16> {

public:

  static const unsigned int size = 16;
  typedef SCALAR_TYPE value_type; 
  typedef SCALAR_TYPE vector_value_type  __attribute__((ext_vector_type(size)));

  union {
    vector_value_type data;
    SCALAR_TYPE           ar[size];
    struct { SCALAR_TYPE  x,y,z,w,s4,s5,s6,s7,s8,s9,sA,sB,sC,sD,sE,sF; };
  };

  __vector_data_container() __CPU_GPU__ { 
    data = static_cast<SCALAR_TYPE>(0); 
  }

  __vector_data_container(vector_value_type v) __CPU_GPU__ { 
    data = v; 
  }

  __attribute__((annotate("user_deserialize")))
  __vector_data_container(const SCALAR_TYPE x, const SCALAR_TYPE y, const SCALAR_TYPE z, const SCALAR_TYPE w
     , const SCALAR_TYPE s4, const SCALAR_TYPE s5, const SCALAR_TYPE s6, const SCALAR_TYPE s7
     , const SCALAR_TYPE s8, const SCALAR_TYPE s9, const SCALAR_TYPE sA, const SCALAR_TYPE sB
     , const SCALAR_TYPE sC, const SCALAR_TYPE sD, const SCALAR_TYPE sE, const SCALAR_TYPE sF) __CPU_GPU__ {
    data = { x,y,z,w,s4,s5,s6,s7,s8,s9,sA,sB,sC,sD,sE,sF };
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    for (auto &component : ar) {
      s.Append(sizeof(SCALAR_TYPE), &component);
    }
  }
};



// Implementation of a generic short vector
template <typename SCALAR_TYPE, unsigned int VECTOR_LENGTH>
class __vector : public __vector_data_container<SCALAR_TYPE, VECTOR_LENGTH>   {

  static_assert((std::is_integral<SCALAR_TYPE>::value
                || std::is_floating_point<SCALAR_TYPE>::value
#if !__HCC_AMP__
                || std::is_same<SCALAR_TYPE,hc::half>::value
#endif
                )
                , "short_vector of this data type is not supported");

  static_assert((VECTOR_LENGTH==1 || VECTOR_LENGTH==2 || VECTOR_LENGTH==3 
                || VECTOR_LENGTH==4 || VECTOR_LENGTH==8 || VECTOR_LENGTH==16)
                  , "short_vector of this size is not supported");

public:
  typedef SCALAR_TYPE value_type;
  static const unsigned int size = VECTOR_LENGTH;
  typedef __vector<value_type,size> __scalartype_N;
  typedef value_type vector_value_type  __attribute__((ext_vector_type(size)));
  typedef __vector_data_container<value_type,size> vector_container_type;

private:
  typedef value_type v1_type_internal  __attribute__((ext_vector_type(1)));
  typedef value_type v2_type_internal  __attribute__((ext_vector_type(2)));
  typedef value_type v3_type_internal  __attribute__((ext_vector_type(3)));
  typedef value_type v4_type_internal  __attribute__((ext_vector_type(4)));
  typedef value_type v8_type_internal  __attribute__((ext_vector_type(8)));
  typedef value_type v16_type_internal  __attribute__((ext_vector_type(16)));


public:

  __vector() __CPU_GPU__ { } 

  // the vector type overloaded constructor below already covers this scalar case
  //__vector(value_type value) __CPU_GPU__ { data = { static_cast<value_type>(value), static_cast<value_type>(value)}; }
  __vector(const vector_value_type& value) __CPU_GPU__ : vector_container_type(value) {}

  __vector(const __scalartype_N& other) __CPU_GPU__ : vector_container_type(other.data) { }

  // component-wise constructor
  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==2,value_type>::type > 
  __vector(value_type x, value_type y) __CPU_GPU__ : vector_container_type(x,y) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==3,value_type>::type > 
  __vector(value_type x, value_type y, value_type z) __CPU_GPU__ : vector_container_type(x,y,z) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==4,value_type>::type > 
  __vector(value_type x, value_type y, value_type z, value_type w) __CPU_GPU__ : vector_container_type(x,y,z,w) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==8,value_type>::type > 
  __vector(value_type x, value_type y
           , value_type z, value_type w
           , value_type s4, value_type s5
           , value_type s6, value_type s7) __CPU_GPU__ : vector_container_type(x,y,z,w
                                                                               ,s4,s5,s6,s7) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==16,value_type>::type > 
  __vector(value_type x, value_type y
          , value_type z, value_type w
          , value_type s4, value_type s5
          , value_type s6, value_type s7
          , value_type s8, value_type s9
          , value_type sA, value_type sB
          , value_type sC, value_type sD
          , value_type sE, value_type sF) __CPU_GPU__ : vector_container_type(x,y,z,w,s4,s5,s6,s7,s8
                                                                              ,s9,sA,sB,sC,sD,sE,sF) { }

  // conversion constructor from other short vector types
  template <typename ST>
  explicit __vector(const  __vector<ST,1>& other) __CPU_GPU__ 
             : vector_container_type(other.x) {}

  template <typename ST>
  explicit __vector(const  __vector<ST,2>& other) __CPU_GPU__ 
             : vector_container_type(other.x, other.y) { }

  template < typename ST>
  explicit __vector(const  __vector<ST,3>& other) __CPU_GPU__ 
             : vector_container_type(other.x, other.y, other.z) { }
  
  template <typename ST>
  explicit __vector(const  __vector<ST,4>& other) __CPU_GPU__
             : vector_container_type(other.x, other.y, other.z, other.w) { }
  
  template <typename ST>
  explicit __vector(const  __vector<ST,8>& other) __CPU_GPU__ 
             : vector_container_type(other.x, other.y, other.z, other.w
                                    , other.s4, other.s5, other.s6, other.s7) { }
  
   template <typename ST>
  explicit __vector(const  __vector<ST,16>& other)  __CPU_GPU__ 
             : vector_container_type(other.x, other.y, other.z, other.w
                                    , other.s4, other.s5, other.s6, other.s7
                                    , other.s8, other.s9, other.sA, other.sB
                                    , other.sC, other.sD, other.sE, other.sF) { }

  // one-component accessors

#define DECLARE_VECTOR_ONE_COMPONENT_GET_SET(N,MIN_V_SIZE) \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  value_type get_s ##N() const __CPU_GPU__ { return this->data.s ##N; } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_s ##N(value_type v) const __CPU_GPU__ { this->data.s ##N = v; }

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
  __vector<value_type, 2> get_ ##C0 ##C1 () { return create_vector2(this->data.C0 ## C1); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  __vector<value_type, 2> get_ ##C1 ##C0 () { return create_vector2(this->data.C1 ## C0); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C0 ##C1 (const __vector<value_type, 2>& v) { this->data.C0 ## C1 = v.get_vector();  } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C1 ##C0 (const __vector<value_type, 2>& v) { this->data.C1 ## C0 = v.get_vector();  } 

  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,y,2)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,z,3)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,w,4)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,z,3)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,w,4)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(w,z,4)


  // three-component accessors
#define DECLARE_VECTOR_THREE_COMPONENT_GET_SET_PAIR(C0,C1,C2,MIN_V_SIZE) \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  __vector<value_type, 3> get_ ##C0 ##C1 ## C2 () { return create_vector3(this->data.C0 ## C1 ## C2); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C0 ##C1 ##C2 (const __vector<value_type, 3>& v) { this->data.C0 ## C1 ## C2 = v.get_vector(); }  

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
  __vector<value_type, 4> get_ ##C0 ##C1 ## C2 ## C3 () { return create_vector4(this->data.C0 ## C1 ## C2 ## C3); } \
  template <typename T = __scalartype_N ,class = typename std::enable_if<T::size>=MIN_V_SIZE,value_type>::type > \
  void set_ ##C0 ##C1 ##C2 ##C3 (const __vector<value_type, 4>& v) { this->data.C0 ## C1 ## C2 ## C3 = v.get_vector(); }  

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


  vector_value_type get_vector() const __CPU_GPU__ { return this->data;  }
  void set_vector(vector_value_type v)  __CPU_GPU__ { this->data = v; }

  __scalartype_N& operator=(const __scalartype_N& rhs) __CPU_GPU__ { 
    this->data = rhs.data;
    return *this;
  }

  __scalartype_N& operator++() __CPU_GPU__ { 
     this->data += static_cast<vector_value_type>(static_cast<value_type>(1)); 
     return *this; 
  }
  __scalartype_N operator++(int) __CPU_GPU__ { 
    __scalartype_N r(*this);
    operator++();
    return r;
  }
  __scalartype_N& operator--() __CPU_GPU__ { 
    this->data -= static_cast<vector_value_type>(static_cast<value_type>(1)); 
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
    this->data += rhs.data;
    return *this;
  }

  __scalartype_N& operator-=(const __scalartype_N& rhs) __CPU_GPU__ { 
    this->data -= rhs.data;
    return *this;
  }
 
  __scalartype_N& operator*=(const __scalartype_N& rhs) __CPU_GPU__ { 
    this->data *= rhs.data;
    return *this;
  }
 
  __scalartype_N& operator/=(const __scalartype_N& rhs) __CPU_GPU__ { 
    this->data /= rhs.data;
    return *this;
  }

  __scalartype_N operator-() __CPU_GPU__ {
    static_assert(std::is_signed<value_type>::value, "operator- can only support short vector of signed integral or floating-point types.");
    __scalartype_N r;
    r.data = -this->data;
    return r;
  }

  __scalartype_N operator~() __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator~ can only support short vector of integral types.");
    __scalartype_N r;
    r.data = ~this->data;
    return r;
  }

  __scalartype_N operator%(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator% can only support short vector of integral types.");
    __scalartype_N r;
    r.data = this->data%lhs.data;
    return r;
  }
  __scalartype_N& operator%=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this%lhs;
    return *this;
  }

  __scalartype_N operator^(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator^ can only support integral short vector.");
    __scalartype_N r;
    r.data = this->data^lhs.data;
    return r;
  }
  __scalartype_N& operator^=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this^lhs;
    return *this;
  }

  __scalartype_N operator|(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator| can only support integral short vector.");
    __scalartype_N r;
    r.data = this->data|lhs.data;
    return r;
  }
  __scalartype_N& operator|=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this|lhs;
    return *this;
  }

  __scalartype_N operator&(const __scalartype_N& lhs) __CPU_GPU__ { 
   static_assert(std::is_integral<value_type>::value, "operator& can only support integral short vector.");
    __scalartype_N r;
    r.data = this->data&lhs.data;
    return r;
  }
  __scalartype_N& operator&=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this&lhs;
    return *this;
  }

  __scalartype_N operator>>(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator>> can only support integral short vector.");
    __scalartype_N r;
    r.data = this->data>>lhs.data;
    return r;
  }
  __scalartype_N& operator>>=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this>>lhs;
    return *this;
  }

  __scalartype_N operator<<(const __scalartype_N& lhs) __CPU_GPU__ { 
    static_assert(std::is_integral<value_type>::value, "operator<< can only support integral short vector.");
    __scalartype_N r;
    r.data = this->data<<lhs.data;
    return r;
  }
  __scalartype_N& operator<<=(const __scalartype_N& lhs) __CPU_GPU__ { 
    *this = *this<<lhs;
    return *this;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==1,value_type>::type >
  bool operator==(const __vector<value_type, 1>& rhs) __CPU_GPU__ { 
    return (this->data.x == rhs.data.x); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==2,value_type>::type >
  bool operator==(const __vector<value_type, 2>& rhs) __CPU_GPU__ { 
    return (this->data.x == rhs.data.x 
         && this->data.y == rhs.data.y); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==3,value_type>::type >
  bool operator==(const __vector<value_type, 3>& rhs) __CPU_GPU__ { 
    return   ((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && (this->data.s2 == rhs.data.s2);

  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==4,value_type>::type >
  bool operator==(const __vector<value_type, 4>& rhs) __CPU_GPU__ { 
    return   ((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && ((this->data.s2 == rhs.data.s2) && (this->data.s3 == rhs.data.s3));

  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==8,value_type>::type >
  bool operator==(const __vector<value_type, 8>& rhs) __CPU_GPU__ {
    return    (((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && ((this->data.s2 == rhs.data.s2) && (this->data.s3 == rhs.data.s3)))
            &&  
              (((this->data.s4 == rhs.data.s4) && (this->data.s5 == rhs.data.s5))
              && ((this->data.s6 == rhs.data.s6) && (this->data.s7 == rhs.data.s7)))
              ;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==16,value_type>::type >
  bool operator==(const __vector<value_type, 16>& rhs) __CPU_GPU__ {

    return (   (((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && ((this->data.s2 == rhs.data.s2) && (this->data.s3 == rhs.data.s3)))
            &&  
              (((this->data.s4 == rhs.data.s4) && (this->data.s5 == rhs.data.s5))
              && ((this->data.s6 == rhs.data.s6) && (this->data.s7 == rhs.data.s7)))
           )
           &&
           (  (((this->data.s8 == rhs.data.s8) && (this->data.s9 == rhs.data.s9))
              && ((this->data.sA == rhs.data.sA) && (this->data.sB == rhs.data.sB)))
            &&  
              (((this->data.sC == rhs.data.sC) && (this->data.sD == rhs.data.sD))
              && ((this->data.sE == rhs.data.sE) && (this->data.sF == rhs.data.sF)))
           )
           ;
  }

  bool operator!=(const __scalartype_N& rhs) __CPU_GPU__ { return !(*this==rhs); }

private:

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

// scalar * vector
template <typename SCALAR_TYPE1, typename SCALAR_TYPE2, unsigned int VECTOR_LENGTH>
typename std::enable_if<std::is_scalar<SCALAR_TYPE1>::value, __vector<SCALAR_TYPE2,VECTOR_LENGTH> >::type
operator*(const SCALAR_TYPE1& lhs,
          const __vector<SCALAR_TYPE2,VECTOR_LENGTH>& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE2,VECTOR_LENGTH> r(rhs.get_vector() * static_cast<SCALAR_TYPE2>(lhs));
  return r;
}

// vector * scalar
template <typename SCALAR_TYPE1, typename SCALAR_TYPE2, unsigned int VECTOR_LENGTH>
typename std::enable_if<std::is_scalar<SCALAR_TYPE2>::value, __vector<SCALAR_TYPE1,VECTOR_LENGTH> >::type
operator*(const __vector<SCALAR_TYPE1,VECTOR_LENGTH>& lhs,
          const SCALAR_TYPE2& rhs) __CPU_GPU__ {
  __vector<SCALAR_TYPE1,VECTOR_LENGTH> r(lhs.get_vector() * static_cast<SCALAR_TYPE1>(rhs));
  return r;
}

// Specialization for norm, unorm
template <bool normIsSigned, unsigned int VECTOR_LENGTH>
class __vector<__amp_norm_template<normIsSigned>,VECTOR_LENGTH> :
         public  __vector_data_container<float, VECTOR_LENGTH>  {

  static_assert((VECTOR_LENGTH==1 || VECTOR_LENGTH==2 || VECTOR_LENGTH==3 
                || VECTOR_LENGTH==4 || VECTOR_LENGTH==8 || VECTOR_LENGTH==16)
                  , "short_vector of this size is not supported");

public:
  typedef __amp_norm_template<normIsSigned> value_type;
  static const unsigned int size = VECTOR_LENGTH;
  typedef __vector<value_type,size> __scalartype_N;
  typedef float vector_value_type  __attribute__((ext_vector_type(size)));
  typedef __vector_data_container<float,size> vector_container_type;

private:
  typedef float v1_type_internal  __attribute__((ext_vector_type(1)));
  typedef float v2_type_internal  __attribute__((ext_vector_type(2)));
  typedef float v3_type_internal  __attribute__((ext_vector_type(3)));
  typedef float v4_type_internal  __attribute__((ext_vector_type(4)));
  typedef float v8_type_internal  __attribute__((ext_vector_type(8)));
  typedef float v16_type_internal  __attribute__((ext_vector_type(16)));

  v1_type_internal clamp(v1_type_internal v) __CPU_GPU__ {
    return { value_type(v.s0) };
  }

  v2_type_internal clamp(v2_type_internal v) __CPU_GPU__ {
    return { value_type(v.s0)
            ,value_type(v.s1)
            };
  }

  v4_type_internal clamp(v4_type_internal v) __CPU_GPU__ {
    return { value_type(v.s0)
            ,value_type(v.s1)
            ,value_type(v.s2)
            ,value_type(v.s3)
            };
  }

  v8_type_internal clamp(v8_type_internal v) __CPU_GPU__ {
    return { value_type(v.s0)
            ,value_type(v.s1)
            ,value_type(v.s2)
            ,value_type(v.s3)
            ,value_type(v.s4)
            ,value_type(v.s5)
            ,value_type(v.s6)
            ,value_type(v.s7)
            };
  }

  v16_type_internal clamp(v16_type_internal v) __CPU_GPU__ {
    return { value_type(v.s0)
            ,value_type(v.s1)
            ,value_type(v.s2)
            ,value_type(v.s3)
            ,value_type(v.s4)
            ,value_type(v.s5)
            ,value_type(v.s6)
            ,value_type(v.s7)
            ,value_type(v.s8)
            ,value_type(v.s9)
            ,value_type(v.sA)
            ,value_type(v.sB)
            ,value_type(v.sC)
            ,value_type(v.sD)
            ,value_type(v.sE)
            ,value_type(v.sF)
            };
  }

public:

  __vector() __CPU_GPU__ { }

  // the vector type overloaded constructor below already covers this scalar case
  //__vector(value_type value) __CPU_GPU__ { data = { static_cast<value_type>(value), static_cast<value_type>(value)}; }
  __vector(const vector_value_type& value) __CPU_GPU__   { set_vector(value); }

  __vector(const __scalartype_N& other) __CPU_GPU__ : vector_container_type(other.data) {  }

  // component-wise constructor
  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==2,value_type>::type > 
  __vector(value_type x, value_type y) __CPU_GPU__ : vector_container_type(x,y) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==3,value_type>::type > 
  __vector(value_type x, value_type y, value_type z) __CPU_GPU__ : vector_container_type(x,y,z) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==4,value_type>::type > 
  __vector(value_type x, value_type y, value_type z, value_type w) __CPU_GPU__ : vector_container_type(x,y,z,w) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==8,value_type>::type > 
  __vector(value_type x, value_type y
           , value_type z, value_type w
           , value_type s4, value_type s5
           , value_type s6, value_type s7) __CPU_GPU__ : vector_container_type(x,y,z,w
                                                                              ,s4,s5,s6,s7) { }

  template<typename T = __scalartype_N
          ,class = typename std::enable_if<T::size==16,value_type>::type > 
  __vector(value_type x, value_type y
          , value_type z, value_type w
          , value_type s4, value_type s5
          , value_type s6, value_type s7
          , value_type s8, value_type s9
          , value_type sA, value_type sB
          , value_type sC, value_type sD
          , value_type sE, value_type sF) __CPU_GPU__ : vector_container_type(x,y,z,w,s4,s5,s6,s7,s8
                                                                              ,s9,sA,sB,sC,sD,sE,sF)  { }

  
  // conversion constructor from other short vector types
  template <typename ST>
  explicit __vector(const  __vector<ST,1>& other)  __CPU_GPU__ { this->data = { value_type(other.get_s0()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,2>& other)  __CPU_GPU__ { this->data = { value_type(other.get_s0())
                                                                               ,value_type(other.get_s1()) }; }

  template < typename ST>
  explicit __vector(const  __vector<ST,3>& other)  __CPU_GPU__ { this->data = { value_type(other.get_s0())
                                                                               ,value_type(other.get_s1())
                                                                               ,value_type(other.get_s2()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,4>& other)  __CPU_GPU__ { this->data = { value_type(other.get_s0())
                                                                               ,value_type(other.get_s1())
                                                                               ,value_type(other.get_s2()) 
                                                                               ,value_type(other.get_s3()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,8>& other)  __CPU_GPU__ { this->data = { value_type(other.get_s0())
                                                                               ,value_type(other.get_s1())
                                                                               ,value_type(other.get_s2()) 
                                                                               ,value_type(other.get_s3()) 
                                                                               ,value_type(other.get_s4())
                                                                               ,value_type(other.get_s5())
                                                                               ,value_type(other.get_s6()) 
                                                                               ,value_type(other.get_s7()) }; }

  template <typename ST>
  explicit __vector(const  __vector<ST,16>& other)  __CPU_GPU__ { this->data = { value_type(other.get_s0())
                                                                                ,value_type(other.get_s1())
                                                                                ,value_type(other.get_s2()) 
                                                                                ,value_type(other.get_s3()) 
                                                                                ,value_type(other.get_s4())
                                                                                ,value_type(other.get_s5())
                                                                                ,value_type(other.get_s6()) 
                                                                                ,value_type(other.get_s7()) 
                                                                                ,value_type(other.get_s8())
                                                                                ,value_type(other.get_s9())
                                                                                ,value_type(other.get_sA()) 
                                                                                ,value_type(other.get_sB()) 
                                                                                ,value_type(other.get_sC())
                                                                                ,value_type(other.get_sD())
                                                                                ,value_type(other.get_sE()) 
                                                                                ,value_type(other.get_sF()) }; }



  // one-component accessors
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
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,y,2)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,z,3)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(x,w,4)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,z,3)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(y,w,4)
  DECLARE_VECTOR_TWO_COMPONENT_GET_SET(w,z,4)


  // three-component accessors
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,y,z,3)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,y,w,4)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(x,z,w,4)
  DECLARE_VECTOR_THREE_COMPONENT_GET_SET(y,z,w,4) 


  // four-component accessors
  DECLARE_VECTOR_FOUR_COMPONENT_GET_SET(x,y,z,w,4);

  vector_value_type get_vector() const __CPU_GPU__ { return this->data; }
  void set_vector(vector_value_type v)  __CPU_GPU__ { this->data = clamp(v); }

  __scalartype_N& operator=(const __scalartype_N& rhs) __CPU_GPU__ { 
    this->data = rhs.data;
    return *this;
  }

  __scalartype_N& operator++() __CPU_GPU__ { 
     set_vector(this->data + static_cast<vector_value_type>(static_cast<value_type>(1))); 
     return *this; 
  }
  __scalartype_N operator++(int) __CPU_GPU__ { 
    __scalartype_N r(*this);
    operator++();
    return r;
  }
  __scalartype_N& operator--() __CPU_GPU__ { 
    set_vector(this->data - static_cast<vector_value_type>(static_cast<value_type>(1))); 
    return *this;
  }
  __scalartype_N operator--(int) __CPU_GPU__ { 
    __scalartype_N r(*this);
    operator--();
    return r;
  }

  __scalartype_N  operator+(const __scalartype_N& rhs) __CPU_GPU__ {
    __scalartype_N r;   
    r.set_vector(this->data+rhs.data);
    return r;
  }
  __scalartype_N& operator+=(const __scalartype_N& rhs) __CPU_GPU__ { 
    set_vector(this->data + rhs.data);
    return *this;
  }

  __scalartype_N& operator-=(const __scalartype_N& rhs) __CPU_GPU__ { 
    set_vector(this->data - rhs.data);
    return *this;
  }
 
  __scalartype_N& operator*=(const __scalartype_N& rhs) __CPU_GPU__ { 
    set_vector(this->data * rhs.data);
    return *this;
  }
 
  __scalartype_N& operator/=(const __scalartype_N& rhs) __CPU_GPU__ { 
    set_vector(this->data / rhs.data);
    return *this;
  }

  __scalartype_N operator-() __CPU_GPU__ {
    static_assert(normIsSigned, "operator- can only support short vector of signed integral or floating-point types.");
    __scalartype_N r;
    r.data = -this->data;
    return r;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==1,value_type>::type >
  bool operator==(const __vector<value_type, 1>& rhs) __CPU_GPU__ { 
    return (this->data.x == rhs.data.x); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==2,value_type>::type >
  bool operator==(const __vector<value_type, 2>& rhs) __CPU_GPU__ { 
    return (this->data.x == rhs.data.x 
         && this->data.y == rhs.data.y); 
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==3,value_type>::type >
  bool operator==(const __vector<value_type, 3>& rhs) __CPU_GPU__ { 
    return   ((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && (this->data.s2 == rhs.data.s2);

  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==4,value_type>::type >
  bool operator==(const __vector<value_type, 4>& rhs) __CPU_GPU__ { 
    return   ((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && ((this->data.s2 == rhs.data.s2) && (this->data.s3 == rhs.data.s3));

  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==8,value_type>::type >
  bool operator==(const __vector<value_type, 8>& rhs) __CPU_GPU__ {
    return    (((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && ((this->data.s2 == rhs.data.s2) && (this->data.s3 == rhs.data.s3)))
            &&  
              (((this->data.s4 == rhs.data.s4) && (this->data.s5 == rhs.data.s5))
              && ((this->data.s6 == rhs.data.s6) && (this->data.s7 == rhs.data.s7)))
              ;
  }

  template <typename T = __scalartype_N
            , class = typename std::enable_if<T::size==16,value_type>::type >
  bool operator==(const __vector<value_type, 16>& rhs) __CPU_GPU__ {

    return (   (((this->data.s0 == rhs.data.s0) && (this->data.s1 == rhs.data.s1))
              && ((this->data.s2 == rhs.data.s2) && (this->data.s3 == rhs.data.s3)))
            &&  
              (((this->data.s4 == rhs.data.s4) && (this->data.s5 == rhs.data.s5))
              && ((this->data.s6 == rhs.data.s6) && (this->data.s7 == rhs.data.s7)))
           )
           &&
           (  (((this->data.s8 == rhs.data.s8) && (this->data.s9 == rhs.data.s9))
              && ((this->data.sA == rhs.data.sA) && (this->data.sB == rhs.data.sB)))
            &&  
              (((this->data.sC == rhs.data.sC) && (this->data.sD == rhs.data.sD))
              && ((this->data.sE == rhs.data.sE) && (this->data.sF == rhs.data.sF)))
           )
           ;
  }

  bool operator!=(const __scalartype_N& rhs) __CPU_GPU__ { return !(*this==rhs); }

private:

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

