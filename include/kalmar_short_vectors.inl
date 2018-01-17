#ifndef _KALMAR_SHORT_VECTORS_H
#define _KALMAR_SHORT_VECTORS_H

class norm;
class unorm;

// Do not rely on macro rescanning and further replacement 

// FIXME: The explicit keyword doesn't work if we define constructor outside
//        the class definition (bug of AST inlining?)

#define NORM_COMMON_PRIVATE_MEMBER(F) \
friend class F; \
float Value; 

// FIXME: C() __CPU_GPU__'s behavior is not specified in Specification
/// C& operator=(const C& other) __CPU_GPU__ do not need to check self-
/// assignment for accerlation on modern CPU
#define NORM_COMMON_PUBLIC_MEMBER(C) \
C() __CPU_GPU__ { set(Value); } \
\
~C() __CPU_GPU__ {} \
\
C(const C& other) __CPU_GPU__ { Value = other.Value; } \
\
C& operator=(const C& other) __CPU_GPU__ \
{ \
  Value = other.Value; \
  return *this; \
} \
\
operator float(void) const __CPU_GPU__ { return Value; } \
\
C& operator+=(const C& other) __CPU_GPU__ \
{ \
  float Res = Value; \
  Res += other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator-=(const C& other) __CPU_GPU__ \
{ \
  float Res = Value; \
  Res -= other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator*=(const C& other) __CPU_GPU__ \
{ \
  float Res = Value; \
  Res *= other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator/=(const C& other) __CPU_GPU__ \
{ \
  float Res = Value; \
  Res /= other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator++() __CPU_GPU__ \
{ \
  float Res = Value; \
  ++Res; \
  set(Res); \
  return *this; \
} \
\
C operator++(int) __CPU_GPU__ \
{ \
  C Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
C& operator--() __CPU_GPU__ \
{ \
  float Res = Value; \
  --Res; \
  set(Res); \
  return *this; \
} \
\
C operator--(int) __CPU_GPU__ \
{ \
  C Ret(*this); \
  operator--(); \
  return Ret; \
}

#if !__HCC_AMP__

#define NORM_CONVERSION_CTOR(C) \
\
explicit C(float v) __CPU_GPU__ { set(v); } \
\
explicit C(unsigned int v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(int v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(double v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(char v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(short v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(long v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(long long int v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(unsigned char v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(unsigned short v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(unsigned long v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(unsigned long long int v) __CPU_GPU__ { set(static_cast<float>(v)); } \

#else

#define NORM_CONVERSION_CTOR(C) \
\
explicit C(float v) __CPU_GPU__ { set(v); } \
\
explicit C(unsigned int v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(int v) __CPU_GPU__ { set(static_cast<float>(v)); } \
\
explicit C(double v) __CPU_GPU__ { set(static_cast<float>(v)); } \

#endif

// C++ AMP Specification 10.7 norm
class norm
{
private:
  void set(float v) __CPU_GPU__
  {
    v = v < -1.0f ? -1.0f : v;
    v = v > 1.0f ? 1.0f : v;
    Value = v;
  }

public:
  NORM_COMMON_PRIVATE_MEMBER(unorm)

public:
  norm(const unorm& other) __CPU_GPU__;

  norm operator-() __CPU_GPU__
  {
    norm Ret;
    Ret.Value = -Value;
    return Ret;
  }

  NORM_COMMON_PUBLIC_MEMBER(norm)

  NORM_CONVERSION_CTOR(norm)
};

// C++ AMP Specification 10.7 unorm
class unorm
{
private:
  void set(float v) __CPU_GPU__
  {
    v = v < 0.0f ? 0.0f : v;
    v = v > 1.0f ? 1.0f : v;
    Value = v;
  }
public:
  NORM_COMMON_PRIVATE_MEMBER(norm)

public:
  explicit unorm(const norm& other) __CPU_GPU__ { set(other.Value); }

  NORM_COMMON_PUBLIC_MEMBER(unorm)

  NORM_CONVERSION_CTOR(unorm)
};

inline norm::norm(const unorm& other) __CPU_GPU__
{
  set(other.Value);
}

#undef NORM_COMMON_PRIVATE_MEMBER
#undef NORM_COMMON_PUBLIC_MEMBER

#define NORM_OPERATOR(C) \
inline C operator+(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return C(static_cast<float>(lhs) + static_cast<float>(rhs)); \
} \
\
inline C operator-(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return C(static_cast<float>(lhs) - static_cast<float>(rhs)); \
} \
\
inline C operator*(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return C(static_cast<float>(lhs) * static_cast<float>(rhs)); \
} \
\
inline C operator/(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return C(static_cast<float>(lhs) / static_cast<float>(rhs)); \
} \
\
inline bool operator==(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return static_cast<float>(lhs) == static_cast<float>(rhs); \
} \
\
inline bool operator!=(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return static_cast<float>(lhs) != static_cast<float>(rhs); \
} \
\
inline bool operator>(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return static_cast<float>(lhs) > static_cast<float>(rhs); \
} \
\
inline bool operator<(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return static_cast<float>(lhs) < static_cast<float>(rhs); \
} \
\
inline bool operator>=(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return static_cast<float>(lhs) >= static_cast<float>(rhs); \
} \
\
inline bool operator<=(const C& lhs, const C& rhs) __CPU_GPU__ \
{ \
  return static_cast<float>(lhs) <= static_cast<float>(rhs); \
}

NORM_OPERATOR(unorm)

NORM_OPERATOR(norm)

#undef NORM_OPERATOR

#define UNORM_MIN ((unorm)0.0f)
#define UNORM_MAX ((unorm)1.0f)
#define UNORM_ZERO ((norm)0.0f)
#define NORM_ZERO ((norm)0.0f)
#define NORM_MIN ((norm)-1.0f)
#define NORM_MAX ((norm)1.0f)

// C++ AMP Specification 10.8 short vector types

// How to Define short vector types (Layout):
//   Class Declaration (10.8.1 Synopsis)
//   Explicit Conversion Constructor Definitions (10.8.2.2)
//   Operators between Two References (10.8.1 Synopsis)
//
// Class Declaration:
//   class scalartype_N
//   {
//   private:
//     SCALARTYPE_N_COMMON_PRIVATE_MEMBER
//
//   public:
//     SCALARTYPE_N_COMMON_PUBLIC_MEMBER
//     /* scalartype_N specific public member */
//     SINGLE_COMPONENT_ACCESS
//     SCALARTYPE_N_REFERENCE_SINGLE_COMPONENT_ACCESS
//     TWO_COMPONENT_ACCESS
//     THREE_COMPONENT_ACCESS
//     FOUR_COMPONENT_ACCESS
//   };
//
// Operators between Two References:
//   SCALARTYPE_N_OPERATOR
//   /* scalartype_N specific operator */

class int_2;
class int_3;
class int_4;
class uint_2;
class uint_3;
class uint_4;
class float_2;
class float_3;
class float_4;
class double_2;
class double_3;
class double_4;
class norm_2;
class norm_3;
class norm_4;
class unorm_2;
class unorm_3;
class unorm_4;

#if !__HCC_AMP__
// additional short vector types not specified in C++AMP
class int_1;
class uint_1;
class float_1;
class double_1;
class char_1;
class char_2;
class char_3;
class char_4;
class uchar_1;
class uchar_2;
class uchar_3;
class uchar_4;
class short_1;
class short_2;
class short_3;
class short_4;
class ushort_1;
class ushort_2;
class ushort_3;
class ushort_4;
class long_1;
class long_2;
class long_3;
class long_4;
class ulong_1;
class ulong_2;
class ulong_3;
class ulong_4;
class longlong_1;
class longlong_2;
class longlong_3;
class longlong_4;
class ulonglong_1;
class ulonglong_2;
class ulonglong_3;
class ulonglong_4;
#endif

typedef int_2 int2;
typedef int_3 int3;
typedef int_4 int4;
typedef uint_2 uint2;
typedef uint_3 uint3;
typedef uint_4 uint4;
typedef float_2 float2;
typedef float_3 float3;
typedef float_4 float4;
typedef double_2 double2;
typedef double_3 double3;
typedef double_4 double4;
typedef norm_2 norm2;
typedef norm_3 norm3;
typedef norm_4 norm4;
typedef unorm_2 unorm2;
typedef unorm_3 unorm3;
typedef unorm_4 unorm4;

#if !__HCC_AMP__
// additional short vector types not specified in C++AMP
typedef int_1 int1;
typedef uint_1 uint1;
typedef float_1 float1;
typedef double_1 double1;
typedef char_1 char1;
typedef char_2 char2;
typedef char_3 char3;
typedef char_4 char4;
typedef uchar_1 uchar1;
typedef uchar_2 uchar2;
typedef uchar_3 uchar3;
typedef uchar_4 uchar4;
typedef short_1 short1;
typedef short_2 short2;
typedef short_3 short3;
typedef short_4 short4;
typedef ushort_1 ushort1;
typedef ushort_2 ushort2;
typedef ushort_3 ushort3;
typedef ushort_4 ushort4;
typedef long_1 long1;
typedef long_2 long2;
typedef long_3 long3;
typedef long_4 long4;
typedef ulong_1 ulong1;
typedef ulong_2 ulong2;
typedef ulong_3 ulong3;
typedef ulong_4 ulong4;
typedef longlong_1 longlong1;
typedef longlong_2 longlong2;
typedef longlong_3 longlong3;
typedef longlong_4 longlong4;
typedef ulonglong_1 ulonglong1;
typedef ulonglong_2 ulonglong2;
typedef ulonglong_3 ulonglong3;
typedef ulonglong_4 ulonglong4;
#endif

//   Class Declaration (10.8.1 Synopsis)

#define SINGLE_COMPONENT_ACCESS(ST, Dim) \
ST get ## _ ## Dim() const __CPU_GPU__ { return Dim; } \
\
void set ## _ ## Dim(ST v) __CPU_GPU__ { Dim = v; }

#define TWO_COMPONENT_ACCESS(ST_2, Dim1, Dim2) \
ST_2 get_ ## Dim1 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_2(Dim1, Dim2); \
} \
\
ST_2 get_ ## Dim2 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_2(Dim2, Dim1); \
} \
\
void set_ ## Dim1 ## Dim2(ST_2 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
} \
void set_ ## Dim2 ## Dim1(ST_2 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
}

#define THREE_COMPONENT_ACCESS(ST_3, Dim1, Dim2, Dim3) \
ST_3 get_ ## Dim1 ## Dim2 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_3(Dim1, Dim2, Dim3); \
} \
\
ST_3 get_ ## Dim1 ## Dim3 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_3(Dim1, Dim3, Dim2); \
} \
\
ST_3 get_ ## Dim2 ## Dim1 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_3(Dim2, Dim1, Dim3); \
} \
\
ST_3 get_ ## Dim2 ## Dim3 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_3(Dim2, Dim3, Dim1); \
} \
\
ST_3 get_ ## Dim3 ## Dim1 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_3(Dim3, Dim1, Dim2); \
} \
\
ST_3 get_ ## Dim3 ## Dim2 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_3(Dim3, Dim2, Dim1); \
} \
\
void set_ ## Dim1 ## Dim2 ## Dim3(ST_3 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim3 = v.get_z(); \
} \
\
void set_ ## Dim1 ## Dim3 ## Dim2(ST_3 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim2 = v.get_z(); \
} \
\
void set_ ## Dim2 ## Dim1 ## Dim3(ST_3 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim3 = v.get_z(); \
} \
\
void set_ ## Dim2 ## Dim3 ## Dim1(ST_3 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim1 = v.get_z(); \
} \
\
void set_ ## Dim3 ## Dim1 ## Dim2(ST_3 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim2 = v.get_z(); \
} \
\
void set_ ## Dim3 ## Dim2 ## Dim1(ST_3 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim1 = v.get_z(); \
}

#define FOUR_COMPONENT_ACCESS(ST_4, Dim1, Dim2, Dim3, Dim4) \
ST_4 get_ ## Dim1 ## Dim2 ## Dim3 ## Dim4() const __CPU_GPU__ \
{ \
  return ST_4(Dim1, Dim2, Dim3, Dim4); \
} \
\
ST_4 get_ ## Dim1 ## Dim2 ## Dim4 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_4(Dim1, Dim2, Dim4, Dim3); \
} \
\
ST_4 get_ ## Dim1 ## Dim3 ## Dim2 ## Dim4() const __CPU_GPU__ \
{ \
  return ST_4(Dim1, Dim3, Dim2, Dim4); \
} \
\
ST_4 get_ ## Dim1 ## Dim3 ## Dim4 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_4(Dim1, Dim3, Dim4, Dim2); \
} \
\
ST_4 get_ ## Dim1 ## Dim4 ## Dim2 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_4(Dim1, Dim4, Dim2, Dim3); \
} \
\
ST_4 get_ ## Dim1 ## Dim4 ## Dim3 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_4(Dim1, Dim4, Dim3, Dim2); \
} \
\
ST_4 get_ ## Dim2 ## Dim1 ## Dim3 ## Dim4() const __CPU_GPU__ \
{ \
  return ST_4(Dim2, Dim1, Dim3, Dim4); \
} \
\
ST_4 get_ ## Dim2 ## Dim1 ## Dim4 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_4(Dim2, Dim1, Dim4, Dim3); \
} \
\
ST_4 get_ ## Dim2 ## Dim3 ## Dim1 ## Dim4() const __CPU_GPU__ \
{ \
  return ST_4(Dim2, Dim3, Dim1, Dim4); \
} \
\
ST_4 get_ ## Dim2 ## Dim3 ## Dim4 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_4(Dim2, Dim3, Dim4, Dim1); \
} \
\
ST_4 get_ ## Dim2 ## Dim4 ## Dim1 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_4(Dim2, Dim4, Dim1, Dim3); \
} \
\
ST_4 get_ ## Dim2 ## Dim4 ## Dim3 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_4(Dim2, Dim4, Dim3, Dim1); \
} \
\
ST_4 get_ ## Dim3 ## Dim1 ## Dim2 ## Dim4() const __CPU_GPU__ \
{ \
  return ST_4(Dim3, Dim1, Dim2, Dim4); \
} \
\
ST_4 get_ ## Dim3 ## Dim1 ## Dim4 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_4(Dim3, Dim1, Dim4, Dim2); \
} \
\
ST_4 get_ ## Dim3 ## Dim2 ## Dim1 ## Dim4() const __CPU_GPU__ \
{ \
  return ST_4(Dim3, Dim2, Dim1, Dim4); \
} \
\
ST_4 get_ ## Dim3 ## Dim2 ## Dim4 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_4(Dim3, Dim2, Dim4, Dim1); \
} \
\
ST_4 get_ ## Dim3 ## Dim4 ## Dim1 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_4(Dim3, Dim4, Dim1, Dim2); \
} \
\
ST_4 get_ ## Dim3 ## Dim4 ## Dim2 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_4(Dim3, Dim4, Dim2, Dim1); \
} \
\
ST_4 get_ ## Dim4 ## Dim1 ## Dim2 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_4(Dim4, Dim1, Dim2, Dim3); \
} \
\
ST_4 get_ ## Dim4 ## Dim1 ## Dim3 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_4(Dim4, Dim1, Dim3, Dim2); \
} \
\
ST_4 get_ ## Dim4 ## Dim2 ## Dim1 ## Dim3() const __CPU_GPU__ \
{ \
  return ST_4(Dim4, Dim2, Dim1, Dim3); \
} \
\
ST_4 get_ ## Dim4 ## Dim2 ## Dim3 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_4(Dim4, Dim2, Dim3, Dim1); \
} \
\
ST_4 get_ ## Dim4 ## Dim3 ## Dim1 ## Dim2() const __CPU_GPU__ \
{ \
  return ST_4(Dim4, Dim3, Dim1, Dim2); \
} \
\
ST_4 get_ ## Dim4 ## Dim3 ## Dim2 ## Dim1() const __CPU_GPU__ \
{ \
  return ST_4(Dim4, Dim3, Dim2, Dim1); \
} \
\
void set_ ## Dim1 ## Dim2 ## Dim3 ## Dim4(ST_4 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim2 ## Dim4 ## Dim3(ST_4 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim3 ## Dim2 ## Dim4(ST_4 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim3 ## Dim4 ## Dim2(ST_4 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim4 ## Dim2 ## Dim3(ST_4 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim4 ## Dim3 ## Dim2(ST_4 v) __CPU_GPU__ \
{ \
  Dim1 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim1 ## Dim3 ## Dim4(ST_4 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim1 ## Dim4 ## Dim3(ST_4 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim3 ## Dim1 ## Dim4(ST_4 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim3 ## Dim4 ## Dim1(ST_4 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim4 ## Dim1 ## Dim3(ST_4 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim4 ## Dim3 ## Dim1(ST_4 v) __CPU_GPU__ \
{ \
  Dim2 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim1 ## Dim2 ## Dim4(ST_4 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim1 ## Dim4 ## Dim2(ST_4 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim2 ## Dim1 ## Dim4(ST_4 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim2 ## Dim4 ## Dim1(ST_4 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim4 ## Dim1 ## Dim2(ST_4 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim4 ## Dim2 ## Dim1(ST_4 v) __CPU_GPU__ \
{ \
  Dim3 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim1 ## Dim2 ## Dim3(ST_4 v) __CPU_GPU__ \
{ \
  Dim4 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim1 ## Dim3 ## Dim2(ST_4 v) __CPU_GPU__ \
{ \
  Dim4 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim2 ## Dim1 ## Dim3(ST_4 v) __CPU_GPU__ \
{ \
  Dim4 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim2 ## Dim3 ## Dim1(ST_4 v) __CPU_GPU__ \
{ \
  Dim4 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim3 ## Dim1 ## Dim2(ST_4 v) __CPU_GPU__ \
{ \
  Dim4 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim3 ## Dim2 ## Dim1(ST_4 v) __CPU_GPU__ \
{ \
  Dim4 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim1 = v.get_w(); \
}

#define SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() __CPU_GPU__ { return x; } \
\
ST& ref_r() __CPU_GPU__ { return x; }

#define SCALARTYPE_1_COMMON_PUBLIC_MEMBER(ST, ST_1) \
ST x; \
typedef ST value_type; \
static const int size = 1; \
\
ST_1() __CPU_GPU__ {} \
\
~ST_1() __CPU_GPU__ {} \
\
ST_1(ST value) __CPU_GPU__ \
{ \
  x = value; \
} \
\
ST_1(const ST_1&  other) __CPU_GPU__ \
{ \
  x = other.x; \
} \
\
ST_1& operator=(const ST_1& other) __CPU_GPU__ \
{ \
  x = other.x; \
  return *this; \
} \
\
ST_1& operator++() __CPU_GPU__ \
{ \
  ++x; \
  return *this; \
} \
\
ST_1 operator++(int) __CPU_GPU__ \
{ \
  ST_1 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_1& operator--() __CPU_GPU__ \
{ \
  --x; \
  return *this; \
} \
\
ST_1 operator--(int) __CPU_GPU__ \
{ \
  ST_1 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_1& operator+=(const ST_1& rhs) __CPU_GPU__ \
{ \
  x += rhs.x; \
  return *this; \
} \
\
ST_1& operator-=(const ST_1& rhs) __CPU_GPU__ \
{ \
  x -= rhs.x; \
  return *this; \
} \
\
ST_1& operator*=(const ST_1& rhs) __CPU_GPU__ \
{ \
  x *= rhs.x; \
  return *this; \
} \
\
ST_1& operator/=(const ST_1& rhs) __CPU_GPU__ \
{ \
  x /= rhs.x; \
  return *this; \
}

#if !__HCC_AMP__

#define SCALARTYPE_1_CONVERSION_CTOR(ST_1, \
ST_1_o1, ST_1_o2, ST_1_o3, ST_1_o4, ST_1_o5, \
ST_1_o6, ST_1_o7, ST_1_o8, ST_1_o9, ST_1_o10, ST_1_o11) \
\
explicit ST_1(const ST_1_o1& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o2& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o3& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o4& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o5& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o6& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o7& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o8& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o9& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o10& other) __CPU_GPU__; \
\
explicit ST_1(const ST_1_o11& other) __CPU_GPU__;

#endif // if !__HCC_AMP__

#if !__HCC_AMP__
class int_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(int, int_1)

  SCALARTYPE_1_CONVERSION_CTOR(int_1,
    uint_1, float_1, double_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  int_1 operator-() const __CPU_GPU__ { return int_1(-x); }

  int_1 operator~() const __CPU_GPU__ { return int_1(~x); }

  int_1& operator%=(const int_1& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    return *this;
  }

  int_1& operator^=(const int_1& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    return *this;
  }

  int_1& operator|=(const int_1& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    return *this;
  }

  int_1& operator&=(const int_1& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    return *this;
  }

  int_1& operator>>=(const int_1& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    return *this;
  }

  int_1& operator<<=(const int_1& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    return *this;
  }
  
  SINGLE_COMPONENT_ACCESS(int, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(int)

};

class uint_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(unsigned int, uint_1)

  SCALARTYPE_1_CONVERSION_CTOR(uint_1,
    int_1, float_1, double_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  uint_1 operator~() const __CPU_GPU__ { return uint_1(~x); }

  uint_1& operator%=(const uint_1& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    return *this;
  }

  uint_1& operator^=(const uint_1& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    return *this;
  }

  uint_1& operator|=(const uint_1& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    return *this;
  }

  uint_1& operator&=(const uint_1& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    return *this;
  }

  uint_1& operator>>=(const uint_1& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    return *this;
  }

  uint_1& operator<<=(const uint_1& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    return *this;
  }
 
  SINGLE_COMPONENT_ACCESS(unsigned int, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned int)

};

class float_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(float, float_1)

  SCALARTYPE_1_CONVERSION_CTOR(float_1,
    int_1, uint_1, double_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  float_1 operator-() const __CPU_GPU__ { return float_1(-x); }

  SINGLE_COMPONENT_ACCESS(float, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(float)

};

class double_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(double, double_1)

  SCALARTYPE_1_CONVERSION_CTOR(double_1,
    int_1, uint_1, float_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  double_1 operator-() const __CPU_GPU__ { return double_1(-x); }

  SINGLE_COMPONENT_ACCESS(double, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(double)
};

class char_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(char, char_1)

  SCALARTYPE_1_CONVERSION_CTOR(char_1,
    int_1, uint_1, float_1,
    double_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  char_1 operator-() const __CPU_GPU__ { return char_1(-x); }

  SINGLE_COMPONENT_ACCESS(char, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(char)
};

class uchar_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(unsigned char, uchar_1)

  SCALARTYPE_1_CONVERSION_CTOR(uchar_1,
    int_1, uint_1, float_1,
    double_1, char_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  SINGLE_COMPONENT_ACCESS(unsigned char, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned char)
};

class short_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(short, short_1)

  SCALARTYPE_1_CONVERSION_CTOR(short_1,
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

  short_1 operator-() const __CPU_GPU__ { return short_1(-x); }

  SINGLE_COMPONENT_ACCESS(short, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(short)
};

class ushort_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(unsigned short, ushort_1)

  SCALARTYPE_1_CONVERSION_CTOR(ushort_1,
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, long_1, ulong_1, longlong_1, ulonglong_1)
    
  SINGLE_COMPONENT_ACCESS(unsigned short, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned short)
};

class long_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(long, long_1)

  SCALARTYPE_1_CONVERSION_CTOR(long_1,
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, ulong_1, longlong_1, ulonglong_1)
    
  long_1 operator-() const __CPU_GPU__ { return long_1(-x); }

  SINGLE_COMPONENT_ACCESS(long, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(long)
};

class ulong_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(unsigned long, ulong_1)

  SCALARTYPE_1_CONVERSION_CTOR(ulong_1,
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, long_1, longlong_1, ulonglong_1)
    
  SINGLE_COMPONENT_ACCESS(unsigned long, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long)
};

class longlong_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(long long int, longlong_1)

  SCALARTYPE_1_CONVERSION_CTOR(longlong_1,
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, ulonglong_1)
    
  longlong_1 operator-() const __CPU_GPU__ { return longlong_1(-x); }

  SINGLE_COMPONENT_ACCESS(long long int, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(long long int)
};

class ulonglong_1
{
public:
  SCALARTYPE_1_COMMON_PUBLIC_MEMBER(unsigned long long int, ulonglong_1)

  SCALARTYPE_1_CONVERSION_CTOR(ulonglong_1,
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1)
    
  SINGLE_COMPONENT_ACCESS(unsigned long long int, x)

  SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long long int)
};

#endif // if !__HCC_AMP__

#undef SCALARTYPE_1_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_1_COMMON_PUBLIC_MEMBER

#define SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() __CPU_GPU__ { return x; } \
\
ST& ref_y() __CPU_GPU__ { return y; } \
\
ST& ref_r() __CPU_GPU__ { return x; } \
\
ST& ref_g() __CPU_GPU__ { return y; }

#define SCALARTYPE_2_COMMON_PUBLIC_MEMBER(ST, ST_2) \
ST x; \
ST y; \
typedef ST value_type; \
static const int size = 2; \
\
ST_2() __CPU_GPU__ {} \
\
~ST_2() __CPU_GPU__ {} \
\
ST_2(ST value) __CPU_GPU__ \
{ \
  x = value; \
  y = value; \
} \
\
ST_2(const ST_2&  other) __CPU_GPU__ \
{ \
  x = other.x; \
  y = other.y; \
} \
\
ST_2(ST v1, ST v2) __CPU_GPU__ \
{ \
  x = v1; \
  y = v2; \
} \
\
ST_2& operator=(const ST_2& other) __CPU_GPU__ \
{ \
  x = other.x; \
  y = other.y; \
  return *this; \
} \
\
ST_2& operator++() __CPU_GPU__ \
{ \
  ++x; \
  ++y; \
  return *this; \
} \
\
ST_2 operator++(int) __CPU_GPU__ \
{ \
  ST_2 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_2& operator--() __CPU_GPU__ \
{ \
  --x; \
  --y; \
  return *this; \
} \
\
ST_2 operator--(int) __CPU_GPU__ \
{ \
  ST_2 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_2& operator+=(const ST_2& rhs) __CPU_GPU__ \
{ \
  x += rhs.x; \
  y += rhs.y; \
  return *this; \
} \
\
ST_2& operator-=(const ST_2& rhs) __CPU_GPU__ \
{ \
  x -= rhs.x; \
  y -= rhs.y; \
  return *this; \
} \
\
ST_2& operator*=(const ST_2& rhs) __CPU_GPU__ \
{ \
  x *= rhs.x; \
  y *= rhs.y; \
  return *this; \
} \
\
ST_2& operator/=(const ST_2& rhs) __CPU_GPU__ \
{ \
  x /= rhs.x; \
  y /= rhs.y; \
  return *this; \
}

#if !__HCC_AMP__

#define SCALARTYPE_2_CONVERSION_CTOR(ST_2, \
ST_2_o1, ST_2_o2, ST_2_o3, ST_2_o4, ST_2_o5, \
ST_2_o6, ST_2_o7, ST_2_o8, ST_2_o9, ST_2_o10, ST_2_o11, ST_2_o12, ST_2_o13) \
\
explicit ST_2(const ST_2_o1& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o2& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o3& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o4& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o5& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o6& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o7& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o8& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o9& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o10& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o11& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o12& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o13& other) __CPU_GPU__;

#else

#define SCALARTYPE_2_CONVERSION_CTOR(ST_2, \
ST_2_o1, ST_2_o2, ST_2_o3, ST_2_o4, ST_2_o5) \
\
explicit ST_2(const ST_2_o1& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o2& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o3& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o4& other) __CPU_GPU__; \
\
explicit ST_2(const ST_2_o5& other) __CPU_GPU__;

#endif // if !__HCC_AMP__

class int_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(int, int_2)

#if !__HCC_AMP__
  SCALARTYPE_2_CONVERSION_CTOR(int_2,
    uint_2, float_2, double_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
#else
  SCALARTYPE_2_CONVERSION_CTOR(int_2,
    uint_2, float_2, double_2, norm_2, unorm_2)
#endif

  int_2 operator-() const __CPU_GPU__ { return int_2(-x, -y); }

  int_2 operator~() const __CPU_GPU__ { return int_2(~x, ~y); }

  int_2& operator%=(const int_2& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    y %= rhs.y;
    return *this;
  }

  int_2& operator^=(const int_2& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    y ^= rhs.y;
    return *this;
  }

  int_2& operator|=(const int_2& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    y |= rhs.y;
    return *this;
  }

  int_2& operator&=(const int_2& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    y &= rhs.y;
    return *this;
  }

  int_2& operator>>=(const int_2& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    y >>= rhs.y;
    return *this;
  }

  int_2& operator<<=(const int_2& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    y <<= rhs.y;
    return *this;
  }
  
  SINGLE_COMPONENT_ACCESS(int, x)
  SINGLE_COMPONENT_ACCESS(int, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(int)

  TWO_COMPONENT_ACCESS(int_2, x, y)
};

class uint_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unsigned int, uint_2)

#if !__HCC_AMP__
  SCALARTYPE_2_CONVERSION_CTOR(uint_2,
    int_2, float_2, double_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
#else
  SCALARTYPE_2_CONVERSION_CTOR(uint_2,
    int_2, float_2, double_2, norm_2, unorm_2)
#endif
 
  uint_2 operator~() const __CPU_GPU__ { return uint_2(~x, ~y); }

  uint_2& operator%=(const uint_2& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    y %= rhs.y;
    return *this;
  }

  uint_2& operator^=(const uint_2& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    y ^= rhs.y;
    return *this;
  }

  uint_2& operator|=(const uint_2& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    y |= rhs.y;
    return *this;
  }

  uint_2& operator&=(const uint_2& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    y &= rhs.y;
    return *this;
  }

  uint_2& operator>>=(const uint_2& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    y >>= rhs.y;
    return *this;
  }

  uint_2& operator<<=(const uint_2& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    y <<= rhs.y;
    return *this;
  }
 
  SINGLE_COMPONENT_ACCESS(unsigned int, x)
  SINGLE_COMPONENT_ACCESS(unsigned int, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned int)

  TWO_COMPONENT_ACCESS(uint_2, x, y)
};

class float_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(float, float_2)

#if !__HCC_AMP__
  SCALARTYPE_2_CONVERSION_CTOR(float_2,
    int_2, uint_2, double_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
#else
  SCALARTYPE_2_CONVERSION_CTOR(float_2,
    int_2, uint_2, double_2, norm_2, unorm_2)
#endif
  
  float_2 operator-() const __CPU_GPU__ { return float_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(float, x)
  SINGLE_COMPONENT_ACCESS(float, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(float)

  TWO_COMPONENT_ACCESS(float_2, x, y)
};

class double_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(double, double_2)

#if !__HCC_AMP__
  SCALARTYPE_2_CONVERSION_CTOR(double_2,
    int_2, uint_2, float_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
#else
  SCALARTYPE_2_CONVERSION_CTOR(double_2,
    int_2, uint_2, float_2, norm_2, unorm_2)
#endif
  
  double_2 operator-() const __CPU_GPU__ { return double_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(double, x)
  SINGLE_COMPONENT_ACCESS(double, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(double)

  TWO_COMPONENT_ACCESS(double_2, x, y)
};

class norm_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(norm, norm_2)

#if !__HCC_AMP__
  SCALARTYPE_2_CONVERSION_CTOR(norm_2,
    int_2, uint_2, float_2, double_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
#else
  SCALARTYPE_2_CONVERSION_CTOR(norm_2,
    int_2, uint_2, float_2, double_2, unorm_2)
#endif

#if __GNUG__
  // for some reason g++ will mistakenly treat x, y as type float
  // so we need to explicitly cast them to norm type here
  norm_2 operator-() const __CPU_GPU__ { return norm2(-(norm)x, -(norm)y); }
#else
  norm_2 operator-() const __CPU_GPU__ { return norm_2(-x, -y); }
#endif
  
  SINGLE_COMPONENT_ACCESS(norm, x)
  SINGLE_COMPONENT_ACCESS(norm, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(norm)

  TWO_COMPONENT_ACCESS(norm_2, x, y)
};

class unorm_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unorm, unorm_2)

#if !__HCC_AMP__
  SCALARTYPE_2_CONVERSION_CTOR(unorm_2,
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
#else
  SCALARTYPE_2_CONVERSION_CTOR(unorm_2,
    int_2, uint_2, float_2, double_2, norm_2)
#endif

  SINGLE_COMPONENT_ACCESS(unorm, x)
  SINGLE_COMPONENT_ACCESS(unorm, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unorm)

  TWO_COMPONENT_ACCESS(unorm_2, x, y)
};

// additional types not specified in C++AMP
#if !__HCC_AMP__
class char_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(char, char_2)

  SCALARTYPE_2_CONVERSION_CTOR(char_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
    
  char_2 operator-() const __CPU_GPU__ { return char_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(char, x)
  SINGLE_COMPONENT_ACCESS(char, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(char)

  TWO_COMPONENT_ACCESS(char_2, x, y)
};

class uchar_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unsigned char, uchar_2)

  SCALARTYPE_2_CONVERSION_CTOR(uchar_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, char_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
    
  SINGLE_COMPONENT_ACCESS(unsigned char, x)
  SINGLE_COMPONENT_ACCESS(unsigned char, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned char)

  TWO_COMPONENT_ACCESS(uchar_2, x, y)
};

class short_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(short, short_2)

  SCALARTYPE_2_CONVERSION_CTOR(short_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, uchar_2, char_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)
    
  short_2 operator-() const __CPU_GPU__ { return short_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(short, x)
  SINGLE_COMPONENT_ACCESS(short, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(short)

  TWO_COMPONENT_ACCESS(short_2, x, y)
};

class ushort_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unsigned short, ushort_2)

  SCALARTYPE_2_CONVERSION_CTOR(ushort_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, char_2, short_2, uchar_2, long_2, ulong_2, longlong_2, ulonglong_2)
    
  SINGLE_COMPONENT_ACCESS(unsigned short, x)
  SINGLE_COMPONENT_ACCESS(unsigned short, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned short)

  TWO_COMPONENT_ACCESS(ushort_2, x, y)
};

class long_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(long, long_2)

  SCALARTYPE_2_CONVERSION_CTOR(long_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, uchar_2, char_2, ushort_2, short_2, ulong_2, longlong_2, ulonglong_2)
    
  long_2 operator-() const __CPU_GPU__ { return long_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(long, x)
  SINGLE_COMPONENT_ACCESS(long, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(long)

  TWO_COMPONENT_ACCESS(long_2, x, y)
};

class ulong_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unsigned long, ulong_2)

  SCALARTYPE_2_CONVERSION_CTOR(ulong_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, char_2, short_2, uchar_2, long_2, ushort_2, longlong_2, ulonglong_2)
    
  SINGLE_COMPONENT_ACCESS(unsigned long, x)
  SINGLE_COMPONENT_ACCESS(unsigned long, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long)

  TWO_COMPONENT_ACCESS(ulong_2, x, y)
};

class longlong_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(long long int, longlong_2)

  SCALARTYPE_2_CONVERSION_CTOR(longlong_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, uchar_2, char_2, ushort_2, short_2, ulong_2, long_2, ulonglong_2)
    
  longlong_2 operator-() const __CPU_GPU__ { return longlong_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(long long int, x)
  SINGLE_COMPONENT_ACCESS(long long int, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(long long int)

  TWO_COMPONENT_ACCESS(longlong_2, x, y)
};

class ulonglong_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unsigned long long int, ulonglong_2)

  SCALARTYPE_2_CONVERSION_CTOR(ulonglong_2,
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, char_2, short_2, uchar_2, long_2, ushort_2, longlong_2, ulong_2)
    
  SINGLE_COMPONENT_ACCESS(unsigned long long int, x)
  SINGLE_COMPONENT_ACCESS(unsigned long long int, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long long int)

  TWO_COMPONENT_ACCESS(ulonglong_2, x, y)
};

#endif // if !__HCC_AMP__

#undef SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_2_COMMON_PUBLIC_MEMBER

#define SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() __CPU_GPU__ { return x; } \
\
ST& ref_y() __CPU_GPU__ { return y; } \
\
ST& ref_z() __CPU_GPU__ { return z; } \
\
ST& ref_r() __CPU_GPU__ { return x; } \
\
ST& ref_g() __CPU_GPU__ { return y; } \
\
ST& ref_b() __CPU_GPU__ { return z; }

#define SCALARTYPE_3_COMMON_PUBLIC_MEMBER(ST, ST_3) \
ST x; \
ST y; \
ST z; \
typedef ST value_type; \
static const int size = 3; \
\
ST_3() __CPU_GPU__ {} \
\
~ST_3() __CPU_GPU__ {} \
\
ST_3(ST value) __CPU_GPU__ \
{ \
  x = value; \
  y = value; \
  z = value; \
} \
\
ST_3(const ST_3&  other) __CPU_GPU__ \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
} \
\
ST_3(ST v1, ST v2, ST v3) __CPU_GPU__ \
{ \
  x = v1; \
  y = v2; \
  z = v3; \
} \
\
ST_3& operator=(const ST_3& other) __CPU_GPU__ \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
  return *this; \
} \
\
ST_3& operator++() __CPU_GPU__ \
{ \
  ++x; \
  ++y; \
  ++z; \
  return *this; \
} \
\
ST_3 operator++(int) __CPU_GPU__ \
{ \
  ST_3 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_3& operator--() __CPU_GPU__ \
{ \
  --x; \
  --y; \
  --z; \
  return *this; \
} \
\
ST_3 operator--(int) __CPU_GPU__ \
{ \
  ST_3 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_3& operator+=(const ST_3& rhs) __CPU_GPU__ \
{ \
  x += rhs.x; \
  y += rhs.y; \
  z += rhs.z; \
  return *this; \
} \
\
ST_3& operator-=(const ST_3& rhs) __CPU_GPU__ \
{ \
  x -= rhs.x; \
  y -= rhs.y; \
  z -= rhs.z; \
  return *this; \
} \
\
ST_3& operator*=(const ST_3& rhs) __CPU_GPU__ \
{ \
  x *= rhs.x; \
  y *= rhs.y; \
  z *= rhs.z; \
  return *this; \
} \
\
ST_3& operator/=(const ST_3& rhs) __CPU_GPU__ \
{ \
  x /= rhs.x; \
  y /= rhs.y; \
  z /= rhs.z; \
  return *this; \
}

#if !__HCC_AMP__

#define SCALARTYPE_3_CONVERSION_CTOR(ST_3, \
ST_3_o1, ST_3_o2, ST_3_o3, ST_3_o4, ST_3_o5, \
ST_3_o6, ST_3_o7, ST_3_o8, ST_3_o9, ST_3_o10, ST_3_o11, ST_3_o12, ST_3_o13) \
\
explicit ST_3(const ST_3_o1& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o2& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o3& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o4& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o5& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o6& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o7& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o8& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o9& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o10& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o11& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o12& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o13& other) __CPU_GPU__;

#else

#define SCALARTYPE_3_CONVERSION_CTOR(ST_3, \
ST_3_o1, ST_3_o2, ST_3_o3, ST_3_o4, ST_3_o5) \
\
explicit ST_3(const ST_3_o1& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o2& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o3& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o4& other) __CPU_GPU__; \
\
explicit ST_3(const ST_3_o5& other) __CPU_GPU__;

#endif // if !__HCC_AMP__


class int_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(int, int_3)

#if !__HCC_AMP__
  SCALARTYPE_3_CONVERSION_CTOR(int_3,
    uint_3, float_3, double_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)
#else
  SCALARTYPE_3_CONVERSION_CTOR(int_3,
    uint_3, float_3, double_3, norm_3, unorm_3)
#endif

  int_3 operator-() const __CPU_GPU__ { return int_3(-x, -y, -z); }

  int_3 operator~() const __CPU_GPU__ { return int_3(~x, ~y, -z); }

  int_3& operator%=(const int_3& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    return *this;
  }

  int_3& operator^=(const int_3& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    return *this;
  }

  int_3& operator|=(const int_3& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    return *this;
  }

  int_3& operator&=(const int_3& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    return *this;
  }

  int_3& operator>>=(const int_3& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    return *this;
  }

  int_3& operator<<=(const int_3& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    y <<= rhs.y;
    z <<= rhs.z;
    return *this;
  }
  
  SINGLE_COMPONENT_ACCESS(int, x)
  SINGLE_COMPONENT_ACCESS(int, y)
  SINGLE_COMPONENT_ACCESS(int, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(int)

  TWO_COMPONENT_ACCESS(int_2, x, y)
  TWO_COMPONENT_ACCESS(int_2, x, z)
  TWO_COMPONENT_ACCESS(int_2, y, z)

  THREE_COMPONENT_ACCESS(int_3, x, y, z)
};

class uint_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unsigned int, uint_3)

#if !__HCC_AMP__
  SCALARTYPE_3_CONVERSION_CTOR(uint_3,
    int_3, float_3, double_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)
#else
  SCALARTYPE_3_CONVERSION_CTOR(uint_3,
    int_3, float_3, double_3, norm_3, unorm_3)
#endif
 
  uint_3 operator~() const __CPU_GPU__ { return uint_3(~x, ~y, ~z); }

  uint_3& operator%=(const uint_3& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    return *this;
  }

  uint_3& operator^=(const uint_3& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    return *this;
  }

  uint_3& operator|=(const uint_3& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    return *this;
  }

  uint_3& operator&=(const uint_3& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    return *this;
  }

  uint_3& operator>>=(const uint_3& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    return *this;
  }

  uint_3& operator<<=(const uint_3& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    y <<= rhs.y;
    z <<= rhs.z;
    return *this;
  }
 
  SINGLE_COMPONENT_ACCESS(unsigned int, x)
  SINGLE_COMPONENT_ACCESS(unsigned int, y)
  SINGLE_COMPONENT_ACCESS(unsigned int, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned int)

  TWO_COMPONENT_ACCESS(uint_2, x, y)
  TWO_COMPONENT_ACCESS(uint_2, x, z)
  TWO_COMPONENT_ACCESS(uint_2, y, z)

  THREE_COMPONENT_ACCESS(uint_3, x, y, z)
};

class float_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(float, float_3)

#if !__HCC_AMP__
  SCALARTYPE_3_CONVERSION_CTOR(float_3,
    int_3, uint_3, double_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)
#else
  SCALARTYPE_3_CONVERSION_CTOR(float_3,
    int_3, uint_3, double_3, norm_3, unorm_3)
#endif
  
  float_3 operator-() const __CPU_GPU__ { return float_3(-x, -y, -z); }

  SINGLE_COMPONENT_ACCESS(float, x)
  SINGLE_COMPONENT_ACCESS(float, y)
  SINGLE_COMPONENT_ACCESS(float, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(float)

  TWO_COMPONENT_ACCESS(float_2, x, y)
  TWO_COMPONENT_ACCESS(float_2, x, z)
  TWO_COMPONENT_ACCESS(float_2, y, z)

  THREE_COMPONENT_ACCESS(float_3, x, y, z)
};

class double_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(double, double_3)

#if !__HCC_AMP__
  SCALARTYPE_3_CONVERSION_CTOR(double_3,
    int_3, uint_3, float_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)
#else
  SCALARTYPE_3_CONVERSION_CTOR(double_3,
    int_3, uint_3, float_3, norm_3, unorm_3)
#endif
  
  double_3 operator-() const __CPU_GPU__ { return double_3(-x, -y, -z); }

  SINGLE_COMPONENT_ACCESS(double, x)
  SINGLE_COMPONENT_ACCESS(double, y)
  SINGLE_COMPONENT_ACCESS(double, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(double)

  TWO_COMPONENT_ACCESS(double_2, x, y)
  TWO_COMPONENT_ACCESS(double_2, x, z)
  TWO_COMPONENT_ACCESS(double_2, y, z)

  THREE_COMPONENT_ACCESS(double_3, x, y, z)
};

class norm_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(norm, norm_3)

#if !__HCC_AMP__
  SCALARTYPE_3_CONVERSION_CTOR(norm_3,
    int_3, uint_3, float_3, double_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)
#else
  SCALARTYPE_3_CONVERSION_CTOR(norm_3,
    int_3, uint_3, float_3, double_3, unorm_3)
#endif

#if __GNUG__
  // for some reason g++ will mistakenly treat x, y, z as type float
  // so we need to explicitly cast them to norm type here
  norm_3 operator-() const __CPU_GPU__ { return norm_3(-(norm)x, -(norm)y, -(norm)z); }
#else
  norm_3 operator-() const __CPU_GPU__ { return norm_3(-x, -y, -z); }
#endif
  
  SINGLE_COMPONENT_ACCESS(norm, x)
  SINGLE_COMPONENT_ACCESS(norm, y)
  SINGLE_COMPONENT_ACCESS(norm, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(norm)

  TWO_COMPONENT_ACCESS(norm_2, x, y)
  TWO_COMPONENT_ACCESS(norm_2, x, z)
  TWO_COMPONENT_ACCESS(norm_2, y, z)

  THREE_COMPONENT_ACCESS(norm_3, x, y, z)
};

class unorm_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unorm, unorm_3)

#if !__HCC_AMP__
  SCALARTYPE_3_CONVERSION_CTOR(unorm_3,
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)
#else
  SCALARTYPE_3_CONVERSION_CTOR(unorm_3,
    int_3, uint_3, float_3, double_3, norm_3)
#endif

  SINGLE_COMPONENT_ACCESS(unorm, x)
  SINGLE_COMPONENT_ACCESS(unorm, y)
  SINGLE_COMPONENT_ACCESS(unorm, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unorm)

  TWO_COMPONENT_ACCESS(unorm_2, x, y)
  TWO_COMPONENT_ACCESS(unorm_2, x, z)
  TWO_COMPONENT_ACCESS(unorm_2, y, z)

  THREE_COMPONENT_ACCESS(unorm_3, x, y, z)
};

#if !__HCC_AMP__

class char_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(char, char_3)

  SCALARTYPE_3_CONVERSION_CTOR(char_3,
    int_3, uint_3, float_3, double_3, unorm_3,
    norm_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

  char_3 operator-() const __CPU_GPU__ { return char_3(-x, -y, -z); }
  
  SINGLE_COMPONENT_ACCESS(char, x)
  SINGLE_COMPONENT_ACCESS(char, y)
  SINGLE_COMPONENT_ACCESS(char, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(char)

  TWO_COMPONENT_ACCESS(char_2, x, y)
  TWO_COMPONENT_ACCESS(char_2, x, z)
  TWO_COMPONENT_ACCESS(char_2, y, z)

  THREE_COMPONENT_ACCESS(char_3, x, y, z)
};

class uchar_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unsigned char, uchar_3)

  SCALARTYPE_3_CONVERSION_CTOR(uchar_3,
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, unorm_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

  SINGLE_COMPONENT_ACCESS(unsigned char, x)
  SINGLE_COMPONENT_ACCESS(unsigned char, y)
  SINGLE_COMPONENT_ACCESS(unsigned char, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned char)

  TWO_COMPONENT_ACCESS(uchar_2, x, y)
  TWO_COMPONENT_ACCESS(uchar_2, x, z)
  TWO_COMPONENT_ACCESS(uchar_2, y, z)

  THREE_COMPONENT_ACCESS(uchar_3, x, y, z)
};

class short_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(short, short_3)

  SCALARTYPE_3_CONVERSION_CTOR(short_3,
    int_3, uint_3, float_3, double_3, unorm_3,
    norm_3, uchar_3, char_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

  short_3 operator-() const __CPU_GPU__ { return short_3(-x, -y, -z); }
  
  SINGLE_COMPONENT_ACCESS(short, x)
  SINGLE_COMPONENT_ACCESS(short, y)
  SINGLE_COMPONENT_ACCESS(short, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(short)

  TWO_COMPONENT_ACCESS(short_2, x, y)
  TWO_COMPONENT_ACCESS(short_2, x, z)
  TWO_COMPONENT_ACCESS(short_2, y, z)

  THREE_COMPONENT_ACCESS(short_3, x, y, z)
};

class ushort_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unsigned short, ushort_3)

  SCALARTYPE_3_CONVERSION_CTOR(ushort_3,
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, unorm_3, short_3, uchar_3, long_3, ulong_3, longlong_3, ulonglong_3)

  SINGLE_COMPONENT_ACCESS(unsigned short, x)
  SINGLE_COMPONENT_ACCESS(unsigned short, y)
  SINGLE_COMPONENT_ACCESS(unsigned short, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned short)

  TWO_COMPONENT_ACCESS(ushort_2, x, y)
  TWO_COMPONENT_ACCESS(ushort_2, x, z)
  TWO_COMPONENT_ACCESS(ushort_2, y, z)

  THREE_COMPONENT_ACCESS(ushort_3, x, y, z)
};

class long_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(long, long_3)

  SCALARTYPE_3_CONVERSION_CTOR(long_3,
    int_3, uint_3, float_3, double_3, unorm_3,
    norm_3, uchar_3, short_3, ushort_3, char_3, ulong_3, longlong_3, ulonglong_3)

  long_3 operator-() const __CPU_GPU__ { return long_3(-x, -y, -z); }
  
  SINGLE_COMPONENT_ACCESS(long, x)
  SINGLE_COMPONENT_ACCESS(long, y)
  SINGLE_COMPONENT_ACCESS(long, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(long)

  TWO_COMPONENT_ACCESS(long_2, x, y)
  TWO_COMPONENT_ACCESS(long_2, x, z)
  TWO_COMPONENT_ACCESS(long_2, y, z)

  THREE_COMPONENT_ACCESS(long_3, x, y, z)
};

class ulong_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unsigned long, ulong_3)

  SCALARTYPE_3_CONVERSION_CTOR(ulong_3,
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, unorm_3, short_3, ushort_3, long_3, uchar_3, longlong_3, ulonglong_3)

  SINGLE_COMPONENT_ACCESS(unsigned long, x)
  SINGLE_COMPONENT_ACCESS(unsigned long, y)
  SINGLE_COMPONENT_ACCESS(unsigned long, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long)

  TWO_COMPONENT_ACCESS(ulong_2, x, y)
  TWO_COMPONENT_ACCESS(ulong_2, x, z)
  TWO_COMPONENT_ACCESS(ulong_2, y, z)

  THREE_COMPONENT_ACCESS(ulong_3, x, y, z)
};

class longlong_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(long long int, longlong_3)

  SCALARTYPE_3_CONVERSION_CTOR(longlong_3,
    int_3, uint_3, float_3, double_3, unorm_3,
    norm_3, uchar_3, short_3, ushort_3, char_3, ulong_3, long_3, ulonglong_3)

  longlong_3 operator-() const __CPU_GPU__ { return longlong_3(-x, -y, -z); }
  
  SINGLE_COMPONENT_ACCESS(long long int, x)
  SINGLE_COMPONENT_ACCESS(long long int, y)
  SINGLE_COMPONENT_ACCESS(long long int, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(long long int)

  TWO_COMPONENT_ACCESS(longlong_2, x, y)
  TWO_COMPONENT_ACCESS(longlong_2, x, z)
  TWO_COMPONENT_ACCESS(longlong_2, y, z)

  THREE_COMPONENT_ACCESS(longlong_3, x, y, z)
};

class ulonglong_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unsigned long long int, ulonglong_3)

  SCALARTYPE_3_CONVERSION_CTOR(ulonglong_3,
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, unorm_3, short_3, ushort_3, long_3, uchar_3, longlong_3, ulong_3)

  SINGLE_COMPONENT_ACCESS(unsigned long long int, x)
  SINGLE_COMPONENT_ACCESS(unsigned long long int, y)
  SINGLE_COMPONENT_ACCESS(unsigned long long int, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long long int)

  TWO_COMPONENT_ACCESS(ulonglong_2, x, y)
  TWO_COMPONENT_ACCESS(ulonglong_2, x, z)
  TWO_COMPONENT_ACCESS(ulonglong_2, y, z)

  THREE_COMPONENT_ACCESS(ulonglong_3, x, y, z)
};

#endif // if !__HCC_AMP__

#undef SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_3_COMMON_PUBLIC_MEMBER

#define SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() __CPU_GPU__ { return x; } \
\
ST& ref_y() __CPU_GPU__ { return y; } \
\
ST& ref_z() __CPU_GPU__ { return z; } \
\
ST& ref_w() __CPU_GPU__ { return w; } \
\
ST& ref_r() __CPU_GPU__ { return x; } \
\
ST& ref_g() __CPU_GPU__ { return y; } \
\
ST& ref_b() __CPU_GPU__ { return z; } \
\
ST& ref_a() __CPU_GPU__ { return w; }

#define SCALARTYPE_4_COMMON_PUBLIC_MEMBER(ST, ST_4) \
ST x; \
ST y; \
ST z; \
ST w; \
typedef ST value_type; \
static const int size = 4; \
\
ST_4() __CPU_GPU__ {} \
\
~ST_4() __CPU_GPU__ {} \
\
ST_4(ST value) __CPU_GPU__ \
{ \
  x = value; \
  y = value; \
  z = value; \
  w = value; \
} \
\
ST_4(const ST_4&  other) __CPU_GPU__ \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
  w = other.w; \
} \
\
ST_4(ST v1, ST v2, ST v3, ST v4) __CPU_GPU__ \
{ \
  x = v1; \
  y = v2; \
  z = v3; \
  w = v4; \
} \
\
ST_4& operator=(const ST_4& other) __CPU_GPU__ \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
  w = other.w; \
  return *this; \
} \
\
ST_4& operator++() __CPU_GPU__ \
{ \
  ++x; \
  ++y; \
  ++z; \
  ++w; \
  return *this; \
} \
\
ST_4 operator++(int) __CPU_GPU__ \
{ \
  ST_4 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_4& operator--() __CPU_GPU__ \
{ \
  --x; \
  --y; \
  --z; \
  --w; \
  return *this; \
} \
\
ST_4 operator--(int) __CPU_GPU__ \
{ \
  ST_4 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_4& operator+=(const ST_4& rhs) __CPU_GPU__ \
{ \
  x += rhs.x; \
  y += rhs.y; \
  z += rhs.z; \
  w += rhs.w; \
  return *this; \
} \
\
ST_4& operator-=(const ST_4& rhs) __CPU_GPU__ \
{ \
  x -= rhs.x; \
  y -= rhs.y; \
  z -= rhs.z; \
  w -= rhs.w; \
  return *this; \
} \
\
ST_4& operator*=(const ST_4& rhs) __CPU_GPU__ \
{ \
  x *= rhs.x; \
  y *= rhs.y; \
  z *= rhs.z; \
  w *= rhs.w; \
  return *this; \
} \
\
ST_4& operator/=(const ST_4& rhs) __CPU_GPU__ \
{ \
  x /= rhs.x; \
  y /= rhs.y; \
  z /= rhs.z; \
  w /= rhs.w; \
  return *this; \
}

#if !__HCC_AMP__

#define SCALARTYPE_4_CONVERSION_CTOR(ST_4, \
ST_4_o1, ST_4_o2, ST_4_o3, ST_4_o4, ST_4_o5, \
ST_4_o6, ST_4_o7, ST_4_o8, ST_4_o9, ST_4_o10, ST_4_o11, ST_4_o12, ST_4_o13) \
\
explicit ST_4(const ST_4_o1& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o2& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o3& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o4& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o5& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o6& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o7& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o8& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o9& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o10& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o11& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o12& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o13& other) __CPU_GPU__;

#else

#define SCALARTYPE_4_CONVERSION_CTOR(ST_4, \
ST_4_o1, ST_4_o2, ST_4_o3, ST_4_o4, ST_4_o5) \
\
explicit ST_4(const ST_4_o1& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o2& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o3& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o4& other) __CPU_GPU__; \
\
explicit ST_4(const ST_4_o5& other) __CPU_GPU__; \

#endif // if !__HCC_AMP__

class int_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(int, int_4)

#if !__HCC_AMP__
  SCALARTYPE_4_CONVERSION_CTOR(int_4,
    uint_4, float_4, double_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)
#else
  SCALARTYPE_4_CONVERSION_CTOR(int_4,
    uint_4, float_4, double_4, norm_4, unorm_4) 
#endif

  int_4 operator-() const __CPU_GPU__ { return int_4(-x, -y, -z, -w); }

  int_4 operator~() const __CPU_GPU__ { return int_4(~x, ~y, -z, -w); }

  int_4& operator%=(const int_4& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    w %= rhs.w;
    return *this;
  }

  int_4& operator^=(const int_4& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    w ^= rhs.w;
    return *this;
  }

  int_4& operator|=(const int_4& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    w |= rhs.w;
    return *this;
  }

  int_4& operator&=(const int_4& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    w &= rhs.w;
    return *this;
  }

  int_4& operator>>=(const int_4& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    w >>= rhs.w;
    return *this;
  }

  int_4& operator<<=(const int_4& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    y <<= rhs.y;
    z <<= rhs.z;
    w <<= rhs.w;
    return *this;
  }
  
  SINGLE_COMPONENT_ACCESS(int, x)
  SINGLE_COMPONENT_ACCESS(int, y)
  SINGLE_COMPONENT_ACCESS(int, z)
  SINGLE_COMPONENT_ACCESS(int, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(int)

  TWO_COMPONENT_ACCESS(int_2, x, y)
  TWO_COMPONENT_ACCESS(int_2, x, z)
  TWO_COMPONENT_ACCESS(int_2, x, w)
  TWO_COMPONENT_ACCESS(int_2, y, z)
  TWO_COMPONENT_ACCESS(int_2, y, w)
  TWO_COMPONENT_ACCESS(int_2, z, w)

  THREE_COMPONENT_ACCESS(int_3, x, y, z)
  THREE_COMPONENT_ACCESS(int_3, x, y, w)
  THREE_COMPONENT_ACCESS(int_3, x, z, w)
  THREE_COMPONENT_ACCESS(int_3, y, z, w)

  FOUR_COMPONENT_ACCESS(int_4, x, y, z, w)
};

class uint_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unsigned int, uint_4)

#if !__HCC_AMP__
  SCALARTYPE_4_CONVERSION_CTOR(uint_4,
    int_4, float_4, double_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)
#else
  SCALARTYPE_4_CONVERSION_CTOR(uint_4,
    int_4, float_4, double_4, norm_4, unorm_4) 
#endif
 
  uint_4 operator~() const __CPU_GPU__ { return uint_4(~x, ~y, ~z, -w); }

  uint_4& operator%=(const uint_4& rhs) __CPU_GPU__
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    w %= rhs.w;
    return *this;
  }

  uint_4& operator^=(const uint_4& rhs) __CPU_GPU__
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    w ^= rhs.w;
    return *this;
  }

  uint_4& operator|=(const uint_4& rhs) __CPU_GPU__
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    w |= rhs.w;
    return *this;
  }

  uint_4& operator&=(const uint_4& rhs) __CPU_GPU__
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    w &= rhs.w;
    return *this;
  }

  uint_4& operator>>=(const uint_4& rhs) __CPU_GPU__
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    w >>= rhs.w;
    return *this;
  }

  uint_4& operator<<=(const uint_4& rhs) __CPU_GPU__
  {
    x <<= rhs.x;
    y <<= rhs.y;
    z <<= rhs.z;
    w <<= rhs.w;
    return *this;
  }
 
  SINGLE_COMPONENT_ACCESS(unsigned int, x)
  SINGLE_COMPONENT_ACCESS(unsigned int, y)
  SINGLE_COMPONENT_ACCESS(unsigned int, z)
  SINGLE_COMPONENT_ACCESS(unsigned int, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned int)

  TWO_COMPONENT_ACCESS(uint_2, x, y)
  TWO_COMPONENT_ACCESS(uint_2, x, z)
  TWO_COMPONENT_ACCESS(uint_2, x, w)
  TWO_COMPONENT_ACCESS(uint_2, y, z)
  TWO_COMPONENT_ACCESS(uint_2, y, w)
  TWO_COMPONENT_ACCESS(uint_2, z, w)

  THREE_COMPONENT_ACCESS(uint_3, x, y, z)
  THREE_COMPONENT_ACCESS(uint_3, x, y, w)
  THREE_COMPONENT_ACCESS(uint_3, x, z, w)
  THREE_COMPONENT_ACCESS(uint_3, y, z, w)

  FOUR_COMPONENT_ACCESS(uint_4, x, y, z, w)
};

class float_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(float, float_4)

#if !__HCC_AMP__
  SCALARTYPE_4_CONVERSION_CTOR(float_4,
    int_4, uint_4, double_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)
#else
  SCALARTYPE_4_CONVERSION_CTOR(float_4,
    int_4, uint_4, double_4, norm_4, unorm_4) 
#endif
  
  float_4 operator-() const __CPU_GPU__ { return float_4(-x, -y, -z, -w); }

  SINGLE_COMPONENT_ACCESS(float, x)
  SINGLE_COMPONENT_ACCESS(float, y)
  SINGLE_COMPONENT_ACCESS(float, z)
  SINGLE_COMPONENT_ACCESS(float, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(float)

  TWO_COMPONENT_ACCESS(float_2, x, y)
  TWO_COMPONENT_ACCESS(float_2, x, z)
  TWO_COMPONENT_ACCESS(float_2, x, w)
  TWO_COMPONENT_ACCESS(float_2, y, z)
  TWO_COMPONENT_ACCESS(float_2, y, w)
  TWO_COMPONENT_ACCESS(float_2, z, w)

  THREE_COMPONENT_ACCESS(float_3, x, y, z)
  THREE_COMPONENT_ACCESS(float_3, x, y, w)
  THREE_COMPONENT_ACCESS(float_3, x, z, w)
  THREE_COMPONENT_ACCESS(float_3, y, z, w)

  FOUR_COMPONENT_ACCESS(float_4, x, y, z, w)
};

class double_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(double, double_4)

#if !__HCC_AMP__
  SCALARTYPE_4_CONVERSION_CTOR(double_4,
    int_4, uint_4, float_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)
#else
  SCALARTYPE_4_CONVERSION_CTOR(double_4,
    int_4, uint_4, float_4, norm_4, unorm_4) 
#endif
  
  double_4 operator-() const __CPU_GPU__ { return double_4(-x, -y, -z, -w); }

  SINGLE_COMPONENT_ACCESS(double, x)
  SINGLE_COMPONENT_ACCESS(double, y)
  SINGLE_COMPONENT_ACCESS(double, z)
  SINGLE_COMPONENT_ACCESS(double, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(double)

  TWO_COMPONENT_ACCESS(double_2, x, y)
  TWO_COMPONENT_ACCESS(double_2, x, z)
  TWO_COMPONENT_ACCESS(double_2, x, w)
  TWO_COMPONENT_ACCESS(double_2, y, z)
  TWO_COMPONENT_ACCESS(double_2, y, w)
  TWO_COMPONENT_ACCESS(double_2, z, w)

  THREE_COMPONENT_ACCESS(double_3, x, y, z)
  THREE_COMPONENT_ACCESS(double_3, x, y, w)
  THREE_COMPONENT_ACCESS(double_3, x, z, w)
  THREE_COMPONENT_ACCESS(double_3, y, z, w)

  FOUR_COMPONENT_ACCESS(double_4, x, y, z, w)
};

class norm_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(norm, norm_4)

#if !__HCC_AMP__
  SCALARTYPE_4_CONVERSION_CTOR(norm_4,
    int_4, uint_4, float_4, double_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)
#else
  SCALARTYPE_4_CONVERSION_CTOR(norm_4,
    int_4, uint_4, float_4, double_4, unorm_4) 
#endif

#if __GNUG__
  // for some reason g++ will mistakenly treat x, y, z, w as type float
  // so we need to explicitly cast them to norm type here
  norm_4 operator-() const __CPU_GPU__ { return norm_4(-(norm)x, -(norm)y, -(norm)z, -(norm)w); }
#else
  norm_4 operator-() const __CPU_GPU__ { return norm_4(-x, -y, -z, -w); }
#endif
  
  SINGLE_COMPONENT_ACCESS(norm, x)
  SINGLE_COMPONENT_ACCESS(norm, y)
  SINGLE_COMPONENT_ACCESS(norm, z)
  SINGLE_COMPONENT_ACCESS(norm, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(norm)

  TWO_COMPONENT_ACCESS(norm_2, x, y)
  TWO_COMPONENT_ACCESS(norm_2, x, z)
  TWO_COMPONENT_ACCESS(norm_2, x, w)
  TWO_COMPONENT_ACCESS(norm_2, y, z)
  TWO_COMPONENT_ACCESS(norm_2, y, w)
  TWO_COMPONENT_ACCESS(norm_2, z, w)

  THREE_COMPONENT_ACCESS(norm_3, x, y, z)
  THREE_COMPONENT_ACCESS(norm_3, x, y, w)
  THREE_COMPONENT_ACCESS(norm_3, x, z, w)
  THREE_COMPONENT_ACCESS(norm_3, y, z, w)

  FOUR_COMPONENT_ACCESS(norm_4, x, y, z, w)
};

class unorm_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unorm, unorm_4)

#if !__HCC_AMP__
  SCALARTYPE_4_CONVERSION_CTOR(unorm_4,
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)
#else
  SCALARTYPE_4_CONVERSION_CTOR(unorm_4,
    int_4, uint_4, float_4, double_4, norm_4) 
#endif

  SINGLE_COMPONENT_ACCESS(unorm, x)
  SINGLE_COMPONENT_ACCESS(unorm, y)
  SINGLE_COMPONENT_ACCESS(unorm, z)
  SINGLE_COMPONENT_ACCESS(unorm, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(unorm)

  TWO_COMPONENT_ACCESS(unorm_2, x, y)
  TWO_COMPONENT_ACCESS(unorm_2, x, z)
  TWO_COMPONENT_ACCESS(unorm_2, x, w)
  TWO_COMPONENT_ACCESS(unorm_2, y, z)
  TWO_COMPONENT_ACCESS(unorm_2, y, w)
  TWO_COMPONENT_ACCESS(unorm_2, z, w)

  THREE_COMPONENT_ACCESS(unorm_3, x, y, z)
  THREE_COMPONENT_ACCESS(unorm_3, x, y, w)
  THREE_COMPONENT_ACCESS(unorm_3, x, z, w)
  THREE_COMPONENT_ACCESS(unorm_3, y, z, w)

  FOUR_COMPONENT_ACCESS(unorm_4, x, y, z, w)
};

#if !__HCC_AMP__

class char_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(char, char_4)

  SCALARTYPE_4_CONVERSION_CTOR(char_4,
    int_4, uint_4, float_4, double_4, unorm_4,
    norm_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

  char_4 operator-() const __CPU_GPU__ { return char_4(-x, -y, -z, -w); }
  
  SINGLE_COMPONENT_ACCESS(char, x)
  SINGLE_COMPONENT_ACCESS(char, y)
  SINGLE_COMPONENT_ACCESS(char, z)
  SINGLE_COMPONENT_ACCESS(char, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(char)

  TWO_COMPONENT_ACCESS(char_2, x, y)
  TWO_COMPONENT_ACCESS(char_2, x, z)
  TWO_COMPONENT_ACCESS(char_2, x, w)
  TWO_COMPONENT_ACCESS(char_2, y, z)
  TWO_COMPONENT_ACCESS(char_2, y, w)
  TWO_COMPONENT_ACCESS(char_2, z, w)

  THREE_COMPONENT_ACCESS(char_3, x, y, z)
  THREE_COMPONENT_ACCESS(char_3, x, y, w)
  THREE_COMPONENT_ACCESS(char_3, x, z, w)
  THREE_COMPONENT_ACCESS(char_3, y, z, w)

  FOUR_COMPONENT_ACCESS(char_4, x, y, z, w)
};

class uchar_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unsigned char, uchar_4)

  SCALARTYPE_4_CONVERSION_CTOR(uchar_4,
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, unorm_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

  SINGLE_COMPONENT_ACCESS(unsigned char, x)
  SINGLE_COMPONENT_ACCESS(unsigned char, y)
  SINGLE_COMPONENT_ACCESS(unsigned char, z)
  SINGLE_COMPONENT_ACCESS(unsigned char, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned char)

  TWO_COMPONENT_ACCESS(uchar_2, x, y)
  TWO_COMPONENT_ACCESS(uchar_2, x, z)
  TWO_COMPONENT_ACCESS(uchar_2, x, w)
  TWO_COMPONENT_ACCESS(uchar_2, y, z)
  TWO_COMPONENT_ACCESS(uchar_2, y, w)
  TWO_COMPONENT_ACCESS(uchar_2, z, w)

  THREE_COMPONENT_ACCESS(uchar_3, x, y, z)
  THREE_COMPONENT_ACCESS(uchar_3, x, y, w)
  THREE_COMPONENT_ACCESS(uchar_3, x, z, w)
  THREE_COMPONENT_ACCESS(uchar_3, y, z, w)

  FOUR_COMPONENT_ACCESS(uchar_4, x, y, z, w)
};

class short_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(short, short_4)

  SCALARTYPE_4_CONVERSION_CTOR(short_4,
    int_4, uint_4, float_4, double_4, unorm_4,
    norm_4, uchar_4, char_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

  short_4 operator-() const __CPU_GPU__ { return short_4(-x, -y, -z, -w); }
  
  SINGLE_COMPONENT_ACCESS(short, x)
  SINGLE_COMPONENT_ACCESS(short, y)
  SINGLE_COMPONENT_ACCESS(short, z)
  SINGLE_COMPONENT_ACCESS(short, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(short)

  TWO_COMPONENT_ACCESS(short_2, x, y)
  TWO_COMPONENT_ACCESS(short_2, x, z)
  TWO_COMPONENT_ACCESS(short_2, x, w)
  TWO_COMPONENT_ACCESS(short_2, y, z)
  TWO_COMPONENT_ACCESS(short_2, y, w)
  TWO_COMPONENT_ACCESS(short_2, z, w)

  THREE_COMPONENT_ACCESS(short_3, x, y, z)
  THREE_COMPONENT_ACCESS(short_3, x, y, w)
  THREE_COMPONENT_ACCESS(short_3, x, z, w)
  THREE_COMPONENT_ACCESS(short_3, y, z, w)

  FOUR_COMPONENT_ACCESS(short_4, x, y, z, w)
};

class ushort_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unsigned short, ushort_4)

  SCALARTYPE_4_CONVERSION_CTOR(ushort_4,
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, unorm_4, short_4, uchar_4, long_4, ulong_4, longlong_4, ulonglong_4)

  SINGLE_COMPONENT_ACCESS(unsigned short, x)
  SINGLE_COMPONENT_ACCESS(unsigned short, y)
  SINGLE_COMPONENT_ACCESS(unsigned short, z)
  SINGLE_COMPONENT_ACCESS(unsigned short, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned short)

  TWO_COMPONENT_ACCESS(ushort_2, x, y)
  TWO_COMPONENT_ACCESS(ushort_2, x, z)
  TWO_COMPONENT_ACCESS(ushort_2, x, w)
  TWO_COMPONENT_ACCESS(ushort_2, y, z)
  TWO_COMPONENT_ACCESS(ushort_2, y, w)
  TWO_COMPONENT_ACCESS(ushort_2, z, w)

  THREE_COMPONENT_ACCESS(ushort_3, x, y, z)
  THREE_COMPONENT_ACCESS(ushort_3, x, y, w)
  THREE_COMPONENT_ACCESS(ushort_3, x, z, w)
  THREE_COMPONENT_ACCESS(ushort_3, y, z, w)

  FOUR_COMPONENT_ACCESS(ushort_4, x, y, z, w)
};

class long_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(long, long_4)

  SCALARTYPE_4_CONVERSION_CTOR(long_4,
    int_4, uint_4, float_4, double_4, unorm_4,
    norm_4, uchar_4, short_4, ushort_4, char_4, ulong_4, longlong_4, ulonglong_4)

  long_4 operator-() const __CPU_GPU__ { return long_4(-x, -y, -z, -w); }
  
  SINGLE_COMPONENT_ACCESS(long, x)
  SINGLE_COMPONENT_ACCESS(long, y)
  SINGLE_COMPONENT_ACCESS(long, z)
  SINGLE_COMPONENT_ACCESS(long, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(long)

  TWO_COMPONENT_ACCESS(long_2, x, y)
  TWO_COMPONENT_ACCESS(long_2, x, z)
  TWO_COMPONENT_ACCESS(long_2, x, w)
  TWO_COMPONENT_ACCESS(long_2, y, z)
  TWO_COMPONENT_ACCESS(long_2, y, w)
  TWO_COMPONENT_ACCESS(long_2, z, w)

  THREE_COMPONENT_ACCESS(long_3, x, y, z)
  THREE_COMPONENT_ACCESS(long_3, x, y, w)
  THREE_COMPONENT_ACCESS(long_3, x, z, w)
  THREE_COMPONENT_ACCESS(long_3, y, z, w)

  FOUR_COMPONENT_ACCESS(long_4, x, y, z, w)
};

class ulong_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unsigned long, ulong_4)

  SCALARTYPE_4_CONVERSION_CTOR(ulong_4,
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, unorm_4, short_4, ushort_4, long_4, uchar_4, longlong_4, ulonglong_4)

  SINGLE_COMPONENT_ACCESS(unsigned long, x)
  SINGLE_COMPONENT_ACCESS(unsigned long, y)
  SINGLE_COMPONENT_ACCESS(unsigned long, z)
  SINGLE_COMPONENT_ACCESS(unsigned long, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long)

  TWO_COMPONENT_ACCESS(ulong_2, x, y)
  TWO_COMPONENT_ACCESS(ulong_2, x, z)
  TWO_COMPONENT_ACCESS(ulong_2, x, w)
  TWO_COMPONENT_ACCESS(ulong_2, y, z)
  TWO_COMPONENT_ACCESS(ulong_2, y, w)
  TWO_COMPONENT_ACCESS(ulong_2, z, w)

  THREE_COMPONENT_ACCESS(ulong_3, x, y, z)
  THREE_COMPONENT_ACCESS(ulong_3, x, y, w)
  THREE_COMPONENT_ACCESS(ulong_3, x, z, w)
  THREE_COMPONENT_ACCESS(ulong_3, y, z, w)

  FOUR_COMPONENT_ACCESS(ulong_4, x, y, z, w)
};

class longlong_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(long long int, longlong_4)

  SCALARTYPE_4_CONVERSION_CTOR(longlong_4,
    int_4, uint_4, float_4, double_4, unorm_4,
    norm_4, uchar_4, short_4, ushort_4, char_4, ulong_4, long_4, ulonglong_4)

  longlong_4 operator-() const __CPU_GPU__ { return longlong_4(-x, -y, -z, -w); }
  
  SINGLE_COMPONENT_ACCESS(long long int, x)
  SINGLE_COMPONENT_ACCESS(long long int, y)
  SINGLE_COMPONENT_ACCESS(long long int, z)
  SINGLE_COMPONENT_ACCESS(long long int, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(long long int)

  TWO_COMPONENT_ACCESS(longlong_2, x, y)
  TWO_COMPONENT_ACCESS(longlong_2, x, z)
  TWO_COMPONENT_ACCESS(longlong_2, x, w)
  TWO_COMPONENT_ACCESS(longlong_2, y, z)
  TWO_COMPONENT_ACCESS(longlong_2, y, w)
  TWO_COMPONENT_ACCESS(longlong_2, z, w)

  THREE_COMPONENT_ACCESS(longlong_3, x, y, z)
  THREE_COMPONENT_ACCESS(longlong_3, x, y, w)
  THREE_COMPONENT_ACCESS(longlong_3, x, z, w)
  THREE_COMPONENT_ACCESS(longlong_3, y, z, w)

  FOUR_COMPONENT_ACCESS(longlong_4, x, y, z, w)
};

class ulonglong_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unsigned long long int, ulonglong_4)

  SCALARTYPE_4_CONVERSION_CTOR(ulonglong_4,
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, unorm_4, short_4, ushort_4, long_4, uchar_4, longlong_4, ulong_4)

  SINGLE_COMPONENT_ACCESS(unsigned long long int, x)
  SINGLE_COMPONENT_ACCESS(unsigned long long int, y)
  SINGLE_COMPONENT_ACCESS(unsigned long long int, z)
  SINGLE_COMPONENT_ACCESS(unsigned long long int, w)

  SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(unsigned long long int)

  TWO_COMPONENT_ACCESS(ulonglong_2, x, y)
  TWO_COMPONENT_ACCESS(ulonglong_2, x, z)
  TWO_COMPONENT_ACCESS(ulonglong_2, x, w)
  TWO_COMPONENT_ACCESS(ulonglong_2, y, z)
  TWO_COMPONENT_ACCESS(ulonglong_2, y, w)
  TWO_COMPONENT_ACCESS(ulonglong_2, z, w)

  THREE_COMPONENT_ACCESS(ulonglong_3, x, y, z)
  THREE_COMPONENT_ACCESS(ulonglong_3, x, y, w)
  THREE_COMPONENT_ACCESS(ulonglong_3, x, z, w)
  THREE_COMPONENT_ACCESS(ulonglong_3, y, z, w)

  FOUR_COMPONENT_ACCESS(ulonglong_4, x, y, z, w)
};

#endif // if !__HCC_AMP__

#undef SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_4_COMMON_PUBLIC_MEMBER

#undef SINGLE_COMPONENT_ACCESS
#undef TWO_COMPONENT_ACCESS
#undef THREE_COMPONENT_ACCESS
#undef FOUR_COMPONENT_ACCESS

//   Explicit Conversion Constructor Definitions (10.8.2.2)

#if !__HCC_AMP__

#define SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_1, \
ST_1_o1, ST_1_o2, ST_1_o3, ST_1_o4, ST_1_o5, \
ST_1_o6, ST_1_o7, ST_1_o8, ST_1_o9, ST_1_o10, ST_1_o11) \
inline ST_1::ST_1(const ST_1_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
\
inline ST_1::ST_1(const ST_1_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
\
inline ST_1::ST_1(const ST_1_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
\
inline ST_1::ST_1(const ST_1_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
\
inline ST_1::ST_1(const ST_1_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
inline ST_1::ST_1(const ST_1_o6& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
inline ST_1::ST_1(const ST_1_o7& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
inline ST_1::ST_1(const ST_1_o8& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
inline ST_1::ST_1(const ST_1_o9& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
inline ST_1::ST_1(const ST_1_o10& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
} \
inline ST_1::ST_1(const ST_1_o11& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
}

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_1, 
    uint_1, float_1, double_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_1, 
    int_1, float_1, double_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_1, 
    int_1, uint_1, double_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_1, 
    int_1, uint_1, float_1,
    char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(char, char_1, 
    int_1, uint_1, float_1,
    double_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned char, uchar_1, 
    int_1, uint_1, float_1,
     double_1, char_1, short_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(short, short_1, 
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, ushort_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned short, ushort_1, 
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, long_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(long, long_1, 
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, ulong_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long, ulong_1, 
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, long_1, longlong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(long long int, longlong_1, 
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, ulonglong_1)

SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long long int, ulonglong_1, 
    int_1, uint_1, float_1,
    double_1, char_1, uchar_1, short_1, ushort_1, long_1, ulong_1, longlong_1)

#undef SCALARTYPE_1_EXPLICIT_CONVERSION_CONSTRUCTORS

#endif // if !__HCC_AMP__

#if !__HCC_AMP__

#define SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_2, \
ST_2_o1, ST_2_o2, ST_2_o3, ST_2_o4, ST_2_o5, \
ST_2_o6, ST_2_o7, ST_2_o8, ST_2_o9, ST_2_o10, ST_2_o11, ST_2_o12, ST_2_o13) \
inline ST_2::ST_2(const ST_2_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o6& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o7& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o8& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o9& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o10& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o11& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o12& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
inline ST_2::ST_2(const ST_2_o13& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
}

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_2, 
    uint_2, float_2, double_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_2, 
    int_2, float_2, double_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_2, 
    int_2, uint_2, double_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_2, 
    int_2, uint_2, float_2, norm_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(norm, norm_2, 
    int_2, uint_2, float_2, double_2, unorm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unorm, unorm_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(char, char_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    unorm_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned char, uchar_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, unorm_2, short_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(short, short_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, unorm_2, ushort_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned short, ushort_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, unorm_2, long_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(long, long_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, ushort_2, unorm_2, ulong_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long, ulong_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, unorm_2, longlong_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(long long int, longlong_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, unorm_2, ulonglong_2)

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long long int, ulonglong_2, 
    int_2, uint_2, float_2, double_2, norm_2,
    char_2, uchar_2, short_2, ushort_2, long_2, ulong_2, longlong_2, unorm_2)

#undef SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS

#else // if !__HCC_AMP__

#define SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_2, \
ST_2_o1, ST_2_o2, ST_2_o3, ST_2_o4, ST_2_o5) \
inline ST_2::ST_2(const ST_2_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
inline ST_2::ST_2(const ST_2_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
}

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_2, 
    uint_2, float_2, double_2, norm_2, unorm_2) 

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_2, 
    int_2, float_2, double_2, norm_2, unorm_2) 

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_2, 
    int_2, uint_2, double_2, norm_2, unorm_2) 

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_2, 
    int_2, uint_2, float_2, norm_2, unorm_2) 

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(norm, norm_2, 
    int_2, uint_2, float_2, double_2, unorm_2) 

SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(unorm, unorm_2, 
    int_2, uint_2, float_2, double_2, norm_2) 

#undef SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS

#endif // if !__HCC_AMP__

#if !__HCC_AMP__

#define SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_3, \
ST_3_o1, ST_3_o2, ST_3_o3, ST_3_o4, ST_3_o5, \
ST_3_o6, ST_3_o7, ST_3_o8, ST_3_o9, ST_3_o10, ST_3_o11, ST_3_o12, ST_3_o13) \
inline ST_3::ST_3(const ST_3_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o6& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o7& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o8& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o9& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o10& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o11& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o12& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
inline ST_3::ST_3(const ST_3_o13& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
}

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_3, 
    uint_3, float_3, double_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_3, 
    int_3, float_3, double_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_3, 
    int_3, uint_3, double_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_3, 
    int_3, uint_3, float_3, norm_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(norm, norm_3, 
    int_3, uint_3, float_3, double_3, unorm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unorm, unorm_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(char, char_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    unorm_3, uchar_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned char, uchar_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, unorm_3, short_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(short, short_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, unorm_3, ushort_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned short, ushort_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, unorm_3, long_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(long, long_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, ushort_3, unorm_3, ulong_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long, ulong_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, unorm_3, longlong_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(long long int, longlong_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, ushort_3, unorm_3, ulong_3, long_3, ulonglong_3)

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long long int, ulonglong_3, 
    int_3, uint_3, float_3, double_3, norm_3,
    char_3, uchar_3, short_3, ushort_3, long_3, unorm_3, longlong_3, ulong_3)

#undef SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS

#else // if !__HCC_AMP__

#define SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_3, \
ST_3_o1, ST_3_o2, ST_3_o3, ST_3_o4, ST_3_o5) \
inline ST_3::ST_3(const ST_3_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
inline ST_3::ST_3(const ST_3_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
}

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_3, 
    uint_3, float_3, double_3, norm_3, unorm_3) 

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_3, 
    int_3, float_3, double_3, norm_3, unorm_3) 

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_3, 
    int_3, uint_3, double_3, norm_3, unorm_3) 

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_3, 
    int_3, uint_3, float_3, norm_3, unorm_3) 

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(norm, norm_3, 
    int_3, uint_3, float_3, double_3, unorm_3) 

SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(unorm, unorm_3, 
    int_3, uint_3, float_3, double_3, norm_3) 

#undef SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS

#endif // if !__HCC_AMP__

#if !__HCC_AMP__

#define SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_4, \
ST_4_o1, ST_4_o2, ST_4_o3, ST_4_o4, ST_4_o5, \
ST_4_o6, ST_4_o7, ST_4_o8, ST_4_o9, ST_4_o10, ST_4_o11, ST_4_o12, ST_4_o13) \
inline ST_4::ST_4(const ST_4_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o6& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o7& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o8& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o9& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o10& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o11& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o12& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
inline ST_4::ST_4(const ST_4_o13& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
}

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_4, 
    uint_4, float_4, double_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_4, 
    int_4, float_4, double_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_4, 
    int_4, uint_4, double_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_4, 
    int_4, uint_4, float_4, norm_4, unorm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(norm, norm_4, 
    int_4, uint_4, float_4, double_4, unorm_4, 
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unorm, unorm_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(char, char_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    unorm_4, uchar_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned char, uchar_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, unorm_4, short_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(short, short_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, unorm_4, ushort_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned short, ushort_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, unorm_4, long_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(long, long_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, ushort_4, unorm_4, ulong_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long, ulong_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, unorm_4, longlong_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(long long int, longlong_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, ushort_4, unorm_4, ulong_4, long_4, ulonglong_4)

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned long long int, ulonglong_4, 
    int_4, uint_4, float_4, double_4, norm_4,
    char_4, uchar_4, short_4, ushort_4, long_4, unorm_4, longlong_4, ulong_4)

#undef SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS

#else // if !__HCC_AMP__

#define SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_4, \
ST_4_o1, ST_4_o2, ST_4_o3, ST_4_o4, ST_4_o5) \
inline ST_4::ST_4(const ST_4_o1& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o2& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o3& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o4& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
inline ST_4::ST_4(const ST_4_o5& other) __CPU_GPU__ \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
}

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(int, int_4, 
    uint_4, float_4, double_4, norm_4, unorm_4) 

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unsigned int, uint_4, 
    int_4, float_4, double_4, norm_4, unorm_4) 

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(float, float_4, 
    int_4, uint_4, double_4, norm_4, unorm_4) 

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(double, double_4, 
    int_4, uint_4, float_4, norm_4, unorm_4) 

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(norm, norm_4, 
    int_4, uint_4, float_4, double_4, unorm_4) 

SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(unorm, unorm_4, 
    int_4, uint_4, float_4, double_4, norm_4) 

#undef SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS

#endif

//   Operators between Two References (10.8.1 Synopsis)

#if !__HCC_AMP__

#define SCALARTYPE_1_OPERATOR(ST_1) \
inline ST_1 operator+(const ST_1& lhs, const ST_1& rhs) __CPU_GPU__ \
{ \
  return ST_1(lhs.get_x() + rhs.get_x()); \
} \
\
inline ST_1 operator-(const ST_1& lhs, const ST_1& rhs) __CPU_GPU__ \
{ \
  return ST_1(lhs.get_x() - rhs.get_x()); \
} \
\
inline ST_1 operator*(const ST_1& lhs, const ST_1& rhs) __CPU_GPU__ \
{ \
  return ST_1(lhs.get_x() * rhs.get_x()); \
} \
\
inline ST_1 operator/(const ST_1& lhs, const ST_1& rhs) __CPU_GPU__ \
{ \
  return ST_1(lhs.get_x() / rhs.get_x()); \
} \
\
inline bool operator==(const ST_1& lhs, const ST_1& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() == rhs.get_x()); \
} \
\
inline bool operator!=(const ST_1& lhs, const ST_1& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() != rhs.get_x()); \
}

SCALARTYPE_1_OPERATOR(int_1)

SCALARTYPE_1_OPERATOR(uint_1)

SCALARTYPE_1_OPERATOR(float_1)

SCALARTYPE_1_OPERATOR(double_1)

SCALARTYPE_1_OPERATOR(char_1)

SCALARTYPE_1_OPERATOR(uchar_1)

SCALARTYPE_1_OPERATOR(short_1)

SCALARTYPE_1_OPERATOR(ushort_1)

SCALARTYPE_1_OPERATOR(long_1)

SCALARTYPE_1_OPERATOR(ulong_1)

SCALARTYPE_1_OPERATOR(longlong_1)

SCALARTYPE_1_OPERATOR(ulonglong_1)

#undef SCALARTYPE_1_OPERATOR

inline int_1 operator%(const int_1& lhs, const int_1& rhs) __CPU_GPU__
{
  return int_1(lhs.get_x() % rhs.get_x());
}

inline int_1 operator^(const int_1& lhs, const int_1& rhs) __CPU_GPU__
{
  return int_1(lhs.get_x() ^ rhs.get_x());
}

inline int_1 operator|(const int_1& lhs, const int_1& rhs) __CPU_GPU__
{
  return int_1(lhs.get_x() | rhs.get_x());
}

inline int_1 operator&(const int_1& lhs, const int_1& rhs) __CPU_GPU__
{
  return int_1(lhs.get_x() & rhs.get_x());
}

inline int_1 operator<<(const int_1& lhs, const int_1& rhs) __CPU_GPU__
{
  return int_1(lhs.get_x() << rhs.get_x());
}

inline int_1 operator>>(const int_1& lhs, const int_1& rhs) __CPU_GPU__
{
  return int_1(lhs.get_x() >> rhs.get_x());
}

inline uint_1 operator%(const uint_1& lhs, const uint_1& rhs) __CPU_GPU__
{
  return uint_1(lhs.get_x() % rhs.get_x());
}

inline uint_1 operator^(const uint_1& lhs, const uint_1& rhs) __CPU_GPU__
{
  return uint_1(lhs.get_x() ^ rhs.get_x());
}

inline uint_1 operator|(const uint_1& lhs, const uint_1& rhs) __CPU_GPU__
{
  return uint_1(lhs.get_x() | rhs.get_x());
}

inline uint_1 operator&(const uint_1& lhs, const uint_1& rhs) __CPU_GPU__
{
  return uint_1(lhs.get_x() & rhs.get_x());
}

inline uint_1 operator<<(const uint_1& lhs, const uint_1& rhs) __CPU_GPU__
{
  return uint_1(lhs.get_x() << rhs.get_x());
}

inline uint_1 operator>>(const uint_1& lhs, const uint_1& rhs) __CPU_GPU__
{
  return uint_1(lhs.get_x() >> rhs.get_x());
}

#endif // if !__HCC_AMP__

#define SCALARTYPE_2_OPERATOR(ST_2) \
inline ST_2 operator+(const ST_2& lhs, const ST_2& rhs) __CPU_GPU__ \
{ \
  return ST_2(lhs.get_x() + rhs.get_x(), lhs.get_y() + rhs.get_y()); \
} \
\
inline ST_2 operator-(const ST_2& lhs, const ST_2& rhs) __CPU_GPU__ \
{ \
  return ST_2(lhs.get_x() - rhs.get_x(), lhs.get_y() - rhs.get_y()); \
} \
\
inline ST_2 operator*(const ST_2& lhs, const ST_2& rhs) __CPU_GPU__ \
{ \
  return ST_2(lhs.get_x() * rhs.get_x(), lhs.get_y() * rhs.get_y()); \
} \
\
inline ST_2 operator/(const ST_2& lhs, const ST_2& rhs) __CPU_GPU__ \
{ \
  return ST_2(lhs.get_x() / rhs.get_x(), lhs.get_y() / rhs.get_y()); \
} \
\
inline bool operator==(const ST_2& lhs, const ST_2& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() == rhs.get_x()) && (lhs.get_y() == rhs.get_y()); \
} \
\
inline bool operator!=(const ST_2& lhs, const ST_2& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() != rhs.get_x()) || (lhs.get_y() != rhs.get_y()); \
}

SCALARTYPE_2_OPERATOR(int_2)

SCALARTYPE_2_OPERATOR(uint_2)

SCALARTYPE_2_OPERATOR(float_2)

SCALARTYPE_2_OPERATOR(double_2)

SCALARTYPE_2_OPERATOR(norm_2)

SCALARTYPE_2_OPERATOR(unorm_2)

#if !__HCC_AMP__

SCALARTYPE_2_OPERATOR(char_2)

SCALARTYPE_2_OPERATOR(uchar_2)

SCALARTYPE_2_OPERATOR(short_2)

SCALARTYPE_2_OPERATOR(ushort_2)

SCALARTYPE_2_OPERATOR(long_2)

SCALARTYPE_2_OPERATOR(ulong_2)

SCALARTYPE_2_OPERATOR(longlong_2)

SCALARTYPE_2_OPERATOR(ulonglong_2)

#endif // if !__HCC_AMP__

#undef SCALARTYPE_2_OPERATOR

inline int_2 operator%(const int_2& lhs, const int_2& rhs) __CPU_GPU__
{
  return int_2(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y());
}

inline int_2 operator^(const int_2& lhs, const int_2& rhs) __CPU_GPU__
{
  return int_2(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y());
}

inline int_2 operator|(const int_2& lhs, const int_2& rhs) __CPU_GPU__
{
  return int_2(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y());
}

inline int_2 operator&(const int_2& lhs, const int_2& rhs) __CPU_GPU__
{
  return int_2(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y());
}

inline int_2 operator<<(const int_2& lhs, const int_2& rhs) __CPU_GPU__
{
  return int_2(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y());
}

inline int_2 operator>>(const int_2& lhs, const int_2& rhs) __CPU_GPU__
{
  return int_2(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y());
}

inline uint_2 operator%(const uint_2& lhs, const uint_2& rhs) __CPU_GPU__
{
  return uint_2(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y());
}

inline uint_2 operator^(const uint_2& lhs, const uint_2& rhs) __CPU_GPU__
{
  return uint_2(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y());
}

inline uint_2 operator|(const uint_2& lhs, const uint_2& rhs) __CPU_GPU__
{
  return uint_2(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y());
}

inline uint_2 operator&(const uint_2& lhs, const uint_2& rhs) __CPU_GPU__
{
  return uint_2(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y());
}

inline uint_2 operator<<(const uint_2& lhs, const uint_2& rhs) __CPU_GPU__
{
  return uint_2(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y());
}

inline uint_2 operator>>(const uint_2& lhs, const uint_2& rhs) __CPU_GPU__
{
  return uint_2(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y());
}

#define SCALARTYPE_3_OPERATOR(ST_3) \
inline ST_3 operator+(const ST_3& lhs, const ST_3& rhs) __CPU_GPU__ \
{ \
  return ST_3(lhs.get_x() + rhs.get_x(), lhs.get_y() + rhs.get_y(), \
               lhs.get_z() + rhs.get_z()); \
} \
\
inline ST_3 operator-(const ST_3& lhs, const ST_3& rhs) __CPU_GPU__ \
{ \
  return ST_3(lhs.get_x() - rhs.get_x(), lhs.get_y() - rhs.get_y(), \
               lhs.get_z() - rhs.get_z()); \
} \
\
inline ST_3 operator*(const ST_3& lhs, const ST_3& rhs) __CPU_GPU__ \
{ \
  return ST_3(lhs.get_x() * rhs.get_x(), lhs.get_y() * rhs.get_y(), \
               lhs.get_z() * rhs.get_z()); \
} \
\
inline ST_3 operator/(const ST_3& lhs, const ST_3& rhs) __CPU_GPU__ \
{ \
  return ST_3(lhs.get_x() / rhs.get_x(), lhs.get_y() / rhs.get_y(), \
               lhs.get_z() / rhs.get_z()); \
} \
\
inline bool operator==(const ST_3& lhs, const ST_3& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() == rhs.get_x()) && (lhs.get_y() == rhs.get_y()) \
           && (lhs.get_z() == rhs.get_z()); \
} \
\
inline bool operator!=(const ST_3& lhs, const ST_3& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() != rhs.get_x()) || (lhs.get_y() != rhs.get_y()) \
           || (lhs.get_z() != rhs.get_z()); \
}

SCALARTYPE_3_OPERATOR(int_3)

SCALARTYPE_3_OPERATOR(uint_3)

SCALARTYPE_3_OPERATOR(float_3)

SCALARTYPE_3_OPERATOR(double_3)

SCALARTYPE_3_OPERATOR(norm_3)

SCALARTYPE_3_OPERATOR(unorm_3)

#if !__HCC_AMP__

SCALARTYPE_3_OPERATOR(char_3)

SCALARTYPE_3_OPERATOR(uchar_3)

SCALARTYPE_3_OPERATOR(short_3)

SCALARTYPE_3_OPERATOR(ushort_3)

SCALARTYPE_3_OPERATOR(long_3)

SCALARTYPE_3_OPERATOR(ulong_3)

SCALARTYPE_3_OPERATOR(longlong_3)

SCALARTYPE_3_OPERATOR(ulonglong_3)

#endif // if !__HCC_AMP__

#undef SCALARTYPE_3_OPERATOR

inline int_3 operator%(const int_3& lhs, const int_3& rhs) __CPU_GPU__
{
  return int_3(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                lhs.get_z() % rhs.get_z());
}

inline int_3 operator^(const int_3& lhs, const int_3& rhs) __CPU_GPU__
{
  return int_3(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                lhs.get_z() ^ rhs.get_z());
}

inline int_3 operator|(const int_3& lhs, const int_3& rhs) __CPU_GPU__
{
  return int_3(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                lhs.get_z() | rhs.get_z());
}

inline int_3 operator&(const int_3& lhs, const int_3& rhs) __CPU_GPU__
{
  return int_3(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                lhs.get_z() & rhs.get_z());
}

inline int_3 operator<<(const int_3& lhs, const int_3& rhs) __CPU_GPU__
{
  return int_3(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                lhs.get_z() << rhs.get_z());
}

inline int_3 operator>>(const int_3& lhs, const int_3& rhs) __CPU_GPU__
{
  return int_3(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                lhs.get_z() >> rhs.get_z());
}

inline uint_3 operator%(const uint_3& lhs, const uint_3& rhs) __CPU_GPU__
{
  return uint_3(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                 lhs.get_z() % rhs.get_z());
}

inline uint_3 operator^(const uint_3& lhs, const uint_3& rhs) __CPU_GPU__
{
  return uint_3(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                 lhs.get_z() ^ rhs.get_z());
}

inline uint_3 operator|(const uint_3& lhs, const uint_3& rhs) __CPU_GPU__
{
  return uint_3(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                 lhs.get_z() | rhs.get_z());
}

inline uint_3 operator&(const uint_3& lhs, const uint_3& rhs) __CPU_GPU__
{
  return uint_3(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                 lhs.get_z() & rhs.get_z());
}

inline uint_3 operator<<(const uint_3& lhs, const uint_3& rhs) __CPU_GPU__
{
  return uint_3(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                 lhs.get_z() << rhs.get_z());
}

inline uint_3 operator>>(const uint_3& lhs, const uint_3& rhs) __CPU_GPU__
{
  return uint_3(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                 lhs.get_z() >> rhs.get_z());
}

#define SCALARTYPE_4_OPERATOR(ST_4) \
inline ST_4 operator+(const ST_4& lhs, const ST_4& rhs) __CPU_GPU__ \
{ \
  return ST_4(lhs.get_x() + rhs.get_x(), lhs.get_y() + rhs.get_y(), \
               lhs.get_z() + rhs.get_z(), lhs.get_w() + rhs.get_w()); \
} \
\
inline ST_4 operator-(const ST_4& lhs, const ST_4& rhs) __CPU_GPU__ \
{ \
  return ST_4(lhs.get_x() - rhs.get_x(), lhs.get_y() - rhs.get_y(), \
               lhs.get_z() - rhs.get_z(), lhs.get_w() - rhs.get_w()); \
} \
\
inline ST_4 operator*(const ST_4& lhs, const ST_4& rhs) __CPU_GPU__ \
{ \
  return ST_4(lhs.get_x() * rhs.get_x(), lhs.get_y() * rhs.get_y(), \
               lhs.get_z() * rhs.get_z(), lhs.get_w() * rhs.get_w()); \
} \
\
inline ST_4 operator/(const ST_4& lhs, const ST_4& rhs) __CPU_GPU__ \
{ \
  return ST_4(lhs.get_x() / rhs.get_x(), lhs.get_y() / rhs.get_y(), \
               lhs.get_z() / rhs.get_z(), lhs.get_w() / rhs.get_w()); \
} \
\
inline bool operator==(const ST_4& lhs, const ST_4& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() == rhs.get_x()) && (lhs.get_y() == rhs.get_y()) \
           && (lhs.get_z() == rhs.get_z()) && (lhs.get_w() == rhs.get_w()); \
} \
\
inline bool operator!=(const ST_4& lhs, const ST_4& rhs) __CPU_GPU__ \
{ \
  return (lhs.get_x() != rhs.get_x()) || (lhs.get_y() != rhs.get_y()) \
           || (lhs.get_z() != rhs.get_z()) || (lhs.get_w() != rhs.get_w()); \
}

SCALARTYPE_4_OPERATOR(int_4)

SCALARTYPE_4_OPERATOR(uint_4)

SCALARTYPE_4_OPERATOR(float_4)

SCALARTYPE_4_OPERATOR(double_4)

SCALARTYPE_4_OPERATOR(norm_4)

SCALARTYPE_4_OPERATOR(unorm_4)

#if !__HCC_AMP__

SCALARTYPE_4_OPERATOR(char_4)

SCALARTYPE_4_OPERATOR(uchar_4)

SCALARTYPE_4_OPERATOR(short_4)

SCALARTYPE_4_OPERATOR(ushort_4)

SCALARTYPE_4_OPERATOR(long_4)

SCALARTYPE_4_OPERATOR(ulong_4)

SCALARTYPE_4_OPERATOR(longlong_4)

SCALARTYPE_4_OPERATOR(ulonglong_4)

#endif // if !__HCC_AMP__

#undef SCALARTYPE_4_OPERATOR

inline int_4 operator%(const int_4& lhs, const int_4& rhs) __CPU_GPU__
{
  return int_4(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                lhs.get_z() % rhs.get_z(), lhs.get_w() % rhs.get_w());
}

inline int_4 operator^(const int_4& lhs, const int_4& rhs) __CPU_GPU__
{
  return int_4(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                lhs.get_z() ^ rhs.get_z(), lhs.get_w() ^ rhs.get_w());
}

inline int_4 operator|(const int_4& lhs, const int_4& rhs) __CPU_GPU__
{
  return int_4(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                lhs.get_z() | rhs.get_z(), lhs.get_w() | rhs.get_w());
}

inline int_4 operator&(const int_4& lhs, const int_4& rhs) __CPU_GPU__
{
  return int_4(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                lhs.get_z() & rhs.get_z(), lhs.get_w() & rhs.get_w());
}

inline int_4 operator<<(const int_4& lhs, const int_4& rhs) __CPU_GPU__
{
  return int_4(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                lhs.get_z() << rhs.get_z(), lhs.get_w() << rhs.get_w());
}

inline int_4 operator>>(const int_4& lhs, const int_4& rhs) __CPU_GPU__
{
  return int_4(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                lhs.get_z() >> rhs.get_z(), lhs.get_w() >> rhs.get_w());
}

inline uint_4 operator%(const uint_4& lhs, const uint_4& rhs) __CPU_GPU__
{
  return uint_4(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                 lhs.get_z() % rhs.get_z(), lhs.get_w() % rhs.get_w());
}

inline uint_4 operator^(const uint_4& lhs, const uint_4& rhs) __CPU_GPU__
{
  return uint_4(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                 lhs.get_z() ^ rhs.get_z(), lhs.get_w() ^ rhs.get_w());
}

inline uint_4 operator|(const uint_4& lhs, const uint_4& rhs) __CPU_GPU__
{
  return uint_4(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                 lhs.get_z() | rhs.get_z(), lhs.get_w() | rhs.get_w());
}

inline uint_4 operator&(const uint_4& lhs, const uint_4& rhs) __CPU_GPU__
{
  return uint_4(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                 lhs.get_z() & rhs.get_z(), lhs.get_w() & rhs.get_w());
}

inline uint_4 operator<<(const uint_4& lhs, const uint_4& rhs) __CPU_GPU__
{
  return uint_4(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                 lhs.get_z() << rhs.get_z(), lhs.get_w() << rhs.get_w());
}

inline uint_4 operator>>(const uint_4& lhs, const uint_4& rhs) __CPU_GPU__
{
  return uint_4(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                 lhs.get_z() >> rhs.get_z(), lhs.get_w() >> rhs.get_w());
}

// C++ AMP Specification 10.9 short_vector
template<typename scalar_type, int size> struct short_vector
{
  short_vector()
  {
    // FIXME: Bug of Clang, passed under ICC 13 and VC++ 2012
    // static_assert(false, "short_vector is not supported for this scalar type (T) and length (N)");
  }
};

#define SHORT_VECTOR(ST, S, ST_S) \
template<> \
struct short_vector<ST, S> \
{ \
  typedef ST_S type; \
};

#if !__HCC_AMP__
SHORT_VECTOR(unsigned int, 1, uint_1)
#else
SHORT_VECTOR(unsigned int, 1, unsigned int)
#endif

SHORT_VECTOR(unsigned int, 2, uint_2)

SHORT_VECTOR(unsigned int, 3, uint_3)

SHORT_VECTOR(unsigned int, 4, uint_4)

#if !__HCC_AMP__
SHORT_VECTOR(int, 1, int_1)
#else
SHORT_VECTOR(int, 1, int)
#endif

SHORT_VECTOR(int, 2, int_2)

SHORT_VECTOR(int, 3, int_3)

SHORT_VECTOR(int, 4, int_4)

#if !__HCC_AMP__
SHORT_VECTOR(float, 1, float_1)
#else
SHORT_VECTOR(float, 1, float)
#endif

SHORT_VECTOR(float, 2, float_2)

SHORT_VECTOR(float, 3, float_3)

SHORT_VECTOR(float, 4, float_4)

SHORT_VECTOR(unorm, 1, unorm)

SHORT_VECTOR(unorm, 2, unorm_2)

SHORT_VECTOR(unorm, 3, unorm_3)

SHORT_VECTOR(unorm, 4, unorm_4)

SHORT_VECTOR(norm, 1, norm)

SHORT_VECTOR(norm, 2, norm_2)

SHORT_VECTOR(norm, 3, norm_3)

SHORT_VECTOR(norm, 4, norm_4)

#if !__HCC_AMP__
SHORT_VECTOR(double, 1, double_1)
#else
SHORT_VECTOR(double, 1, double)
#endif

SHORT_VECTOR(double, 2, double_2)

SHORT_VECTOR(double, 3, double_3)

SHORT_VECTOR(double, 4, double_4)

#if !__HCC_AMP__

SHORT_VECTOR(char, 1, char_1)

SHORT_VECTOR(char, 2, char_2)

SHORT_VECTOR(char, 3, char_3)

SHORT_VECTOR(char, 4, char_4)

SHORT_VECTOR(unsigned char, 1, uchar_1)

SHORT_VECTOR(unsigned char, 2, uchar_2)

SHORT_VECTOR(unsigned char, 3, uchar_3)

SHORT_VECTOR(unsigned char, 4, uchar_4)

SHORT_VECTOR(short, 1, short_1)

SHORT_VECTOR(short, 2, short_2)

SHORT_VECTOR(short, 3, short_3)

SHORT_VECTOR(short, 4, short_4)

SHORT_VECTOR(unsigned short, 1, ushort_1)

SHORT_VECTOR(unsigned short, 2, ushort_2)

SHORT_VECTOR(unsigned short, 3, ushort_3)

SHORT_VECTOR(unsigned short, 4, ushort_4)

SHORT_VECTOR(long, 1, long_1)

SHORT_VECTOR(long, 2, long_2)

SHORT_VECTOR(long, 3, long_3)

SHORT_VECTOR(long, 4, long_4)

SHORT_VECTOR(unsigned long, 1, ulong_1)

SHORT_VECTOR(unsigned long, 2, ulong_2)

SHORT_VECTOR(unsigned long, 3, ulong_3)

SHORT_VECTOR(unsigned long, 4, ulong_4)

SHORT_VECTOR(long long int, 1, longlong_1)

SHORT_VECTOR(long long int, 2, longlong_2)

SHORT_VECTOR(long long int, 3, longlong_3)

SHORT_VECTOR(long long int, 4, longlong_4)

SHORT_VECTOR(unsigned long long int, 1, ulonglong_1)

SHORT_VECTOR(unsigned long long int, 2, ulonglong_2)

SHORT_VECTOR(unsigned long long int, 3, ulonglong_3)

SHORT_VECTOR(unsigned long long int, 4, ulonglong_4)

#endif // if !__HCC_AMP__

#undef SHORT_VECTOR

// C++ AMP Specification 10.10 short_vector_traits
template<typename type> struct short_vector_traits
{
  short_vector_traits()
  {
    // FIXME: Bug of Clang, passed under ICC 13 and VC++ 2012
    // static_assert(false, "short_vector_traits is not supported for this type (type)");
  }
};

#define SHORT_VECTOR_TRAITS(ST, S, ST_S) \
template<> \
struct short_vector_traits<ST_S> \
{ \
  typedef ST value_type; \
  static int const size = S; \
};

#if !__HCC_AMP__
SHORT_VECTOR_TRAITS(unsigned int, 1, uint_1)
#else
SHORT_VECTOR_TRAITS(unsigned int, 1, unsigned int)
#endif

SHORT_VECTOR_TRAITS(unsigned int, 2, uint_2)

SHORT_VECTOR_TRAITS(unsigned int, 3, uint_3)

SHORT_VECTOR_TRAITS(unsigned int, 4, uint_4)

#if !__HCC_AMP__
SHORT_VECTOR_TRAITS(int, 1, int_1)
#else
SHORT_VECTOR_TRAITS(int, 1, int)
#endif

SHORT_VECTOR_TRAITS(int, 2, int_2)

SHORT_VECTOR_TRAITS(int, 3, int_3)

SHORT_VECTOR_TRAITS(int, 4, int_4)

#if !__HCC_AMP__
SHORT_VECTOR_TRAITS(float, 1, float_1)
#else
SHORT_VECTOR_TRAITS(float, 1, float)
#endif

SHORT_VECTOR_TRAITS(float, 2, float_2)

SHORT_VECTOR_TRAITS(float, 3, float_3)

SHORT_VECTOR_TRAITS(float, 4, float_4)

SHORT_VECTOR_TRAITS(unorm, 1, unorm)

SHORT_VECTOR_TRAITS(unorm, 2, unorm_2)

SHORT_VECTOR_TRAITS(unorm, 3, unorm_3)

SHORT_VECTOR_TRAITS(unorm, 4, unorm_4)

SHORT_VECTOR_TRAITS(norm, 1, norm)

SHORT_VECTOR_TRAITS(norm, 2, norm_2)

SHORT_VECTOR_TRAITS(norm, 3, norm_3)

SHORT_VECTOR_TRAITS(norm, 4, norm_4)

#if !__HCC_AMP__
SHORT_VECTOR_TRAITS(double, 1, double_1)
#else
SHORT_VECTOR_TRAITS(double, 1, double)
#endif

SHORT_VECTOR_TRAITS(double, 2, double_2)

SHORT_VECTOR_TRAITS(double, 3, double_3)

SHORT_VECTOR_TRAITS(double, 4, double_4)

#if !__HCC_AMP__

SHORT_VECTOR_TRAITS(char, 1, char_1)

SHORT_VECTOR_TRAITS(char, 2, char_2)

SHORT_VECTOR_TRAITS(char, 3, char_3)

SHORT_VECTOR_TRAITS(char, 4, char_4)

SHORT_VECTOR_TRAITS(unsigned char, 1, uchar_1)

SHORT_VECTOR_TRAITS(unsigned char, 2, uchar_2)

SHORT_VECTOR_TRAITS(unsigned char, 3, uchar_3)

SHORT_VECTOR_TRAITS(unsigned char, 4, uchar_4)

SHORT_VECTOR_TRAITS(short, 1, short_1)

SHORT_VECTOR_TRAITS(short, 2, short_2)

SHORT_VECTOR_TRAITS(short, 3, short_3)

SHORT_VECTOR_TRAITS(short, 4, short_4)

SHORT_VECTOR_TRAITS(unsigned short, 1, ushort_1)

SHORT_VECTOR_TRAITS(unsigned short, 2, ushort_2)

SHORT_VECTOR_TRAITS(unsigned short, 3, ushort_3)

SHORT_VECTOR_TRAITS(unsigned short, 4, ushort_4)

SHORT_VECTOR_TRAITS(long, 1, long_1)

SHORT_VECTOR_TRAITS(long, 2, long_2)

SHORT_VECTOR_TRAITS(long, 3, long_3)

SHORT_VECTOR_TRAITS(long, 4, long_4)

SHORT_VECTOR_TRAITS(unsigned long, 1, ulong_1)

SHORT_VECTOR_TRAITS(unsigned long, 2, ulong_2)

SHORT_VECTOR_TRAITS(unsigned long, 3, ulong_3)

SHORT_VECTOR_TRAITS(unsigned long, 4, ulong_4)

SHORT_VECTOR_TRAITS(long long int, 1, longlong_1)

SHORT_VECTOR_TRAITS(long long int, 2, longlong_2)

SHORT_VECTOR_TRAITS(long long int, 3, longlong_3)

SHORT_VECTOR_TRAITS(long long int, 4, longlong_4)

SHORT_VECTOR_TRAITS(unsigned long long int, 1, ulonglong_1)

SHORT_VECTOR_TRAITS(unsigned long long int, 2, ulonglong_2)

SHORT_VECTOR_TRAITS(unsigned long long int, 3, ulonglong_3)

SHORT_VECTOR_TRAITS(unsigned long long int, 4, ulonglong_4)

#endif // if !__HCC_AMP__

#undef SHORT_VECTOR_TRAITS

#endif // _KALMAR_SHORT_VECTORS_H
