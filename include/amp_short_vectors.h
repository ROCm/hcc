#pragma once

#ifndef _AMP_SHORT_VECTORS_H
#define _AMP_SHORT_VECTORS_H

namespace Concurrency
{
namespace graphics
{

class norm;
class unorm;

// Do not rely on macro rescanning and further replacement 

// FIXME: The explicit keyword doesn't work if we define constructor outside
//        the class definition (bug of AST inlining?)

#define NORM_COMMON_PRIVATE_MEMBER(F) \
friend class F; \
float Value; 

// FIXME: C() restrict(cpu, amp)'s behavior is not specified in Specification
/// C& operator=(const C& other) restrict(cpu, amp) do not need to check self-
/// assignment for accerlation on modern CPU
#define NORM_COMMON_PUBLIC_MEMBER(C) \
C() restrict(cpu, amp) { set(Value); } \
\
explicit C(float v) restrict(cpu, amp) { set(v); } \
\
explicit C(unsigned int v) restrict(cpu, amp) { set(static_cast<float>(v)); } \
\
explicit C(int v) restrict(cpu, amp) { set(static_cast<float>(v)); } \
\
explicit C(double v) restrict(cpu, amp) { set(static_cast<float>(v)); } \
\
C(const C& other) restrict(cpu, amp) { Value = other.Value; } \
\
C& operator=(const C& other) restrict(cpu, amp) \
{ \
  Value = other.Value; \
  return *this; \
} \
\
operator float(void) const restrict(cpu, amp) { return Value; } \
\
C& operator+=(const C& other) restrict(cpu, amp) \
{ \
  float Res = Value; \
  Res += other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator-=(const C& other) restrict(cpu, amp) \
{ \
  float Res = Value; \
  Res -= other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator*=(const C& other) restrict(cpu, amp) \
{ \
  float Res = Value; \
  Res *= other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator/=(const C& other) restrict(cpu, amp) \
{ \
  float Res = Value; \
  Res /= other.Value; \
  set(Res); \
  return *this; \
} \
\
C& operator++() restrict(cpu, amp) \
{ \
  float Res = Value; \
  ++Res; \
  set(Res); \
  return *this; \
} \
\
C operator++(int) restrict(cpu, amp) \
{ \
  C Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
C& operator--() restrict(cpu, amp) \
{ \
  float Res = Value; \
  --Res; \
  set(Res); \
  return *this; \
} \
\
C operator--(int) restrict(cpu, amp) \
{ \
  C Ret(*this); \
  operator--(); \
  return Ret; \
}

// C++ AMP Specification 10.7 norm
class norm
{
private:
  void set(float v) restrict(cpu, amp)
  {
    v = v < -1.0f ? -1.0f : v;
    v = v > 1.0f ? 1.0f : v;
    Value = v;
  }

public:
  NORM_COMMON_PRIVATE_MEMBER(unorm)

public:
  norm(const unorm& other) restrict(cpu, amp);

  norm operator-() restrict(cpu, amp)
  {
    norm Ret;
    Ret.Value = -Value;
    return Ret;
  }

  NORM_COMMON_PUBLIC_MEMBER(norm)
};

// C++ AMP Specification 10.7 unorm
class unorm
{
private:
  void set(float v) restrict(cpu, amp)
  {
    v = v < 0.0f ? 0.0f : v;
    v = v > 1.0f ? 1.0f : v;
    Value = v;
  }
public:
  NORM_COMMON_PRIVATE_MEMBER(norm)

public:
  explicit unorm(const norm& other) restrict(cpu, amp) { set(other.Value); }

  NORM_COMMON_PUBLIC_MEMBER(unorm)
};

norm::norm(const unorm& other) restrict(cpu, amp)
{
  set(other.Value);
}

#undef NORM_COMMON_PRIVATE_MEMBER
#undef NORM_COMMON_PUBLIC_MEMBER

#define NORM_OPERATOR(C) \
C operator+(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return C(static_cast<float>(lhs) + static_cast<float>(rhs)); \
} \
\
C operator-(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return C(static_cast<float>(lhs) - static_cast<float>(rhs)); \
} \
\
C operator*(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return C(static_cast<float>(lhs) * static_cast<float>(rhs)); \
} \
\
C operator/(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return C(static_cast<float>(lhs) / static_cast<float>(rhs)); \
} \
\
bool operator==(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return static_cast<float>(lhs) == static_cast<float>(rhs); \
} \
\
bool operator!=(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return static_cast<float>(lhs) != static_cast<float>(rhs); \
} \
\
bool operator>(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return static_cast<float>(lhs) > static_cast<float>(rhs); \
} \
\
bool operator<(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return static_cast<float>(lhs) < static_cast<float>(rhs); \
} \
\
bool operator>=(const C& lhs, const C& rhs) restrict(cpu, amp) \
{ \
  return static_cast<float>(lhs) >= static_cast<float>(rhs); \
} \
\
bool operator<=(const C& lhs, const C& rhs) restrict(cpu, amp) \
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


//   Class Declaration (10.8.1 Synopsis)

#define SINGLE_COMPONENT_ACCESS(ST, Dim) \
ST get ## _ ## Dim() const restrict(cpu, amp) { return Dim; } \
\
void set ## _ ## Dim(ST v) restrict(cpu, amp) { Dim = v; }

#define TWO_COMPONENT_ACCESS(ST_2, Dim1, Dim2) \
ST_2 get_ ## Dim1 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_2(Dim1, Dim2); \
} \
\
ST_2 get_ ## Dim2 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_2(Dim2, Dim1); \
} \
\
void set_ ## Dim1 ## Dim2(ST_2 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
} \
void set_ ## Dim2 ## Dim1(ST_2 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
}

#define THREE_COMPONENT_ACCESS(ST_3, Dim1, Dim2, Dim3) \
ST_3 get_ ## Dim1 ## Dim2 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_3(Dim1, Dim2, Dim3); \
} \
\
ST_3 get_ ## Dim1 ## Dim3 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_3(Dim1, Dim3, Dim2); \
} \
\
ST_3 get_ ## Dim2 ## Dim1 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_3(Dim2, Dim1, Dim3); \
} \
\
ST_3 get_ ## Dim2 ## Dim3 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_3(Dim2, Dim3, Dim1); \
} \
\
ST_3 get_ ## Dim3 ## Dim1 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_3(Dim3, Dim1, Dim2); \
} \
\
ST_3 get_ ## Dim3 ## Dim2 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_3(Dim3, Dim2, Dim1); \
} \
\
void set_ ## Dim1 ## Dim2 ## Dim3(ST_3 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim3 = v.get_z(); \
} \
\
void set_ ## Dim1 ## Dim3 ## Dim2(ST_3 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim2 = v.get_z(); \
} \
\
void set_ ## Dim2 ## Dim1 ## Dim3(ST_3 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim3 = v.get_z(); \
} \
\
void set_ ## Dim2 ## Dim3 ## Dim1(ST_3 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim1 = v.get_z(); \
} \
\
void set_ ## Dim3 ## Dim1 ## Dim2(ST_3 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim2 = v.get_z(); \
} \
\
void set_ ## Dim3 ## Dim2 ## Dim1(ST_3 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim1 = v.get_z(); \
}

#define FOUR_COMPONENT_ACCESS(ST_4, Dim1, Dim2, Dim3, Dim4) \
ST_4 get_ ## Dim1 ## Dim2 ## Dim3 ## Dim4() const restrict(cpu, amp) \
{ \
  return ST_4(Dim1, Dim2, Dim3, Dim4); \
} \
\
ST_4 get_ ## Dim1 ## Dim2 ## Dim4 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_4(Dim1, Dim2, Dim4, Dim3); \
} \
\
ST_4 get_ ## Dim1 ## Dim3 ## Dim2 ## Dim4() const restrict(cpu, amp) \
{ \
  return ST_4(Dim1, Dim3, Dim2, Dim4); \
} \
\
ST_4 get_ ## Dim1 ## Dim3 ## Dim4 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_4(Dim1, Dim3, Dim4, Dim2); \
} \
\
ST_4 get_ ## Dim1 ## Dim4 ## Dim2 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_4(Dim1, Dim4, Dim2, Dim3); \
} \
\
ST_4 get_ ## Dim1 ## Dim4 ## Dim3 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_4(Dim1, Dim4, Dim3, Dim2); \
} \
\
ST_4 get_ ## Dim2 ## Dim1 ## Dim3 ## Dim4() const restrict(cpu, amp) \
{ \
  return ST_4(Dim2, Dim1, Dim3, Dim4); \
} \
\
ST_4 get_ ## Dim2 ## Dim1 ## Dim4 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_4(Dim2, Dim1, Dim4, Dim3); \
} \
\
ST_4 get_ ## Dim2 ## Dim3 ## Dim1 ## Dim4() const restrict(cpu, amp) \
{ \
  return ST_4(Dim2, Dim3, Dim1, Dim4); \
} \
\
ST_4 get_ ## Dim2 ## Dim3 ## Dim4 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_4(Dim2, Dim3, Dim4, Dim1); \
} \
\
ST_4 get_ ## Dim2 ## Dim4 ## Dim1 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_4(Dim2, Dim4, Dim1, Dim3); \
} \
\
ST_4 get_ ## Dim2 ## Dim4 ## Dim3 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_4(Dim2, Dim4, Dim3, Dim1); \
} \
\
ST_4 get_ ## Dim3 ## Dim1 ## Dim2 ## Dim4() const restrict(cpu, amp) \
{ \
  return ST_4(Dim3, Dim1, Dim2, Dim4); \
} \
\
ST_4 get_ ## Dim3 ## Dim1 ## Dim4 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_4(Dim3, Dim1, Dim4, Dim2); \
} \
\
ST_4 get_ ## Dim3 ## Dim2 ## Dim1 ## Dim4() const restrict(cpu, amp) \
{ \
  return ST_4(Dim3, Dim2, Dim1, Dim4); \
} \
\
ST_4 get_ ## Dim3 ## Dim2 ## Dim4 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_4(Dim3, Dim2, Dim4, Dim1); \
} \
\
ST_4 get_ ## Dim3 ## Dim4 ## Dim1 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_4(Dim3, Dim4, Dim1, Dim2); \
} \
\
ST_4 get_ ## Dim3 ## Dim4 ## Dim2 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_4(Dim3, Dim4, Dim2, Dim1); \
} \
\
ST_4 get_ ## Dim4 ## Dim1 ## Dim2 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_4(Dim4, Dim1, Dim2, Dim3); \
} \
\
ST_4 get_ ## Dim4 ## Dim1 ## Dim3 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_4(Dim4, Dim1, Dim3, Dim2); \
} \
\
ST_4 get_ ## Dim4 ## Dim2 ## Dim1 ## Dim3() const restrict(cpu, amp) \
{ \
  return ST_4(Dim4, Dim2, Dim1, Dim3); \
} \
\
ST_4 get_ ## Dim4 ## Dim2 ## Dim3 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_4(Dim4, Dim2, Dim3, Dim1); \
} \
\
ST_4 get_ ## Dim4 ## Dim3 ## Dim1 ## Dim2() const restrict(cpu, amp) \
{ \
  return ST_4(Dim4, Dim3, Dim1, Dim2); \
} \
\
ST_4 get_ ## Dim4 ## Dim3 ## Dim2 ## Dim1() const restrict(cpu, amp) \
{ \
  return ST_4(Dim4, Dim3, Dim2, Dim1); \
} \
\
void set_ ## Dim1 ## Dim2 ## Dim3 ## Dim4(ST_4 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim2 ## Dim4 ## Dim3(ST_4 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim3 ## Dim2 ## Dim4(ST_4 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim3 ## Dim4 ## Dim2(ST_4 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim4 ## Dim2 ## Dim3(ST_4 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim1 ## Dim4 ## Dim3 ## Dim2(ST_4 v) restrict(cpu, amp) \
{ \
  Dim1 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim1 ## Dim3 ## Dim4(ST_4 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim1 ## Dim4 ## Dim3(ST_4 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim3 ## Dim1 ## Dim4(ST_4 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim3 ## Dim4 ## Dim1(ST_4 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim4 ## Dim1 ## Dim3(ST_4 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim2 ## Dim4 ## Dim3 ## Dim1(ST_4 v) restrict(cpu, amp) \
{ \
  Dim2 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim1 ## Dim2 ## Dim4(ST_4 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim1 ## Dim4 ## Dim2(ST_4 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim2 ## Dim1 ## Dim4(ST_4 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim4 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim2 ## Dim4 ## Dim1(ST_4 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim4 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim4 ## Dim1 ## Dim2(ST_4 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim3 ## Dim4 ## Dim2 ## Dim1(ST_4 v) restrict(cpu, amp) \
{ \
  Dim3 = v.get_x(); \
  Dim4 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim1 ## Dim2 ## Dim3(ST_4 v) restrict(cpu, amp) \
{ \
  Dim4 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim1 ## Dim3 ## Dim2(ST_4 v) restrict(cpu, amp) \
{ \
  Dim4 = v.get_x(); \
  Dim1 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim2 ## Dim1 ## Dim3(ST_4 v) restrict(cpu, amp) \
{ \
  Dim4 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim3 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim2 ## Dim3 ## Dim1(ST_4 v) restrict(cpu, amp) \
{ \
  Dim4 = v.get_x(); \
  Dim2 = v.get_y(); \
  Dim3 = v.get_z(); \
  Dim1 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim3 ## Dim1 ## Dim2(ST_4 v) restrict(cpu, amp) \
{ \
  Dim4 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim1 = v.get_z(); \
  Dim2 = v.get_w(); \
} \
\
void set_ ## Dim4 ## Dim3 ## Dim2 ## Dim1(ST_4 v) restrict(cpu, amp) \
{ \
  Dim4 = v.get_x(); \
  Dim3 = v.get_y(); \
  Dim2 = v.get_z(); \
  Dim1 = v.get_w(); \
}

#define SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() restrict(cpu, amp) { return x; } \
\
ST& ref_y() restrict(cpu, amp) { return y; } \
\
ST& ref_r() restrict(cpu, amp) { return x; } \
\
ST& ref_g() restrict(cpu, amp) { return y; }


#define SCALARTYPE_2_COMMON_PUBLIC_MEMBER(ST, ST_2, \
ST_2_o1, ST_2_o2, ST_2_o3, ST_2_o4, ST_2_o5) \
ST x; \
ST y; \
typedef ST value_type; \
static const int size = 2; \
\
ST_2() restrict(cpu, amp) {} \
\
ST_2(ST value) restrict(cpu, amp) \
{ \
  x = value; \
  y = value; \
} \
\
ST_2(const ST_2&  other) restrict(cpu, amp) \
{ \
  x = other.x; \
  y = other.y; \
} \
\
ST_2(ST v1, ST v2) restrict (cpu, amp) \
{ \
  x = v1; \
  y = v2; \
} \
\
explicit ST_2(const ST_2_o1& other) restrict(cpu, amp); \
\
explicit ST_2(const ST_2_o2& other) restrict(cpu, amp); \
\
explicit ST_2(const ST_2_o3& other) restrict(cpu, amp); \
\
explicit ST_2(const ST_2_o4& other) restrict(cpu, amp); \
\
explicit ST_2(const ST_2_o5& other) restrict(cpu, amp); \
\
ST_2& operator=(const ST_2& other) restrict(cpu, amp) \
{ \
  x = other.x; \
  y = other.y; \
  return *this; \
} \
\
ST_2& operator++() restrict(cpu, amp) \
{ \
  ++x; \
  ++y; \
  return *this; \
} \
\
ST_2 operator++(int) restrict(cpu, amp) \
{ \
  ST_2 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_2& operator--() restrict(cpu, amp) \
{ \
  --x; \
  --y; \
  return *this; \
} \
\
ST_2 operator--(int) restrict(cpu, amp) \
{ \
  ST_2 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_2& operator+=(const ST_2& rhs) restrict(cpu, amp) \
{ \
  x += rhs.x; \
  y += rhs.y; \
  return *this; \
} \
\
ST_2& operator-=(const ST_2& rhs) restrict(cpu, amp) \
{ \
  x -= rhs.x; \
  y -= rhs.y; \
  return *this; \
} \
\
ST_2& operator*=(const ST_2& rhs) restrict(cpu, amp) \
{ \
  x *= rhs.x; \
  y *= rhs.y; \
  return *this; \
} \
\
ST_2& operator/=(const ST_2& rhs) restrict(cpu, amp) \
{ \
  x /= rhs.x; \
  y /= rhs.y; \
  return *this; \
}

class int_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(int, int_2, 
    uint_2, float_2, double_2, norm_2, unorm_2) 

  int_2 operator-() const restrict(cpu, amp) { return int_2(-x, -y); }

  int_2 operator~() const restrict(cpu, amp) { return int_2(~x, ~y); }

  int_2& operator%=(const int_2& rhs) restrict(cpu, amp)
  {
    x %= rhs.x;
    y %= rhs.y;
    return *this;
  }

  int_2& operator^=(const int_2& rhs) restrict(cpu, amp)
  {
    x ^= rhs.x;
    y ^= rhs.y;
    return *this;
  }

  int_2& operator|=(const int_2& rhs) restrict(cpu, amp)
  {
    x |= rhs.x;
    y |= rhs.y;
    return *this;
  }

  int_2& operator&=(const int_2& rhs) restrict(cpu, amp)
  {
    x &= rhs.x;
    y &= rhs.y;
    return *this;
  }

  int_2& operator>>=(const int_2& rhs) restrict(cpu, amp)
  {
    x >>= rhs.x;
    y >>= rhs.y;
    return *this;
  }

  int_2& operator<<=(const int_2& rhs) restrict(cpu, amp)
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
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unsigned int, uint_2, 
    int_2, float_2, double_2, norm_2, unorm_2) 
 
  uint_2 operator~() const restrict(cpu, amp) { return uint_2(~x, ~y); }

  uint_2& operator%=(const uint_2& rhs) restrict(cpu, amp)
  {
    x %= rhs.x;
    y %= rhs.y;
    return *this;
  }

  uint_2& operator^=(const uint_2& rhs) restrict(cpu, amp)
  {
    x ^= rhs.x;
    y ^= rhs.y;
    return *this;
  }

  uint_2& operator|=(const uint_2& rhs) restrict(cpu, amp)
  {
    x |= rhs.x;
    y |= rhs.y;
    return *this;
  }

  uint_2& operator&=(const uint_2& rhs) restrict(cpu, amp)
  {
    x &= rhs.x;
    y &= rhs.y;
    return *this;
  }

  uint_2& operator>>=(const uint_2& rhs) restrict(cpu, amp)
  {
    x >>= rhs.x;
    y >>= rhs.y;
    return *this;
  }

  uint_2& operator<<=(const uint_2& rhs) restrict(cpu, amp)
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
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(float, float_2, 
    int_2, uint_2, double_2, norm_2, unorm_2) 
  
  float_2 operator-() const restrict(cpu, amp) { return float_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(float, x)
  SINGLE_COMPONENT_ACCESS(float, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(float)

  TWO_COMPONENT_ACCESS(float_2, x, y)
};

class double_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(double, double_2, 
    int_2, uint_2, float_2, norm_2, unorm_2) 
  
  double_2 operator-() const restrict(cpu, amp) { return double_2(-x, -y); }

  SINGLE_COMPONENT_ACCESS(double, x)
  SINGLE_COMPONENT_ACCESS(double, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(double)

  TWO_COMPONENT_ACCESS(double_2, x, y)
};

class norm_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(norm, norm_2, 
    int_2, uint_2, float_2, double_2, unorm_2) 

  norm_2 operator-() const restrict(cpu, amp) { return norm_2(-x, -y); }
  
  SINGLE_COMPONENT_ACCESS(norm, x)
  SINGLE_COMPONENT_ACCESS(norm, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(norm)

  TWO_COMPONENT_ACCESS(norm_2, x, y)
};

class unorm_2
{
public:
  SCALARTYPE_2_COMMON_PUBLIC_MEMBER(unorm, unorm_2, 
    int_2, uint_2, float_2, double_2, norm_2) 

  SINGLE_COMPONENT_ACCESS(unorm, x)
  SINGLE_COMPONENT_ACCESS(unorm, y)

  SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS(unorm)

  TWO_COMPONENT_ACCESS(unorm_2, x, y)
};

#undef SCALARTYPE_2_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_2_COMMON_PUBLIC_MEMBER

#define SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() restrict(cpu, amp) { return x; } \
\
ST& ref_y() restrict(cpu, amp) { return y; } \
\
ST& ref_z() restrict(cpu, amp) { return z; } \
\
ST& ref_r() restrict(cpu, amp) { return x; } \
\
ST& ref_g() restrict(cpu, amp) { return y; } \
\
ST& ref_b() restrict(cpu, amp) { return z; }

#define SCALARTYPE_3_COMMON_PUBLIC_MEMBER(ST, ST_3, \
ST_3_o1, ST_3_o2, ST_3_o3, ST_3_o4, ST_3_o5) \
ST x; \
ST y; \
ST z; \
typedef ST value_type; \
static const int size = 3; \
\
ST_3() restrict(cpu, amp) {} \
\
ST_3(ST value) restrict(cpu, amp) \
{ \
  x = value; \
  y = value; \
  z = value; \
} \
\
ST_3(const ST_3&  other) restrict(cpu, amp) \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
} \
\
ST_3(ST v1, ST v2, ST v3) restrict (cpu, amp) \
{ \
  x = v1; \
  y = v2; \
  z = v3; \
} \
\
explicit ST_3(const ST_3_o1& other) restrict(cpu, amp); \
\
explicit ST_3(const ST_3_o2& other) restrict(cpu, amp); \
\
explicit ST_3(const ST_3_o3& other) restrict(cpu, amp); \
\
explicit ST_3(const ST_3_o4& other) restrict(cpu, amp); \
\
explicit ST_3(const ST_3_o5& other) restrict(cpu, amp); \
\
ST_3& operator=(const ST_3& other) restrict(cpu, amp) \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
  return *this; \
} \
\
ST_3& operator++() restrict(cpu, amp) \
{ \
  ++x; \
  ++y; \
  ++z; \
  return *this; \
} \
\
ST_3 operator++(int) restrict(cpu, amp) \
{ \
  ST_3 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_3& operator--() restrict(cpu, amp) \
{ \
  --x; \
  --y; \
  --z; \
  return *this; \
} \
\
ST_3 operator--(int) restrict(cpu, amp) \
{ \
  ST_3 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_3& operator+=(const ST_3& rhs) restrict(cpu, amp) \
{ \
  x += rhs.x; \
  y += rhs.y; \
  z += rhs.z; \
  return *this; \
} \
\
ST_3& operator-=(const ST_3& rhs) restrict(cpu, amp) \
{ \
  x -= rhs.x; \
  y -= rhs.y; \
  z -= rhs.z; \
  return *this; \
} \
\
ST_3& operator*=(const ST_3& rhs) restrict(cpu, amp) \
{ \
  x *= rhs.x; \
  y *= rhs.y; \
  z *= rhs.z; \
  return *this; \
} \
\
ST_3& operator/=(const ST_3& rhs) restrict(cpu, amp) \
{ \
  x /= rhs.x; \
  y /= rhs.y; \
  z /= rhs.z; \
  return *this; \
}

class int_3
{
public:
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(int, int_3, 
    uint_3, float_3, double_3, norm_3, unorm_3) 

  int_3 operator-() const restrict(cpu, amp) { return int_3(-x, -y, -z); }

  int_3 operator~() const restrict(cpu, amp) { return int_3(~x, ~y, -z); }

  int_3& operator%=(const int_3& rhs) restrict(cpu, amp)
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    return *this;
  }

  int_3& operator^=(const int_3& rhs) restrict(cpu, amp)
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    return *this;
  }

  int_3& operator|=(const int_3& rhs) restrict(cpu, amp)
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    return *this;
  }

  int_3& operator&=(const int_3& rhs) restrict(cpu, amp)
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    return *this;
  }

  int_3& operator>>=(const int_3& rhs) restrict(cpu, amp)
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    return *this;
  }

  int_3& operator<<=(const int_3& rhs) restrict(cpu, amp)
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
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unsigned int, uint_3, 
    int_3, float_3, double_3, norm_3, unorm_3) 
 
  uint_3 operator~() const restrict(cpu, amp) { return uint_3(~x, ~y, ~z); }

  uint_3& operator%=(const uint_3& rhs) restrict(cpu, amp)
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    return *this;
  }

  uint_3& operator^=(const uint_3& rhs) restrict(cpu, amp)
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    return *this;
  }

  uint_3& operator|=(const uint_3& rhs) restrict(cpu, amp)
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    return *this;
  }

  uint_3& operator&=(const uint_3& rhs) restrict(cpu, amp)
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    return *this;
  }

  uint_3& operator>>=(const uint_3& rhs) restrict(cpu, amp)
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    return *this;
  }

  uint_3& operator<<=(const uint_3& rhs) restrict(cpu, amp)
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
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(float, float_3, 
    int_3, uint_3, double_3, norm_3, unorm_3) 
  
  float_3 operator-() const restrict(cpu, amp) { return float_3(-x, -y, -z); }

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
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(double, double_3, 
    int_3, uint_3, float_3, norm_3, unorm_3) 
  
  double_3 operator-() const restrict(cpu, amp) { return double_3(-x, -y, -z); }

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
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(norm, norm_3, 
    int_3, uint_3, float_3, double_3, unorm_3) 

  norm_3 operator-() const restrict(cpu, amp) { return norm_3(-x, -y, -z); }
  
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
  SCALARTYPE_3_COMMON_PUBLIC_MEMBER(unorm, unorm_3, 
    int_3, uint_3, float_3, double_3, norm_3) 

  SINGLE_COMPONENT_ACCESS(unorm, x)
  SINGLE_COMPONENT_ACCESS(unorm, y)
  SINGLE_COMPONENT_ACCESS(unorm, z)

  SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS(unorm)

  TWO_COMPONENT_ACCESS(unorm_2, x, y)
  TWO_COMPONENT_ACCESS(unorm_2, x, z)
  TWO_COMPONENT_ACCESS(unorm_2, y, z)

  THREE_COMPONENT_ACCESS(unorm_3, x, y, z)
};

#undef SCALARTYPE_3_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_3_COMMON_PUBLIC_MEMBER

#define SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS(ST) \
ST& ref_x() restrict(cpu, amp) { return x; } \
\
ST& ref_y() restrict(cpu, amp) { return y; } \
\
ST& ref_z() restrict(cpu, amp) { return z; } \
\
ST& ref_w() restrict(cpu, amp) { return w; } \
\
ST& ref_r() restrict(cpu, amp) { return x; } \
\
ST& ref_g() restrict(cpu, amp) { return y; } \
\
ST& ref_b() restrict(cpu, amp) { return z; } \
\
ST& ref_a() restrict(cpu, amp) { return w; }

#define SCALARTYPE_4_COMMON_PUBLIC_MEMBER(ST, ST_4, \
ST_4_o1, ST_4_o2, ST_4_o3, ST_4_o4, ST_4_o5) \
ST x; \
ST y; \
ST z; \
ST w; \
typedef ST value_type; \
static const int size = 4; \
\
ST_4() restrict(cpu, amp) {} \
\
ST_4(ST value) restrict(cpu, amp) \
{ \
  x = value; \
  y = value; \
  z = value; \
  w = value; \
} \
\
ST_4(const ST_4&  other) restrict(cpu, amp) \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
  w = other.w; \
} \
\
ST_4(ST v1, ST v2, ST v3, ST v4) restrict (cpu, amp) \
{ \
  x = v1; \
  y = v2; \
  z = v3; \
  w = v4; \
} \
\
explicit ST_4(const ST_4_o1& other) restrict(cpu, amp); \
\
explicit ST_4(const ST_4_o2& other) restrict(cpu, amp); \
\
explicit ST_4(const ST_4_o3& other) restrict(cpu, amp); \
\
explicit ST_4(const ST_4_o4& other) restrict(cpu, amp); \
\
explicit ST_4(const ST_4_o5& other) restrict(cpu, amp); \
\
ST_4& operator=(const ST_4& other) restrict(cpu, amp) \
{ \
  x = other.x; \
  y = other.y; \
  z = other.z; \
  w = other.w; \
  return *this; \
} \
\
ST_4& operator++() restrict(cpu, amp) \
{ \
  ++x; \
  ++y; \
  ++z; \
  ++w; \
  return *this; \
} \
\
ST_4 operator++(int) restrict(cpu, amp) \
{ \
  ST_4 Ret(*this); \
  operator++(); \
  return Ret; \
} \
\
ST_4& operator--() restrict(cpu, amp) \
{ \
  --x; \
  --y; \
  --z; \
  --w; \
  return *this; \
} \
\
ST_4 operator--(int) restrict(cpu, amp) \
{ \
  ST_4 Ret(*this); \
  operator--(); \
  return Ret; \
} \
\
ST_4& operator+=(const ST_4& rhs) restrict(cpu, amp) \
{ \
  x += rhs.x; \
  y += rhs.y; \
  z += rhs.z; \
  w += rhs.w; \
  return *this; \
} \
\
ST_4& operator-=(const ST_4& rhs) restrict(cpu, amp) \
{ \
  x -= rhs.x; \
  y -= rhs.y; \
  z -= rhs.z; \
  w -= rhs.w; \
  return *this; \
} \
\
ST_4& operator*=(const ST_4& rhs) restrict(cpu, amp) \
{ \
  x *= rhs.x; \
  y *= rhs.y; \
  z *= rhs.z; \
  w *= rhs.w; \
  return *this; \
} \
\
ST_4& operator/=(const ST_4& rhs) restrict(cpu, amp) \
{ \
  x /= rhs.x; \
  y /= rhs.y; \
  z /= rhs.z; \
  w /= rhs.w; \
  return *this; \
}

class int_4
{
public:
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(int, int_4, 
    uint_4, float_4, double_4, norm_4, unorm_4) 

  int_4 operator-() const restrict(cpu, amp) { return int_4(-x, -y, -z, -w); }

  int_4 operator~() const restrict(cpu, amp) { return int_4(~x, ~y, -z, -w); }

  int_4& operator%=(const int_4& rhs) restrict(cpu, amp)
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    w %= rhs.w;
    return *this;
  }

  int_4& operator^=(const int_4& rhs) restrict(cpu, amp)
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    w ^= rhs.w;
    return *this;
  }

  int_4& operator|=(const int_4& rhs) restrict(cpu, amp)
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    w |= rhs.w;
    return *this;
  }

  int_4& operator&=(const int_4& rhs) restrict(cpu, amp)
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    w &= rhs.w;
    return *this;
  }

  int_4& operator>>=(const int_4& rhs) restrict(cpu, amp)
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    w >>= rhs.w;
    return *this;
  }

  int_4& operator<<=(const int_4& rhs) restrict(cpu, amp)
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
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unsigned int, uint_4, 
    int_4, float_4, double_4, norm_4, unorm_4) 
 
  uint_4 operator~() const restrict(cpu, amp) { return uint_4(~x, ~y, ~z, -w); }

  uint_4& operator%=(const uint_4& rhs) restrict(cpu, amp)
  {
    x %= rhs.x;
    y %= rhs.y;
    z %= rhs.z;
    w %= rhs.w;
    return *this;
  }

  uint_4& operator^=(const uint_4& rhs) restrict(cpu, amp)
  {
    x ^= rhs.x;
    y ^= rhs.y;
    z ^= rhs.z;
    w ^= rhs.w;
    return *this;
  }

  uint_4& operator|=(const uint_4& rhs) restrict(cpu, amp)
  {
    x |= rhs.x;
    y |= rhs.y;
    z |= rhs.z;
    w |= rhs.w;
    return *this;
  }

  uint_4& operator&=(const uint_4& rhs) restrict(cpu, amp)
  {
    x &= rhs.x;
    y &= rhs.y;
    z &= rhs.z;
    w &= rhs.w;
    return *this;
  }

  uint_4& operator>>=(const uint_4& rhs) restrict(cpu, amp)
  {
    x >>= rhs.x;
    y >>= rhs.y;
    z >>= rhs.z;
    w >>= rhs.w;
    return *this;
  }

  uint_4& operator<<=(const uint_4& rhs) restrict(cpu, amp)
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
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(float, float_4, 
    int_4, uint_4, double_4, norm_4, unorm_4) 
  
  float_4 operator-() const restrict(cpu, amp) { return float_4(-x, -y, -z, -w); }

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
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(double, double_4, 
    int_4, uint_4, float_4, norm_4, unorm_4) 
  
  double_4 operator-() const restrict(cpu, amp) { return double_4(-x, -y, -z, -w); }

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
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(norm, norm_4, 
    int_4, uint_4, float_4, double_4, unorm_4) 

  norm_4 operator-() const restrict(cpu, amp) { return norm_4(-x, -y, -z, -w); }
  
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
  SCALARTYPE_4_COMMON_PUBLIC_MEMBER(unorm, unorm_4, 
    int_4, uint_4, float_4, double_4, norm_4) 

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

#undef SCALARTYPE_4_REFERENCE_SINGLE_COMPONENT_ACCESS
#undef SCALARTYPE_4_COMMON_PUBLIC_MEMBER

#undef SINGLE_COMPONENT_ACCESS
#undef TWO_COMPONENT_ACCESS
#undef THREE_COMPONENT_ACCESS
#undef FOUR_COMPONENT_ACCESS

//   Explicit Conversion Constructor Definitions (10.8.2.2)

#define SCALARTYPE_2_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_2, \
ST_2_o1, ST_2_o2, ST_2_o3, ST_2_o4, ST_2_o5) \
ST_2::ST_2(const ST_2_o1& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
ST_2::ST_2(const ST_2_o2& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
ST_2::ST_2(const ST_2_o3& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
ST_2::ST_2(const ST_2_o4& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
} \
\
ST_2::ST_2(const ST_2_o5& other) restrict(cpu, amp) \
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

#define SCALARTYPE_3_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_3, \
ST_3_o1, ST_3_o2, ST_3_o3, ST_3_o4, ST_3_o5) \
ST_3::ST_3(const ST_3_o1& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
ST_3::ST_3(const ST_3_o2& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
ST_3::ST_3(const ST_3_o3& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
ST_3::ST_3(const ST_3_o4& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
} \
\
ST_3::ST_3(const ST_3_o5& other) restrict(cpu, amp) \
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

#define SCALARTYPE_4_EXPLICIT_CONVERSION_CONSTRUCTORS(ST, ST_4, \
ST_4_o1, ST_4_o2, ST_4_o3, ST_4_o4, ST_4_o5) \
ST_4::ST_4(const ST_4_o1& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
ST_4::ST_4(const ST_4_o2& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
ST_4::ST_4(const ST_4_o3& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
ST_4::ST_4(const ST_4_o4& other) restrict(cpu, amp) \
{ \
  x = static_cast<ST>(other.get_x()); \
  y = static_cast<ST>(other.get_y()); \
  z = static_cast<ST>(other.get_z()); \
  w = static_cast<ST>(other.get_w()); \
} \
\
ST_4::ST_4(const ST_4_o5& other) restrict(cpu, amp) \
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

//   Operators between Two References (10.8.1 Synopsis)

#define SCALARTYPE_2_OPERATOR(ST_2) \
ST_2 operator+(const ST_2& lhs, const ST_2& rhs) restrict(cpu, amp) \
{ \
  return ST_2(lhs.get_x() + rhs.get_x(), lhs.get_y() + rhs.get_y()); \
} \
\
ST_2 operator-(const ST_2& lhs, const ST_2& rhs) restrict(cpu, amp) \
{ \
  return ST_2(lhs.get_x() - rhs.get_x(), lhs.get_y() - rhs.get_y()); \
} \
\
ST_2 operator*(const ST_2& lhs, const ST_2& rhs) restrict(cpu, amp) \
{ \
  return ST_2(lhs.get_x() * rhs.get_x(), lhs.get_y() * rhs.get_y()); \
} \
\
ST_2 operator/(const ST_2& lhs, const ST_2& rhs) restrict(cpu, amp) \
{ \
  return ST_2(lhs.get_x() / rhs.get_x(), lhs.get_y() / rhs.get_y()); \
} \
\
bool operator==(const ST_2& lhs, const ST_2& rhs) restrict(cpu, amp) \
{ \
  return (lhs.get_x() == rhs.get_x()) && (lhs.get_y() == rhs.get_y()); \
} \
\
bool operator!=(const ST_2& lhs, const ST_2& rhs) restrict(cpu, amp) \
{ \
  return (lhs.get_x() != rhs.get_x()) || (lhs.get_y() != rhs.get_y()); \
}

SCALARTYPE_2_OPERATOR(int_2)

SCALARTYPE_2_OPERATOR(uint_2)

SCALARTYPE_2_OPERATOR(float_2)

SCALARTYPE_2_OPERATOR(double_2)

SCALARTYPE_2_OPERATOR(norm_2)

SCALARTYPE_2_OPERATOR(unorm_2)

#undef SCALARTYPE_2_OPERATOR

int_2 operator%(const int_2& lhs, const int_2& rhs) restrict(cpu, amp)
{
  return int_2(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y());
}

int_2 operator^(const int_2& lhs, const int_2& rhs) restrict(cpu, amp)
{
  return int_2(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y());
}

int_2 operator|(const int_2& lhs, const int_2& rhs) restrict(cpu, amp)
{
  return int_2(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y());
}

int_2 operator&(const int_2& lhs, const int_2& rhs) restrict(cpu, amp)
{
  return int_2(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y());
}

int_2 operator<<(const int_2& lhs, const int_2& rhs) restrict(cpu, amp)
{
  return int_2(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y());
}

int_2 operator>>(const int_2& lhs, const int_2& rhs) restrict(cpu, amp)
{
  return int_2(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y());
}

uint_2 operator%(const uint_2& lhs, const uint_2& rhs) restrict(cpu, amp)
{
  return uint_2(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y());
}

uint_2 operator^(const uint_2& lhs, const uint_2& rhs) restrict(cpu, amp)
{
  return uint_2(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y());
}

uint_2 operator|(const uint_2& lhs, const uint_2& rhs) restrict(cpu, amp)
{
  return uint_2(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y());
}

uint_2 operator&(const uint_2& lhs, const uint_2& rhs) restrict(cpu, amp)
{
  return uint_2(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y());
}

uint_2 operator<<(const uint_2& lhs, const uint_2& rhs) restrict(cpu, amp)
{
  return uint_2(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y());
}

uint_2 operator>>(const uint_2& lhs, const uint_2& rhs) restrict(cpu, amp)
{
  return uint_2(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y());
}

#define SCALARTYPE_3_OPERATOR(ST_3) \
ST_3 operator+(const ST_3& lhs, const ST_3& rhs) restrict(cpu, amp) \
{ \
  return ST_3(lhs.get_x() + rhs.get_x(), lhs.get_y() + rhs.get_y(), \
               lhs.get_z() + rhs.get_z()); \
} \
\
ST_3 operator-(const ST_3& lhs, const ST_3& rhs) restrict(cpu, amp) \
{ \
  return ST_3(lhs.get_x() - rhs.get_x(), lhs.get_y() - rhs.get_y(), \
               lhs.get_z() - rhs.get_z()); \
} \
\
ST_3 operator*(const ST_3& lhs, const ST_3& rhs) restrict(cpu, amp) \
{ \
  return ST_3(lhs.get_x() * rhs.get_x(), lhs.get_y() * rhs.get_y(), \
               lhs.get_z() * rhs.get_z()); \
} \
\
ST_3 operator/(const ST_3& lhs, const ST_3& rhs) restrict(cpu, amp) \
{ \
  return ST_3(lhs.get_x() / rhs.get_x(), lhs.get_y() / rhs.get_y(), \
               lhs.get_z() / rhs.get_z()); \
} \
\
bool operator==(const ST_3& lhs, const ST_3& rhs) restrict(cpu, amp) \
{ \
  return (lhs.get_x() == rhs.get_x()) && (lhs.get_y() == rhs.get_y()) \
           && (lhs.get_z() == rhs.get_z()); \
} \
\
bool operator!=(const ST_3& lhs, const ST_3& rhs) restrict(cpu, amp) \
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

#undef SCALARTYPE_3_OPERATOR

int_3 operator%(const int_3& lhs, const int_3& rhs) restrict(cpu, amp)
{
  return int_3(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                lhs.get_z() % rhs.get_z());
}

int_3 operator^(const int_3& lhs, const int_3& rhs) restrict(cpu, amp)
{
  return int_3(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                lhs.get_z() ^ rhs.get_z());
}

int_3 operator|(const int_3& lhs, const int_3& rhs) restrict(cpu, amp)
{
  return int_3(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                lhs.get_z() | rhs.get_z());
}

int_3 operator&(const int_3& lhs, const int_3& rhs) restrict(cpu, amp)
{
  return int_3(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                lhs.get_z() & rhs.get_z());
}

int_3 operator<<(const int_3& lhs, const int_3& rhs) restrict(cpu, amp)
{
  return int_3(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                lhs.get_z() << rhs.get_z());
}

int_3 operator>>(const int_3& lhs, const int_3& rhs) restrict(cpu, amp)
{
  return int_3(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                lhs.get_z() >> rhs.get_z());
}

uint_3 operator%(const uint_3& lhs, const uint_3& rhs) restrict(cpu, amp)
{
  return uint_3(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                 lhs.get_z() % rhs.get_z());
}

uint_3 operator^(const uint_3& lhs, const uint_3& rhs) restrict(cpu, amp)
{
  return uint_3(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                 lhs.get_z() ^ rhs.get_z());
}

uint_3 operator|(const uint_3& lhs, const uint_3& rhs) restrict(cpu, amp)
{
  return uint_3(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                 lhs.get_z() | rhs.get_z());
}

uint_3 operator&(const uint_3& lhs, const uint_3& rhs) restrict(cpu, amp)
{
  return uint_3(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                 lhs.get_z() & rhs.get_z());
}

uint_3 operator<<(const uint_3& lhs, const uint_3& rhs) restrict(cpu, amp)
{
  return uint_3(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                 lhs.get_z() << rhs.get_z());
}

uint_3 operator>>(const uint_3& lhs, const uint_3& rhs) restrict(cpu, amp)
{
  return uint_3(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                 lhs.get_z() >> rhs.get_z());
}

#define SCALARTYPE_4_OPERATOR(ST_4) \
ST_4 operator+(const ST_4& lhs, const ST_4& rhs) restrict(cpu, amp) \
{ \
  return ST_4(lhs.get_x() + rhs.get_x(), lhs.get_y() + rhs.get_y(), \
               lhs.get_z() + rhs.get_z(), lhs.get_w() + rhs.get_w()); \
} \
\
ST_4 operator-(const ST_4& lhs, const ST_4& rhs) restrict(cpu, amp) \
{ \
  return ST_4(lhs.get_x() - rhs.get_x(), lhs.get_y() - rhs.get_y(), \
               lhs.get_z() - rhs.get_z(), lhs.get_w() - rhs.get_w()); \
} \
\
ST_4 operator*(const ST_4& lhs, const ST_4& rhs) restrict(cpu, amp) \
{ \
  return ST_4(lhs.get_x() * rhs.get_x(), lhs.get_y() * rhs.get_y(), \
               lhs.get_z() * rhs.get_z(), lhs.get_w() * rhs.get_w()); \
} \
\
ST_4 operator/(const ST_4& lhs, const ST_4& rhs) restrict(cpu, amp) \
{ \
  return ST_4(lhs.get_x() / rhs.get_x(), lhs.get_y() / rhs.get_y(), \
               lhs.get_z() / rhs.get_z(), lhs.get_w() / rhs.get_w()); \
} \
\
bool operator==(const ST_4& lhs, const ST_4& rhs) restrict(cpu, amp) \
{ \
  return (lhs.get_x() == rhs.get_x()) && (lhs.get_y() == rhs.get_y()) \
           && (lhs.get_z() == rhs.get_z()) && (lhs.get_w() == rhs.get_w()); \
} \
\
bool operator!=(const ST_4& lhs, const ST_4& rhs) restrict(cpu, amp) \
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

#undef SCALARTYPE_4_OPERATOR

int_4 operator%(const int_4& lhs, const int_4& rhs) restrict(cpu, amp)
{
  return int_4(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                lhs.get_z() % rhs.get_z(), lhs.get_w() % rhs.get_w());
}

int_4 operator^(const int_4& lhs, const int_4& rhs) restrict(cpu, amp)
{
  return int_4(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                lhs.get_z() ^ rhs.get_z(), lhs.get_w() ^ rhs.get_w());
}

int_4 operator|(const int_4& lhs, const int_4& rhs) restrict(cpu, amp)
{
  return int_4(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                lhs.get_z() | rhs.get_z(), lhs.get_w() | rhs.get_w());
}

int_4 operator&(const int_4& lhs, const int_4& rhs) restrict(cpu, amp)
{
  return int_4(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                lhs.get_z() & rhs.get_z(), lhs.get_w() & rhs.get_w());
}

int_4 operator<<(const int_4& lhs, const int_4& rhs) restrict(cpu, amp)
{
  return int_4(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                lhs.get_z() << rhs.get_z(), lhs.get_w() << rhs.get_w());
}

int_4 operator>>(const int_4& lhs, const int_4& rhs) restrict(cpu, amp)
{
  return int_4(lhs.get_x() >> rhs.get_x(), lhs.get_y() >> rhs.get_y(),
                lhs.get_z() >> rhs.get_z(), lhs.get_w() >> rhs.get_w());
}

uint_4 operator%(const uint_4& lhs, const uint_4& rhs) restrict(cpu, amp)
{
  return uint_4(lhs.get_x() % rhs.get_x(), lhs.get_y() % rhs.get_y(),
                 lhs.get_z() % rhs.get_z(), lhs.get_w() % rhs.get_w());
}

uint_4 operator^(const uint_4& lhs, const uint_4& rhs) restrict(cpu, amp)
{
  return uint_4(lhs.get_x() ^ rhs.get_x(), lhs.get_y() ^ rhs.get_y(),
                 lhs.get_z() ^ rhs.get_z(), lhs.get_w() ^ rhs.get_w());
}

uint_4 operator|(const uint_4& lhs, const uint_4& rhs) restrict(cpu, amp)
{
  return uint_4(lhs.get_x() | rhs.get_x(), lhs.get_y() | rhs.get_y(),
                 lhs.get_z() | rhs.get_z(), lhs.get_w() | rhs.get_w());
}

uint_4 operator&(const uint_4& lhs, const uint_4& rhs) restrict(cpu, amp)
{
  return uint_4(lhs.get_x() & rhs.get_x(), lhs.get_y() & rhs.get_y(),
                 lhs.get_z() & rhs.get_z(), lhs.get_w() & rhs.get_w());
}

uint_4 operator<<(const uint_4& lhs, const uint_4& rhs) restrict(cpu, amp)
{
  return uint_4(lhs.get_x() << rhs.get_x(), lhs.get_y() << rhs.get_y(),
                 lhs.get_z() << rhs.get_z(), lhs.get_w() << rhs.get_w());
}

uint_4 operator>>(const uint_4& lhs, const uint_4& rhs) restrict(cpu, amp)
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

SHORT_VECTOR(unsigned int, 1, unsigned int)

SHORT_VECTOR(unsigned int, 2, uint_2)

SHORT_VECTOR(unsigned int, 3, uint_3)

SHORT_VECTOR(unsigned int, 4, uint_4)

SHORT_VECTOR(int, 1, int)

SHORT_VECTOR(int, 2, int_2)

SHORT_VECTOR(int, 3, int_3)

SHORT_VECTOR(int, 4, int_4)

SHORT_VECTOR(float, 1, float)

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

SHORT_VECTOR(double, 1, double)

SHORT_VECTOR(double, 2, double_2)

SHORT_VECTOR(double, 3, double_3)

SHORT_VECTOR(double, 4, double_4)

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

SHORT_VECTOR_TRAITS(unsigned int, 1, unsigned int)

SHORT_VECTOR_TRAITS(unsigned int, 2, uint_2)

SHORT_VECTOR_TRAITS(unsigned int, 3, uint_3)

SHORT_VECTOR_TRAITS(unsigned int, 4, uint_4)

SHORT_VECTOR_TRAITS(int, 1, int)

SHORT_VECTOR_TRAITS(int, 2, int_2)

SHORT_VECTOR_TRAITS(int, 3, int_3)

SHORT_VECTOR_TRAITS(int, 4, int_4)

SHORT_VECTOR_TRAITS(float, 1, float)

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

SHORT_VECTOR_TRAITS(double, 1, double)

SHORT_VECTOR_TRAITS(double, 2, double_2)

SHORT_VECTOR_TRAITS(double, 3, double_3)

SHORT_VECTOR_TRAITS(double, 4, double_4)

#undef SHORT_VECTOR_TRAITS

} // namespace graphics
} // namespace Concurrency

#endif // _AMP_SHORT_VECTORS_H
