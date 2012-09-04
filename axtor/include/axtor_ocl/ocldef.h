#include <float.h>
#include <stddef.h>
// #include <stdint.h>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define INT_FUNCTIONS_ONE_ARG(name) \
__attribute__((overloadable, pure)) \
  int16 name (int16); \
  __attribute__((overloadable, pure)) \
  int8 name (int8); \
  __attribute__((overloadable, pure)) \
  int4 name (int4); \
  __attribute__((overloadable, pure)) \
  int3 name (int3); \
  __attribute__((overloadable, pure)) \
  int2 name (int2); \
  __attribute__((overloadable, pure)) \
  int name (int)

#define FLOAT_FUNCTIONS_ONE_ARG(name) \
  __attribute__((overloadable, pure)) \
  float16 name (float16); \
  __attribute__((overloadable, pure)) \
  float8 name (float8); \
  __attribute__((overloadable, pure)) \
  float4 name (float4); \
  __attribute__((overloadable, pure)) \
  float3 name (float3); \
  __attribute__((overloadable, pure)) \
  float2 name (float2); \
  __attribute__((overloadable, pure)) \
  float name (float); \
  __attribute__((overloadable, pure)) \
  double16 name (double16); \
  __attribute__((overloadable, pure)) \
  double8 name (double8); \
  __attribute__((overloadable, pure)) \
  double4 name (double4); \
  __attribute__((overloadable, pure)) \
  double3 name (double3); \
  __attribute__((overloadable, pure)) \
  double2 name (double2); \
  __attribute__((overloadable, pure)) \
  double name (double)

#define INT_FUNCTIONS_TWO_ARGS(name) \
  __attribute__((overloadable, pure)) \
  int16 name (int16, int16); \
  __attribute__((overloadable, pure)) \
  int8 name (int8, int8); \
  __attribute__((overloadable, pure)) \
  int4 name (int4, int4); \
  __attribute__((overloadable, pure)) \
  int3 name (int3, int3); \
  __attribute__((overloadable, pure)) \
  int2 name (int2, int2); \
  __attribute__((overloadable, pure)) \
  int name (int , int)

#define FLOAT_FUNCTIONS_TWO_ARGS(name) \
  __attribute__((overloadable, pure)) \
  float16 name (float16, float16); \
  __attribute__((overloadable, pure)) \
  float8 name (float8, float8); \
  __attribute__((overloadable, pure)) \
  float4 name (float4, float4); \
  __attribute__((overloadable, pure)) \
  float3 name (float3, float3); \
  __attribute__((overloadable, pure)) \
  float2 name (float2, float2); \
  __attribute__((overloadable, pure)) \
  float name (float , float); \
  __attribute__((overloadable, pure)) \
  double16 name (double16, double16); \
  __attribute__((overloadable, pure)) \
  double8 name (double8, double8); \
  __attribute__((overloadable, pure)) \
  double4 name (double4, double4); \
  __attribute__((overloadable, pure)) \
  double3 name (double3, double3); \
  __attribute__((overloadable, pure)) \
  double2 name (double2, double2); \
  __attribute__((overloadable, pure)) \
  double name (double, double)

#define FLOAT_FUNCTIONS_THREE_ARGS(name) \
  __attribute__((overloadable, pure)) \
  float16 name (float16, float16, float16); \
  __attribute__((overloadable, pure)) \
  float8 name (float8, float8, float8); \
  __attribute__((overloadable, pure)) \
  float4 name (float4, float4, float4); \
  __attribute__((overloadable, pure)) \
  float3 name (float3, float3, float3); \
  __attribute__((overloadable, pure)) \
  float2 name (float2, float2, float2); \
  __attribute__((overloadable, pure)) \
  float name (float , float, float); \
  __attribute__((overloadable, pure)) \
  double16 name (double16, double16, double16); \
  __attribute__((overloadable, pure)) \
  double8 name (double8, double8, double8); \
  __attribute__((overloadable, pure)) \
  double4 name (double4, double4, double4); \
  __attribute__((overloadable, pure)) \
  double3 name (double3, double3, double3); \
  __attribute__((overloadable, pure)) \
  double2 name (double2, double2, double2); \
  __attribute__((overloadable, pure)) \
  double name (double, double, double)


#define FUNCTIONS_TWO_ARGS(name) \
	FLOAT_FUNCTIONS_TWO_ARGS(name); \
	INT_FUNCTIONS_TWO_ARGS(name)

#define CONVERT_FUNCTION(to, from) \
  __attribute__((overloadable, pure)) \
  to convert_##to (from)

// These macros are defined in math.h, but because we
// cannot include it, define them here. Definitions picked-up
// from GNU math.h.
#define M_E            2.71828182845904523540f  // e          
#define M_LOG2E        1.44269504088896340740f  // log_2 e    
#define M_LOG10E       0.43429448190325182765f  // log_10 e   
#define M_LN2          0.69314718055994530942f  // log_e 2    
#define M_LN10         2.30258509299404568402f  // log_e 10   
#define M_PI           3.14159265358979323846f  // pi         
#define M_PI_2         1.57079632679489661923f  // pi/2       
#define M_PI_4         0.78539816339744830962f  // pi/4       
#define M_1_PI         0.31830988618379067154f  // 1/pi       
#define M_2_PI         0.63661977236758134308f  // 2/pi       
#define M_2_SQRTPI     1.12837916709551257390f  // 2/sqrt(pi) 
#define M_SQRT2        1.41421356237309504880f  // sqrt(2)    
#define M_SQRT1_2      0.70710678118654752440f  // 1/sqrt(2)

// Synchronization Macros.
#define CLK_LOCAL_MEM_FENCE  0
#define CLK_GLOBAL_MEM_FENCE 1

// Built-in Scalar Data Types.
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

// Built-in Vector Data Types.
typedef __attribute__((ext_vector_type(16))) uint uint16;
typedef __attribute__((ext_vector_type(2))) uint uint2;
typedef __attribute__((ext_vector_type(3))) uint uint3;
typedef __attribute__((ext_vector_type(4))) uint uint4;
typedef __attribute__((ext_vector_type(8))) uint uint8;
typedef __attribute__((ext_vector_type(16))) uchar uchar16;
typedef __attribute__((ext_vector_type(2))) uchar uchar2;
typedef __attribute__((ext_vector_type(3))) uchar uchar3;
typedef __attribute__((ext_vector_type(4))) uchar uchar4;
typedef __attribute__((ext_vector_type(8))) uchar uchar8;
typedef __attribute__((ext_vector_type(16))) char char16;
typedef __attribute__((ext_vector_type(2))) char char2;
typedef __attribute__((ext_vector_type(3))) char char3;
typedef __attribute__((ext_vector_type(4))) char char4;
typedef __attribute__((ext_vector_type(8))) char char8;
typedef __attribute__((ext_vector_type(16))) float float16;
typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(3))) float float3;
typedef __attribute__((ext_vector_type(4))) float float4;
typedef __attribute__((ext_vector_type(8))) float float8;
typedef __attribute__((ext_vector_type(16))) int int16;
typedef __attribute__((ext_vector_type(2))) int int2;
typedef __attribute__((ext_vector_type(3))) int int3;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(8))) int int8;
typedef __attribute__((ext_vector_type(16))) long long16;
typedef __attribute__((ext_vector_type(2))) long long2;
typedef __attribute__((ext_vector_type(3))) long long3;
typedef __attribute__((ext_vector_type(4))) long long4;
typedef __attribute__((ext_vector_type(8))) long long8;
typedef __attribute__((ext_vector_type(16))) short short16;
typedef __attribute__((ext_vector_type(2))) short short2;
typedef __attribute__((ext_vector_type(3))) short short3;
typedef __attribute__((ext_vector_type(4))) short short4;
typedef __attribute__((ext_vector_type(8))) short short8;
typedef __attribute__((ext_vector_type(16))) double double16;
typedef __attribute__((ext_vector_type(2))) double double2;
typedef __attribute__((ext_vector_type(3))) double double3;
typedef __attribute__((ext_vector_type(4))) double double4;
typedef __attribute__((ext_vector_type(8))) double double8;

// Other derived types.
typedef ulong cl_mem_fence_flags;

// Work-Item Functions.
uint get_work_dim();
size_t get_global_size(uint dimindx);
size_t get_global_id(uint dimindx);
size_t get_local_size(uint dimindx);
size_t get_local_id(uint dimindx);
size_t get_num_groups(uint dimindx);
size_t get_group_id(uint dimindx);
size_t get_global_offset(uint dimindx);

// Synchronization Functions.
void barrier(cl_mem_fence_flags flag);
void mem_fence(cl_mem_fence_flags flags);

#define INT_ATOMIC_TWO_32(NAME) \
__attribute__((overloadable)) \
	int NAME(volatile __local int* p, int cmp, int val); \
__attribute__((overloadable)) \
	int NAME(volatile __global int* p, int cmp, int val); \
\
__attribute__((overloadable)) \
  uint NAME(volatile __local uint* p, uint cmp, uint val); \
__attribute__((overloadable)) \
  uint NAME(volatile __global uint* p, uint cmp, uint val);

#define INT_ATOMIC_TWO_64(NAME) \
__attribute__((overloadable)) \
  long NAME(volatile __local long* p, long cmp, long val); \
__attribute__((overloadable)) \
  long NAME(volatile __global long* p, long cmp, long val); \
\
  __attribute__((overloadable)) \
  ulong NAME(volatile __local ulong* p, ulong cmp, ulong val); \
__attribute__((overloadable)) \
  ulong NAME(volatile __global ulong* p, ulong cmp, ulong val);

#define INT_ATOMIC_ONE_32(NAME) \
__attribute__((overloadable)) \
	int NAME(volatile __local int* p, int val); \
__attribute__((overloadable)) \
	int NAME(volatile __global int* p, int val); \
\
__attribute__((overloadable)) \
  uint NAME(volatile __local uint* p, uint val); \
__attribute__((overloadable)) \
  uint NAME(volatile __global uint* p, uint val);

#define INT_ATOMIC_ONE_64(NAME) \
__attribute__((overloadable)) \
  long NAME(volatile __local long* p, long val); \
__attribute__((overloadable)) \
  long NAME(volatile __global long* p, long val); \
\
  __attribute__((overloadable)) \
  ulong NAME(volatile __local ulong* p, ulong val); \
__attribute__((overloadable)) \
  ulong NAME(volatile __global ulong* p, ulong val);

#define INT_ATOMIC_ZERO_32(NAME) \
__attribute__((overloadable)) \
	int NAME(volatile __local int* p); \
__attribute__((overloadable)) \
	int NAME(volatile __global int* p); \
\
__attribute__((overloadable)) \
  uint NAME(volatile __local uint* p); \
__attribute__((overloadable)) \
  uint NAME(volatile __global uint* p);

#define INT_ATOMIC_ZERO_64(NAME) \
__attribute__((overloadable)) \
	long NAME(volatile __local long* p); \
__attribute__((overloadable)) \
	long NAME(volatile __global long* p); \
\
__attribute__((overloadable)) \
  ulong NAME(volatile __local ulong* p); \
__attribute__((overloadable)) \
  ulong NAME(volatile __global ulong* p);

// aliases for 32-bit builtins in the atom_* namespace that is actually for 64-bit builtins
#define INT_ATOMIC_TWO_32_ALIASES(NAME) \
		INT_ATOMIC_TWO_32(atomic_##NAME) \
		INT_ATOMIC_TWO_32(atom_##NAME)

#define INT_ATOMIC_ONE_32_ALIASES(NAME) \
		INT_ATOMIC_ONE_32(atomic_##NAME) \
		INT_ATOMIC_ONE_32(atom_##NAME)

#define INT_ATOMIC_ZERO_32_ALIASES(NAME) \
		INT_ATOMIC_ZERO_32(atomic_##NAME) \
		INT_ATOMIC_ZERO_32(atom_##NAME)

// Atom(ic) Functions
INT_ATOMIC_TWO_32_ALIASES(cmpxchg)
INT_ATOMIC_ONE_32_ALIASES(add)
INT_ATOMIC_ONE_32_ALIASES(sub)
INT_ATOMIC_ONE_32_ALIASES(xchg)
INT_ATOMIC_ONE_32_ALIASES(min)
INT_ATOMIC_ONE_32_ALIASES(max)
INT_ATOMIC_ONE_32_ALIASES(and)
INT_ATOMIC_ONE_32_ALIASES(or)
INT_ATOMIC_ONE_32_ALIASES(xor)
INT_ATOMIC_ZERO_32_ALIASES(inc)
INT_ATOMIC_ZERO_32_ALIASES(dec)

INT_ATOMIC_TWO_64(atom_cmpxchg)
INT_ATOMIC_ONE_64(atom_add)
INT_ATOMIC_ONE_64(atom_sub)
INT_ATOMIC_ONE_64(atom_xchg)
INT_ATOMIC_ONE_64(atom_min)
INT_ATOMIC_ONE_64(atom_max)
INT_ATOMIC_ONE_64(atom_and)
INT_ATOMIC_ONE_64(atom_or)
INT_ATOMIC_ONE_64(atom_xor)
INT_ATOMIC_ZERO_64(atom_inc)
INT_ATOMIC_ZERO_64(atom_dec)


__attribute__((overloadable))
float atomic_xchg(volatile __global float * p, float val);


//Predicates
__attribute__((overloadable, pure))
  int all(int4);



// Math functions.
// WARNING There are more definitions than needed.
FLOAT_FUNCTIONS_ONE_ARG(acos);
FLOAT_FUNCTIONS_ONE_ARG(acosh);
FLOAT_FUNCTIONS_ONE_ARG(acospi);
FLOAT_FUNCTIONS_ONE_ARG(cos);
FLOAT_FUNCTIONS_ONE_ARG(exp);
FLOAT_FUNCTIONS_ONE_ARG(asin);
FLOAT_FUNCTIONS_ONE_ARG(asinh);
FLOAT_FUNCTIONS_ONE_ARG(asinpi);
FLOAT_FUNCTIONS_ONE_ARG(sin);
FLOAT_FUNCTIONS_ONE_ARG(sqrt);
FLOAT_FUNCTIONS_ONE_ARG(fabs);
FLOAT_FUNCTIONS_ONE_ARG(log);
FLOAT_FUNCTIONS_ONE_ARG(log2);
FLOAT_FUNCTIONS_ONE_ARG(rint);
FLOAT_FUNCTIONS_ONE_ARG(native_sqrt);
FLOAT_FUNCTIONS_ONE_ARG(native_sin);
FLOAT_FUNCTIONS_ONE_ARG(native_cos);
FLOAT_FUNCTIONS_ONE_ARG(native_recip);
FLOAT_FUNCTIONS_ONE_ARG(normalize);
FLOAT_FUNCTIONS_ONE_ARG(fast_normalize);
FLOAT_FUNCTIONS_ONE_ARG(rsqrt);
FLOAT_FUNCTIONS_ONE_ARG(floor);
FLOAT_FUNCTIONS_ONE_ARG(native_log2);
FLOAT_FUNCTIONS_ONE_ARG(atan);
FLOAT_FUNCTIONS_ONE_ARG(ceil);
FLOAT_FUNCTIONS_ONE_ARG(floor);


FLOAT_FUNCTIONS_TWO_ARGS(pow);
FLOAT_FUNCTIONS_TWO_ARGS(hypot);
FLOAT_FUNCTIONS_TWO_ARGS(native_divide);

// Geometric
FLOAT_FUNCTIONS_TWO_ARGS(cross);

FUNCTIONS_TWO_ARGS(max);
FUNCTIONS_TWO_ARGS(min);

FLOAT_FUNCTIONS_THREE_ARGS(mix);
FLOAT_FUNCTIONS_THREE_ARGS(fma);
FLOAT_FUNCTIONS_THREE_ARGS(mad);
FLOAT_FUNCTIONS_THREE_ARGS(clamp);

CONVERT_FUNCTION(float4, int4);
CONVERT_FUNCTION(float4, uint4);
CONVERT_FUNCTION(float4, char4);
CONVERT_FUNCTION(float4, uchar4);
CONVERT_FUNCTION(int4, uint4);
CONVERT_FUNCTION(int4, char4);
CONVERT_FUNCTION(int4, uchar4);
CONVERT_FUNCTION(uint4, uchar4);
CONVERT_FUNCTION(float, uint);
CONVERT_FUNCTION(uint, float);
CONVERT_FUNCTION(uchar4, int4);
CONVERT_FUNCTION(uchar4, float4);
CONVERT_FUNCTION(double4, uint4);
CONVERT_FUNCTION(int8, double8);

__attribute__((overloadable, pure))
  uchar4 convert_uchar4_sat(float4);

__attribute__((overloadable, pure))
  uint mul24(uint, uint);
__attribute__((overloadable, pure))
  int mul24(int, int);

// Geometrib built-in functions

#define FLOAT_FUNCTION_TWO_ARGS_RED(name) \
		__attribute__((overloadable, pure)) \
		  float name(float4, float4); \
	    __attribute__((overloadable, pure)) \
		  float name(float3, float3); \
		__attribute__((overloadable, pure)) \
		  float name(float2, float2); \
		__attribute__((overloadable, pure)) \
		  float name(double4, double4); \
		__attribute__((overloadable, pure)) \
		  float name(double3, double3); \
		__attribute__((overloadable, pure)) \
		  float name(double2, double2);

#define FLOAT_FUNCTION_ONE_ARG_RED(name) \
		__attribute__((overloadable, pure)) \
		  float name(float4); \
	    __attribute__((overloadable, pure)) \
		  float name(float3); \
		__attribute__((overloadable, pure)) \
		  float name(float2); \
		__attribute__((overloadable, pure)) \
		  float name(double4); \
		__attribute__((overloadable, pure)) \
		  float name(double3); \
		__attribute__((overloadable, pure)) \
		  float name(double2);


FLOAT_FUNCTION_TWO_ARGS_RED(dot)
FLOAT_FUNCTION_TWO_ARGS_RED(distance)
FLOAT_FUNCTION_TWO_ARGS_RED(fast_distance)
FLOAT_FUNCTION_ONE_ARG_RED(length)
FLOAT_FUNCTION_ONE_ARG_RED(fast_length)
