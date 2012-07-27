/* xtgmath.h internal header */
#pragma once
#ifndef _XTGMATH
#define _XTGMATH
#ifndef RC_INVOKED

 #if defined(__cplusplus)
#include <xtr1common>

 #pragma pack(push,_CRT_PACKING)
 #pragma warning(push,3)
 #pragma push_macro("new")
 #undef new

_STD_BEGIN
template<class _Ty>
	struct _Promote_to_float
	{	// promote integral to double
	typedef typename conditional<is_integral<_Ty>::value,
		double, _Ty>::type type;
	};

template<class _Ty1,
	class _Ty2>
	struct _Common_float_type
	{	// find type for two-argument math function
	typedef typename _Promote_to_float<_Ty1>::type _Ty1f;
	typedef typename _Promote_to_float<_Ty2>::type _Ty2f;
	typedef typename conditional<is_same<_Ty1f, long double>::value
		|| is_same<_Ty2f, long double>::value, long double,
		typename conditional<is_same<_Ty1f, double>::value
			|| is_same<_Ty2f, double>::value, double,
			float>::type>::type type;
	};
_STD_END

#define _CRTDEFAULT
#define _CRTSPECIAL	_CRTIMP

#define _GENERIC_MATH1(FUN, CRTTYPE) \
extern "C" CRTTYPE double __cdecl FUN(_In_ double); \
template<class _Ty> inline \
	typename _STD enable_if< _STD is_integral<_Ty>::value, double>::type \
	FUN(_Ty _Left) \
	{ \
	return (_CSTD FUN((double)_Left)); \
	}

#define _GENERIC_MATH1X(FUN, ARG2, CRTTYPE) \
extern "C" CRTTYPE double __cdecl FUN(_In_ double, ARG2); \
template<class _Ty> inline \
	typename _STD enable_if< _STD is_integral<_Ty>::value, double>::type \
	FUN(_Ty _Left, ARG2 _Arg2) \
	{ \
	return (_CSTD FUN((double)_Left, _Arg2)); \
	}

#define _GENERIC_MATH2(FUN, CRTTYPE) \
extern "C" CRTTYPE double __cdecl FUN(_In_ double, _In_ double); \
template<class _Ty1, \
	class _Ty2> inline \
	typename _STD _Common_float_type<_Ty1, _Ty2>::type \
	FUN(_Ty1 _Left, _Ty2 _Right) \
	{ \
	typedef typename _STD _Common_float_type<_Ty1, _Ty2>::type type; \
	return (_CSTD FUN((type)_Left, (type)_Right)); \
	}

_C_STD_BEGIN
extern "C" double __cdecl pow(_In_ double, _In_ double);
float __CRTDECL  pow(_In_ float, _In_ float);
long double __CRTDECL  pow(_In_ long double, _In_ long double);

template<class _Ty1,
	class _Ty2> inline
	typename _STD enable_if< _STD _Is_numeric<_Ty1>::value
		&& _STD _Is_numeric<_Ty2>::value,
		typename _STD _Common_float_type<_Ty1, _Ty2>::type>::type
	pow(const _Ty1 _Left, const _Ty2 _Right)
	{	// bring mixed types to a common type
	typedef typename _STD _Common_float_type<_Ty1, _Ty2>::type type;
	return (_CSTD pow(type(_Left), type(_Right)));
	}

//_GENERIC_MATH1(abs, _CRTDEFAULT)	// has integer overloads
_GENERIC_MATH1(acos, _CRTDEFAULT)
_GENERIC_MATH1(asin, _CRTDEFAULT)
_GENERIC_MATH1(atan, _CRTDEFAULT)
_GENERIC_MATH2(atan2, _CRTDEFAULT)
_GENERIC_MATH1(ceil, _CRTSPECIAL)
_GENERIC_MATH1(cos, _CRTDEFAULT)
_GENERIC_MATH1(cosh, _CRTDEFAULT)
_GENERIC_MATH1(exp, _CRTDEFAULT)
//_GENERIC_MATH1(fabs, _CRT_JIT_INTRINSIC)	// not required
_GENERIC_MATH1(floor, _CRTSPECIAL)
_GENERIC_MATH2(fmod, _CRTDEFAULT)
_GENERIC_MATH1X(frexp, _Out_ int *, _CRTSPECIAL)
_GENERIC_MATH1X(ldexp, _In_ int, _CRTSPECIAL)
_GENERIC_MATH1(log, _CRTDEFAULT)
_GENERIC_MATH1(log10, _CRTDEFAULT)
//_GENERIC_MATH1(modf, _CRTDEFAULT)		// types must match
//_GENERIC_MATH2(pow, _CRTDEFAULT)	// hand crafted
_GENERIC_MATH1(sin, _CRTDEFAULT)
_GENERIC_MATH1(sinh, _CRTDEFAULT)
_GENERIC_MATH1(sqrt, _CRTDEFAULT)
_GENERIC_MATH1(tan, _CRTDEFAULT)
_GENERIC_MATH1(tanh, _CRTDEFAULT)
_C_STD_END
 #endif /* defined(__cplusplus) */

 #pragma pop_macro("new")
 #pragma warning(pop)
 #pragma pack(pop)
#endif /* RC_INVOKED */
#endif /* _XTGMATH */

/*
 * Copyright (c) 1992-2012 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V6.00:0009 */
