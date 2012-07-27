/* ymath.h internal header */
#pragma once
#ifndef _YMATH
#define _YMATH
#ifndef RC_INVOKED
#include <yvals.h>
_C_STD_BEGIN
 #ifndef _M_CEE_PURE
_C_LIB_DECL
 #endif /* _M_CEE_PURE */

#pragma pack(push, _CRT_PACKING)
#pragma push_macro("new")
#undef new

		/* MACROS FOR _Dtest RETURN (0 => ZERO) */
#define _DENORM		(-2)	/* C9X only */
#define _FINITE		(-1)
#define _INFCODE	1
#define _NANCODE	2

		/* MACROS FOR _Feraise ARGUMENT */
#define _FE_DIVBYZERO	0x04
#define _FE_INEXACT	0x20
#define _FE_INVALID	0x01
#define _FE_OVERFLOW	0x08
#define _FE_UNDERFLOW	0x10

		/* TYPE DEFINITIONS */
typedef union
	{	/* pun float types as integer array */
	unsigned short _Word[8];
	float _Float;
	double _Double;
	long double _Long_double;
	} _Dconst;

		/* ERROR REPORTING */
void __CLRCALL_PURE_OR_CDECL _Feraise(int);

		/* double DECLARATIONS */
_CRTIMP2_PURE double __CLRCALL_PURE_OR_CDECL _Cosh(double, double);
_CRTIMP2_PURE short __CLRCALL_PURE_OR_CDECL _Dtest(double *);
_CRTIMP2_PURE short __CLRCALL_PURE_OR_CDECL _Exp(double *, double, short);
_CRTIMP2_PURE double __CLRCALL_PURE_OR_CDECL _Sinh(double, double);
extern _CRTIMP2_PURE /* const */ _Dconst _Denorm, _Hugeval, _Inf,
	_Nan, _Snan;

		/* float DECLARATIONS */
_CRTIMP2_PURE float __CLRCALL_PURE_OR_CDECL _FCosh(float, float);
_CRTIMP2_PURE short __CLRCALL_PURE_OR_CDECL _FDtest(float *);
_CRTIMP2_PURE short __CLRCALL_PURE_OR_CDECL _FExp(float *, float, short);
_CRTIMP2_PURE float __CLRCALL_PURE_OR_CDECL _FSinh(float, float);
extern _CRTIMP2_PURE /* const */ _Dconst _FDenorm, _FInf, _FNan, _FSnan;

		/* long double DECLARATIONS */
_CRTIMP2_PURE long double __CLRCALL_PURE_OR_CDECL _LCosh(long double, long double);
_CRTIMP2_PURE short __CLRCALL_PURE_OR_CDECL _LDtest(long double *);
_CRTIMP2_PURE short __CLRCALL_PURE_OR_CDECL _LExp(long double *, long double, short);
_CRTIMP2_PURE long double __CLRCALL_PURE_OR_CDECL _LSinh(long double, long double);
extern _CRTIMP2_PURE /* const */ _Dconst _LDenorm, _LInf, _LNan, _LSnan;

 #ifndef _M_CEE_PURE
_END_C_LIB_DECL
 #endif /* _M_CEE_PURE */

#pragma pop_macro("new")
#pragma pack(pop)
_C_STD_END
#endif /* RC_INVOKED */
#endif /* _YMATH */

/*
 * Copyright (c) 1992-2012 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V6.00:0009 */
