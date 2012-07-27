/***
*math.h - definitions and declarations for math library
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file contains constant definitions and external subroutine
*       declarations for the math subroutine library.
*       [ANSI/System V]
*
*       [Public]
*
****/

#ifndef _INC_MATH
#define _INC_MATH

#include <crtdefs.h>

/*
 * Currently, all MS C compilers for Win32 platforms default to 8 byte
 * alignment.
 */
#pragma pack(push,_CRT_PACKING)

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __assembler /* Protect from assembler */

/* Definition of _exception struct - this struct is passed to the matherr
 * routine when a floating point exception is detected
 */

#ifndef _EXCEPTION_DEFINED
struct _exception {
        int type;       /* exception type - see below */
        char *name;     /* name of function where error occured */
        double arg1;    /* first argument to function */
        double arg2;    /* second argument (if any) to function */
        double retval;  /* value to be returned by function */
        } ;

#define _EXCEPTION_DEFINED
#endif


/* Definition of a _complex struct to be used by those who use cabs and
 * want type checking on their argument
 */

#ifndef _COMPLEX_DEFINED
struct _complex {
        double x,y; /* real and imaginary parts */
        } ;

#if     !__STDC__ && !defined (__cplusplus)
/* Non-ANSI name for compatibility */
#define complex _complex
#endif

#define _COMPLEX_DEFINED
#endif
#endif  /* __assembler */


/* Constant definitions for the exception type passed in the _exception struct
 */

#define _DOMAIN     1   /* argument domain error */
#define _SING       2   /* argument singularity */
#define _OVERFLOW   3   /* overflow range error */
#define _UNDERFLOW  4   /* underflow range error */
#define _TLOSS      5   /* total loss of precision */
#define _PLOSS      6   /* partial loss of precision */

#define EDOM        33
#define ERANGE      34


/* Definitions of _HUGE and HUGE_VAL - respectively the XENIX and ANSI names
 * for a value returned in case of error by a number of the floating point
 * math routines
 */
#ifndef __assembler /* Protect from assembler */
#if !defined(_M_CEE_PURE)
_CRTIMP extern double _HUGE;
#else
const double _HUGE = System::Double::PositiveInfinity;
#endif /* !defined(_M_CEE_PURE) */
#endif  /* __assembler */

#define HUGE_VAL _HUGE

/* Function prototypes */

#if     !defined(__assembler)   /* Protect from assembler */
#ifndef _CRT_ABS_DEFINED
#define _CRT_ABS_DEFINED
        int     __cdecl abs(_In_ int _X);
        long    __cdecl labs(_In_ long _X);
        long long __cdecl llabs(_In_ long long _X);
#endif

        double  __cdecl acos(_In_ double _X);
        double  __cdecl asin(_In_ double _X);
        double  __cdecl atan(_In_ double _X);
        double  __cdecl atan2(_In_ double _Y, _In_ double _X);
#ifndef _SIGN_DEFINED
_Check_return_ _CRTIMP double __cdecl _copysign (_In_ double _Number, _In_ double _Sign);
_Check_return_ _CRTIMP double __cdecl _chgsign (_In_ double _X);
#define _SIGN_DEFINED
#endif
        double  __cdecl cos(_In_ double _X);
        double  __cdecl cosh(_In_ double _X);
        double  __cdecl exp(_In_ double _X);
_CRT_JIT_INTRINSIC double  __cdecl fabs(_In_ double _X);
        double  __cdecl fmod(_In_ double _X, _In_ double _Y);
        double  __cdecl log(_In_ double _X);
        double  __cdecl log10(_In_ double _X);
        double  __cdecl pow(_In_ double _X, _In_ double _Y);
        double  __cdecl sin(_In_ double _X);
        double  __cdecl sinh(_In_ double _X);
        double  __cdecl tan(_In_ double _X);
        double  __cdecl tanh(_In_ double _X);
_CRT_JIT_INTRINSIC  double  __cdecl sqrt(_In_ double _X);
#ifndef _CRT_ATOF_DEFINED
#define _CRT_ATOF_DEFINED
_Check_return_ _CRTIMP double  __cdecl atof(_In_z_ const char *_String);
_Check_return_ _CRTIMP double  __cdecl _atof_l(_In_z_ const char *_String, _In_opt_ _locale_t _Locale);
#endif

_CRTIMP double  __cdecl _cabs(_In_ struct _complex _Complex_value);
_CRTIMP double  __cdecl ceil(_In_ double _X);
_CRTIMP double  __cdecl floor(_In_ double _X);
_CRTIMP double  __cdecl frexp(_In_ double _X, _Out_ int * _Y);
_CRTIMP double  __cdecl _hypot(_In_ double _X, _In_ double _Y);
_CRTIMP float   __cdecl _hypotf(_In_ float _X, _In_ float _Y);
_CRTIMP double  __cdecl _j0(_In_ double _X );
_CRTIMP double  __cdecl _j1(_In_ double _X );
_CRTIMP double  __cdecl _jn(int _X, _In_ double _Y);
_CRTIMP double  __cdecl ldexp(_In_ double _X, _In_ int _Y);
#ifndef _CRT_MATHERR_DEFINED
#define _CRT_MATHERR_DEFINED
#if defined(MRTDLL) || defined(_M_CEE_PURE)
        int     __CRTDECL _matherr(_Inout_ struct _exception * _Except);
#else
        int     __cdecl _matherr(_Inout_ struct _exception * _Except);
#endif
#endif
_CRTIMP double  __cdecl modf(_In_ double _X, _Out_ double * _Y);

_CRTIMP double  __cdecl _y0(_In_ double _X);
_CRTIMP double  __cdecl _y1(_In_ double _X);
_CRTIMP double  __cdecl _yn(_In_ int _X, _In_ double _Y);

/* hypot and hypotf are now part of the C99 Standard */
#if !defined(RC_INVOKED) && !defined(__midl)

static __inline double __CRTDECL hypot(_In_ double _X, _In_ double _Y)
{
    return _hypot(_X, _Y);
}

static __inline float __CRTDECL hypotf(_In_ float _X, _In_ float _Y)
{
    return _hypotf(_X, _Y);
}

#endif

#if     defined(_M_IX86)

_CRTIMP int     __cdecl _set_SSE2_enable(_In_ int _Flag);

#endif

#if defined(_M_X64)

/* ANSI C, 4.5 Mathematics        */

/* 4.5.2 Trigonometric functions */

_CRTIMP float  __cdecl acosf( _In_ float _X);
_CRTIMP float  __cdecl asinf( _In_ float _X);
_CRTIMP float  __cdecl atanf( _In_ float _X);
_CRTIMP float  __cdecl atan2f( _In_ float  _Y, _In_ float  _X);
_CRTIMP float  __cdecl cosf( _In_ float _X);
_CRTIMP float  __cdecl sinf( _In_ float _X);
_CRTIMP float  __cdecl tanf( _In_ float _X);

/* 4.5.3 Hyperbolic functions */
_CRTIMP float  __cdecl coshf( _In_ float _X);
_CRTIMP float  __cdecl sinhf( _In_ float _X);
_CRTIMP float  __cdecl tanhf( _In_ float _X);

/* 4.5.4 Exponential and logarithmic functions */
_CRTIMP float  __cdecl expf( _In_ float  _X);
_CRTIMP float  __cdecl logf( _In_ float  _X);
_CRTIMP float  __cdecl log10f( _In_ float  _X);
_CRTIMP float  __cdecl modff( _In_ float  _X, _Out_ float*  _Y);

/* 4.5.5 Power functions */
_CRTIMP float  __cdecl powf( _In_ float _X, _In_ float _Y);
_CRTIMP float  __cdecl sqrtf( _In_ float  _X);

/* 4.5.6 Nearest integer, absolute value, and remainder functions */
_CRTIMP float  __cdecl ceilf( _In_ float  _X);
_CRTIMP float  __cdecl floorf( _In_ float  _X);
_CRTIMP float  __cdecl fmodf( _In_ float  _X, _In_ float _Y);

_CRTIMP float __cdecl _copysignf (_In_ float _Number, _In_ float _Sign);
_CRTIMP float __cdecl _chgsignf (_In_ float _X);
_CRTIMP float __cdecl _logbf(_In_ float _X);
_CRTIMP float __cdecl _nextafterf(_In_ float _X, _In_ float _Y);
_CRTIMP int    __cdecl _finitef(_In_ float _X);
_CRTIMP int    __cdecl _isnanf(_In_ float _X);
_CRTIMP int    __cdecl _fpclassf(_In_ float _X);

#endif /* _M_X64 */

#if defined (_M_ARM)
/* ANSI C, 4.5 Mathematics        */

/* 4.5.2 Trigonometric functions */

_CRTIMP float  __cdecl acosf( _In_ float _X);
_CRTIMP float  __cdecl asinf( _In_ float _X);
_CRTIMP float  __cdecl atanf( _In_ float _X);
_CRTIMP float  __cdecl atan2f( _In_ float  _Y, _In_ float  _X);
_CRTIMP float  __cdecl cosf( _In_ float _X);
_CRTIMP float  __cdecl sinf( _In_ float _X);
_CRTIMP float  __cdecl tanf( _In_ float _X);

/* 4.5.3 Hyperbolic functions */
_CRTIMP float  __cdecl coshf( _In_ float _X);
_CRTIMP float  __cdecl sinhf( _In_ float _X);
_CRTIMP float  __cdecl tanhf( _In_ float _X);

/* 4.5.4 Exponential and logarithmic functions */
_CRTIMP float  __cdecl expf( _In_ float  _X);
_CRTIMP float  __cdecl logf( _In_ float  _X);
_CRTIMP float  __cdecl log10f( _In_ float  _X);
_CRTIMP float  __cdecl modff( _In_ float  _X, _Out_ float*  _Y);

/* 4.5.5 Power functions */
_CRTIMP float  __cdecl powf( _In_ float _X, _In_ float _Y);
_CRTIMP float  __cdecl sqrtf( _In_ float  _X);

/* 4.5.6 Nearest integer, absolute value, and remainder functions */
_CRTIMP float  __cdecl ceilf( _In_ float  _X);
_CRT_JIT_INTRINSIC  _CRTIMP float  __cdecl fabsf( _In_ float  _X);
_CRTIMP float  __cdecl floorf( _In_ float  _X);
_CRTIMP float  __cdecl fmodf( _In_ float  _X, _In_ float _Y);
_CRTIMP float __cdecl _logbf(_In_ float _X);
_CRTIMP float __cdecl _copysignf (_In_ float _Number, _In_ float _Sign);
_CRTIMP float __cdecl _chgsignf (_In_ float _X);
_CRTIMP float __cdecl _logbf(_In_ float _X);
_CRTIMP int    __cdecl _finitef(_In_ float _X);

#endif /* _M_ARM */

/* Macros defining long double functions to be their double counterparts
 * (long double is synonymous with double in this implementation).
 */

#ifndef __cplusplus
#define acosl(x)        ((long double)acos((double)(x)))
#define asinl(x)        ((long double)asin((double)(x)))
#define atanl(x)        ((long double)atan((double)(x)))
#define atan2l(y,x)     ((long double)atan2((double)(y), (double)(x)))
#define ceill(x)        ((long double)ceil((double)(x)))
#define cosl(x)         ((long double)cos((double)(x)))
#define coshl(x)        ((long double)cosh((double)(x)))
#define expl(x)         ((long double)exp((double)(x)))
#define fabsl(x)        ((long double)fabs((double)(x)))
#define floorl(x)       ((long double)floor((double)(x)))
#define fmodl(x,y)      ((long double)fmod((double)(x), (double)(y)))
#define frexpl(x,y)     ((long double)frexp((double)(x), (y)))
#define _hypotl(x,y)    ((long double)_hypot((double)(x), (double)(y)))
#define hypotl(x,y)     ((long double)_hypot((double)(x), (double)(y)))
#define ldexpl(x,y)     ((long double)ldexp((double)(x), (y)))
#define logl(x)         ((long double)log((double)(x)))
#define log10l(x)       ((long double)log10((double)(x)))
#define _matherrl       _matherr
#define modfl(x,y)      ((long double)modf((double)(x), (double *)(y)))
#define powl(x,y)       ((long double)pow((double)(x), (double)(y)))
#define sinl(x)         ((long double)sin((double)(x)))
#define sinhl(x)        ((long double)sinh((double)(x)))
#define sqrtl(x)        ((long double)sqrt((double)(x)))
#define tanl(x)         ((long double)tan((double)(x)))
#define tanhl(x)        ((long double)tanh((double)(x)))
#define _chgsignl(x)    ((long double)_chgsign((double)(x)))
#define _copysignl(x,y) ((long double)_copysign((double)(x), (double)(y)))

#define frexpf(x,y) ((float)frexp((double)(x),(y)))

#if !defined(_M_ARM)
#define fabsf(x)    ((float)fabs((double)(x)))
#endif
#define ldexpf(x,y) ((float)ldexp((double)(x),(y)))

#if !defined (_M_X64) && !defined(_M_ARM)

#define acosf(x)    ((float)acos((double)(x)))
#define asinf(x)    ((float)asin((double)(x)))
#define atanf(x)    ((float)atan((double)(x)))
#define atan2f(y,x) ((float)atan2((double)(y), (double)(x)))
#define ceilf(x)    ((float)ceil((double)(x)))
#define cosf(x)     ((float)cos((double)(x)))
#define coshf(x)    ((float)cosh((double)(x)))
#define expf(x)     ((float)exp((double)(x)))
#define floorf(x)   ((float)floor((double)(x)))
#define fmodf(x,y)  ((float)fmod((double)(x), (double)(y)))
#define logf(x)     ((float)log((double)(x)))
#define log10f(x)   ((float)log10((double)(x)))
#define modff(x,y)  ((float)modf((double)(x), (double *)(y)))
#define powf(x,y)   ((float)pow((double)(x), (double)(y)))
#define sinf(x)     ((float)sin((double)(x)))
#define sinhf(x)    ((float)sinh((double)(x)))
#define sqrtf(x)    ((float)sqrt((double)(x)))
#define tanf(x)     ((float)tan((double)(x)))
#define tanhf(x)    ((float)tanh((double)(x)))

#endif /* !defined (_M_X64) && !defined(_M_ARM) */

#else   /* __cplusplus */
inline long double acosl(_In_ long double _X)
        {return (acos((double)_X)); }
inline long double asinl(_In_ long double _X)
        {return (asin((double)_X)); }
inline long double atanl(_In_ long double _X)
        {return (atan((double)_X)); }
inline long double atan2l(_In_ long double _Y, _In_ long double _X)
        {return (atan2((double)_Y, (double)_X)); }
inline long double ceill(_In_ long double _X)
        {return (ceil((double)_X)); }
inline long double cosl(_In_ long double _X)
        {return (cos((double)_X)); }
inline long double coshl(_In_ long double _X)
        {return (cosh((double)_X)); }
inline long double expl(_In_ long double _X)
        {return (exp((double)_X)); }
inline long double fabsl(_In_ long double _X)
        {return (fabs((double)_X)); }
inline long double floorl(_In_ long double _X)
        {return (floor((double)_X)); }
inline long double fmodl(_In_ long double _X, _In_ long double _Y)
        {return (fmod((double)_X, (double)_Y)); }
inline long double frexpl(_In_ long double _X, _Out_ int *_Y)
        {return (frexp((double)_X, _Y)); }
inline long double ldexpl(_In_ long double _X, _In_ int _Y)
        {return (ldexp((double)_X, _Y)); }
inline long double logl(_In_ long double _X)
        {return (log((double)_X)); }
inline long double log10l(_In_ long double _X)
        {return (log10((double)_X)); }
inline long double modfl(_In_ long double _X, _Out_ long double *_Y)
        {double _Di, _Df = modf((double)_X, &_Di);
        *_Y = (long double)_Di;
        return (_Df); }
inline long double powl(_In_ long double _X, _In_ long double _Y)
        {return (pow((double)_X, (double)_Y)); }
inline long double sinl(_In_ long double _X)
        {return (sin((double)_X)); }
inline long double sinhl(_In_ long double _X)
        {return (sinh((double)_X)); }
inline long double sqrtl(_In_ long double _X)
        {return (sqrt((double)_X)); }
inline long double tanl(_In_ long double _X)
        {return (tan((double)_X)); }

inline long double tanhl(_In_ long double _X)
        {return (tanh((double)_X)); }

inline long double _chgsignl(_In_ long double _Number)
{
    return _chgsign(static_cast<double>(_Number)); 
}

inline long double _copysignl(_In_ long double _Number, _In_ long double _Sign)
{
    return _copysign(static_cast<double>(_Number), static_cast<double>(_Sign)); 
}

inline float frexpf(_In_ float _X, _Out_ int *_Y)
        {return ((float)frexp((double)_X, _Y)); }

#if     !defined(_M_ARM)
inline float fabsf(_In_ float _X)
        {return ((float)fabs((double)_X)); }
#endif  /* !defined(_M_ARM) */

inline float ldexpf(_In_ float _X, _In_ int _Y)
        {return ((float)ldexp((double)_X, _Y)); }

#if     !defined(_M_X64) && !defined(_M_ARM)
inline float acosf(_In_ float _X)
        {return ((float)acos((double)_X)); }
inline float asinf(_In_ float _X)
        {return ((float)asin((double)_X)); }
inline float atanf(_In_ float _X)
        {return ((float)atan((double)_X)); }
inline float atan2f(_In_ float _Y, _In_ float _X)
        {return ((float)atan2((double)_Y, (double)_X)); }
inline float ceilf(_In_ float _X)
        {return ((float)ceil((double)_X)); }
inline float cosf(_In_ float _X)
        {return ((float)cos((double)_X)); }
inline float coshf(_In_ float _X)
        {return ((float)cosh((double)_X)); }
inline float expf(_In_ float _X)
        {return ((float)exp((double)_X)); }
inline float floorf(_In_ float _X)
        {return ((float)floor((double)_X)); }
inline float fmodf(_In_ float _X, _In_ float _Y)
        {return ((float)fmod((double)_X, (double)_Y)); }
inline float logf(_In_ float _X)
        {return ((float)log((double)_X)); }
inline float log10f(_In_ float _X)
        {return ((float)log10((double)_X)); }
inline float modff(_In_ float _X, _Out_ float *_Y)
        { double _Di, _Df = modf((double)_X, &_Di);
        *_Y = (float)_Di;
        return ((float)_Df); }
inline float powf(_In_ float _X, _In_ float _Y)
        {return ((float)pow((double)_X, (double)_Y)); }
inline float sinf(_In_ float _X)
        {return ((float)sin((double)_X)); }
inline float sinhf(_In_ float _X)
        {return ((float)sinh((double)_X)); }
inline float sqrtf(_In_ float _X)
        {return ((float)sqrt((double)_X)); }
inline float tanf(_In_ float _X)
        {return ((float)tan((double)_X)); }
inline float tanhf(_In_ float _X)
        {return ((float)tanh((double)_X)); }
#endif  /* !defined(_M_X64) && !defined(_M_ARM)*/
#endif  /* __cplusplus */
#endif  /* __assembler */

#if     !__STDC__

/* Non-ANSI names for compatibility */

#define DOMAIN      _DOMAIN
#define SING        _SING
#define OVERFLOW    _OVERFLOW
#define UNDERFLOW   _UNDERFLOW
#define TLOSS       _TLOSS
#define PLOSS       _PLOSS

#define matherr     _matherr

#ifndef __assembler /* Protect from assembler */

#if !defined(_M_CEE_PURE)
_CRTIMP extern double HUGE;
#else
    const double HUGE = _HUGE;
#endif /* !defined(_M_CEE_PURE) */

_CRT_NONSTDC_DEPRECATE(_cabs) _CRTIMP double  __cdecl cabs(_In_ struct _complex _X);
_CRT_NONSTDC_DEPRECATE(_j0) _CRTIMP double  __cdecl j0(_In_ double _X);
_CRT_NONSTDC_DEPRECATE(_j1) _CRTIMP double  __cdecl j1(_In_ double _X);
_CRT_NONSTDC_DEPRECATE(_jn) _CRTIMP double  __cdecl jn(_In_ int _X, _In_ double _Y);
_CRT_NONSTDC_DEPRECATE(_y0) _CRTIMP double  __cdecl y0(_In_ double _X);
_CRT_NONSTDC_DEPRECATE(_y1) _CRTIMP double  __cdecl y1(_In_ double _X);
_CRT_NONSTDC_DEPRECATE(_yn) _CRTIMP double  __cdecl yn(_In_ int _X, _In_ double _Y);

#endif  /* __assembler */

#endif  /* __STDC__ */

#ifdef  __cplusplus
}

extern "C++" {

template<class _Ty> inline
        _Ty _Pow_int(_Ty _X, int _Y)
        {unsigned int _N;
        if (_Y >= 0)
                _N = (unsigned int)_Y;
        else
                _N = (unsigned int)(-_Y);
        for (_Ty _Z = _Ty(1); ; _X *= _X)
                {if ((_N & 1) != 0)
                        _Z *= _X;
                if ((_N >>= 1) == 0)
                        return (_Y < 0 ? _Ty(1) / _Z : _Z); }}

inline double __CRTDECL abs(_In_ double _X)
        {return (fabs(_X)); }
inline double __CRTDECL pow(_In_ double _X, _In_ int _Y)
        {return (_Pow_int(_X, _Y)); }
inline float __CRTDECL abs(_In_ float _X)
        {return (fabsf(_X)); }
inline float __CRTDECL acos(_In_ float _X)
        {return (acosf(_X)); }
inline float __CRTDECL asin(_In_ float _X)
        {return (asinf(_X)); }
inline float __CRTDECL atan(_In_ float _X)
        {return (atanf(_X)); }
inline float __CRTDECL atan2(_In_ float _Y, _In_ float _X)
        {return (atan2f(_Y, _X)); }
inline float __CRTDECL ceil(_In_ float _X)
        {return (ceilf(_X)); }
inline float __CRTDECL cos(_In_ float _X)
        {return (cosf(_X)); }
inline float __CRTDECL cosh(_In_ float _X)
        {return (coshf(_X)); }
inline float __CRTDECL exp(_In_ float _X)
        {return (expf(_X)); }
inline float __CRTDECL fabs(_In_ float _X)
        {return (fabsf(_X)); }
inline float __CRTDECL floor(_In_ float _X)
        {return (floorf(_X)); }
inline float __CRTDECL fmod(_In_ float _X, _In_ float _Y)
        {return (fmodf(_X, _Y)); }
inline float __CRTDECL frexp(_In_ float _X, _Out_ int * _Y)
        {return (frexpf(_X, _Y)); }
inline float __CRTDECL ldexp(_In_ float _X, _In_ int _Y)
        {return (ldexpf(_X, _Y)); }
inline float __CRTDECL log(_In_ float _X)
        {return (logf(_X)); }
inline float __CRTDECL log10(_In_ float _X)
        {return (log10f(_X)); }
inline float __CRTDECL modf(_In_ float _X, _Out_ float * _Y)
        {return (modff(_X, _Y)); }
inline float __CRTDECL pow(_In_ float _X, _In_ float _Y)
        {return (powf(_X, _Y)); }
inline float __CRTDECL pow(_In_ float _X, _In_ int _Y)
        {return (_Pow_int(_X, _Y)); }
inline float __CRTDECL sin(_In_ float _X)
        {return (sinf(_X)); }
inline float __CRTDECL sinh(_In_ float _X)
        {return (sinhf(_X)); }
inline float __CRTDECL sqrt(_In_ float _X)
        {return (sqrtf(_X)); }
inline float __CRTDECL tan(_In_ float _X)
        {return (tanf(_X)); }
inline float __CRTDECL tanh(_In_ float _X)
        {return (tanhf(_X)); }
inline long double __CRTDECL abs(_In_ long double _X)
        {return (fabsl(_X)); }
inline long double __CRTDECL acos(_In_ long double _X)
        {return (acosl(_X)); }
inline long double __CRTDECL asin(_In_ long double _X)
        {return (asinl(_X)); }
inline long double __CRTDECL atan(_In_ long double _X)
        {return (atanl(_X)); }
inline long double __CRTDECL atan2(_In_ long double _Y, _In_ long double _X)
        {return (atan2l(_Y, _X)); }
inline long double __CRTDECL ceil(_In_ long double _X)
        {return (ceill(_X)); }
inline long double __CRTDECL cos(_In_ long double _X)
        {return (cosl(_X)); }
inline long double __CRTDECL cosh(_In_ long double _X)
        {return (coshl(_X)); }
inline long double __CRTDECL exp(_In_ long double _X)
        {return (expl(_X)); }
inline long double __CRTDECL fabs(_In_ long double _X)
        {return (fabsl(_X)); }
inline long double __CRTDECL floor(_In_ long double _X)
        {return (floorl(_X)); }
inline long double __CRTDECL fmod(_In_ long double _X, _In_ long double _Y)
        {return (fmodl(_X, _Y)); }
inline long double __CRTDECL frexp(_In_ long double _X, _Out_ int * _Y)
        {return (frexpl(_X, _Y)); }
inline long double __CRTDECL ldexp(_In_ long double _X, _In_ int _Y)
        {return (ldexpl(_X, _Y)); }
inline long double __CRTDECL log(_In_ long double _X)
        {return (logl(_X)); }
inline long double __CRTDECL log10(_In_ long double _X)
        {return (log10l(_X)); }
inline long double __CRTDECL modf(_In_ long double _X, _Out_ long double * _Y)
        {return (modfl(_X, _Y)); }
inline long double __CRTDECL pow(_In_ long double _X, _In_ long double _Y)
        {return (powl(_X, _Y)); }
inline long double __CRTDECL pow(_In_ long double _X, _In_ int _Y)
        {return (_Pow_int(_X, _Y)); }
inline long double __CRTDECL sin(_In_ long double _X)
        {return (sinl(_X)); }
inline long double __CRTDECL sinh(_In_ long double _X)
        {return (sinhl(_X)); }
inline long double __CRTDECL sqrt(_In_ long double _X)
        {return (sqrtl(_X)); }
inline long double __CRTDECL tan(_In_ long double _X)
        {return (tanl(_X)); }
inline long double __CRTDECL tanh(_In_ long double _X)
        {return (tanhl(_X)); }

}
#endif  /* __cplusplus */

#pragma pack(pop)

#endif  /* _INC_MATH */

#if defined(_USE_MATH_DEFINES) && !defined(_MATH_DEFINES_DEFINED)
#define _MATH_DEFINES_DEFINED

/* Define _USE_MATH_DEFINES before including math.h to expose these macro
 * definitions for common math constants.  These are placed under an #ifdef
 * since these commonly-defined names are not part of the C/C++ standards.
 */

/* Definitions of useful mathematical constants
 * M_E        - e
 * M_LOG2E    - log2(e)
 * M_LOG10E   - log10(e)
 * M_LN2      - ln(2)
 * M_LN10     - ln(10)
 * M_PI       - pi
 * M_PI_2     - pi/2
 * M_PI_4     - pi/4
 * M_1_PI     - 1/pi
 * M_2_PI     - 2/pi
 * M_2_SQRTPI - 2/sqrt(pi)
 * M_SQRT2    - sqrt(2)
 * M_SQRT1_2  - 1/sqrt(2)
 */

#define M_E        2.71828182845904523536
#define M_LOG2E    1.44269504088896340736
#define M_LOG10E   0.434294481903251827651
#define M_LN2      0.693147180559945309417
#define M_LN10     2.30258509299404568402
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_PI_4     0.785398163397448309616
#define M_1_PI     0.318309886183790671538
#define M_2_PI     0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2    1.41421356237309504880
#define M_SQRT1_2  0.707106781186547524401

#endif  /* _USE_MATH_DEFINES */

