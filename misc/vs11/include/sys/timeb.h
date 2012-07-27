/***
*sys/timeb.h - definition/declarations for _ftime()
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file define the _ftime() function and the types it uses.
*       [System V]
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_TIMEB
#define _INC_TIMEB

#include <crtdefs.h>

#if     !defined(_WIN32)
#error ERROR: Only Win32 target supported!
#endif


#pragma pack(push,_CRT_PACKING)

#ifdef  __cplusplus
extern "C" {
#endif



/* Define _CRTIMP */

#ifndef _CRTIMP
#ifdef  _DLL
#define _CRTIMP __declspec(dllimport)
#else   /* ndef _DLL */
#define _CRTIMP
#endif  /* _DLL */
#endif  /* _CRTIMP */


#if !defined(_W64)
#if !defined(__midl) && (defined(_X86_) || defined(_M_IX86))
#define _W64 __w64
#else
#define _W64
#endif
#endif

#ifndef _TIME32_T_DEFINED
typedef _W64 long __time32_t;   /* 32-bit time value */
#define _TIME32_T_DEFINED
#endif

#ifndef _TIME64_T_DEFINED
typedef __int64 __time64_t;     /* 64-bit time value */
#define _TIME64_T_DEFINED
#endif

#ifndef _TIME_T_DEFINED
#ifdef _USE_32BIT_TIME_T
typedef __time32_t time_t;      /* time value */
#else
typedef __time64_t time_t;      /* time value */
#endif
#define _TIME_T_DEFINED         /* avoid multiple def's of time_t */
#endif

/* Structure returned by _ftime system call */

#ifndef _TIMEB_DEFINED
struct __timeb32 {
        __time32_t time;
        unsigned short millitm;
        short timezone;
        short dstflag;
        };

#if     !__STDC__

/* Non-ANSI name for compatibility */
struct timeb {
        time_t time;
        unsigned short millitm;
        short timezone;
        short dstflag;
        };


#endif

struct __timeb64 {
        __time64_t time;
        unsigned short millitm;
        short timezone;
        short dstflag;
        };

#ifdef  _USE_32BIT_TIME_T

#define _timeb      __timeb32
#define _ftime      _ftime32
#define _ftime_s    _ftime32_s

#else

#define _timeb      __timeb64
#define _ftime      _ftime64
#define _ftime_s    _ftime64_s

#endif
#define _TIMEB_DEFINED
#endif

#include <crtdefs.h>

/* Function prototypes */

_CRT_INSECURE_DEPRECATE(_ftime32_s) _CRTIMP void __cdecl _ftime32(_Out_ struct __timeb32 * _Time);
_CRTIMP errno_t __cdecl _ftime32_s(_Out_ struct __timeb32 * _Time);
_CRT_INSECURE_DEPRECATE(_ftime64_s) _CRTIMP void __cdecl _ftime64(_Out_ struct __timeb64 * _Time);
_CRTIMP errno_t __cdecl _ftime64_s(_Out_ struct __timeb64 * _Time);

#if !defined(RC_INVOKED) && !defined(__midl)
#include <sys/timeb.inl>
#endif

#ifdef  __cplusplus
}
#endif

#pragma pack(pop)

#endif  /* _INC_TIMEB */
