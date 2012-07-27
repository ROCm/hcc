/***
*sys/utime.h - definitions/declarations for utime()
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file defines the structure used by the utime routine to set
*       new file access and modification times.  NOTE - MS-DOS
*       does not recognize access time, so this field will
*       always be ignored and the modification time field will be
*       used to set the new time.
*
*       [Public]
*
****/

#pragma once

#include <crtdefs.h>

#ifndef _INC_UTIME
#define _INC_UTIME

#if     !defined(_WIN32)
#error ERROR: Only Win32 target supported!
#endif

#include <crtdefs.h>


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


#ifndef _WCHAR_T_DEFINED
typedef unsigned short wchar_t;
#define _WCHAR_T_DEFINED
#endif

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

/* define struct used by _utime() function */

#ifndef _UTIMBUF_DEFINED

struct _utimbuf {
        time_t actime;          /* access time */
        time_t modtime;         /* modification time */
        };

struct __utimbuf32 {
        __time32_t actime;      /* access time */
        __time32_t modtime;     /* modification time */
        };

struct __utimbuf64 {
        __time64_t actime;      /* access time */
        __time64_t modtime;     /* modification time */
        };

#if     !__STDC__
/* Non-ANSI name for compatibility */
struct utimbuf {
        time_t actime;          /* access time */
        time_t modtime;         /* modification time */
        };
        
struct utimbuf32 {
        __time32_t actime;      /* access time */
        __time32_t modtime;     /* modification time */
        };

#endif /* !__STDC__ */

#define _UTIMBUF_DEFINED
#endif


/* Function Prototypes */

_CRTIMP int __cdecl _utime32(_In_z_ const char * _Filename, _In_opt_ struct __utimbuf32 * _Time);

_CRTIMP int __cdecl _futime32(_In_ int _FileDes, _In_opt_ struct __utimbuf32 * _Time);

/* Wide Function Prototypes */
_CRTIMP int __cdecl _wutime32(_In_z_ const wchar_t * _Filename, _In_opt_ struct __utimbuf32 * _Time);

_CRTIMP int __cdecl _utime64(_In_z_ const char * _Filename, _In_opt_ struct __utimbuf64 * _Time);
_CRTIMP int __cdecl _futime64(_In_ int _FileDes, _In_opt_ struct __utimbuf64 * _Time);
_CRTIMP int __cdecl _wutime64(_In_z_ const wchar_t * _Filename, _In_opt_ struct __utimbuf64 * _Time);

#if !defined(RC_INVOKED) && !defined(__midl)
#include <sys/utime.inl>
#endif

#ifdef  __cplusplus
}
#endif

#pragma pack(pop)

#endif  /* _INC_UTIME */
