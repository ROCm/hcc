/***
*varargs.h - XENIX style macros for variable argument functions
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file defines XENIX style macros for accessing arguments of a
*       function which takes a variable number of arguments.
*       [System V]
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_VARARGS
#define _INC_VARARGS

#if     !defined(_WIN32)
#error ERROR: Only Win32 target supported!
#endif


#include <crtdefs.h>

/*
 * Currently, all MS C compilers for Win32 platforms default to 8 byte
 * alignment.
 */
#pragma pack(push,_CRT_PACKING)

#ifdef  __cplusplus
extern "C" {
#endif

#if     __STDC__
#error varargs.h incompatible with ANSI (use stdarg.h)
#endif

#if !defined(_W64)
#if !defined(__midl) && (defined(_X86_) || defined(_M_IX86))
#define _W64 __w64
#else
#define _W64
#endif
#endif

#ifndef _UINTPTR_T_DEFINED
#ifdef  _WIN64
typedef unsigned __int64    uintptr_t;
#else
typedef _W64 unsigned int   uintptr_t;
#endif
#define _UINTPTR_T_DEFINED
#endif

#ifndef _VA_LIST_DEFINED
#ifdef _M_CEE_PURE
typedef System::ArgIterator va_list;
#else
typedef char *  va_list;
#endif
#define _VA_LIST_DEFINED
#endif

#ifndef va_arg

#if     defined(_M_CEE)

#error varargs.h not supported when targetting _M_CEE (use stdarg.h)

#elif   defined(_M_IX86)

/*
 * define a macro to compute the size of a type, variable or expression,
 * rounded up to the nearest multiple of sizeof(int). This number is its
 * size as function argument (Intel architecture). Note that the macro
 * depends on sizeof(int) being a power of 2!
 */
#define _INTSIZEOF(n)    ( (sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1) )

#define va_dcl va_list va_alist;
#define va_start(ap) ap = (va_list)&va_alist
#define va_arg(ap,t)    ( *(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)) )
#define va_end(ap) ap = (va_list)0

#elif defined(_M_ARM)

#define _VA_ALIGN       4
#define _APALIGN(t,ap)  ( ((va_list)0 - (ap)) & (__alignof(t) - 1) )
#define va_dcl          va_list va_alist;

#define _SLOTSIZEOF(t)  ( (sizeof(t) + _VA_ALIGN - 1) & ~(_VA_ALIGN - 1) )
#define va_start(ap)    ( ap = (va_list)&va_alist )
#define va_arg(ap,t)    (*(t *)((ap += _SLOTSIZEOF(t) + _APALIGN(t,ap)) \
                        - _SLOTSIZEOF(t)))
#define va_end(ap)      ( ap = (va_list)0 )

#elif defined(_M_X64)

extern void __cdecl __va_start(va_list *, ...);
#define va_dcl          va_list va_alist;
#define va_start(ap)    ( __va_start(&ap, 0) )
#define va_arg(ap, t)   \
    ( ( sizeof(t) > sizeof(__int64) || ( sizeof(t) & (sizeof(t) - 1) ) != 0 ) \
        ? **(t **)( ( ap += sizeof(__int64) ) - sizeof(__int64) ) \
        :  *(t  *)( ( ap += sizeof(__int64) ) - sizeof(__int64) ) )
#define va_end(ap)      ( ap = (va_list)0 )

#else

/* A guess at the proper definitions for other platforms */

#define _INTSIZEOF(n)    ( (sizeof(n) + sizeof(int) - 1) & ~(sizeof(int) - 1) )

#define va_dcl va_list va_alist;
#define va_start(ap) ap = (va_list)&va_alist
#define va_arg(ap,t) ( *(t *)((ap += _INTSIZEOF(t)) - _INTSIZEOF(t)) )
#define va_end(ap) ap = (va_list)0

#endif

#endif


#ifdef  __cplusplus
}
#endif

#pragma pack(pop)

#endif  /* _INC_VARARGS */
