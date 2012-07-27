/***
*wstat.inl - inline definitions for low-level file handling and I/O functions
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file contains the function definitions for the low-level
*       file handling and I/O functions.
*
*       [Public]
*
****/

#pragma once

#if !defined(__CRTDECL)
#if defined(_M_CEE_PURE)
#define __CRTDECL
#else
#define __CRTDECL   __cdecl
#endif
#endif

#ifndef _INC_WSTAT_INL
#define _INC_WSTAT_INL

/* _STATIC_ASSERT is for enforcing boolean/integral conditions at compile time.
   Since it is purely a compile-time mechanism that generates no code, the check
   is left in even if _DEBUG is not defined. */

#ifndef _STATIC_ASSERT
#define _STATIC_ASSERT(expr) typedef char __static_assert_t[ (expr) ]
#endif

#ifdef _USE_32BIT_TIME_T
static __inline int __CRTDECL _wstat(const wchar_t * _Filename, struct _stat * _Stat)
{
    _STATIC_ASSERT( sizeof(struct _stat) == sizeof(struct _stat32) );
    return _wstat32(_Filename,(struct _stat32 *)_Stat);
}
#else
static __inline int __CRTDECL _wstat(const wchar_t * _Filename, struct _stat * _Stat)
{
    _STATIC_ASSERT( sizeof(struct _stat) == sizeof(struct _stat64i32) );
    return _wstat64i32(_Filename,(struct _stat64i32 *)_Stat);
}
#endif /* _USE_32BIT_TIME_T */

#ifdef _USE_32BIT_TIME_T
static __inline int __CRTDECL _wstati64(const wchar_t * _Filename, struct _stati64 * _Stat)
{
    _STATIC_ASSERT( sizeof(struct _stati64) == sizeof(struct _stat32i64) );
    return _wstat32i64(_Filename,(struct _stat32i64*)_Stat);
}
#else
static __inline int __CRTDECL _wstati64(const wchar_t * _Filename, struct _stati64 * _Stat)
{
    _STATIC_ASSERT( sizeof(struct _stati64) == sizeof(struct _stat64) );
    return _wstat64(_Filename,(struct _stat64*)_Stat);
}
#endif /* _USE_32BIT_TIME_T */

#endif
