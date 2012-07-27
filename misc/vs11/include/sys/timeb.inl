/***
*timeb.inl - inline definitions for low-level file handling and I/O functions
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file contains the definition of the _ftime() function.
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

#ifndef _INC_TIMEB_INL
#define _INC_TIMEB_INL

/* _STATIC_ASSERT is for enforcing boolean/integral conditions at compile time.
   Since it is purely a compile-time mechanism that generates no code, the check
   is left in even if _DEBUG is not defined. */

#ifndef _STATIC_ASSERT
#define _STATIC_ASSERT(expr) typedef char __static_assert_t[ (expr) ]
#endif

#if     !__STDC__

/* Non-ANSI name for compatibility */

#pragma warning(push)
#pragma warning(disable:4996)

#ifdef _USE_32BIT_TIME_T
static __inline void __CRTDECL ftime(struct timeb * _Tmb)
{
    _STATIC_ASSERT( sizeof(struct timeb) == sizeof(struct __timeb32) );
    _ftime32((struct __timeb32 *)_Tmb);
}
#else
static __inline void __CRTDECL ftime(struct timeb * _Tmb)
{
    _STATIC_ASSERT( sizeof(struct timeb) == sizeof(struct __timeb64) );
    _ftime64((struct __timeb64 *)_Tmb);
}
#endif /* _USE_32BIT_TIME_T */

#pragma warning(pop)

#endif /* !__STDC__ */

#endif
