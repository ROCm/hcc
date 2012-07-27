/***
*wtime.inl - inline definitions for wctime()
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file contains the definition of wctime().
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

#ifndef _INC_WTIME_INL
#define _INC_WTIME_INL
#ifndef RC_INVOKED

#pragma warning(push)
#pragma warning(disable:4996)

#ifdef _USE_32BIT_TIME_T
static __inline wchar_t * __CRTDECL _wctime(const time_t * _Time)
{
#pragma warning( push )
#pragma warning( disable : 4996 )
    return _wctime32(_Time);
#pragma warning( pop )
}

static __inline errno_t __CRTDECL _wctime_s(wchar_t *_Buffer, size_t _SizeInWords, const time_t * _Time)
{
    return _wctime32_s(_Buffer, _SizeInWords, _Time);
}
#else
static __inline wchar_t * __CRTDECL _wctime(const time_t * _Time)
{
#pragma warning( push )
#pragma warning( disable : 4996 )
    return _wctime64(_Time);
#pragma warning( pop )
}

static __inline errno_t __CRTDECL _wctime_s(wchar_t *_Buffer, size_t _SizeInWords, const time_t * _Time)
{
    return _wctime64_s(_Buffer, _SizeInWords, _Time);
}
#endif /* _USE_32BIT_TIME_T */

#pragma warning(pop)

#endif /* RC_INVOKED */
#endif /* _INC_WTIME_INL */
