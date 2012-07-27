/* xtimec.h -- header for high-resolution time functions */
#pragma once
#ifndef _THR_XTIMEC_H
#define _THR_XTIMEC_H
#ifndef RC_INVOKED
#include <thr/xthrcommon.h>
#include <time.h>

 #pragma pack(push,_CRT_PACKING)
 #pragma warning(push,3)
 #pragma push_macro("new")
 #undef new

#ifdef __cplusplus
extern "C" {	// C linkage
#endif /* __cplusplus */

enum {	/* define TIME_UTC */
	TIME_UTC = 1
	};

typedef struct xtime
	{	/* store time with nanosecond resolution */
	time_t sec;
	long nsec;
	} xtime;

_CRTIMP2_PURE int __cdecl xtime_get(xtime*, int);

_CRTIMP2_PURE long __cdecl _Xtime_diff_to_millis(const xtime *);
_CRTIMP2_PURE long __cdecl _Xtime_diff_to_millis2(const xtime*, const xtime *);
_CRTIMP2_PURE _LONGLONG __cdecl _Xtime_get_ticks();
#define _XTIME_NSECS_PER_TICK	100
#define _XTIME_TICKS_PER_TIME_T	(_LONGLONG)10000000

#ifdef __cplusplus
} // extern "C"
#endif
 #pragma pop_macro("new")
 #pragma warning(pop)
 #pragma pack(pop)
#endif /* RC_INVOKED */
#endif	/* _THR_XTIMEC_H */

/*
 * This file is derived from software bearing the following
 * restrictions:
 *
 * (c) Copyright William E. Kempf 2001
 *
 * Permission to use, copy, modify, distribute and sell this
 * software and its documentation for any purpose is hereby
 * granted without fee, provided that the above copyright
 * notice appear in all copies and that both that copyright
 * notice and this permission notice appear in supporting
 * documentation. William E. Kempf makes no representations
 * about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

/*
 * Copyright (c) 1992-2012 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V6.00:0009 */
