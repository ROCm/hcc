/***
*use_ansi.h - pragmas for ANSI Standard C++ libraries
*
*	Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This header is intended to force the use of the appropriate ANSI
*       Standard C++ libraries whenever it is included.
*
*       [Public]
*
****/

#pragma once

#ifndef _USE_ANSI_CPP
#define _USE_ANSI_CPP

#undef _CRT_NOPRAGMA_LIBS

#ifndef _CRT_NOPRAGMA_LIBS

#if !defined(_M_CEE_PURE) && !defined(MRTDLL)

#if defined(_DLL) && !defined(_STATIC_CPPLIB)
#ifdef _DEBUG
#pragma comment(lib,"msvcprtd")
#else	/* _DEBUG */
#pragma comment(lib,"msvcprt")
#endif	/* _DEBUG */

#else	/* _DLL && !STATIC_CPPLIB */
#ifdef _DEBUG
#if _ITERATOR_DEBUG_LEVEL == 0
#pragma comment(lib,"libcpmtd0")
#elif _ITERATOR_DEBUG_LEVEL == 1
#pragma comment(lib,"libcpmtd1")
#else	/* _ITERATOR_DEBUG_LEVEL */
#ifdef _GUARDED_CRT
#pragma comment(lib,"libcpmtgd")
#else /* _GUARDED_CRT */
#pragma comment(lib,"libcpmtd")
#endif	/* _GUARDED_CRT */
#endif	/* _ITERATOR_DEBUG_LEVEL */
#else	/* _DEBUG */
#if _ITERATOR_DEBUG_LEVEL == 0
#ifdef _GUARDED_CRT
#pragma comment(lib,"libcpmtg")
#else /* _GUARDED_CRT */
#pragma comment(lib,"libcpmt")
#endif	/* _GUARDED_CRT */
#else	/* _ITERATOR_DEBUG_LEVEL */
#pragma comment(lib,"libcpmt1")
#endif	/* _ITERATOR_DEBUG_LEVEL */
#endif	/* _DEBUG */
#endif	/* _DLL && !STATIC_CPPLIB */

#endif /* !defined(_M_CEE_PURE) && !defined(MRTDLL) */

#endif  /* _CRT_NOPRAGMA_LIBS */

#endif	/* _USE_ANSI_CPP */
