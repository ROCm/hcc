/***
*wctype.h - declarations for wide character functions
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*       Created from wchar.h January 1996 by P.J. Plauger
*
*Purpose:
*       This file contains the types, macros and function declarations for
*       all ctype-style wide-character functions.  They may also be declared in
*       wchar.h.
*       [ISO]
*
*       Note: keep in sync with ctype.h and wchar.h.
*
*       [Public]
*
****/


#pragma once

#ifndef _INC_WCTYPE
#define _INC_WCTYPE

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

/* Define _CRTIMP2 */
#ifndef _CRTIMP2
#if defined(_DLL) && !defined(_STATIC_CPPLIB)
#define _CRTIMP2 __declspec(dllimport)
#else   /* ndef _DLL && !STATIC_CPPLIB */
#define _CRTIMP2
#endif  /* _DLL && !STATIC_CPPLIB */
#endif  /* _CRTIMP2 */

#ifndef _WCHAR_T_DEFINED
typedef unsigned short wchar_t;
#define _WCHAR_T_DEFINED
#endif

#ifndef _WCTYPE_T_DEFINED
typedef unsigned short wint_t;
typedef unsigned short wctype_t;
#define _WCTYPE_T_DEFINED
#endif


#ifndef WEOF
#define WEOF (wint_t)(0xFFFF)
#endif

/*
 * This declaration allows the user access to the ctype look-up
 * array _ctype defined in ctype.obj by simply including ctype.h
 */
#ifndef _CRT_CTYPEDATA_DEFINED
#define _CRT_CTYPEDATA_DEFINED
#ifndef _CTYPE_DISABLE_MACROS

#ifndef __PCTYPE_FUNC
#if defined(_CRT_DISABLE_PERFCRIT_LOCKS) && !defined(_DLL)
#define __PCTYPE_FUNC  _pctype
#else
#define __PCTYPE_FUNC   __pctype_func()
#endif
#endif  /* __PCTYPE_FUNC */

_CRTIMP const unsigned short * __cdecl __pctype_func(void);
#if !defined(_M_CEE_PURE)
_CRTIMP extern const unsigned short *_pctype;
#else
#define _pctype (__pctype_func())
#endif /* !defined(_M_CEE_PURE) */
#endif  /* _CTYPE_DISABLE_MACROS */
#endif

#ifndef _CRT_WCTYPEDATA_DEFINED
#define _CRT_WCTYPEDATA_DEFINED
#ifndef _CTYPE_DISABLE_MACROS
#if !defined(_M_CEE_PURE)
_CRTIMP extern const unsigned short _wctype[];
#endif /* !defined(_M_CEE_PURE) */

_CRTIMP const wctype_t * __cdecl __pwctype_func(void);
#if !defined(_M_CEE_PURE)
_CRTIMP extern const wctype_t *_pwctype;
#else
#define _pwctype (__pwctype_func())
#endif /* !defined(_M_CEE_PURE) */
#endif  /* _CTYPE_DISABLE_MACROS */
#endif



/* set bit masks for the possible character types */

#define _UPPER          0x1     /* upper case letter */
#define _LOWER          0x2     /* lower case letter */
#define _DIGIT          0x4     /* digit[0-9] */
#define _SPACE          0x8     /* tab, carriage return, newline, */
                                /* vertical tab or form feed */
#define _PUNCT          0x10    /* punctuation character */
#define _CONTROL        0x20    /* control character */
#define _BLANK          0x40    /* space char */
#define _HEX            0x80    /* hexadecimal digit */

#define _LEADBYTE       0x8000                  /* multibyte leadbyte */
#define _ALPHA          (0x0100|_UPPER|_LOWER)  /* alphabetic character */


/* Function prototypes */

#ifndef _WCTYPE_DEFINED

/* Character classification function prototypes */
/* also declared in ctype.h */

_CRTIMP int __cdecl iswalpha(wint_t);
_CRTIMP int __cdecl iswupper(wint_t);
_CRTIMP int __cdecl iswlower(wint_t);
_CRTIMP int __cdecl iswdigit(wint_t);
_CRTIMP int __cdecl iswxdigit(wint_t);
_CRTIMP int __cdecl iswspace(wint_t);
_CRTIMP int __cdecl iswpunct(wint_t);
_CRTIMP int __cdecl iswalnum(wint_t);
_CRTIMP int __cdecl iswprint(wint_t);
_CRTIMP int __cdecl iswgraph(wint_t);
_CRTIMP int __cdecl iswcntrl(wint_t);
_CRTIMP int __cdecl iswascii(wint_t);
_CRTIMP int __cdecl isleadbyte(int);

_CRTIMP wint_t __cdecl towupper(wint_t);
_CRTIMP wint_t __cdecl towlower(wint_t);

_CRTIMP int __cdecl iswctype(wint_t, wctype_t);

_CRTIMP int __cdecl __iswcsymf(wint_t);
_CRTIMP int __cdecl __iswcsym(wint_t);

#ifdef _CRT_USE_WINAPI_FAMILY_DESKTOP_APP
_CRT_OBSOLETE(iswctype) _CRTIMP int __cdecl is_wctype(wint_t, wctype_t);
#endif /* _CRT_USE_WINAPI_FAMILY_DESKTOP_APP */

#define _WCTYPE_DEFINED
#endif

#ifndef _WCTYPE_INLINE_DEFINED
#if !defined(__cplusplus) || defined(_M_CEE_PURE) || defined(MRTDLL)
#define iswalpha(_c)    ( iswctype(_c,_ALPHA) )
#define iswupper(_c)    ( iswctype(_c,_UPPER) )
#define iswlower(_c)    ( iswctype(_c,_LOWER) )
#define iswdigit(_c)    ( iswctype(_c,_DIGIT) )
#define iswxdigit(_c)   ( iswctype(_c,_HEX) )
#define iswspace(_c)    ( iswctype(_c,_SPACE) )
#define iswpunct(_c)    ( iswctype(_c,_PUNCT) )
#define iswalnum(_c)    ( iswctype(_c,_ALPHA|_DIGIT) )
#define iswprint(_c)    ( iswctype(_c,_BLANK|_PUNCT|_ALPHA|_DIGIT) )
#define iswgraph(_c)    ( iswctype(_c,_PUNCT|_ALPHA|_DIGIT) )
#define iswcntrl(_c)    ( iswctype(_c,_CONTROL) )
#define iswascii(_c)    ( (unsigned)(_c) < 0x80 )

#define isleadbyte(c) (__pctype_func()[(unsigned char)(c)] & _LEADBYTE)
#else   /* __cplusplus */
inline int __cdecl iswalpha(wint_t _C) {return (iswctype(_C,_ALPHA)); }
inline int __cdecl iswupper(wint_t _C) {return (iswctype(_C,_UPPER)); }
inline int __cdecl iswlower(wint_t _C) {return (iswctype(_C,_LOWER)); }
inline int __cdecl iswdigit(wint_t _C) {return (iswctype(_C,_DIGIT)); }
inline int __cdecl iswxdigit(wint_t _C) {return (iswctype(_C,_HEX)); }
inline int __cdecl iswspace(wint_t _C) {return (iswctype(_C,_SPACE)); }
inline int __cdecl iswpunct(wint_t _C) {return (iswctype(_C,_PUNCT)); }
inline int __cdecl iswalnum(wint_t _C) {return (iswctype(_C,_ALPHA|_DIGIT)); }
inline int __cdecl iswprint(wint_t _C)
        {return (iswctype(_C,_BLANK|_PUNCT|_ALPHA|_DIGIT)); }
inline int __cdecl iswgraph(wint_t _C)
        {return (iswctype(_C,_PUNCT|_ALPHA|_DIGIT)); }
inline int __cdecl iswcntrl(wint_t _C) {return (iswctype(_C,_CONTROL)); }
inline int __cdecl iswascii(wint_t _C) {return ((unsigned)(_C) < 0x80); }

inline int __cdecl isleadbyte(int _C)
        {return (__pctype_func()[(unsigned char)(_C)] & _LEADBYTE); }
#endif  /* __cplusplus */
#define _WCTYPE_INLINE_DEFINED
#endif  /* _WCTYPE_INLINE_DEFINED */

typedef wchar_t wctrans_t;
_MRTIMP2 wint_t __cdecl towctrans(wint_t, wctrans_t);
_MRTIMP2 wctrans_t __cdecl wctrans(const char *);
_MRTIMP2 wctype_t __cdecl wctype(const char *);


#ifdef  __cplusplus
}
#endif

#pragma pack(pop)

#endif  /* _INC_WCTYPE */
