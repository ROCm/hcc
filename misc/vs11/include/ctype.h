/***
*ctype.h - character conversion macros and ctype macros
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       Defines macros for character classification/conversion.
*       [ANSI/System V]
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_CTYPE
#define _INC_CTYPE

#include <crtdefs.h>

#ifdef  __cplusplus
extern "C" {
#endif

#ifndef WEOF
#define WEOF (wint_t)(0xFFFF)
#endif

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

#ifndef _CTYPE_DISABLE_MACROS
#endif  /* _CTYPE_DISABLE_MACROS */




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


/* character classification function prototypes */

#ifndef _CTYPE_DEFINED

_Check_return_ _CRTIMP int __cdecl _isctype(_In_ int _C, _In_ int _Type);
_Check_return_ _CRTIMP int __cdecl _isctype_l(_In_ int _C, _In_ int _Type, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl isalpha(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isalpha_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl isupper(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isupper_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl islower(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _islower_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl isdigit(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isdigit_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl isxdigit(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isxdigit_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl isspace(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isspace_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl ispunct(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _ispunct_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl isalnum(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isalnum_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl isprint(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isprint_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl isgraph(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isgraph_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iscntrl(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _iscntrl_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl toupper(_In_ int _C);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl tolower(_In_ int _C);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl _tolower(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _tolower_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRT_JIT_INTRINSIC _CRTIMP int __cdecl _toupper(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _toupper_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl __isascii(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl __toascii(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl __iscsymf(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl __iscsym(_In_ int _C);
#define _CTYPE_DEFINED
#endif

#ifndef _WCTYPE_DEFINED

/* wide function prototypes, also declared in wchar.h  */

/* character classification function prototypes */

_Check_return_ _CRTIMP int __cdecl iswalpha(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswalpha_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswupper(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswupper_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswlower(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswlower_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswdigit(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswdigit_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswxdigit(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswxdigit_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswspace(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswspace_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswpunct(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswpunct_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswalnum(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswalnum_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswprint(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswprint_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswgraph(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswgraph_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswcntrl(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswcntrl_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl iswascii(_In_ wint_t _C);

_Check_return_ _CRTIMP wint_t __cdecl towupper(_In_ wint_t _C);
_Check_return_ _CRTIMP wint_t __cdecl _towupper_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP wint_t __cdecl towlower(_In_ wint_t _C);
_Check_return_ _CRTIMP wint_t __cdecl _towlower_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale); 
_Check_return_ _CRTIMP int __cdecl iswctype(_In_ wint_t _C, _In_ wctype_t _Type);
_Check_return_ _CRTIMP int __cdecl _iswctype_l(_In_ wint_t _C, _In_ wctype_t _Type, _In_opt_ _locale_t _Locale);

_Check_return_ _CRTIMP int __cdecl __iswcsymf(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswcsymf_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);
_Check_return_ _CRTIMP int __cdecl __iswcsym(_In_ wint_t _C);
_Check_return_ _CRTIMP int __cdecl _iswcsym_l(_In_ wint_t _C, _In_opt_ _locale_t _Locale);

#ifdef _CRT_USE_WINAPI_FAMILY_DESKTOP_APP
_Check_return_ _CRTIMP int __cdecl isleadbyte(_In_ int _C);
_Check_return_ _CRTIMP int __cdecl _isleadbyte_l(_In_ int _C, _In_opt_ _locale_t _Locale);
_CRT_OBSOLETE(iswctype) _CRTIMP int __cdecl is_wctype(_In_ wint_t _C, _In_ wctype_t _Type);
#endif /* _CRT_USE_WINAPI_FAMILY_DESKTOP_APP */

#define _WCTYPE_DEFINED
#endif

/* the character classification macro definitions */

#ifndef _CTYPE_DISABLE_MACROS

/*
 * Maximum number of bytes in multi-byte character in the current locale
 * (also defined in stdlib.h).
 */
#ifndef MB_CUR_MAX

#if defined(_CRT_DISABLE_PERFCRIT_LOCKS) && !defined(_DLL)
#define MB_CUR_MAX __mb_cur_max
#else
#define MB_CUR_MAX ___mb_cur_max_func()
#endif
#if !defined(_M_CEE_PURE)
/* No data exports in pure code */
_CRTIMP extern int __mb_cur_max;
#else
#define __mb_cur_max (___mb_cur_max_func())
#endif /* !defined(_M_CEE_PURE) */
_CRTIMP int __cdecl ___mb_cur_max_func(void);
_CRTIMP int __cdecl ___mb_cur_max_l_func(_locale_t);
#endif  /* MB_CUR_MAX */

/* Introduced to detect error when character testing functions are called
 * with illegal input of integer.
 */
#ifdef _DEBUG
_CRTIMP int __cdecl _chvalidator(_In_ int _Ch, _In_ int _Mask);
#define __chvalidchk(a,b)       _chvalidator(a,b)
#else
#define __chvalidchk(a,b)       (__PCTYPE_FUNC[(a)] & (b))
#endif



#if defined(_CRT_DISABLE_PERFCRIT_LOCKS) && !defined(_DLL)
#ifndef __cplusplus
#define isalpha(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_ALPHA) : __chvalidchk(_c, _ALPHA))
#define isupper(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_UPPER) : __chvalidchk(_c, _UPPER))
#define islower(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_LOWER) : __chvalidchk(_c, _LOWER))
#define isdigit(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_DIGIT) : __chvalidchk(_c, _DIGIT))
#define isxdigit(_c)    (MB_CUR_MAX > 1 ? _isctype(_c,_HEX)   : __chvalidchk(_c, _HEX))
#define isspace(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_SPACE) : __chvalidchk(_c, _SPACE))
#define ispunct(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_PUNCT) : __chvalidchk(_c, _PUNCT))
#define isalnum(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_ALPHA|_DIGIT) : __chvalidchk(_c, (_ALPHA|_DIGIT)))
#define isprint(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_BLANK|_PUNCT|_ALPHA|_DIGIT) : __chvalidchk(_c, (_BLANK|_PUNCT|_ALPHA|_DIGIT)))
#define isgraph(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_PUNCT|_ALPHA|_DIGIT) : __chvalidchk(_c, (_PUNCT|_ALPHA|_DIGIT)))
#define iscntrl(_c)     (MB_CUR_MAX > 1 ? _isctype(_c,_CONTROL) : __chvalidchk(_c, _CONTROL))
#elif   0         /* Pending ANSI C++ integration */
_Check_return_ inline int isalpha(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_ALPHA) : __chvalidchk(_C, _ALPHA)); }
_Check_return_ inline int isupper(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_UPPER) : __chvalidchk(_C, _UPPER)); }
_Check_return_ inline int islower(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_LOWER) : __chvalidchk(_C, _LOWER)); }
_Check_return_ inline int isdigit(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_DIGIT) : __chvalidchk(_C, _DIGIT)); }
_Check_return_ inline int isxdigit(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_HEX)   : __chvalidchk(_C, _HEX)); }
_Check_return_ inline int isspace(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_SPACE) : __chvalidchk(_C, _SPACE)); }
_Check_return_ inline int ispunct(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_PUNCT) : __chvalidchk(_C, _PUNCT)); }
_Check_return_ inline int isalnum(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_ALPHA|_DIGIT)
                : __chvalidchk(_C) , (_ALPHA|_DIGIT)); }
_Check_return_ inline int isprint(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_BLANK|_PUNCT|_ALPHA|_DIGIT)
                : __chvalidchk(_C , (_BLANK|_PUNCT|_ALPHA|_DIGIT))); }
_Check_return_ inline int isgraph(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_PUNCT|_ALPHA|_DIGIT)
                : __chvalidchk(_C , (_PUNCT|_ALPHA|_DIGIT))); }
_Check_return_ inline int iscntrl(_In_ int _C)
        {return (MB_CUR_MAX > 1 ? _isctype(_C,_CONTROL)
                : __chvalidchk(_C , _CONTROL)); }
#endif  /* __cplusplus */
#endif  

#ifdef _DEBUG
_CRTIMP int __cdecl _chvalidator_l(_In_opt_ _locale_t, _In_ int _Ch, _In_ int _Mask);
#define _chvalidchk_l(_Char, _Flag, _Locale)  _chvalidator_l(_Locale, _Char, _Flag)
#else
#define _chvalidchk_l(_Char, _Flag, _Locale)  (_Locale==NULL ? __chvalidchk(_Char, _Flag) : ((_locale_t)_Locale)->locinfo->pctype[_Char] & (_Flag))
#endif  /* DEBUG */


#define _ischartype_l(_Char, _Flag, _Locale)    ( ((_Locale)!=NULL && (((_locale_t)(_Locale))->locinfo->mb_cur_max) > 1) ? _isctype_l(_Char, (_Flag), _Locale) : _chvalidchk_l(_Char,_Flag,_Locale))
#define _isalpha_l(_Char, _Locale)      _ischartype_l(_Char, _ALPHA, _Locale)
#define _isupper_l(_Char, _Locale)      _ischartype_l(_Char, _UPPER, _Locale)
#define _islower_l(_Char, _Locale)      _ischartype_l(_Char, _LOWER, _Locale)
#define _isdigit_l(_Char, _Locale)      _ischartype_l(_Char, _DIGIT, _Locale)
#define _isxdigit_l(_Char, _Locale)     _ischartype_l(_Char, _HEX, _Locale)
#define _isspace_l(_Char, _Locale)      _ischartype_l(_Char, _SPACE, _Locale)
#define _ispunct_l(_Char, _Locale)      _ischartype_l(_Char, _PUNCT, _Locale)
#define _isalnum_l(_Char, _Locale)      _ischartype_l(_Char, _ALPHA|_DIGIT, _Locale)
#define _isprint_l(_Char, _Locale)      _ischartype_l(_Char, _BLANK|_PUNCT|_ALPHA|_DIGIT, _Locale)
#define _isgraph_l(_Char, _Locale)      _ischartype_l(_Char, _PUNCT|_ALPHA|_DIGIT, _Locale)
#define _iscntrl_l(_Char, _Locale)      _ischartype_l(_Char, _CONTROL, _Locale)

#define _tolower(_Char)    ( (_Char)-'A'+'a' )
#define _toupper(_Char)    ( (_Char)-'a'+'A' )

#define __isascii(_Char)   ( (unsigned)(_Char) < 0x80 )
#define __toascii(_Char)   ( (_Char) & 0x7f )

#ifndef _WCTYPE_INLINE_DEFINED

#undef _CRT_WCTYPE_NOINLINE

#if !defined(__cplusplus) || defined(_M_CEE_PURE) || defined(MRTDLL) || defined(_CRT_WCTYPE_NOINLINE)
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

#define _iswalpha_l(_c,_p)    ( iswctype(_c,_ALPHA) )
#define _iswupper_l(_c,_p)    ( iswctype(_c,_UPPER) )
#define _iswlower_l(_c,_p)    ( iswctype(_c,_LOWER) )
#define _iswdigit_l(_c,_p)    ( iswctype(_c,_DIGIT) )
#define _iswxdigit_l(_c,_p)   ( iswctype(_c,_HEX) )
#define _iswspace_l(_c,_p)    ( iswctype(_c,_SPACE) )
#define _iswpunct_l(_c,_p)    ( iswctype(_c,_PUNCT) )
#define _iswalnum_l(_c,_p)    ( iswctype(_c,_ALPHA|_DIGIT) )
#define _iswprint_l(_c,_p)    ( iswctype(_c,_BLANK|_PUNCT|_ALPHA|_DIGIT) )
#define _iswgraph_l(_c,_p)    ( iswctype(_c,_PUNCT|_ALPHA|_DIGIT) )
#define _iswcntrl_l(_c,_p)    ( iswctype(_c,_CONTROL) )
#elif   0         /* __cplusplus */
_Check_return_ inline int iswalpha(_In_ wint_t _C) {return (iswctype(_C,_ALPHA)); }
_Check_return_ inline int iswupper(_In_ wint_t _C) {return (iswctype(_C,_UPPER)); }
_Check_return_ inline int iswlower(_In_ wint_t _C) {return (iswctype(_C,_LOWER)); }
_Check_return_ inline int iswdigit(_In_ wint_t _C) {return (iswctype(_C,_DIGIT)); }
_Check_return_ inline int iswxdigit(_In_ wint_t _C) {return (iswctype(_C,_HEX)); }
_Check_return_ inline int iswspace(_In_ wint_t _C) {return (iswctype(_C,_SPACE)); }
_Check_return_ inline int iswpunct(_In_ wint_t _C) {return (iswctype(_C,_PUNCT)); }
_Check_return_ inline int iswalnum(_In_ wint_t _C) {return (iswctype(_C,_ALPHA|_DIGIT)); }
_Check_return_ inline int iswprint(_In_ wint_t _C) {return (iswctype(_C,_BLANK|_PUNCT|_ALPHA|_DIGIT)); }
_Check_return_ inline int iswgraph(_In_ wint_t _C) {return (iswctype(_C,_PUNCT|_ALPHA|_DIGIT)); }
_Check_return_ inline int iswcntrl(_In_ wint_t _C) {return (iswctype(_C,_CONTROL)); }
_Check_return_ inline int iswascii(_In_ wint_t _C) {return ((unsigned)(_C) < 0x80); }

_Check_return_ inline int __CRTDECL _iswalpha_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_ALPHA)); }
_Check_return_ inline int __CRTDECL _iswupper_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_UPPER)); }
_Check_return_ inline int __CRTDECL _iswlower_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_LOWER)); }
_Check_return_ inline int __CRTDECL _iswdigit_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_DIGIT)); }
_Check_return_ inline int __CRTDECL _iswxdigit_l(_In_ wint_t _C, _In_opt_ _locale_t) {return(iswctype(_C,_HEX)); }
_Check_return_ inline int __CRTDECL _iswspace_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_SPACE)); }
_Check_return_ inline int __CRTDECL _iswpunct_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_PUNCT)); }
_Check_return_ inline int __CRTDECL _iswalnum_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_ALPHA|_DIGIT)); }
_Check_return_ inline int __CRTDECL _iswprint_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_BLANK|_PUNCT|_ALPHA|_DIGIT)); }
_Check_return_ inline int __CRTDECL _iswgraph_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_PUNCT|_ALPHA|_DIGIT)); }
_Check_return_ inline int __CRTDECL _iswcntrl_l(_In_ wint_t _C, _In_opt_ _locale_t) {return (iswctype(_C,_CONTROL)); }

_Check_return_ inline int isleadbyte(int _C) {return (__PCTYPE_FUNC[(unsigned char)(_C)] & _LEADBYTE); }
#endif  /* __cplusplus */
#define _WCTYPE_INLINE_DEFINED
#endif  /* _WCTYPE_INLINE_DEFINED */

/* MS C version 2.0 extended ctype macros */

#define __iscsymf(_c)   (isalpha(_c) || ((_c) == '_'))
#define __iscsym(_c)    (isalnum(_c) || ((_c) == '_'))
#define __iswcsymf(_c)  (iswalpha(_c) || ((_c) == '_'))
#define __iswcsym(_c)   (iswalnum(_c) || ((_c) == '_'))

#define _iscsymf_l(_c, _p)   (_isalpha_l(_c, _p) || ((_c) == '_'))
#define _iscsym_l(_c, _p)    (_isalnum_l(_c, _p) || ((_c) == '_'))
#define _iswcsymf_l(_c, _p)  (iswalpha(_c) || ((_c) == '_'))
#define _iswcsym_l(_c, _p)   (iswalnum(_c) || ((_c) == '_'))

#endif  /* _CTYPE_DISABLE_MACROS */


#if     !__STDC__

/* Non-ANSI names for compatibility */

#ifndef _CTYPE_DEFINED
_Check_return_ _CRT_NONSTDC_DEPRECATE(__isascii) _CRTIMP int __cdecl isascii(_In_ int _C);
_Check_return_ _CRT_NONSTDC_DEPRECATE(__toascii) _CRTIMP int __cdecl toascii(_In_ int _C);
_Check_return_ _CRT_NONSTDC_DEPRECATE(__iscsymf) _CRTIMP int __cdecl iscsymf(_In_ int _C);
_Check_return_ _CRT_NONSTDC_DEPRECATE(__iscsym) _CRTIMP int __cdecl iscsym(_In_ int _C);
#else
#define isascii __isascii
#define toascii __toascii
#define iscsymf __iscsymf
#define iscsym  __iscsym
#endif

#endif  /* __STDC__ */

#ifdef  __cplusplus
}
#endif

#endif  /* _INC_CTYPE */
