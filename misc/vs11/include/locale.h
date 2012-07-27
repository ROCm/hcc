/***
*locale.h - definitions/declarations for localization routines
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file defines the structures, values, macros, and functions
*       used by the localization routines.
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_LOCALE
#define _INC_LOCALE

#include <crtdefs.h>

/*
 * Currently, all MS C compilers for Win32 platforms default to 8 byte
 * alignment.
 */
#pragma pack(push,_CRT_PACKING)

#ifdef  __cplusplus
extern "C" {
#endif

/* define NULL pointer value */
#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *)0)
#endif
#endif

/* Locale categories */

#define LC_ALL          0
#define LC_COLLATE      1
#define LC_CTYPE        2
#define LC_MONETARY     3
#define LC_NUMERIC      4
#define LC_TIME         5

#define LC_MIN          LC_ALL
#define LC_MAX          LC_TIME

/* Locale convention structure */

#ifndef _LCONV_DEFINED
struct lconv {
        char *decimal_point;
        char *thousands_sep;
        char *grouping;
        char *int_curr_symbol;
        char *currency_symbol;
        char *mon_decimal_point;
        char *mon_thousands_sep;
        char *mon_grouping;
        char *positive_sign;
        char *negative_sign;
        char int_frac_digits;
        char frac_digits;
        char p_cs_precedes;
        char p_sep_by_space;
        char n_cs_precedes;
        char n_sep_by_space;
        char p_sign_posn;
        char n_sign_posn;
        wchar_t *_W_decimal_point;
        wchar_t *_W_thousands_sep;
        wchar_t *_W_int_curr_symbol;
        wchar_t *_W_currency_symbol;
        wchar_t *_W_mon_decimal_point;
        wchar_t *_W_mon_thousands_sep;
        wchar_t *_W_positive_sign;
        wchar_t *_W_negative_sign;
        };
#define _LCONV_DEFINED
#endif

/* ANSI: char lconv members default is CHAR_MAX which is compile time
   dependent. Defining and using _charmax here causes CRT startup code
   to initialize lconv members properly */

#ifdef  _CHAR_UNSIGNED
extern int _charmax;
extern __inline int __dummy(void) { return _charmax; }
#endif

/* function prototypes */

#ifndef _CONFIG_LOCALE_SWT
#define _ENABLE_PER_THREAD_LOCALE           0x1
#define _DISABLE_PER_THREAD_LOCALE          0x2
#define _ENABLE_PER_THREAD_LOCALE_GLOBAL    0x10
#define _DISABLE_PER_THREAD_LOCALE_GLOBAL   0x20
#define _ENABLE_PER_THREAD_LOCALE_NEW       0x100
#define _DISABLE_PER_THREAD_LOCALE_NEW      0x200
#define _CONFIG_LOCALE_SWT
#endif

_Check_return_opt_ _CRTIMP int __cdecl _configthreadlocale(_In_ int _Flag);
_Check_return_opt_ _CRTIMP char * __cdecl setlocale(_In_ int _Category, _In_opt_z_ const char * _Locale);
_Check_return_opt_ _CRTIMP struct lconv * __cdecl localeconv(void);
_Check_return_opt_ _CRTIMP _locale_t __cdecl _get_current_locale(void);
_Check_return_opt_ _CRTIMP _locale_t __cdecl _create_locale(_In_ int _Category, _In_z_ const char * _Locale);
_CRTIMP void __cdecl _free_locale(_In_opt_ _locale_t _Locale);

#ifdef _CRT_USE_WINAPI_FAMILY_DESKTOP_APP
/* use _get_current_locale, _create_locale and _free_locale, instead of these functions with double leading underscore */
_Check_return_ _CRT_OBSOLETE(_get_current_locale) _CRTIMP _locale_t __cdecl __get_current_locale(void);
_Check_return_ _CRT_OBSOLETE(_create_locale) _CRTIMP _locale_t __cdecl __create_locale(_In_ int _Category, _In_z_ const char * _Locale);
_CRT_OBSOLETE(_free_locale) _CRTIMP void __cdecl __free_locale(_In_opt_ _locale_t _Locale);
#endif /* _CRT_USE_WINAPI_FAMILY_DESKTOP_APP */

#ifndef _WLOCALE_DEFINED

/* wide function prototypes, also declared in wchar.h  */

_Check_return_opt_ _CRTIMP wchar_t * __cdecl _wsetlocale(_In_ int _Category, _In_opt_z_ const wchar_t * _Locale);
_Check_return_opt_ _CRTIMP _locale_t __cdecl _wcreate_locale(_In_ int _Category, _In_z_ const wchar_t * _Locale);

#define _WLOCALE_DEFINED
#endif

#ifdef  __cplusplus
}
#endif

#pragma pack(pop)

#endif  /* _INC_LOCALE */
