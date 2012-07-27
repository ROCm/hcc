/***
*conio.h - console and port I/O declarations
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This include file contains the function declarations for
*       the MS C V2.03 compatible console I/O routines.
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_CONIO
#define _INC_CONIO

#include <crtdefs.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Function prototypes */

#ifdef _CRT_USE_WINAPI_FAMILY_DESKTOP_APP

_Check_return_wat_ _CRTIMP errno_t __cdecl _cgets_s(_Out_writes_z_(_Size)                char * _Buffer, size_t _Size, _Out_ size_t * _SizeRead);
__DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_1(errno_t, _cgets_s, _Out_writes_z_(*_Buffer) char, _Buffer, _Out_ size_t *, _SizeRead)
__DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_0_CGETS(char *, _CRTIMP, _cgets, _Pre_notnull_ _Post_z_, char, _Buffer)
_Check_return_opt_ _CRTIMP int __cdecl _cprintf(_In_z_ _Printf_format_string_ const char * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cprintf_s(_In_z_ _Printf_format_string_ const char * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cputs(_In_z_ const char * _Str);
_Check_return_opt_ _CRT_INSECURE_DEPRECATE(_cscanf_s) _CRTIMP int __cdecl _cscanf(_In_z_ _Scanf_format_string_ const char * _Format, ...);
_Check_return_opt_ _CRT_INSECURE_DEPRECATE(_cscanf_s_l) _CRTIMP int __cdecl _cscanf_l(_In_z_ _Scanf_format_string_params_(0) const char * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cscanf_s(_In_z_ _Scanf_format_string_ const char * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cscanf_s_l(_In_z_ _Scanf_format_string_params_(0) const char * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_ _CRTIMP int __cdecl _getch(void);
_Check_return_ _CRTIMP int __cdecl _getche(void);
_Check_return_opt_ _CRTIMP int __cdecl _vcprintf(_In_z_ _Printf_format_string_ const char * _Format, va_list _ArgList);
_Check_return_opt_ _CRTIMP int __cdecl _vcprintf_s(_In_z_ _Printf_format_string_ const char * _Format, va_list _ArgList);

_Check_return_opt_ _CRTIMP int __cdecl _cprintf_p(_In_z_ _Printf_format_string_ const char * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _vcprintf_p(_In_z_ const char * _Format, va_list _ArgList);

_Check_return_opt_ _CRTIMP int __cdecl _cprintf_l(_In_z_ _Printf_format_string_params_(0) const char * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cprintf_s_l(_In_z_ _Printf_format_string_params_(0) const char * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_opt_ _CRTIMP int __cdecl _vcprintf_l(_In_z_ _Printf_format_string_params_(2) const char * _Format, _In_opt_ _locale_t _Locale, va_list _ArgList);
_Check_return_opt_ _CRTIMP int __cdecl _vcprintf_s_l(_In_z_ _Printf_format_string_params_(2) const char * _Format, _In_opt_ _locale_t _Locale, va_list _ArgList);
_Check_return_opt_ _CRTIMP int __cdecl _cprintf_p_l(_In_z_ _Printf_format_string_params_(0) const char * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_opt_ _CRTIMP int __cdecl _vcprintf_p_l(_In_z_ _Printf_format_string_params_(2) const char * _Format, _In_opt_ _locale_t _Locale, va_list _ArgList);

#if defined(_M_IX86) || defined(_M_X64)
int __cdecl _inp(unsigned short);
unsigned short __cdecl _inpw(unsigned short);
unsigned long __cdecl _inpd(unsigned short);
#endif /* _M_IX86 || _M_X64 */
_CRTIMP int __cdecl _kbhit(void);
#if defined(_M_IX86) || defined(_M_X64)
int __cdecl _outp(unsigned short, int);
unsigned short __cdecl _outpw(unsigned short, unsigned short);
unsigned long __cdecl _outpd(unsigned short, unsigned long);
#endif /* _M_IX86 || _M_X64 */
_CRTIMP int __cdecl _putch(_In_ int _Ch);
_CRTIMP int __cdecl _ungetch(_In_ int _Ch);

_Check_return_ _CRTIMP int __cdecl _getch_nolock(void);
_Check_return_ _CRTIMP int __cdecl _getche_nolock(void);
_CRTIMP int __cdecl _putch_nolock(_In_ int _Ch);
_CRTIMP int __cdecl _ungetch_nolock(_In_ int _Ch);

#ifndef _WCONIO_DEFINED

/* wide function prototypes, also declared in wchar.h */

#ifndef WEOF
#define WEOF (wint_t)(0xFFFF)
#endif

_Check_return_wat_ _CRTIMP errno_t __cdecl _cgetws_s(_Out_writes_to_(_SizeInWords, *_SizeRead) wchar_t * _Buffer, size_t _SizeInWords, _Out_ size_t * _SizeRead);
__DEFINE_CPP_OVERLOAD_SECURE_FUNC_0_1(errno_t, _cgetws_s, _Out_writes_z_(*_Buffer) wchar_t, _Buffer, size_t *, _SizeRead)
__DEFINE_CPP_OVERLOAD_STANDARD_FUNC_0_0_CGETS(wchar_t *, _CRTIMP, _cgetws, _Pre_notnull_ _Post_z_, wchar_t, _Buffer)
_Check_return_ _CRTIMP wint_t __cdecl _getwch(void);
_Check_return_ _CRTIMP wint_t __cdecl _getwche(void);
_Check_return_ _CRTIMP wint_t __cdecl _putwch(wchar_t _WCh);
_Check_return_ _CRTIMP wint_t __cdecl _ungetwch(wint_t _WCh);
_Check_return_opt_ _CRTIMP int __cdecl _cputws(_In_z_ const wchar_t * _String);
_Check_return_opt_ _CRTIMP int __cdecl _cwprintf(_In_z_ _Printf_format_string_ const wchar_t * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cwprintf_s(_In_z_ _Printf_format_string_ const wchar_t * _Format, ...);
_Check_return_opt_ _CRT_INSECURE_DEPRECATE(_cwscanf_s) _CRTIMP int __cdecl _cwscanf(_In_z_ _Scanf_format_string_ const wchar_t * _Format, ...);
_Check_return_opt_ _CRT_INSECURE_DEPRECATE(_cwscanf_s_l) _CRTIMP int __cdecl _cwscanf_l(_In_z_ _Scanf_format_string_params_(0) const wchar_t * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cwscanf_s(_In_z_ _Scanf_format_string_ const wchar_t * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _cwscanf_s_l(_In_z_ _Scanf_format_string_params_(0) const wchar_t * _Format, _In_opt_ _locale_t _Locale, ...);
_Check_return_opt_ _CRTIMP int __cdecl _vcwprintf(_In_z_ _Printf_format_string_ const wchar_t *_Format, va_list _ArgList);
_Check_return_opt_ _CRTIMP int __cdecl _vcwprintf_s(_In_z_ _Printf_format_string_ const wchar_t *_Format, va_list _ArgList);

_Check_return_opt_ _CRTIMP int __cdecl _cwprintf_p(_In_z_ _Printf_format_string_ const wchar_t * _Format, ...);
_Check_return_opt_ _CRTIMP int __cdecl _vcwprintf_p(_In_z_ _Printf_format_string_ const wchar_t*  _Format, va_list _ArgList);

_CRTIMP int __cdecl _cwprintf_l(_In_z_ _Printf_format_string_params_(0) const wchar_t * _Format, _In_opt_ _locale_t _Locale, ...);
_CRTIMP int __cdecl _cwprintf_s_l(_In_z_ _Printf_format_string_params_(0) const wchar_t * _Format, _In_opt_ _locale_t _Locale, ...);
_CRTIMP int __cdecl _vcwprintf_l(_In_z_ _Printf_format_string_params_(2) const wchar_t *_Format, _In_opt_ _locale_t _Locale, va_list _ArgList);
_CRTIMP int __cdecl _vcwprintf_s_l(_In_z_ _Printf_format_string_params_(2) const wchar_t * _Format, _In_opt_ _locale_t _Locale, va_list _ArgList);
_CRTIMP int __cdecl _cwprintf_p_l(_In_z_ _Printf_format_string_params_(0) const wchar_t * _Format, _In_opt_ _locale_t _Locale, ...);
_CRTIMP int __cdecl _vcwprintf_p_l(_In_z_ _Printf_format_string_params_(2) const wchar_t * _Format, _In_opt_ _locale_t _Locale, va_list _ArgList);

_Check_return_opt_ _CRTIMP wint_t __cdecl _putwch_nolock(wchar_t _WCh);
_Check_return_ _CRTIMP wint_t __cdecl _getwch_nolock(void);
_Check_return_ _CRTIMP wint_t __cdecl _getwche_nolock(void);
_Check_return_opt_ _CRTIMP wint_t __cdecl _ungetwch_nolock(wint_t _WCh);

#define _WCONIO_DEFINED
#endif  /* _WCONIO_DEFINED */

#if     !__STDC__

/* Non-ANSI names for compatibility */

#pragma warning(push)
#pragma warning(disable: 4141) /* Using deprecated twice */ 
_Check_return_opt_ _CRT_NONSTDC_DEPRECATE(_cgets) _CRT_INSECURE_DEPRECATE(_cgets_s) _CRTIMP char * __cdecl cgets(_Out_writes_z_(_Inexpressible_(*_Buffer+2)) char * _Buffer);
#pragma warning(pop)
_Check_return_opt_ _CRT_NONSTDC_DEPRECATE(_cprintf) _CRTIMP int __cdecl cprintf(_In_z_ _Printf_format_string_ const char * _Format, ...);
_Check_return_opt_ _CRT_NONSTDC_DEPRECATE(_cputs) _CRTIMP int __cdecl cputs(_In_z_ const char * _Str);
_Check_return_opt_ _CRT_NONSTDC_DEPRECATE(_cscanf) _CRTIMP int __cdecl cscanf(_In_z_ _Scanf_format_string_ const char * _Format, ...);
#if defined(_M_IX86) || defined(_M_X64)
_CRT_NONSTDC_DEPRECATE(_inp) int __cdecl inp(unsigned short);
_CRT_NONSTDC_DEPRECATE(_inpd) unsigned long __cdecl inpd(unsigned short);
_CRT_NONSTDC_DEPRECATE(_inpw) unsigned short __cdecl inpw(unsigned short);
#endif /* _M_IX86 || _M_X64 */
_Check_return_ _CRT_NONSTDC_DEPRECATE(_getch) _CRTIMP int __cdecl getch(void);
_Check_return_ _CRT_NONSTDC_DEPRECATE(_getche) _CRTIMP int __cdecl getche(void);
_Check_return_ _CRT_NONSTDC_DEPRECATE(_kbhit) _CRTIMP int __cdecl kbhit(void);
#if defined(_M_IX86) || defined(_M_X64)
_CRT_NONSTDC_DEPRECATE(_outp) int __cdecl outp(unsigned short, int);
_CRT_NONSTDC_DEPRECATE(_outpd) unsigned long __cdecl outpd(unsigned short, unsigned long);
_CRT_NONSTDC_DEPRECATE(_outpw) unsigned short __cdecl outpw(unsigned short, unsigned short);
#endif /* _M_IX86 || _M_X64 */
_Check_return_opt_ _CRT_NONSTDC_DEPRECATE(_putch) _CRTIMP int __cdecl putch(int _Ch);
_Check_return_opt_ _CRT_NONSTDC_DEPRECATE(_ungetch) _CRTIMP int __cdecl ungetch(int _Ch);

#endif  /* __STDC__ */

#endif /* _CRT_USE_WINAPI_FAMILY_DESKTOP_APP */

#ifdef  __cplusplus
}
#endif

#endif  /* _INC_CONIO */
