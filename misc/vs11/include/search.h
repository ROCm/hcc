/***
*search.h - declarations for searcing/sorting routines
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file contains the declarations for the sorting and
*       searching routines.
*       [System V]
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_SEARCH
#define _INC_SEARCH

#include <crtdefs.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

/* Function prototypes */

#ifndef _CRT_ALGO_DEFINED
#define _CRT_ALGO_DEFINED
#if __STDC_WANT_SECURE_LIB__
_Check_return_ _CRTIMP void * __cdecl bsearch_s(_In_ const void * _Key, _In_reads_bytes_(_NumOfElements * _SizeOfElements) const void * _Base, 
        _In_ rsize_t _NumOfElements, _In_ rsize_t _SizeOfElements,
        _In_ int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
#endif
_Check_return_ _CRTIMP void * __cdecl bsearch(_In_ const void * _Key, _In_reads_bytes_(_NumOfElements * _SizeOfElements) const void * _Base, 
        _In_ size_t _NumOfElements, _In_ size_t _SizeOfElements,
        _In_ int (__cdecl * _PtFuncCompare)(const void *, const void *));

#if __STDC_WANT_SECURE_LIB__
_CRTIMP void __cdecl qsort_s(_Inout_updates_bytes_(_NumOfElements* _SizeOfElements) void * _Base, 
        _In_ rsize_t _NumOfElements, _In_ rsize_t _SizeOfElements,
        _In_ int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void *_Context);
#endif
_CRTIMP void __cdecl qsort(_Inout_updates_bytes_(_NumOfElements * _SizeOfElements) void * _Base, 
	_In_ size_t _NumOfElements, _In_ size_t _SizeOfElements, 
        _In_ int (__cdecl * _PtFuncCompare)(const void *, const void *));
#endif

_Check_return_ _CRTIMP void * __cdecl _lfind_s(_In_ const void * _Key, _In_reads_bytes_((*_NumOfElements) * _SizeOfElements) const void * _Base,
        _Inout_ unsigned int * _NumOfElements, _In_ size_t _SizeOfElements, 
        _In_ int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
_Check_return_ _CRTIMP void * __cdecl _lfind(_In_ const void * _Key, _In_reads_bytes_((*_NumOfElements) * _SizeOfElements) const void * _Base, 
        _Inout_ unsigned int * _NumOfElements, _In_ unsigned int _SizeOfElements, 
	_In_ int (__cdecl * _PtFuncCompare)(const void *, const void *));

_Check_return_ _CRTIMP void * __cdecl _lsearch_s(_In_ const void * _Key, _Inout_updates_bytes_((*_NumOfElements ) * _SizeOfElements) void  * _Base, 
        _Inout_ unsigned int * _NumOfElements, _In_ size_t _SizeOfElements, 
	_In_ int (__cdecl * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
_Check_return_ _CRTIMP void * __cdecl _lsearch(_In_ const void * _Key, _Inout_updates_bytes_((*_NumOfElements ) * _SizeOfElements) void  * _Base, 
        _Inout_ unsigned int * _NumOfElements, _In_ unsigned int _SizeOfElements, 
	_In_ int (__cdecl * _PtFuncCompare)(const void *, const void *));

#if defined(__cplusplus) && defined(_M_CEE)
/*
 * Managed search routines. Note __cplusplus, this is because we only support
 * managed C++.
 */
extern "C++"
{

#if __STDC_WANT_SECURE_LIB__
_Check_return_ void * __clrcall bsearch_s(_In_ const void * _Key, _In_reads_bytes_(_NumOfElements*_SizeOfElements) const void * _Base, 
        _In_ rsize_t _NumOfElements, _In_ rsize_t _SizeOfElements, 
        _In_ int (__clrcall * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
#endif
_Check_return_ void * __clrcall bsearch(_In_ const void * _Key, _In_reads_bytes_(_NumOfElements*_SizeOfElements) const void * _Base, 
        _In_ size_t _NumOfElements, _In_ size_t _SizeOfElements,
        _In_ int (__clrcall * _PtFuncCompare)(const void *, const void *));

_Check_return_ void * __clrcall _lfind_s(_In_ const void * _Key, _In_reads_bytes_(_NumOfElements*_SizeOfElements) const void * _Base, 
        _Inout_ unsigned int * _NumOfElements, _In_ size_t _SizeOfElements, 
        _In_ int (__clrcall * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
_Check_return_ void * __clrcall _lfind(_In_ const void * _Key, _In_reads_bytes_((*_NumOfElements)*_SizeOfElements) const void * _Base, 
        _Inout_ unsigned int * _NumOfElements, _In_ unsigned int _SizeOfElements,
        _In_ int (__clrcall * _PtFuncCompare)(const void *, const void *));

_Check_return_ void * __clrcall _lsearch_s(_In_ const void * _Key, _In_reads_bytes_((*_NumOfElements)*_SizeOfElements) void * _Base, 
        _In_ unsigned int * _NumOfElements, _In_ size_t _SizeOfElements, 
        _In_ int (__clrcall * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
_Check_return_ void * __clrcall _lsearch(_In_ const void * _Key, _Inout_updates_bytes_((*_NumOfElements)*_SizeOfElements) void * _Base, 
        _Inout_ unsigned int * _NumOfElements, _In_ unsigned int _SizeOfElements,
        _In_ int (__clrcall * _PtFuncCompare)(const void *, const void *));

#if __STDC_WANT_SECURE_LIB__
void __clrcall qsort_s(_Inout_updates_bytes_(_NumOfElements*_SizeOfElements) void * _Base, 
        _In_ rsize_t _NumOfElements, _In_ rsize_t _SizeOfElements, 
        _In_ int (__clrcall * _PtFuncCompare)(void *, const void *, const void *), void * _Context);
#endif
void __clrcall qsort(_Inout_updates_bytes_(_NumOfElements*_SizeOfElements) void * _Base, 
        _In_ size_t _NumOfElements, _In_ size_t _SizeOfElements, 
        _In_ int (__clrcall * _PtFuncCompare)(const void *, const void *));

}
#endif


#if     !__STDC__
/* Non-ANSI names for compatibility */

_Check_return_ _CRTIMP _CRT_NONSTDC_DEPRECATE(_lfind) void * __cdecl lfind(_In_ const void * _Key, _In_reads_bytes_((*_NumOfElements) * _SizeOfElements) const void * _Base,
        _Inout_ unsigned int * _NumOfElements, _In_ unsigned int _SizeOfElements, 
	_In_ int (__cdecl * _PtFuncCompare)(const void *, const void *));
_Check_return_ _CRTIMP _CRT_NONSTDC_DEPRECATE(_lsearch) void * __cdecl lsearch(_In_ const void * _Key, _Inout_updates_bytes_((*_NumOfElements) * _SizeOfElements) void  * _Base,
        _Inout_ unsigned int * _NumOfElements, _In_ unsigned int _SizeOfElements, 
	_In_ int (__cdecl * _PtFuncCompare)(const void *, const void *));

#endif  /* __STDC__ */


#ifdef  __cplusplus
}
#endif

#endif  /* _INC_SEARCH */
