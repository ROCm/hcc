/***
*crtwrn.h
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       This file contains the IDs and messages for warnings
*       in the CRT headers.
*
****/

#pragma once

#ifndef _INC_CRTWRN
#define _INC_CRTWRN

#include <crtdefs.h>

#define __CRT_WARNING( _Number, _Description ) \
    message("" __FILE__ "(" _CRT_STRINGIZE(__LINE__) ") : " \
    "warning CRT" _CRT_STRINGIZE(_Number) ": " _CRT_STRINGIZE(_Description))

#define _CRT_WARNING( _Id ) \
    __CRT_WARNING( _CRTWRN_WNUMBER_##_Id, _CRTWRN_MESSAGE_##_Id )

/*
A warning is a 4-digit ID number (_CRTWRN_WNUMBER_*)
followed by a message (_CRTWRN_MESSAGE_*)
Emit a warning by adding the following code to the header file:
    #pragma _CRT_WARNING( id )
*/

/* NAME */
/* #pragma _CRT_WARNING( NAME ) */
/* #define _CRTWRN_WNUMBER_NAME  9999 */
/* #define _CRTWRN_MESSAGE_NAME  description */

/* ID 1001 is obsolete; do not reuse it */

/* ID 1002 is obsolete; do not reuse it */

/* ID 1003 is obsolete; do not reuse it */

/* _NO_SPECIAL_TRANSFER */
/* #pragma _CRT_WARNING( _NO_SPECIAL_TRANSFER ) */
#define _CRTWRN_WNUMBER__NO_SPECIAL_TRANSFER  1004
#define _CRTWRN_MESSAGE__NO_SPECIAL_TRANSFER \
    Special transfer of control routines not defined for this platform

/* ID 1005 is obsolete; do not reuse it */

/* ID 1006 is obsolete; do not reuse it */

/* _DEPRECATE_STATIC_CPPLIB */
/* #pragma push_macro("_STATIC_CPPLIB") */
/* #undef _STATIC_CPPLIB */
/* #pragma _CRT_WARNING( _DEPRECATE_STATIC_CPPLIB ) */
/* #pragma pop_macro("_STATIC_CPPLIB") */
#define _CRTWRN_WNUMBER__DEPRECATE_STATIC_CPPLIB  1007
#define _CRTWRN_MESSAGE__DEPRECATE_STATIC_CPPLIB _STATIC_CPPLIB is deprecated

/* ID 1008 is obsolete; do not reuse it */

#endif /* _INC_CRTWRN */
