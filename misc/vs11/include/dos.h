/***
*dos.h - definitions for MS-DOS interface routines
*
*       Copyright (c) Microsoft Corporation. All rights reserved.
*
*Purpose:
*       Defines the structs and unions used for the direct DOS interface
*       routines; includes macros to access the segment and offset
*       values of far pointers, so that they may be used by the routines; and
*       provides function prototypes for direct DOS interface functions.
*
*       [Public]
*
****/

#pragma once

#ifndef _INC_DOS
#define _INC_DOS

#include <crtdefs.h>

/*
 * Currently, all MS C compilers for Win32 platforms default to 8 byte
 * alignment.
 */
#pragma pack(push,_CRT_PACKING)

#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _DISKFREE_T_DEFINED
/* _getdiskfree structure (duplicated in DIRECT.H) */
struct _diskfree_t {
        unsigned total_clusters;
        unsigned avail_clusters;
        unsigned sectors_per_cluster;
        unsigned bytes_per_sector;
        };

#define _DISKFREE_T_DEFINED
#endif

/* File attribute constants */

#define _A_NORMAL       0x00    /* Normal file - No read/write restrictions */
#define _A_RDONLY       0x01    /* Read only file */
#define _A_HIDDEN       0x02    /* Hidden file */
#define _A_SYSTEM       0x04    /* System file */
#define _A_SUBDIR       0x10    /* Subdirectory */
#define _A_ARCH         0x20    /* Archive file */

/* Function prototypes */
#ifndef _GETDISKFREE_DEFINED
_Check_return_ _CRTIMP unsigned __cdecl _getdiskfree(_In_ unsigned _Drive, _Out_ struct _diskfree_t * _DiskFree);
#define _GETDISKFREE_DEFINED
#endif

#ifdef  _M_IX86
void __cdecl _disable(void);
void __cdecl _enable(void);
#endif  /* _M_IX86 */

#if     !__STDC__
/* Non-ANSI name for compatibility */
#define diskfree_t  _diskfree_t
#endif  /* __STDC__ */

#ifdef  __cplusplus
}
#endif

#pragma pack(pop)

#endif  /* _INC_DOS */
