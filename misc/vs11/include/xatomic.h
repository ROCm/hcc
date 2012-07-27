/* xatomic.h internal header */
#pragma once
#ifndef _XATOMIC_H
#define _XATOMIC_H
#ifndef RC_INVOKED
#include <xatomic0.h>
#include <stddef.h>	// for size_t

 #pragma pack(push,_CRT_PACKING)
 #pragma warning(push,3)
 #pragma push_macro("new")
 #undef new

 #ifndef _CONCAT
  #define _CONCATX(x, y)	x ## y
  #define _CONCAT(x, y)		_CONCATX(x, y)
 #endif /* _CONCAT */

#define ATOMIC_BOOL_LOCK_FREE	\
	(1 <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_CHAR_LOCK_FREE	\
	(1 <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_CHAR16_T_LOCK_FREE	\
	(2 <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_CHAR32_T_LOCK_FREE	\
	(2 <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_WCHAR_T_LOCK_FREE 	\
	(_WCHAR_T_SIZE <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_SHORT_LOCK_FREE	\
	(_SHORT_SIZE <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_INT_LOCK_FREE 	\
	(_INT_SIZE <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_LONG_LOCK_FREE	\
	(_LONG_SIZE <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define _ATOMIC_LLONG_LOCK_FREE	\
	(_LONGLONG_SIZE <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)
#define ATOMIC_POINTER_LOCK_FREE	\
	(_ADDR_SIZE <= _ATOMIC_MAXBYTES_LOCK_FREE ? 2 : 0)

_STD_BEGIN
_EXTERN_C
		/* TYPEDEFS FOR INTERNAL ARITHMETIC TYPES */
typedef unsigned char _Uint1_t;
typedef unsigned short _Uint2_t;
//typedef _Uint32t _Uint4_t;
typedef unsigned _LONGLONG _Uint8_t;

		/* OPERATIONS ON _Atomic_flag_t */
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_flag_test_and_set_locking(
	volatile _Atomic_flag_t *, memory_order);
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_flag_clear_locking(
	volatile _Atomic_flag_t *, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_flag_test_and_set(
	volatile _Atomic_flag_t *, memory_order);
_CRTIMP2_PURE void __cdecl _Atomic_flag_clear(
	volatile _Atomic_flag_t *, memory_order);

 #if _ATOMIC_FLAG_USES_LOCK
  #define _ATOMIC_FLAG_TEST_AND_SET _Atomic_flag_test_and_set_locking
  #define _ATOMIC_FLAG_CLEAR _Atomic_flag_clear_locking

 #else /* _ATOMIC_FLAG_USES_LOCK */
  #define _ATOMIC_FLAG_TEST_AND_SET _Atomic_flag_test_and_set
  #define _ATOMIC_FLAG_CLEAR _Atomic_flag_clear
 #endif /* _ATOMIC_FLAG_USES_LOCK */

		/* GENERIC ATOMIC OPERATIONS */
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_copy(volatile _Atomic_flag_t *_Flag,
	_CSTD size_t _Size,
	volatile void *_Tgt,
	volatile const void *_Src,
	memory_order _Order);
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_exchange(volatile _Atomic_flag_t *_Flag,
	_CSTD size_t _Size,
	volatile void *_Tgt,
	volatile void *_Src,
	memory_order _Order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_weak(
	volatile _Atomic_flag_t *_Flag,
	_CSTD size_t _Size,
	volatile void *_Tgt,
	volatile void *_Exp,
	const volatile void *_Src,
	memory_order _Order1,
	memory_order _Order2);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_strong(
	volatile _Atomic_flag_t *_Flag,
	_CSTD size_t _Size,
	volatile void *_Tgt,
	volatile void *_Exp,
	const volatile void *_Src,
	memory_order _Order1,
	memory_order _Order2);

		/* LOCK_FREE PROPERTY */
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_is_lock_free_1();
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_is_lock_free_2();
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_is_lock_free_4();
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_is_lock_free_8();

		/* LOW-LEVEL LOCK-FREE ATOMIC OPERATIONS */
_CRTIMP2_PURE void __cdecl _Atomic_store_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE void __cdecl _Atomic_store_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE void __cdecl _Atomic_store_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE void __cdecl _Atomic_store_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_load_1(
	volatile _Uint1_t *, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_load_2(
	volatile _Uint2_t *, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_load_4(
	volatile _Uint4_t *, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_load_8(
	volatile _Uint8_t *, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_exchange_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_exchange_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_exchange_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_exchange_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_weak_1(
	volatile _Uint1_t *, _Uint1_t *, _Uint1_t, memory_order, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_weak_2(
	volatile _Uint2_t *, _Uint2_t *, _Uint2_t, memory_order, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_weak_4(
	volatile _Uint4_t *, _Uint4_t *, _Uint4_t, memory_order, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_weak_8(
	volatile _Uint8_t *, _Uint8_t *, _Uint8_t, memory_order, memory_order);

_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_strong_1(
	volatile _Uint1_t *, _Uint1_t *, _Uint1_t, memory_order, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_strong_2(
	volatile _Uint2_t *, _Uint2_t *, _Uint2_t, memory_order, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_strong_4(
	volatile _Uint4_t *, _Uint4_t *, _Uint4_t, memory_order, memory_order);
_CRTIMP2_PURE int __cdecl _Atomic_compare_exchange_strong_8(
	volatile _Uint8_t *, _Uint8_t *, _Uint8_t, memory_order, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_fetch_add_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_fetch_add_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_fetch_add_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_fetch_add_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_fetch_sub_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_fetch_sub_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_fetch_sub_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_fetch_sub_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_fetch_and_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_fetch_and_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_fetch_and_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_fetch_and_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_fetch_or_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_fetch_or_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_fetch_or_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_fetch_or_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __cdecl _Atomic_fetch_xor_1(
	volatile _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __cdecl _Atomic_fetch_xor_2(
	volatile _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __cdecl _Atomic_fetch_xor_4(
	volatile _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __cdecl _Atomic_fetch_xor_8(
	volatile _Uint8_t *, _Uint8_t, memory_order);

		/* LOW-LEVEL LOCKING ATOMIC OPERATIONS */
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_store_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_store_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_store_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_store_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_load_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_load_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_load_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_load_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_exchange_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_exchange_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_exchange_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_exchange_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_weak_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t *, _Uint1_t,
		memory_order, memory_order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_weak_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t *, _Uint2_t,
		memory_order, memory_order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_weak_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t *, _Uint4_t,
		memory_order, memory_order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_weak_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t *, _Uint8_t,
		memory_order, memory_order);

_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_strong_1_locking(
	volatile _Atomic_flag_t *,
		_Uint1_t *, _Uint1_t *, _Uint1_t, memory_order, memory_order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_strong_2_locking(
	volatile _Atomic_flag_t *,
		_Uint2_t *, _Uint2_t *, _Uint2_t, memory_order, memory_order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_strong_4_locking(
	volatile _Atomic_flag_t *,
		_Uint4_t *, _Uint4_t *, _Uint4_t, memory_order, memory_order);
_CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL _Atomic_compare_exchange_strong_8_locking(
	volatile _Atomic_flag_t *,
		_Uint8_t *, _Uint8_t *, _Uint8_t, memory_order, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_add_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_add_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_add_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_add_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_sub_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_sub_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_sub_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_sub_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_and_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_and_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_and_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_and_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_or_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_or_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_or_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_or_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

_CRTIMP2_PURE _Uint1_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_xor_1_locking(
	volatile _Atomic_flag_t *, _Uint1_t *, _Uint1_t, memory_order);
_CRTIMP2_PURE _Uint2_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_xor_2_locking(
	volatile _Atomic_flag_t *, _Uint2_t *, _Uint2_t, memory_order);
_CRTIMP2_PURE _Uint4_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_xor_4_locking(
	volatile _Atomic_flag_t *, _Uint4_t *, _Uint4_t, memory_order);
_CRTIMP2_PURE _Uint8_t __CLRCALL_PURE_OR_CDECL _Atomic_fetch_xor_8_locking(
	volatile _Atomic_flag_t *, _Uint8_t *, _Uint8_t, memory_order);

		/* ATOMIC FENCES */
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_thread_fence_locking(memory_order);
_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL _Atomic_signal_fence_locking(memory_order);
_CRTIMP2_PURE void __cdecl _Atomic_thread_fence(memory_order);
_CRTIMP2_PURE void __cdecl _Atomic_signal_fence(memory_order);

 #if _ATOMIC_FENCE_USES_LOCK
  #define _ATOMIC_THREAD_FENCE _Atomic_thread_fence_locking
  #define _ATOMIC_SIGNAL_FENCE _Atomic_signal_fence_locking

 #else /* _ATOMIC_FENCE_USES_LOCK */
  #define _ATOMIC_THREAD_FENCE _Atomic_thread_fence
  #define _ATOMIC_SIGNAL_FENCE _Atomic_signal_fence
 #endif /* _ATOMIC_FENCE_USES_LOCK */

_END_EXTERN_C
_STD_END
 #pragma pop_macro("new")
 #pragma warning(pop)
 #pragma pack(pop)
#endif /* RC_INVOKED */
#endif /* _XATOMIC_H */

/*
 * Copyright (c) 1992-2012 by P.J. Plauger.  ALL RIGHTS RESERVED.
 * Consult your license regarding permissions and restrictions.
V6.00:0009 */
