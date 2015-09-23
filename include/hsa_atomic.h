#pragma once

#define HSAIL_BUILTIN_GPU __attribute__((hc)) 
#define HSAIL_BUILTIN_CPU __attribute__((cpu)) 

#ifdef __KALMAR_ACCELERATOR__

// fetch_add
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_fetch_add_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_fetch_add_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_fetch_add_int64(int64_t* dest, int64_t val);

// exchange
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_exchange_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_exchange_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_exchange_int64(int64_t* dest, int64_t val);

// compare_exchange
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_compare_exchange_int(int* dest, int compare, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_compare_exchange_unsigned(unsigned int* dest, unsigned int compare, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_compare_exchange_int64(int64_t* dest, int64_t compare, int64_t val);

#else

// fetch_add
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_fetch_add_int(int* dest, int val)
{ return __sync_fetch_and_add(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_fetch_add_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_fetch_and_add(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_fetch_add_int64(int64_t* dest, int64_t val)
{ return __sync_fetch_and_add(dest, val); }

// exchange
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_exchange_int(int* dest, int val)
{ return __sync_swap(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_exchange_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_swap(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_exchange_int64(int64_t* dest, int64_t val)
{ return __sync_swap(dest, val); }

// compare_exchange
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_compare_exchange_int(int* dest, int compare, int val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_compare_exchange_unsigned(unsigned int* dest, unsigned int compare, unsigned int val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_compare_exchange_int64(int64_t* dest, int64_t compare, int64_t val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

#endif
