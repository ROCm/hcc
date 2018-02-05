#pragma once

#define HSAIL_BUILTIN_GPU __attribute__((hc)) 
#define HSAIL_BUILTIN_CPU __attribute__((cpu)) inline

#ifdef __KALMAR_ACCELERATOR__

// fetch_add
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_fetch_add_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_fetch_add_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_fetch_add_int64(int64_t* dest, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_fetch_add_uint64(uint64_t* dest, uint64_t val);

// fetch_sub
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_fetch_sub_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_fetch_sub_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_fetch_sub_int64(int64_t* dest, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_fetch_sub_uint64(uint64_t* dest, uint64_t val);

// fetch_and
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_fetch_and_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_fetch_and_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_fetch_and_int64(int64_t* dest, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_fetch_and_uint64(uint64_t* dest, uint64_t val);

// fetch_or
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_fetch_or_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_fetch_or_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_fetch_or_int64(int64_t* dest, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_fetch_or_uint64(uint64_t* dest, uint64_t val);

// fetch_xor
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_fetch_xor_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_fetch_xor_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_fetch_xor_int64(int64_t* dest, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_fetch_xor_uint64(uint64_t* dest, uint64_t val);

// exchange
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_exchange_int(int* dest, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_exchange_unsigned(unsigned int* dest, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_exchange_int64(int64_t* dest, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_exchange_uint64(uint64_t* dest, uint64_t val);

// compare_exchange
extern "C" HSAIL_BUILTIN_GPU int __hsail_atomic_compare_exchange_int(int* dest, int compare, int val);
extern "C" HSAIL_BUILTIN_GPU unsigned int __hsail_atomic_compare_exchange_unsigned(unsigned int* dest, unsigned int compare, unsigned int val);
extern "C" HSAIL_BUILTIN_GPU int64_t __hsail_atomic_compare_exchange_int64(int64_t* dest, int64_t compare, int64_t val);
extern "C" HSAIL_BUILTIN_GPU uint64_t __hsail_atomic_compare_exchange_uint64(uint64_t* dest, uint64_t compare, uint64_t val);

#else

// fetch_add
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_fetch_add_int(int* dest, int val)
{ return __sync_fetch_and_add(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_fetch_add_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_fetch_and_add(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_fetch_add_int64(int64_t* dest, int64_t val)
{ return __sync_fetch_and_add(dest, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_fetch_add_uint64(uint64_t* dest, uint64_t val)
{ return __sync_fetch_and_add(dest, val); }

// fetch_sub
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_fetch_sub_int(int* dest, int val)
{ return __sync_fetch_and_sub(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_fetch_sub_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_fetch_and_sub(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_fetch_sub_int64(int64_t* dest, int64_t val)
{ return __sync_fetch_and_sub(dest, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_fetch_sub_uint64(uint64_t* dest, uint64_t val)
{ return __sync_fetch_and_sub(dest, val); }

// fetch_and
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_fetch_and_int(int* dest, int val)
{ return __sync_fetch_and_and(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_fetch_and_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_fetch_and_and(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_fetch_and_int64(int64_t* dest, int64_t val)
{ return __sync_fetch_and_and(dest, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_fetch_and_uint64(uint64_t* dest, uint64_t val)
{ return __sync_fetch_and_and(dest, val); }

// fetch_or
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_fetch_or_int(int* dest, int val)
{ return __sync_fetch_and_or(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_fetch_or_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_fetch_and_or(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_fetch_or_int64(int64_t* dest, int64_t val)
{ return __sync_fetch_and_or(dest, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_fetch_or_uint64(uint64_t* dest, uint64_t val)
{ return __sync_fetch_and_or(dest, val); }

// fetch_xor
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_fetch_xor_int(int* dest, int val)
{ return __sync_fetch_and_xor(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_fetch_xor_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_fetch_and_xor(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_fetch_xor_int64(int64_t* dest, int64_t val)
{ return __sync_fetch_and_xor(dest, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_fetch_xor_uint64(uint64_t* dest, uint64_t val)
{ return __sync_fetch_and_xor(dest, val); }

// exchange
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_exchange_int(int* dest, int val)
{ return __sync_swap(dest, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_exchange_unsigned(unsigned int* dest, unsigned int val)
{ return __sync_swap(dest, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_exchange_int64(int64_t* dest, int64_t val)
{ return __sync_swap(dest, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_exchange_uint64(uint64_t* dest, uint64_t val)
{ return __sync_swap(dest, val); }

// compare_exchange
extern "C" HSAIL_BUILTIN_CPU int __hsail_atomic_compare_exchange_int(int* dest, int compare, int val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

extern "C" HSAIL_BUILTIN_CPU unsigned int __hsail_atomic_compare_exchange_unsigned(unsigned int* dest, unsigned int compare, unsigned int val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

extern "C" HSAIL_BUILTIN_CPU int64_t __hsail_atomic_compare_exchange_int64(int64_t* dest, int64_t compare, int64_t val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

extern "C" HSAIL_BUILTIN_CPU uint64_t __hsail_atomic_compare_exchange_uint64(uint64_t* dest, uint64_t compare, uint64_t val)
{ return __sync_val_compare_and_swap(dest, compare, val); }

#endif
