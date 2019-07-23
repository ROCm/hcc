#pragma once

#include <type_traits>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <atomic>
#include <string>
#include <regex>
#include <iostream>
#include <algorithm>

#include "hc_am_internal.hpp"
#include "hsa_atomic.h"

// The printf on the accelerator is only enabled when
// The HCC_ENABLE_ACCELERATOR_PRINTF is defined

// Disabling hc::printf it's broken on certain platform
// and this experimental feature is not longer being maintained.  
#ifdef HCC_ENABLE_ACCELERATOR_PRINTF
  #undef HCC_ENABLE_ACCELERATOR_PRINTF
  #warning "The experimental hc::printf is no longer supported and is disabled"
#endif

// Indicate whether hc::printf is supported
//#define HC_FEATURE_PRINTF (1)

// Enable extra debug messages
#define HC_PRINTF_DEBUG  (0)

namespace hc {

/*
* Supported Types
* Pointer Types
* void*
* const void*
* Integer Types
* uint8_t, int8_t - unsigned char, char
* uint16_t, int16_t - unsigned short, short, uchar16_t, char16_t
* uint32_t, int32_t - unsigned int, int, unsigned long, long, uchar32_t, char32_t
* uint64_t, int64_t - 64 bit uint/ints
* unsigned long long, long long - at least 64 bits
* Floating Point Types
* half - 16 bit fp
* float - 32 bit fp
* double - 64 bit fp
*/

union PrintfPacketData {
  uint32_t        ui;
  int32_t         i;
  uint64_t        uli;
  int64_t         li;
  hc::half        h;
  float           f;
  double          d;
  void*           ptr;
  const void*     cptr;

  // Header offset members (union uses same memory)
  // uia[0] - PrintfPacket buffer offset
  // uia[1] - Printf String buffer offset
  // ali - Using a single atomic offset of 8B, update
  // both uias of 4B using single atomic operation.
  // ull - used to load offsets non-atomically, and
  // required to update atomic_ullong. Non-atomic
  // use of ull will also run faster.
  std::atomic<uint64_t> ali;
  uint32_t        uia[2];
};

enum PrintfPacketDataType {
  // Header types
  PRINTF_BUFFER_SIZE = 0
  ,PRINTF_STRING_BUFFER = 1
  ,PRINTF_STRING_BUFFER_SIZE = 2
  ,PRINTF_OFFSETS = 3
  ,PRINTF_HEADER_SIZE = 4
  ,PRINTF_MIN_SIZE = 5

  // Packet Data types
  ,PRINTF_UNUSED
  ,PRINTF_UINT32_T
  ,PRINTF_INT32_T
  ,PRINTF_UINT64_T
  ,PRINTF_INT64_T
  ,PRINTF_HALF
  ,PRINTF_FLOAT
  ,PRINTF_DOUBLE
  ,PRINTF_VOID_PTR
  ,PRINTF_CONST_VOID_PTR
  ,PRINTF_CHAR_PTR
  ,PRINTF_CONST_CHAR_PTR
};

class PrintfPacket {
public:
  void clear()             [[hc,cpu]] { type = PRINTF_UNUSED; }
  void set(uint32_t d)     [[hc,cpu]] { type = PRINTF_UINT32_T;       data.ui = d; }
  void set(int32_t d)      [[hc,cpu]] { type = PRINTF_INT32_T;        data.i = d; }
  void set(uint64_t d)     [[hc,cpu]] { type = PRINTF_UINT64_T;       data.uli = d; }
  void set(int64_t d)      [[hc,cpu]] { type = PRINTF_INT64_T;        data.li = d; }
  void set(unsigned long long d) [[hc,cpu]] { type = PRINTF_UINT64_T; data.uli = d; }
  void set(long long d)    [[hc,cpu]] { type = PRINTF_INT64_T;        data.li = d; }
  void set(hc::half d)     [[hc,cpu]] { type = PRINTF_HALF;           data.h = d; }
  void set(float d)        [[hc,cpu]] { type = PRINTF_FLOAT;          data.f = d; }
  void set(double d)       [[hc,cpu]] { type = PRINTF_DOUBLE;         data.d = d; }
  void set(void* d)        [[hc,cpu]] { type = PRINTF_VOID_PTR;       data.ptr = d; }
  void set(const void* d)  [[hc,cpu]] { type = PRINTF_CONST_VOID_PTR; data.cptr = d; }
  void set(char* d)        [[hc,cpu]] { type = PRINTF_CHAR_PTR;       data.ptr = d; }
  void set(const char* d)  [[hc,cpu]] { type = PRINTF_CONST_CHAR_PTR; data.cptr = d; }
  PrintfPacketDataType type;
  PrintfPacketData data;
};

// Global printf buffer
// The actual variable is currently defined in mcwamp_hsa.cpp
extern PrintfPacket* printf_buffer;

enum PrintfError {
   PRINTF_SUCCESS = 0
  ,PRINTF_BUFFER_OVERFLOW = 1
  ,PRINTF_STRING_BUFFER_OVERFLOW = 2
  ,PRINTF_UNKNOWN_ERROR = 3
  ,PRINTF_BUFFER_NULLPTR = 4
};

static inline PrintfPacket* createPrintfBuffer(const unsigned int numElements) {
  PrintfPacket* printfBuffer = NULL;
  if (numElements > PRINTF_MIN_SIZE) {
    printfBuffer = hc::internal::am_alloc_host_coherent(sizeof(PrintfPacket) * numElements);

    // Initialize the Header elements of the Printf Buffer
    printfBuffer[PRINTF_BUFFER_SIZE].type = PRINTF_BUFFER_SIZE;
    printfBuffer[PRINTF_BUFFER_SIZE].data.ui = numElements;

    // Header includes a helper string buffer which holds all char* args
    // PrintfPacket is 12 bytes, equivalent string buffer size used
    printfBuffer[PRINTF_STRING_BUFFER].type = PRINTF_STRING_BUFFER;
    printfBuffer[PRINTF_STRING_BUFFER].data.ptr = hc::internal::am_alloc_host_coherent(sizeof(char) * numElements * 12);
    printfBuffer[PRINTF_STRING_BUFFER_SIZE].type = PRINTF_STRING_BUFFER_SIZE;
    printfBuffer[PRINTF_STRING_BUFFER_SIZE].data.ui = numElements * 12;

    // Using one atomic offset to maintain order and atomicity
    printfBuffer[PRINTF_OFFSETS].type = PRINTF_OFFSETS;
    printfBuffer[PRINTF_OFFSETS].data.uia[0] = PRINTF_HEADER_SIZE;
    printfBuffer[PRINTF_OFFSETS].data.uia[1] = 0;
  }
  return printfBuffer;
}

static inline void deletePrintfBuffer(PrintfPacket*& buffer) {
  if (buffer){
    if (buffer[PRINTF_STRING_BUFFER].data.ptr)
      hc::am_free(buffer[PRINTF_STRING_BUFFER].data.ptr);
    hc::am_free(buffer);
  }
  buffer = NULL;
}

static inline unsigned int string_length(const char* str) [[hc,cpu]]{
  unsigned int size = 0;
  while(str[size]!='\0')
    size++;
  return size;
}

static inline void copy_n(char* dest, const char* src, const unsigned int len) [[hc,cpu]] {
  for(unsigned int i=0; i < len; i++){
    dest[i] = src[i];
  }
}

// return the memory size (including '/0') if it's a C-string
template <typename T>
std::size_t mem_size_if_string(typename std::enable_if< std::is_same<T,const char*>::value
                                                        || std::is_same<T,char*>::value, T>::type  s) [[hc,cpu]] {
  return string_length(s) + 1;
}

template <typename T>
std::size_t mem_size_if_string(typename std::enable_if< !std::is_same<T,const char*>::value
                                                         && !std::is_same<T,char*>::value, T>::type  s) [[hc,cpu]] {
  return 0;
}

// get the argument count
static inline void countArg(unsigned int& count_arg, unsigned int& count_char) [[hc,cpu]] {}
template <typename T>
static inline void countArg(unsigned int& count_arg, unsigned int& count_char, const T t) [[hc,cpu]] {
  ++count_arg;
  count_char += mem_size_if_string<T>(t);
}
template <typename T, typename... Rest>
static inline void countArg(unsigned int& count_arg, unsigned int& count_char, const T t, const Rest&... rest) [[hc,cpu]] {
  ++count_arg;
  count_char += mem_size_if_string<T>(t);
  countArg(count_arg, count_char, rest...);
}

template<typename T>
PrintfError process_str_batch(PrintfPacket* queue, int poffset, unsigned int& soffset
, typename std::enable_if< std::is_same<T,const char*>::value || std::is_same<T,char*>::value, T>::type string) [[hc,cpu]] {

  if (queue[poffset].type != PRINTF_CHAR_PTR && queue[poffset].type != PRINTF_CONST_CHAR_PTR)
    return PRINTF_UNKNOWN_ERROR;

  unsigned int str_len = string_length(string);
  unsigned int sb_offset = soffset;
  char* string_buffer = (char*) queue[PRINTF_STRING_BUFFER].data.ptr;
  if (!string_buffer || soffset + str_len + 1 > queue[PRINTF_STRING_BUFFER_SIZE].data.ui){
    return PRINTF_STRING_BUFFER_OVERFLOW;
  }
  copy_n(&string_buffer[sb_offset], string, str_len + 1);
  queue[poffset].set(&string_buffer[sb_offset]);
  soffset += str_len + 1;
  return PRINTF_SUCCESS;
}

template<typename T>
PrintfError process_str_batch(PrintfPacket* queue, int poffset, unsigned int& soffset
, typename std::enable_if< !std::is_same<T,const char*>::value && !std::is_same<T,char*>::value, T>::type data) [[hc,cpu]] {

  if (queue[poffset].type == PRINTF_CHAR_PTR || queue[poffset].type == PRINTF_CONST_CHAR_PTR)
    return PRINTF_UNKNOWN_ERROR;
  else
    return PRINTF_SUCCESS;
}

template <typename T>
static inline PrintfError set_batch(PrintfPacket* queue, int poffset, unsigned int& soffset, const T t) [[hc,cpu]] {
  PrintfError err = PRINTF_SUCCESS;
  queue[poffset].set(t);
  err = process_str_batch<T>(queue, poffset, soffset, t);
  return err;
}

template <typename T, typename... Rest>
static inline PrintfError set_batch(PrintfPacket* queue, int poffset, unsigned int& soffset, const T t, Rest... rest) [[hc,cpu]] {
  PrintfError err = PRINTF_SUCCESS;
  queue[poffset].set(t);

  if ((err = process_str_batch<T>(queue, poffset, soffset, t)) != PRINTF_SUCCESS)
    return err;

  return set_batch(queue, poffset + 1, soffset, rest...);
}

template <typename... All>
static inline PrintfError printf(PrintfPacket* queue, All... all) [[hc,cpu]] {
  unsigned int count_arg = 0;
  unsigned int count_char = 0;
  countArg(count_arg, count_char, all...);

  PrintfError error = PRINTF_SUCCESS;
  PrintfPacketData old_off, try_off;

  if (!queue) {
    error = PRINTF_BUFFER_NULLPTR;
  }
  else if (count_arg + 1 + queue[PRINTF_OFFSETS].data.uia[0] > queue[PRINTF_BUFFER_SIZE].data.ui) {
    error = PRINTF_BUFFER_OVERFLOW;
  }
  else if (!queue[PRINTF_STRING_BUFFER].data.ptr || count_char + queue[PRINTF_OFFSETS].data.uia[1] > queue[PRINTF_STRING_BUFFER_SIZE].data.ui){
    error = PRINTF_STRING_BUFFER_OVERFLOW;
  }
  else {
    do {
      // Suggest an offset and compete with other kernels for a spot.
      // One kernel will make it through at a time. Attempt
      // to win a portion of printf buffer and printf string buffer.
      // Otherwise, update to latest offset values, and try again.
      old_off.uli = queue[PRINTF_OFFSETS].data.ali.load();
      try_off.uia[0] = old_off.uia[0] + count_arg + 1;
      try_off.uia[1] = old_off.uia[1] + count_char;
    } while(!(queue[PRINTF_OFFSETS].data.ali.compare_exchange_weak(old_off.uli, try_off.uli)));

    unsigned int poffset = (unsigned int)old_off.uia[0];
    unsigned int soffset = (unsigned int)old_off.uia[1];

    if (poffset + count_arg + 1 > queue[PRINTF_BUFFER_SIZE].data.ui) {
      error = PRINTF_BUFFER_OVERFLOW;
    }
    else if (soffset + count_char > queue[PRINTF_STRING_BUFFER_SIZE].data.ui){
      error = PRINTF_STRING_BUFFER_OVERFLOW;
    }
    else {
      if (set_batch(queue, poffset, soffset, count_arg, all...) != PRINTF_SUCCESS)
        error = PRINTF_STRING_BUFFER_OVERFLOW;
    }
  }

  return error;
}


// The presence of hc::printf may impact performance even when it's not being called.
// Currently hcc's printf on accelerator is an opt-in feature.  This means that users
// have to define HCC_ENABLE_ACCELERATOR_PRINTF to enable it.
#ifdef HCC_ENABLE_ACCELERATOR_PRINTF

template <typename... All>
static inline PrintfError printf(const char* format_string, All... all) [[hc,cpu]] {
  return printf(hc::printf_buffer, format_string, all...);
}

#else

// this is just a stubs for printf that doesn't do anything
template <typename... All>
static inline PrintfError printf(const char* format_string, All... all) [[hc,cpu]] {
  return PRINTF_SUCCESS;
}

#endif

} // namespace hc
