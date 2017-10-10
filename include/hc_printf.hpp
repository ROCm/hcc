#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <atomic>
#include <string>
#include <regex>
#include <iostream>
#include <algorithm>

#include "hc_am.hpp"
#include "hc.hpp"
#include "hsa_atomic.h"

#define HC_PRINTF_DEBUG  (0)

namespace hc {

union PrintfPacketData {
  unsigned int    ui;
  int             i;
  float           f;
  void*           ptr;
  const void*     cptr;
  
  // Header offset members (union uses same memory)
  // uia[0] - PrintfPacket buffer offset
  // uia[1] - Prtinf String buffer offset
  // al - Using a single atomic offset of 8B, update
  // both uias of 4B using single atomic operation.
  // ull - used to load offsets non-atomically, and
  // required to update atomic_ullong. Non-atomic
  // use of ull will also run faster.
  std::atomic_ullong al;
  unsigned int    uia[2];
  unsigned long long ull;
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
  ,PRINTF_UNSIGNED_INT
  ,PRINTF_SIGNED_INT
  ,PRINTF_FLOAT
  ,PRINTF_VOID_PTR
  ,PRINTF_CONST_VOID_PTR
  ,PRINTF_CHAR_PTR
  ,PRINTF_CONST_CHAR_PTR
};

class PrintfPacket {
public:
  void clear()             [[hc,cpu]] { type = PRINTF_UNUSED; }
  void set(unsigned int d) [[hc,cpu]] { type = PRINTF_UNSIGNED_INT;   data.ui = d; }
  void set(int d)          [[hc,cpu]] { type = PRINTF_SIGNED_INT;     data.i = d; }
  void set(float d)        [[hc,cpu]] { type = PRINTF_FLOAT;          data.f = d; }
  void set(void* d)        [[hc,cpu]] { type = PRINTF_VOID_PTR;       data.ptr = d; }
  void set(const void* d)  [[hc,cpu]] { type = PRINTF_CONST_VOID_PTR; data.cptr = d; }
  void set(char* d)        [[hc,cpu]] { type = PRINTF_CHAR_PTR;       data.ptr = d; }
  void set(const char* d)  [[hc,cpu]] { type = PRINTF_CONST_CHAR_PTR; data.cptr = d; }
  PrintfPacketDataType type;
  PrintfPacketData data;
};

enum PrintfError {
   PRINTF_SUCCESS = 0
  ,PRINTF_BUFFER_OVERFLOW = 1
  ,PRINTF_STRING_BUFFER_OVERFLOW = 2
};

static inline PrintfPacket* createPrintfBuffer(hc::accelerator& a, const unsigned int numElements) {
  PrintfPacket* printfBuffer = NULL;
  if (numElements > PRINTF_MIN_SIZE) {
    printfBuffer = hc::am_alloc(sizeof(PrintfPacket) * numElements, a, amHostCoherent);

    // Initialize the Header elements of the Printf Buffer
    printfBuffer[PRINTF_BUFFER_SIZE].type = PRINTF_BUFFER_SIZE;
    printfBuffer[PRINTF_BUFFER_SIZE].data.ui = numElements;

    // Header includes a helper string buffer which holds all char* args
    // PrintfPacket is 12 bytes, equivalent string buffer size used
    printfBuffer[PRINTF_STRING_BUFFER].type = PRINTF_STRING_BUFFER;
    printfBuffer[PRINTF_STRING_BUFFER].data.ptr = hc::am_alloc(sizeof(char) * numElements * 12, a, amHostCoherent);
    printfBuffer[PRINTF_STRING_BUFFER_SIZE].type = PRINTF_STRING_BUFFER_SIZE;
    printfBuffer[PRINTF_STRING_BUFFER_SIZE].data.ui = numElements * 12;

    // Using one atomic offset to maintain order and atomicity
    printfBuffer[PRINTF_OFFSETS].type = PRINTF_OFFSETS;
    printfBuffer[PRINTF_OFFSETS].data.uia[0] = PRINTF_HEADER_SIZE;
    printfBuffer[PRINTF_OFFSETS].data.uia[1] = 0;
  }
  return printfBuffer;
}

void deletePrintfBuffer(PrintfPacket*& buffer) {
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

static inline void copy_n(char* dest, const char* src, unsigned int len) [[hc,cpu]] {
  for(int i=0; i < len; i++){
    dest[i] = src[i];
  }
}

// get the argument count
static inline void countArg(unsigned int& count_arg, unsigned int& count_char) [[hc,cpu]] {}
template <typename T>
static inline void countArg(unsigned int& count_arg, unsigned int& count_char, const T t) [[hc,cpu]] {
  ++count_arg;
  if (std::is_same<T, char*>::value || std::is_same<T, const char*>::value){
    count_char += string_length((const char*)t) + 1;
  }
}
template <typename T, typename... Rest>
static inline void countArg(unsigned int& count_arg, unsigned int& count_char, const T t, const Rest&... rest) [[hc,cpu]] {
  ++count_arg;
  if (std::is_same<T, char*>::value || std::is_same<T, const char*>::value){
    count_char += string_length((const char*)t) + 1;
  }
  countArg(count_arg, count_char, rest...);
}

static inline PrintfError process_str_batch(PrintfPacket* queue, int poffset, unsigned int& soffset, const char* string) [[hc,cpu]] {
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

template <typename T>
static inline PrintfError set_batch(PrintfPacket* queue, int poffset, unsigned int& soffset, const T t) [[hc,cpu]] {
  PrintfError err = PRINTF_SUCCESS;
  queue[poffset].set(t);
  if (queue[poffset].type == PRINTF_CHAR_PTR || queue[poffset].type == PRINTF_CONST_CHAR_PTR){
    err = process_str_batch(queue, poffset, soffset, (char*)t);
  }
  return err;
}
template <typename T, typename... Rest>
static inline PrintfError set_batch(PrintfPacket* queue, int poffset, unsigned int& soffset, const T t, Rest... rest) [[hc,cpu]] {
  PrintfError err = PRINTF_SUCCESS;
  queue[poffset].set(t);
  if (queue[poffset].type == PRINTF_CHAR_PTR || queue[poffset].type == PRINTF_CONST_CHAR_PTR){
    if ((err = process_str_batch(queue, poffset, soffset, (char*)t)) != PRINTF_SUCCESS)
      return err;
  }
  return set_batch(queue, poffset + 1, soffset, rest...);
}

template <typename... All>
static inline PrintfError printf(PrintfPacket* queue, All... all) [[hc,cpu]] {
  unsigned int count_arg = 0;
  unsigned int count_char = 0;
  countArg(count_arg, count_char, all...);

  PrintfError error = PRINTF_SUCCESS;
  PrintfPacketData old_off, try_off;

  if (!queue || count_arg + 1 + queue[PRINTF_OFFSETS].data.uia[0] > queue[PRINTF_BUFFER_SIZE].data.ui) {
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
      old_off.ull = queue[PRINTF_OFFSETS].data.al.load();
      try_off.uia[0] = old_off.uia[0] + count_arg + 1;
      try_off.uia[1] = old_off.uia[1] + count_char;
    } while(!(queue[PRINTF_OFFSETS].data.al.compare_exchange_weak(old_off.ull, try_off.ull)));

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

// regex for finding format string specifiers
static std::regex specifierPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([diuoxXfFeEgGaAcsp]){1}");
static std::regex signedIntegerPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([cdi]){1}");
static std::regex unsignedIntegerPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([uoxX]){1}");
static std::regex floatPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([fFeEgGaA]){1}");
static std::regex pointerPattern("(%){1}[ps]");
static std::regex doubleAmpersandPattern("(%){2}");

static inline void processPrintfPackets(PrintfPacket* packets, const unsigned int numPackets) {

  for (unsigned int i = 0; i < numPackets; ) {

    unsigned int numPrintfArgs = packets[i++].data.ui;
    if (numPrintfArgs == 0)
      continue;

    // get the format
    unsigned int formatStringIndex = i++;
    assert(packets[formatStringIndex].type == PRINTF_CHAR_PTR
           || packets[formatStringIndex].type == PRINTF_CONST_CHAR_PTR);
    std::string formatString((const char*)packets[formatStringIndex].data.cptr);

    unsigned int formatStringCursor = 0;
    std::smatch specifierMatches;

#if HC_PRINTF_DEBUG
    std::printf("%s:%d \t number of matches = %d\n", __FUNCTION__, __LINE__, (int)specifierMatches.size());
#endif

    for (unsigned int j = 1; j < numPrintfArgs; ++j, ++i) {

      if (!std::regex_search(formatString, specifierMatches, specifierPattern)) {
        // More printf argument than format specifier??
        // Just skip to the next printf request
        break;
      }

      std::string specifier = specifierMatches.str();
#if HC_PRINTF_DEBUG
      std::cout << " (specifier found: " << specifier << ") ";
#endif

      // print the substring before the specifier
      // clean up all the double ampersands
      std::string prefix = specifierMatches.prefix();
      prefix = std::regex_replace(prefix,doubleAmpersandPattern,"%");
      std::printf("%s",prefix.c_str());

      std::smatch specifierTypeMatch;
      if (std::regex_search(specifier, specifierTypeMatch, unsignedIntegerPattern)) {
        std::printf(specifier.c_str(), packets[i].data.ui);
      } else if (std::regex_search(specifier, specifierTypeMatch, signedIntegerPattern)) {
        std::printf(specifier.c_str(), packets[i].data.i);
      } else if (std::regex_search(specifier, specifierTypeMatch, floatPattern)) {
        std::printf(specifier.c_str(), packets[i].data.f);
      } else if (std::regex_search(specifier, specifierTypeMatch, pointerPattern)) {
        std::printf(specifier.c_str(), packets[i].data.cptr);
      }
      else {
        assert(false);
      }
      formatString = specifierMatches.suffix();
    }
    // print the substring after the last specifier
    // clean up all the double ampersands before printing
    formatString = std::regex_replace(formatString,doubleAmpersandPattern,"%");
    std::printf("%s",formatString.c_str());
  }
}

static inline void processPrintfBuffer(PrintfPacket* gpuBuffer) {

  if (gpuBuffer == NULL) return;
  unsigned int bufferSize = gpuBuffer[PRINTF_BUFFER_SIZE].data.ui;
  unsigned int cursor = gpuBuffer[PRINTF_OFFSETS].data.uia[0];
  unsigned int numPackets = ((bufferSize<cursor)?bufferSize:cursor) - PRINTF_HEADER_SIZE;
  if (numPackets > 0) {
    processPrintfPackets(gpuBuffer+PRINTF_HEADER_SIZE, numPackets);
  }
  // reset the printf buffer and string buffer
  gpuBuffer[PRINTF_OFFSETS].data.uia[0] = PRINTF_HEADER_SIZE;
  gpuBuffer[PRINTF_OFFSETS].data.uia[1] = 0;
}


} // namespace hc