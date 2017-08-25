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
  std::atomic_int ai;
};

enum PrintfPacketDataType {
  PRINTF_UNUSED
  ,PRINTF_UNSIGNED_INT
  ,PRINTF_SIGNED_INT
  ,PRINTF_FLOAT
  ,PRINTF_VOID_PTR
  ,PRINTF_CONST_VOID_PTR
  ,PRINTF_CHAR_PTR
  ,PRINTF_CONST_CHAR_PTR
  ,PRINTF_BUFFER_CURSOR
  ,PRINTF_BUFFER_SIZE
  ,PRINTF_STRING_BUFFER
  ,PRINTF_STRING_BUFFER_SIZE
  ,PRINTF_STRING_BUFFER_CURSOR
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
  if (numElements > 6) {
    printfBuffer = hc::am_alloc(sizeof(PrintfPacket) * numElements, a, amHostCoherent);

    // initialize the printf buffer header
    printfBuffer[0].type = PRINTF_BUFFER_SIZE;
    printfBuffer[0].data.ui = numElements;
    printfBuffer[1].type = PRINTF_BUFFER_CURSOR;
    printfBuffer[1].data.ui = 5;
    printfBuffer[2].type = PRINTF_STRING_BUFFER;
    printfBuffer[2].data.ptr = hc::am_alloc(sizeof(char) * numElements * 12, a, amHostCoherent);
    printfBuffer[3].type = PRINTF_STRING_BUFFER_SIZE;
    printfBuffer[3].data.ui = numElements * 12;
    printfBuffer[4].type = PRINTF_STRING_BUFFER_CURSOR;
    printfBuffer[4].data.ui = 0;
  }
  return printfBuffer;
}

void deletePrintfBuffer(PrintfPacket*& buffer) {
  hc::am_free(buffer[2].data.ptr);
  hc::am_free(buffer);
  buffer = NULL;
}

// get the argument count
static inline void countArg(unsigned int& count) [[hc,cpu]] {}
template <typename T>
static inline void countArg(unsigned int& count, const T& t) [[hc,cpu]] { ++count; }
template <typename T, typename... Rest>
static inline void countArg(unsigned int& count, const T& t, const Rest&... rest) [[hc,cpu]] {
  ++count;
  countArg(count,rest...);
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

static inline PrintfError process_str_batch(PrintfPacket* queue, int offset, const char* string) [[hc,cpu]] {
  unsigned int str_len = string_length(string);
  unsigned int sb_offset = queue[4].data.ai.fetch_add(str_len + 1);
  char* string_buffer = (char*) queue[2].data.ptr;
  if (!string_buffer || queue[4].data.ui > queue[3].data.ui){
    return PRINTF_STRING_BUFFER_OVERFLOW;
  }
  copy_n(&string_buffer[sb_offset], string, str_len + 1);
  queue[offset].set(&string_buffer[sb_offset]);
  return PRINTF_SUCCESS;
}

template <typename T>
static inline PrintfError set_batch(PrintfPacket* queue, int offset, const T t) [[hc,cpu]] {
  PrintfError err = PRINTF_SUCCESS;
  queue[offset].set(t);
  if (queue[offset].type == PRINTF_CHAR_PTR || queue[offset].type == PRINTF_CONST_CHAR_PTR){
    err = process_str_batch(queue, offset, (char*)t);
  }
  return err;
}
template <typename T, typename... Rest>
static inline PrintfError set_batch(PrintfPacket* queue, int offset, const T t, Rest... rest) [[hc,cpu]] {
  PrintfError err = PRINTF_SUCCESS;
  queue[offset].set(t);
  if (queue[offset].type == PRINTF_CHAR_PTR || queue[offset].type == PRINTF_CONST_CHAR_PTR){
    if ((err = process_str_batch(queue, offset, (char*)t)) != PRINTF_SUCCESS)
      return err;
  }
  return set_batch(queue, offset + 1, rest...);
}

template <typename... All>
static inline PrintfError printf(PrintfPacket* queue, All... all) [[hc,cpu]] {
  unsigned int count = 0;
  countArg(count, all...);

  PrintfError error = PRINTF_SUCCESS;

  if (!queue || count + 1 + queue[1].data.ui > queue[0].data.ui) {
    error = PRINTF_BUFFER_OVERFLOW;
  } else {

    unsigned int offset = queue[1].data.ai.fetch_add(count + 1);
    if (offset + count + 1 < queue[0].data.ui) {
      if (set_batch(queue, offset, count, all...) != PRINTF_SUCCESS)
        error = PRINTF_STRING_BUFFER_OVERFLOW;
    }
    else {
      error = PRINTF_BUFFER_OVERFLOW;
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
  unsigned int bufferSize = gpuBuffer[0].data.ui;
  unsigned int cursor = gpuBuffer[1].data.ui;
  unsigned int numPackets = ((bufferSize<cursor)?bufferSize:cursor) - 5;
  if (numPackets > 0) {
    processPrintfPackets(gpuBuffer+5, numPackets);
  }
  // reset the printf buffer and string buffer
  gpuBuffer[1].data.ui = 5;
  gpuBuffer[4].data.ui = 0;
}


} // namespace hc

