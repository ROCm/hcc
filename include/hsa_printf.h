#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <atomic>
#include <string>
#include <regex>
#include <iostream>
#include <algorithm>

#include <hc_am.hpp>
#include <hsa_atomic.h>

#define HSA_PRINTF_DEBUG  (0)

union HSAPrintfPacketData {
  unsigned int    ui;
  int             i;
  float           f;
  void*           ptr;
  const void*     cptr;
  std::atomic_int ai;
};

enum HSAPrintfPacketDataType {
  HSA_PRINTF_UNUSED       
  ,HSA_PRINTF_UNSIGNED_INT 
  ,HSA_PRINTF_SIGNED_INT  
  ,HSA_PRINTF_FLOAT       
  ,HSA_PRINTF_VOID_PTR    
  ,HSA_PRINTF_CONST_VOID_PTR
  ,HSA_PRINTF_BUFFER_CURSOR
  ,HSA_PRINTF_BUFFER_SIZE
};

class HSAPrintfPacket {
public:
  void clear()             [[hc,cpu]] { type = HSA_PRINTF_UNUSED; }
  void set(unsigned int d) [[hc,cpu]] { type = HSA_PRINTF_UNSIGNED_INT;   data.ui = d; }
  void set(int d)          [[hc,cpu]] { type = HSA_PRINTF_SIGNED_INT;     data.i = d; }
  void set(float d)        [[hc,cpu]] { type = HSA_PRINTF_FLOAT;          data.f = d; }
  void set(void* d)        [[hc,cpu]] { type = HSA_PRINTF_VOID_PTR;       data.ptr = d; }
  void set(const void* d)  [[hc,cpu]] { type = HSA_PRINTF_CONST_VOID_PTR; data.cptr = d; }
  HSAPrintfPacketDataType type;
  HSAPrintfPacketData data;
};

enum HSAPrintfError {
   HSA_PRINTF_SUCCESS = 0
  ,HSA_PRINTF_BUFFER_OVERFLOW = 1
};

static inline HSAPrintfPacket* createPrintfBuffer(hc::accelerator& a, const unsigned int numElements) {
  HSAPrintfPacket* printfBuffer = NULL;
  if (numElements > 3) {
    printfBuffer = hc::am_alloc(sizeof(HSAPrintfPacket) * numElements, a, 0);

    // initialize the printf buffer header
    HSAPrintfPacket header[2];
    header[0].type = HSA_PRINTF_BUFFER_SIZE;
    header[0].data.ui = numElements;
    header[1].type = HSA_PRINTF_BUFFER_CURSOR;
    header[1].data.ui = 2;
    hc::am_copy(printfBuffer,header,sizeof(HSAPrintfPacket) * 2);
  }
  return printfBuffer;
}

void deletePrintfBuffer(HSAPrintfPacket* buffer) {
  hc::am_free(buffer);
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

template <typename T>
static inline void set_batch(HSAPrintfPacket* queue, int offset, const T t) [[hc,cpu]] {
  queue[offset].set(t);
}
template <typename T, typename... Rest>
static inline void set_batch(HSAPrintfPacket* queue, int offset, const T t, Rest... rest) [[hc,cpu]] {
  queue[offset].set(t);
  set_batch(queue, offset + 1, rest...);
}

template <typename... All>
static inline HSAPrintfError hsa_printf(HSAPrintfPacket* queue, All... all) [[hc,cpu]] {
  unsigned int count = 0;      
  countArg(count, all...);

  HSAPrintfError error = HSA_PRINTF_SUCCESS;

  if (count + 1 + queue[1].data.ai.load() > queue[0].data.ui) {
    error = HSA_PRINTF_BUFFER_OVERFLOW;
  } else {

#if 0
    /*** FIXME: hcc didn't promote the address of the atomic type into global address space ***/
    unsigned int offset = queue[1].data.ai.fetch_add(count + 1);
#endif
    unsigned int offset = __hsail_atomic_fetch_add_unsigned(&(queue[1].data.ui),count + 1);

    if (offset + count + 1 < queue[0].data.ui) { 
      set_batch(queue, offset, count, all...);
    }
    else {
      error = HSA_PRINTF_BUFFER_OVERFLOW;
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

static inline void processPrintfPackets(HSAPrintfPacket* packets, const unsigned int numPackets) {

  for (unsigned int i = 0; i < numPackets; ) {

    unsigned int numPrintfArgs = packets[i++].data.ui;
    if (numPrintfArgs == 0)
      continue;

    // get the format
    unsigned int formatStringIndex = i++;
    assert(packets[formatStringIndex].type == HSA_PRINTF_VOID_PTR
           || packets[formatStringIndex].type == HSA_PRINTF_CONST_VOID_PTR);
    std::string formatString((const char*)packets[formatStringIndex].data.cptr);

    unsigned int formatStringCursor = 0;
    std::smatch specifierMatches;

#if HSA_PRINTF_DEBUG
    printf("%s:%d \t number of matches = %d\n", __FUNCTION__, __LINE__, (int)specifierMatches.size());
#endif
    
    for (unsigned int j = 1; j < numPrintfArgs; ++j, ++i) {

      if (!std::regex_search(formatString, specifierMatches, specifierPattern)) {
        // More printf argument than format specifier??
        // Just skip to the next printf request
        break;
      }

      std::string specifier = specifierMatches.str();
#if HSA_PRINTF_DEBUG
      std::cout << " (specifier found: " << specifier << ") ";
#endif

      // print the substring before the specifier
      // clean up all the double ampersands
      std::string prefix = specifierMatches.prefix();
      prefix = std::regex_replace(prefix,doubleAmpersandPattern,"%");
      printf("%s",prefix.c_str());
      
      std::smatch specifierTypeMatch;
      if (std::regex_search(specifier, specifierTypeMatch, unsignedIntegerPattern)) {
        printf(specifier.c_str(), packets[i].data.ui);
      } else if (std::regex_search(specifier, specifierTypeMatch, signedIntegerPattern)) {
        printf(specifier.c_str(), packets[i].data.i);
      } else if (std::regex_search(specifier, specifierTypeMatch, floatPattern)) {
        printf(specifier.c_str(), packets[i].data.f);
      } else if (std::regex_search(specifier, specifierTypeMatch, pointerPattern)) {
        printf(specifier.c_str(), packets[i].data.cptr);
      }
      else {
        assert(false);
      }
      formatString = specifierMatches.suffix();
    }
    // print the substring after the last specifier
    // clean up all the double ampersands before printing
    formatString = std::regex_replace(formatString,doubleAmpersandPattern,"%");
    printf("%s",formatString.c_str());
  }
}

static inline void processPrintfBuffer(HSAPrintfPacket* gpuBuffer) {

  if (gpuBuffer == NULL) return;

  HSAPrintfPacket header[2];
  hc::am_copy(header, gpuBuffer, sizeof(HSAPrintfPacket)*2);
  unsigned int bufferSize = header[0].data.ui;
  unsigned int cursor = header[1].data.ui;
  unsigned int numPackets = ((bufferSize<cursor)?bufferSize:cursor) - 2;
  if (numPackets > 0) {
    HSAPrintfPacket* hostBuffer = (HSAPrintfPacket*)malloc(sizeof(HSAPrintfPacket) * numPackets);
    if (hostBuffer) {
      hc::am_copy(hostBuffer, gpuBuffer+2, sizeof(HSAPrintfPacket) * numPackets);
      processPrintfPackets(hostBuffer, numPackets);
      free(hostBuffer);
    }
  }
  // reset the printf buffer
  header[1].data.ui = 2;
  hc::am_copy(gpuBuffer,header,sizeof(HSAPrintfPacket) * 2);
}
