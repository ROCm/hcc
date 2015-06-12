#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <atomic>
#include <string>
#include <regex>
#include <iostream>

#define HSA_PRINTF_DEBUG  (0)

union HSAPrintfPacketData {
  unsigned int ui;
  int i;
  float f;
  void* ptr;
  const void* cptr;
};

enum HSAPrintfPacketDataType {
  HSA_PRINTF_UNUSED       // 0
  ,HSA_PRINTF_UNSIGNED_INT // 1
  ,HSA_PRINTF_SIGNED_INT  // 2
  ,HSA_PRINTF_FLOAT       // 3
  ,HSA_PRINTF_VOID_PTR    // 4
  ,HSA_PRINTF_CONST_VOID_PTR  // 5
};

class HSAPrintfPacket {
public:
  void clear() { type = HSA_PRINTF_UNUSED; }
  void set(unsigned int d)  { type = HSA_PRINTF_UNSIGNED_INT;   data.ui = d; }
  void set(int d)           { type = HSA_PRINTF_SIGNED_INT;     data.i = d; }
  void set(float d)         { type = HSA_PRINTF_FLOAT;          data.f = d; }
  void set(void* d)         { type = HSA_PRINTF_VOID_PTR;       data.ptr = d; }
  void set(const void* d)   { type = HSA_PRINTF_CONST_VOID_PTR; data.cptr = d; }
  HSAPrintfPacketDataType type;
  HSAPrintfPacketData data;
};

enum HSAPrintfError {
   HSA_PRINTF_SUCCESS = 0
  ,HSA_PRINTF_BUFFER_OVERFLOW = 1
};

class HSAPrintfPacketQueue {
public:
  HSAPrintfPacketQueue(HSAPrintfPacket* buffer, unsigned int num)
        :queue(buffer),num(num),overflow(false),cursor(0) {}
  HSAPrintfPacket* queue;
  unsigned int num;
  bool overflow;
  std::atomic_int cursor;
};

static inline HSAPrintfPacketQueue* createHSAPrintfPacketQueue(unsigned int num) {
  HSAPrintfPacket* buffer = new HSAPrintfPacket[num];
  HSAPrintfPacketQueue* queue = new HSAPrintfPacketQueue(buffer, num);
  return queue;
}

static inline HSAPrintfPacketQueue* destroyHSAPrintfPacketQueue(HSAPrintfPacketQueue* queue) {
  delete[]  queue->queue;
  delete queue;
  return NULL;
}

static inline void dumpHSAPrintfPacketQueue(const HSAPrintfPacketQueue* q) {
  std::cout << "buffer size: " << q->num << " "
            << "cursor: " << q->cursor.load() << " "
            << "overflow: " << q->overflow << "\n";

#if HSA_PRINTF_DEBUG 
  for (int i = 0; i < q->num / 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      std::cout << q->queue[i * 16 + j].type << " ";
    }
    std::cout << "\n";
  }
#endif
}

// get the argument count
static inline void countArg(unsigned int& count) {}
template <typename T> 
static inline void countArg(unsigned int& count, const T& t) { ++count; }
template <typename T, typename... Rest> 
static inline void countArg(unsigned int& count, const T& t, const Rest&... rest) {
  ++count;
  countArg(count,rest...);
}

template <typename T>
static inline void set_batch(HSAPrintfPacketQueue* queue, int offset, const T t) {
  queue->queue[offset].set(t);
}
template <typename T, typename... Rest>
static inline void set_batch(HSAPrintfPacketQueue* queue, int offset, const T t, Rest... rest) {
  queue->queue[offset].set(t);
  set_batch(queue, offset + 1, rest...);
}

template <typename... All>
static inline HSAPrintfError hsa_printf(HSAPrintfPacketQueue* queue, All... all) restrict(amp,cpu) {
  unsigned int count = 0;      
  countArg(count, all...);

  HSAPrintfError error = HSA_PRINTF_SUCCESS;

  if (count + 1 + queue->cursor.load() > queue->num) {
    queue->overflow = true;
    error = HSA_PRINTF_BUFFER_OVERFLOW;
  } else {
    unsigned int offset = queue->cursor.fetch_add(count + 1);
    set_batch(queue, offset, count, all...);
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

static inline void hsa_process_printf_queue(HSAPrintfPacketQueue* queue) {
  unsigned int numPackets = 0;
  for (unsigned int i = 0; i < queue->cursor.load(); ) {
    numPackets = queue->queue[i++].data.ui;
    if (numPackets == 0)
      continue;

    // get the format
    unsigned int formatStringIndex = i++;
    assert(queue->queue[formatStringIndex].type == HSA_PRINTF_VOID_PTR
           || queue->queue[formatStringIndex].type == HSA_PRINTF_CONST_VOID_PTR);
    std::string formatString((const char*)queue->queue[formatStringIndex].data.cptr);

    unsigned int formatStringCursor = 0;
    std::smatch specifierMatches;

#if HSA_PRINTF_DEBUG
    printf("%s:%d \t number of matches = %d\n", __FUNCTION__, __LINE__, (int)specifierMatches.size());
#endif
    
    for (unsigned int j = 1; j < numPackets; ++j, ++i) {

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
        printf(specifier.c_str(), queue->queue[i].data.ui);
      } else if (std::regex_search(specifier, specifierTypeMatch, signedIntegerPattern)) {
        printf(specifier.c_str(), queue->queue[i].data.i);
      } else if (std::regex_search(specifier, specifierTypeMatch, floatPattern)) {
        printf(specifier.c_str(), queue->queue[i].data.f);
      } else if (std::regex_search(specifier, specifierTypeMatch, pointerPattern)) {
        printf(specifier.c_str(), queue->queue[i].data.cptr);
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

#if HSA_PRINTF_DEBUG
  if (queue->overflow) {
    printf("Overflow detected!\n");
  }
#endif

  // reset internal data
  for (int i = 0; i < queue->cursor.load(); ++i) {
    queue->queue[i].clear();
  }
  queue->overflow = false;
  queue->cursor.store(0);
}
