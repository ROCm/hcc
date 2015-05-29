#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <atomic>
#include <string>
#include <regex>
#include <iostream>

#define HSA_PRINTF_DEBUG  (0)

#define VA_NARGS_IMPL(_1, _2, _3, _4, _5, N, ...) N
#define VA_NARGS(...) VA_NARGS_IMPL(__VA_ARGS__, 5, 4, 3, 2, 1)

#define HSA_PRINTF_IMPL2(count, ...) hsa_printf ## count (__VA_ARGS__)
#define HSA_PRINTF_IMPL(count, ...) HSA_PRINTF_IMPL2(count, __VA_ARGS__) 
#define HSA_PRINTF(...) HSA_PRINTF_IMPL(VA_NARGS(__VA_ARGS__), __VA_ARGS__)

union HSAPrintfPacketData {
  unsigned int ui;
  int i;
  float f;
  void* ptr;
  const void* cptr;
};

enum HSAPrintfPacketDataType {
  HSA_PRINTF_UNSIGNED_INT
  ,HSA_PRINTF_SIGNED_INT
  ,HSA_PRINTF_FLOAT
  ,HSA_PRINTF_VOID_PTR
  ,HSA_PRINTF_CONST_VOID_PTR
};

class HSAPrintfPacket {
public:
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
        :queue(buffer),num(num),cursor(0),overflow(0) {
    lock = 0;
  }
  HSAPrintfPacket* queue;
  unsigned int num;
  unsigned int cursor;
  unsigned int overflow;
  std::atomic_int lock;
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
  std::cout << "num: " << q->num << " "
            << "cursor: " << q->cursor << " "
            << "overflow: " << q->overflow << " "
            << "lock: " << q->lock.load() << "\n";
}

// get the argument count
static inline void countArg(unsigned int& count) {}
template <typename T> 
static inline void countArg(unsigned int& count, const T& t) { count++; }
template <typename T, typename... Rest> 
static inline void countArg(unsigned int& count, const T& t, const Rest&... rest) {
  count++;
  countArg(count,rest...);
}

template <typename T>
static inline void set_batch(HSAPrintfPacketQueue* queue, const T t) {
  queue->queue[queue->cursor++].set(t);
}
template <typename T, typename... Rest>
static inline void set_batch(HSAPrintfPacketQueue* queue, const T t, Rest... rest) {
  queue->queue[queue->cursor++].set(t);
  set_batch(queue, rest...);
}

template <typename... All>
static inline HSAPrintfError hsa_printf(HSAPrintfPacketQueue* queue, const char* format, All... all) restrict(amp,cpu) {
  int unlocked = 0;
  int locked = 1;

  unsigned int count = 0;      
  countArg(count, format, all...);

  HSAPrintfError error = HSA_PRINTF_SUCCESS;

  // disable queue lock for now as HSA cas instruction seems to have trouble
  //while(queue->lock.compare_exchange_strong(unlocked, locked) == false) {
  //  unlocked = 0;
  //}

  if ((count + 1) + queue->cursor >= queue->num) {
    queue->overflow = 1;
    error = HSA_PRINTF_BUFFER_OVERFLOW;
  } else {
    set_batch(queue, count, format, all...);
  }

  // disable queue lock for now as HSA cas instruction seems to have trouble
  //queue->lock.store(unlocked);

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
  int unlocked = 0;
  int locked = 1;

  // disable queue lock for now as HSA cas instruction seems to have trouble
  //while(queue->lock.compare_exchange_strong(unlocked, locked) == false) {
  //  unlocked = 0;
  //}
    
  unsigned int numPackets = 0;
  for (unsigned int i = 0; i < queue->cursor; ) {
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
    
    for (unsigned int j = 1; j < numPackets; j++,i++) {

      if (!std::regex_search(formatString, specifierMatches, specifierPattern)) {
        // More printf argument than format specifier??
        // Just skip to the next printf request
        i = formatStringIndex + numPackets;
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
  queue->overflow = 0;
  queue->cursor = 0;

  // disable queue lock for now as HSA cas instruction seems to have trouble
  //while(queue->lock.compare_exchange_strong(unlocked, locked) == false) {
  //queue->lock.store(unlocked,std::memory_order_release);
}
