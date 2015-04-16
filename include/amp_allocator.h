#pragma once

#include <serialize.h>
namespace Concurrency {

class AMPAllocator {
public:
  AMPAllocator() {}
  virtual ~AMPAllocator() {}
  virtual void init(void*, int) = 0;
  virtual void append(void*, int, void*) = 0;
  virtual void write() = 0;
  virtual void* device_data(void*) = 0;
  virtual void discard(void*) = 0;
  virtual void* getQueue() = 0;
  virtual void read() = 0;
  virtual void free(void*) = 0;
};

AMPAllocator *getAllocator();

} // namespace Concurrency

