#pragma once

#include <serialize.h>
namespace Concurrency {

class AMPAllocator {
public:
  AMPAllocator() {}
  virtual ~AMPAllocator() {}
  virtual void compile() = 0;
  virtual void init(void*, int) = 0;
  virtual void append(Serialize&, void*) = 0;
#if defined(CXXAMP_NV)
  virtual void write() = 0;
  virtual void read() = 0;
#endif
  virtual void free(void*) = 0;
};

AMPAllocator& getAllocator();

} // namespace Concurrency

