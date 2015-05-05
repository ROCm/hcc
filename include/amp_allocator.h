#pragma once

#include <serialize.h>
namespace Concurrency {

struct rw_info;

class AMPAllocator {
public:
  virtual ~AMPAllocator() {}

  void* device_data(void* data) { return _device_data(data); }
  void* getQueue() { return _getQueue(); }
public:

  virtual void* _getQueue() { return nullptr; }
  virtual void* _device_data(void *data) { return nullptr; }
  // overide function
  virtual void regist(int count, void *data, bool hasSrc) = 0;
  virtual void PushArg(void* kernel, int idx, rw_info& data) = 0;
  virtual void amp_write(void *data) = 0;
  virtual void amp_read(void *data) = 0;
  virtual void amp_copy(void *dst, void *src, int n) = 0;
  virtual void unregist(void *data) = 0;
};

AMPAllocator *getAllocator();

struct rw_info
{
    void *data;
    unsigned int discard : 1;
    unsigned int dirty : 1;
    unsigned int hasSrc : 1;

    rw_info(size_t count, void* p = nullptr) : data(p),
    discard(false), dirty(false), hasSrc(data != nullptr) {
        if (!hasSrc)
            data = aligned_alloc(0x1000, count);
#ifdef __AMP_CPU__
        if (!CLAMP::in_cpu_kernel())
#endif
            getAllocator()->regist(count, data, hasSrc);
    }

    void append(void* kernel, int idx, bool isArray) {
        if (!dirty) {
            dirty = true;
            if (!discard || isArray)
                getAllocator()->amp_write(data);
            discard = false;
        }
        getAllocator()->PushArg(kernel, idx, *this);
    }

    void disc() {
        discard = true;
        dirty = false;
    }

    void stash() { dirty = false; }

    void copy(void* dst, size_t count) {
        if (dirty)
            getAllocator()->amp_copy(dst, data, count);
        else
            memmove(dst, data, count);
    }

    void synchronize() {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        if (dirty && !discard) {
            getAllocator()->amp_read(data);
            dirty = false;
        }
    }

    ~rw_info() {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel()) {
            if (!hasSrc)
                ::operator delete(data);
            return;
        }
#endif
        if (hasSrc)
            synchronize();
        getAllocator()->unregist(data);
        if (!hasSrc)
            ::operator delete(data);
    }
};



} // namespace Concurrency

