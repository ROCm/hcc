#pragma once

#include <serialize.h>
namespace Concurrency {

struct rw_info
{
    unsigned int ref : 29;
    unsigned int discard : 1;
    unsigned int dirty : 1;
    unsigned int hasSrc : 1;
};

class AMPAllocator {
public:
  virtual ~AMPAllocator() {}

  void *init(int count, void* data) {
#ifdef __AMP_CPU__
      if (CLAMP::in_cpu_kernel()) {
          if (data == nullptr) {
              data = aligned_alloc(0x1000, count);
              rwq[data] = {1, false, false, false};
          }
          return data;
      }
#endif
      if (count > 0) {
          auto it = rwq.find(data);
          if (std::end(rwq) != it) {
              ++it->second.ref;
              return data;
          }
          bool hasSrc = data != nullptr;
          if (!hasSrc)
              data = aligned_alloc(0x1000, count);
          regist(count, data, hasSrc);
          rwq[data] = {1, false, false, hasSrc};
      }
      return data;
  }

  void append(void* kernel, int idx, std::shared_ptr<void>& mm, bool isArray) {
      void *data = mm.get();
      auto it = rwq.find(data);
      if (it != std::end(rwq)) {
          rw_info& rw = it->second;
          if (!rw.dirty) {
              rw.dirty = true;
              if (!rw.discard || isArray)
                  amp_write(data);
              rw.discard = false;
          }
      }
      PushArg(kernel, idx, mm);
  }

  void discard(void *data) {
      auto it = rwq.find(data);
      if (it != std::end(rwq)) {
          it->second.discard = true;
          it->second.dirty = false;
      }
  }

  void stash(void *data) {
      auto it = rwq.find(data);
      if (it != std::end(rwq))
          it->second.dirty = false;
  }

  void copy(void* dst, void* src, size_t count) {
      auto it = rwq.find(src);
      if (it != std::end(rwq) && it->second.dirty)
          amp_copy(dst, src, count);
      else
          memmove(dst, src, count);
  }

  void sync(void* data) {
#ifdef __AMP_CPU__
      if (CLAMP::in_cpu_kernel())
          return;
#endif
      auto it = rwq.find(data);
      if (it != std::end(rwq)) {
          rw_info& rw = it->second;
          if (rw.dirty && !rw.discard) {
              amp_read(data);
              rw.dirty = false;
          }
      }
  }

  void free(void* data) {
      auto it = rwq.find(data);
      void* p = data;
      if (it != std::end(rwq)) {
          rw_info& rw = it->second;
#ifdef __AMP_CPU__
          if (CLAMP::in_cpu_kernel()) {
              ::operator delete(data);
              rwq.erase(it);
              return;
          }
#endif
          if (rw.hasSrc) {
              sync(data);
              p = nullptr;
          }
          if (!--rw.ref) {
              unregist(data);
              if (p)
                  ::operator delete(p);
              rwq.erase(it);
          }
      }
  }

  void* device_data(void* data) { return _device_data(data); }
  void* getQueue() { return _getQueue(); }
private:
  std::map<void *, rw_info> rwq;

  virtual void* _getQueue() { return nullptr; }
  virtual void* _device_data(void *data) { return nullptr; }
  // overide function
  virtual void regist(int count, void *data, bool hasSrc) = 0;
  virtual void PushArg(void* kernel, int idx, std::shared_ptr<void>& data) = 0;
  virtual void amp_write(void *data) = 0;
  virtual void amp_read(void *data) = 0;
  virtual void amp_copy(void *dst, void *src, int n) = 0;
  virtual void unregist(void *data) = 0;
};

AMPAllocator *getAllocator();

} // namespace Concurrency

