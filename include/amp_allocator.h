#pragma once

#include <serialize.h>
namespace Concurrency {

struct rw_info;

struct obj_info
{
    void* device;
    size_t count;
    int ref;
};

class AMPManager {
    virtual void* create(size_t count, void *data, bool hasSrc) = 0;
    virtual void release(void *data) = 0;
    std::map<void *, obj_info> mem_info;
public:
    virtual ~AMPManager() {}
    virtual void* CreateKernel(const char* fun, void* size, void* source) = 0;
    virtual bool check(size_t *local, size_t dim_ext) = 0;

    void regist(size_t count, void* data, bool hasSrc) {
        auto it = mem_info.find(data);
        if (it == std::end(mem_info)) {
            void* device = create(count, data, hasSrc);
            mem_info[data] = {device, count, 1};
        } else
            ++it->second.ref;
    }

    void unregist(void *data) {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info)) {
            obj_info& obj = it->second;
            if (!--obj.ref) {
                release(obj.device);
                mem_info.erase(it);
            }
        }
    }

    obj_info device_data(void* data) {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info))
            return it->second;
        return obj_info();
    }
};

class AMPAllocator {
protected:
    AMPManager *Man;
    AMPAllocator(AMPManager* Man) : Man(Man) {}
public:
  virtual ~AMPAllocator() {}

  void* getQueue() { return _getQueue(); }

  void regist(size_t count, void* data, bool hasSrc) {
      Man->regist(count, data, hasSrc);
  }

  void unregist(void* data) {
      Man->unregist(data);
  }

  void* CreateKernel(const char* fun, void* size, void* source) {
      return Man->CreateKernel(fun, size, source);
  }

  void* device_data(void* data) { return Man->device_data(data).device; }
  virtual void* _getQueue() { return nullptr; }
  // overide function
  virtual void amp_write(void *data) = 0;
  virtual void amp_read(void *data) = 0;
  virtual void amp_copy(void *dst, void *src, size_t n) = 0;
  virtual void PushArg(void* kernel, int idx, rw_info& data) = 0;
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) = 0;
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

inline void *getDevicePointer(void *ptr) { return getAllocator()->device_data(ptr); }
inline void *getOCLQueue(void *ptr) { return getAllocator()->getQueue(); }

} // namespace Concurrency

