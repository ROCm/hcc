//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include <kalmar_runtime.h>
#include <kalmar_aligned_alloc.h>

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v) {}

namespace Kalmar {

class CPUFallbackQueue final : public KalmarQueue
{
public:

  CPUFallbackQueue(KalmarDevice* pDev) : KalmarQueue(pDev) {}

  void read(void* device, void* dst, size_t count, size_t offset) override {
      if (dst != device)
          memmove(dst, (char*)device + offset, count);
  }

  void write(void* device, const void* src, size_t count, size_t offset, bool blocking) override {
      if (src != device)
          memmove((char*)device + offset, src, count);
  }

  void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) override {
      if (src != dst)
          memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
  }

  void* map(void* device, size_t count, size_t offset, bool modify) override {
      return (char*)device + offset;
  }

  void unmap(void* device, void* addr, size_t count, size_t offset, bool modify) override {}

  void Push(void *kernel, int idx, void* device, bool isConst) override {}
};

class CPUFallbackDevice final : public KalmarDevice
{
public:
    CPUFallbackDevice() : KalmarDevice() {}

    std::wstring get_path() const override { return L"fallback"; }
    std::wstring get_description() const override { return L"CPU Fallback"; }
    size_t get_mem() const override { return 0; }
    bool is_double() const override { return true; }
    bool is_lim_double() const override { return true; }
    bool is_unified() const override { return true; }
    bool is_emulated() const override { return true; }
    uint32_t get_version() const override { return 0; }

    void* create(size_t count, struct rw_info* /* not used */) override {
        return kalmar_aligned_alloc(0x1000, count);
    }
    void release(void *device, struct rw_info* /* not used */ ) override { 
        kalmar_aligned_free(device);
    }
    std::shared_ptr<KalmarQueue> createQueue(execute_order order = execute_in_order) override {
        return std::shared_ptr<KalmarQueue>(new CPUFallbackQueue(this));
    }
};

template <typename T> inline void deleter(T* ptr) { delete ptr; }

class CPUContext final : public KalmarContext
{
public:
    CPUContext() { Devices.push_back(new CPUFallbackDevice); }
    ~CPUContext() { std::for_each(std::begin(Devices), std::end(Devices), deleter<KalmarDevice>); }
};


static CPUContext ctx;

} // namespace Kalmar

extern "C" void *GetContextImpl() {
  return &Kalmar::ctx;
}
