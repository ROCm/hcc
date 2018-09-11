//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <hc_runtime.h>
#include <hc_aligned_alloc.h>

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace detail {

class CPUFallbackQueue final : public HCCQueue
{
public:

  CPUFallbackQueue(HCCDevice* pDev) : HCCQueue(pDev) {}

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

  void wait(hcWaitMode = hcWaitModeBlocked) override {}

    void copy(const void*, void*, size_t) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  void copy_ext(
      const void*,
      void*,
      size_t,
      hcCommandKind,
      const hc::AmPointerInfo&,
      const hc::AmPointerInfo&,
      bool) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  void copy_ext(
      const void*,
      void*,
      size_t,
      hcCommandKind,
      const hc::AmPointerInfo&,
      const hc::AmPointerInfo&,
      const detail::HCCDevice*,
      bool) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  [[noreturn]]
  std::shared_ptr<HCCAsyncOp> detectStreamDeps(hcCommandKind, HCCAsyncOp*) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  void dispatch_hsa_kernel(
    const hsa_kernel_dispatch_packet_t*,
    void*,
    size_t,
    hc::completion_future*,
    const char*) override
  {
    throw std::runtime_error{"Unimplemented."};
  }
  [[noreturn]]
  std::shared_ptr<HCCAsyncOp> EnqueueAsyncCopy(
      const void*, void*, std::size_t) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  [[noreturn]]
  std::shared_ptr<HCCAsyncOp> EnqueueAsyncCopyExt(
      const void*,
      void*,
      size_t,
      hcCommandKind,
      const hc::AmPointerInfo&,
      const hc::AmPointerInfo&,
      const detail::HCCDevice*) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  [[noreturn]]
  std::shared_ptr<HCCAsyncOp> EnqueueMarkerWithDependency(
      int, std::shared_ptr<HCCAsyncOp>*, memory_scope) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  [[noreturn]]
  std::uint32_t GetGroupSegmentSize(void*) override
  {
      throw std::runtime_error{"Unsupported."};
  }
  void LaunchKernel(
      void*,
      std::size_t,
      const std::size_t*,
      const std::size_t*) override
  {
    throw std::runtime_error{"Unsupported."};
  }
  [[noreturn]]
  std::shared_ptr<HCCAsyncOp> LaunchKernelAsync(
      void*,
      std::size_t,
      const std::size_t*,
      const std::size_t*) override
  {
    throw std::runtime_error{"Unsupported."};
  }
  void LaunchKernelWithDynamicGroupMemory(
    void*,
    std::size_t,
    const std::size_t*,
    const std::size_t*,
    std::size_t) override
  {
    throw std::runtime_error{"Unsupported."};
  }
  [[noreturn]]
  std::shared_ptr<HCCAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(
    void*,
    std::size_t,
    const std::size_t*,
    const std::size_t*,
    std::size_t) override
  {
    throw std::runtime_error{"Unimplemented."};
  }
  [[noreturn]]
  bool set_cu_mask(const std::vector<bool>&) override
  {
      throw std::runtime_error{"Unimplemented."};
  }
};

class CPUFallbackDevice final : public HCCDevice
{
public:
    CPUFallbackDevice() : HCCDevice() {}

    std::wstring get_path() const override { return L"fallback"; }
    std::wstring get_description() const override { return L"CPU Fallback"; }
    size_t get_mem() const override { return 0; }
    bool is_double() const override { return true; }
    bool is_lim_double() const override { return true; }
    bool is_unified() const override { return true; }
    bool is_emulated() const override { return true; }
    uint32_t get_version() const override { return 0; }

    void* create(size_t count, struct rw_info* /* not used */) override {
        return hc_aligned_alloc(0x1000, count);
    }
    void release(void *device, struct rw_info* /* not used */ ) override {
        hc_aligned_free(device);
    }
    std::shared_ptr<HCCQueue> createQueue(
        execute_order = execute_in_order) override
    {
        return std::shared_ptr<HCCQueue>(new CPUFallbackQueue(this));
    }

    void BuildProgram(void*, void*) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    [[noreturn]]
    bool check(std::size_t*, std::size_t) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    [[noreturn]]
    void* CreateKernel(
        const char*,
        HCCQueue*,
        std::unique_ptr<void, void (*)(void*)>,
        std::size_t = 0u) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    [[noreturn]]
    void* getSymbolAddress(const char*) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    [[noreturn]]
    bool IsCompatibleKernel(void*, void*) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    [[noreturn]]
    bool is_peer(const HCCDevice*) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    void memcpySymbol(
        const char*,
        void*,
        size_t,
        size_t = 0,
        hcCommandKind = hcMemcpyHostToDevice) override
    {
        throw std::runtime_error{"Unsupported."};
    }
    void memcpySymbol(
        void*,
        void*,
        size_t,
        size_t = 0,
        hcCommandKind = hcMemcpyHostToDevice) override
    {
        throw std::runtime_error{"Unsupported."};
    }
};

template <typename T> inline void deleter(T* ptr) { delete ptr; }

class CPUContext final : public HCCContext
{
public:
    CPUContext() { Devices.push_back(new CPUFallbackDevice); }
    ~CPUContext() { for (auto&& x : Devices) deleter(x); }
};


static CPUContext ctx;

} // namespace detail

extern "C" void *GetContextImpl() {
  return &detail::ctx;
}
