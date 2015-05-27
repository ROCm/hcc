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

#include <amp_runtime.h>

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v) {}

namespace Concurrency {

class CPUFallbackManager final : public AMPManager
{
    std::shared_ptr<AMPAllocator> newAloc();

public:
    void* create(size_t count) override { return aligned_alloc(0x1000, count); }
    void release(void *data) override { ::operator delete(data); }

    CPUFallbackManager() : AMPManager() { cpu_type = access_type_read_write; }

    std::wstring get_path() override { return L"fallback"; }
    std::wstring get_description() override { return L"CPU Fallback"; }
    size_t get_mem() override { return 0; }
    bool is_double() override { return true; }
    bool is_lim_double() override { return true; }
    bool is_unified() override { return true; }
    bool is_emulated() override { return true; }

    std::shared_ptr<AMPAllocator> createAloc() override { return newAloc(); }
};

class CPUFallbackAllocator final : public AMPAllocator
{
    std::map<void*, void*> addrs;
public:
    CPUFallbackAllocator(std::shared_ptr<AMPManager> pMan) : AMPAllocator(pMan) {}
private:
    void Push(void *kernel, int idx, void*& data, void* device, bool isConst) override {
      auto it = addrs.find(data);
      bool find = it != std::end(addrs);
      if (!kernel && !find) {
          addrs[device] = data;
          data = device;
      } else if (kernel && find) {
          data = it->second;
          addrs.erase(it);
      }
  }
};

std::shared_ptr<AMPAllocator> CPUFallbackManager::newAloc() {
    return std::shared_ptr<AMPAllocator>(new CPUFallbackAllocator(shared_from_this()));
}

class CPUContext final : public AMPContext
{
public:
    CPUContext() {
        auto Man = std::shared_ptr<AMPManager>(new CPUFallbackManager);
        default_map[Man] = Man->createAloc();
        Devices.push_back(Man);
        def = Man;
    }
};


static CPUContext ctx;

} // namespace Concurrency

extern "C" void *GetContextImpl() {
  return &Concurrency::ctx;
}
