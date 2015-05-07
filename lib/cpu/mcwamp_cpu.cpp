//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// FIXME this file will place C++AMP Runtime implementation (HSA version)
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include <amp_allocator.h>
#include <amp_runtime.h>

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v) {}

namespace Concurrency {

struct mm_info
{
    std::shared_ptr<void> data;
    int count;
    int ref;
};

class CPUAMPAllocator : public AMPAllocator
{
    void regist(int count, void *data, bool hasSrc) override {
        if (hasSrc) {
            std::shared_ptr<void> p((void*)aligned_alloc(0x1000, count),
                                    [](void *ptr) { ::operator delete(ptr); });
            mem_info[data] = {p, count, 1};
        }
    }
    void PushArg(void *kernel, int idx, std::shared_ptr<void>& data) override {
        auto it = mem_info.find(data.get());
        if (it != std::end(mem_info)) {
            mm_info &rw = it->second;
            if (rw.count != 0) {
                auto iter = mem_info.find(rw.data.get());
                if (iter == std::end(mem_info))
                    mem_info[rw.data.get()] = {data, 0, 1};
                else
                    ++iter->second.ref;
                data = rw.data;
            } else {
                data = mem_info[rw.data.get()].data;
                if (!--rw.ref)
                    mem_info.erase(it);
            }
        }
    }
    void amp_write(void *data) override {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info)) {
            mm_info &rw = it->second;
            memmove(rw.data.get(), data, rw.count);
        }
    }
    void amp_read(void *data) override {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info)) {
            mm_info &rw = it->second;
            memmove(data, rw.data.get(), rw.count);
        }
    }
    void amp_copy(void *dst, void *src, int n) override {
        auto it = mem_info.find(src);
        if (it != std::end(mem_info)) {
            mm_info &rw = it->second;
            src = rw.data.get();
        }
        memmove(dst, src, n);
    }
    void unregist(void *data) override {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info))
            mem_info.erase(it);
    }

    std::map<void *, mm_info> mem_info;
};

static CPUAMPAllocator amp;

CPUAMPAllocator& getCPUAMPAllocator() {
  return amp;
}

} // namespace Concurrency


///
/// kernel compilation / kernel launching
///

extern "C" void *GetAllocatorImpl() {
  return &Concurrency::amp;
}

extern "C" void EnumerateDevicesImpl(int* devices, int* device_number) {
  if (device_number != nullptr) {
    *device_number = 1;
  }
  if (devices != nullptr) {
    devices[0] = AMP_DEVICE_TYPE_CPU;
  }
}

extern "C" void QueryDeviceInfoImpl(const wchar_t* device_path,
  bool* supports_cpu_shared_memory,
  size_t* dedicated_memory,
  bool* supports_limited_double_precision,
  wchar_t* description) {

  const wchar_t des[] = L"CPU";
  wmemcpy(description, des, sizeof(des));
  *supports_cpu_shared_memory = true;
  *supports_limited_double_precision = true;
  *dedicated_memory = 0;
}
