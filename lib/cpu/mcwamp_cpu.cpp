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

struct rw_info
{
  int count;
  bool used;
};
class CPUAMPAllocator : public AMPAllocator
{ 
public:
  void* getQueue() { return nullptr; }
  CPUAMPAllocator() {}
  void init(void *data, int count) {
  }
  void append(void *kernel, int idx, void *data) {
    rwq[data].used = true;
  }
  void write() {
    //std::cerr << "HSAAMPAllocator::write()" << std::endl;
    for (auto& it : rwq) {
      rw_info& rw = it.second;
      if (rw.used) {
      }
    }
  }
  void *device_data(void *) { return nullptr; }
  void discard(void *) {}
  void read() {
    for (auto& it : rwq) {
      rw_info& rw = it.second;
      if (rw.used) {
        if (it.first != mem_info[it.first]) {
        }
        rw.used = false;
      }
    }
  }
  void free(void *data) {
    auto iter = mem_info.find(data);
    if (iter != mem_info.end()) {
      free(iter->second);
      mem_info.erase(iter);
    }
  }
  ~CPUAMPAllocator() {
    mem_info.clear();
    rwq.clear();
  }

  std::map<void *, void*> mem_info;
  std::map<void *, rw_info> rwq;
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
