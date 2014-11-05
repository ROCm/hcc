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

#include <malloc.h> // memalign

#include <amp_allocator.h>

#include "HSAContext.h"

///
/// memory allocator
///
namespace Concurrency {

// forward declaration
namespace CLAMP {
namespace HSA {
extern void RegisterMemory(void*, size_t);
}
}

struct rw_info
{
  int count;
  bool used;
};
class HSAAMPAllocator : public AMPAllocator
{ 
public:
  HSAAMPAllocator() {}
  void compile() {}
  void init(void *data, int count) {
    //std::cerr << "HSAAMPAllocator::init()" << std::endl;
    void *p = memalign(0x1000, count);
    assert(p);
    CLAMP::HSA::RegisterMemory(p, count);
    rwq[data] = {count, false};
    mem_info[data] = p;
    //std::cerr << "add to rwq: " << data << " - " << p << std::endl;
  }
  void append(void *kernel, int idx, void *data) {
    CLAMP::PushArg(kernel, idx, sizeof(void*), &mem_info[data]);
    rwq[data].used = true;
  }
  void write() {
    //std::cerr << "HSAAMPAllocator::write()" << std::endl;
    for (auto& it : rwq) {
      rw_info& rw = it.second;
      if (rw.used) {
        //std::cerr << "copy from: " << mem_info[it.first] << " to: " << it.first << " size: " << rw.count << std::endl;
        memcpy(mem_info[it.first], it.first, rw.count);
      }
    }
  }
  void read() {
    //std::cerr << "HSAAMPAllocator::read()" << std::endl;
    for (auto& it : rwq) {
      rw_info& rw = it.second;
      if (rw.used) {
        //std::cerr << "copy from: " << mem_info[it.first] << " to: " << it.first << " size: " << rw.count << std::endl;
        memcpy(it.first, mem_info[it.first], rw.count);
        rw.used = false;
      }
    }
  }
  void free(void *data) {
    //std::cerr << "HSAAMPAllocator::free()" << std::endl;
    //std::cerr << "data: " << data << std::endl;
    auto iter = mem_info.find(data);
    if (iter != mem_info.end()) {
      free(iter->second);
      mem_info.erase(iter);
    }
  }
  ~HSAAMPAllocator() {
    // FIXME add more proper cleanup
    mem_info.clear();
    rwq.clear();
  }

  std::map<void *, void*> mem_info;
  std::map<void *, rw_info> rwq;
};

static HSAAMPAllocator amp;

HSAAMPAllocator& getHSAAMPAllocator() {
  return amp;
}

AMPAllocator *getAllocator() {
  return &amp;
}

} // namespace Concurrency


///
/// kernel compilation / kernel launching
///

extern "C" char * kernel_source_[] asm ("_binary_kernel_brig_start") __attribute__((weak));
extern "C" char * kernel_size_[] asm ("_binary_kernel_brig_size") __attribute__((weak));

#define AMP_DEVICE_TYPE_CPU (1)
#define AMP_DEVICE_TYPE_GPU (2)

namespace Concurrency {
namespace CLAMP {

std::vector<int> EnumerateDevices() {
    std::vector<int> devices;
    devices.push_back(AMP_DEVICE_TYPE_GPU);
    return devices;
}

void QueryDeviceInfo(const std::wstring& device_path,
    bool& supports_cpu_shared_memory,
    size_t& dedicated_memory,
    bool& supports_limited_double_precision,
    std::wstring& description) {

    description = L"HSA";
    supports_cpu_shared_memory = true;
    supports_limited_double_precision = true;
    dedicated_memory = 0;
}

static HSAContext *context = NULL;

void FinalizeHSAContext() {
  if (context != NULL) {
    context->dispose();
    context = NULL;
  }

  // TBD dispose all Kernel objects

  // TBD dispose all Dispatch objects
}

/* Used only in HSA runtime */
HSAContext *GetOrInitHSAContext(void)
{
  if (!context) {
    //std::cerr << "CLAMP::HSA: create context\n";
    context = HSAContext::Create();
    atexit(FinalizeHSAContext); // register finalizer
  }
  if (!context) {
    std::cerr << "CLAMP::HSA: Unable to create context\n";
    abort();
  }
  return context;
}

static std::map<std::string, HSAContext::Kernel *> __mcw_hsa_kernels;
void *CreateKernel(std::string s)
{
  HSAContext::Kernel *kernel = __mcw_hsa_kernels[s];
  if (!kernel) {
      size_t kernel_size = (size_t)((void *)kernel_size_);
      char *kernel_source = (char*)malloc(kernel_size+1);
      memcpy(kernel_source, kernel_source_, kernel_size);
      kernel_source[kernel_size] = '\0';
      std::string kname = std::string("&")+s;
      //std::cerr << "CLAMP::HSA::Creating kernel: " << kname << "\n";
      kernel = GetOrInitHSAContext()->
          createKernel(kernel_source, kernel_size, kname.c_str());
      if (!kernel) {
          std::cerr << "CLAMP::HSA: Unable to create kernel\n";
          abort();
      } else {
          //std::cerr << "CLAMP::HSA: Created kernel\n";
      }
      __mcw_hsa_kernels[s] = kernel;
  }

  HSAContext::Dispatch *dispatch = GetOrInitHSAContext()->createDispatch(kernel);
  dispatch->clearArgs();
//#define HSAIL_HLC_DEVELOPMENT_COMPILER 1
#ifndef HSAIL_HLC_DEVELOPMENT_COMPILER
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
  dispatch->pushLongArg(0);
#endif
  return dispatch;
}

namespace HSA {
void RegisterMemory(void *p, size_t sz)
{
    //std::cerr << "registering: ptr " << p << " of size " << sz << "\n";
    GetOrInitHSAContext()->registerArrayMemory(p, sz);
}
}

std::future<void> LaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local)
{
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  size_t tmp_local[] = {0, 0, 0};
  if (!local)
      local = tmp_local;

  // FIXME: TBD need to consider allocator with async kernel dispatch

  //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";
  //for (size_t i = 0; i < nr_dim; ++i) {
  //  std::cerr << "g: " << global[i] << " l: " << local[i] << "\n";
  //}
  dispatch->setLaunchAttributes(nr_dim, global, local);
  //std::cerr << "Now real launch\n";
  //kernel->dispatchKernelWaitComplete();

  return dispatch->dispatchKernelAndGetFuture();
}

void LaunchKernel(void *ker, size_t nr_dim, size_t *global, size_t *local)
{
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  size_t tmp_local[] = {0, 0, 0};
  if (!local)
      local = tmp_local;
  //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";
  //for (size_t i = 0; i < nr_dim; ++i) {
  //  std::cerr << "g: " << global[i] << " l: " << local[i] << "\n";
  //}
  HSAAMPAllocator& aloc = getHSAAMPAllocator();
  aloc.write();
  dispatch->setLaunchAttributes(nr_dim, global, local);
  //std::cerr << "Now real launch\n";
  dispatch->dispatchKernelWaitComplete();
  aloc.read();
}

void PushArg(void *ker, int idx, size_t sz, const void *v)
{
  //std::cerr << "pushing:" << ker << " of size " << sz << "\n";
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  void *val = const_cast<void*>(v);
  switch (sz) {
    case sizeof(double):
      dispatch->pushDoubleArg(*reinterpret_cast<double*>(val));
      break;
    case sizeof(int):
      dispatch->pushIntArg(*reinterpret_cast<int*>(val));
      //std::cerr << "(int) value = " << *reinterpret_cast<int*>(val) <<"\n";
      break;
    case sizeof(unsigned char):
      dispatch->pushBooleanArg(*reinterpret_cast<unsigned char*>(val));
      break;
    default:
      assert(0 && "Unsupported kernel argument size");
  }
}
} // namespace CLAMP
} // namespace Concurrency
