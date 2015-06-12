//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C++AMP Runtime implementation (HSA version)
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include <amp_allocator.h>
#include <amp_runtime.h>

#include "HSAContext.h"

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v);
extern "C" void PushArgPtrImpl(void *ker, int idx, size_t sz, const void *v);

namespace Concurrency {
namespace CLAMP {

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


void RegisterMemory(void *p, size_t sz)
{
    //std::cerr << "registering: ptr " << p << " of size " << sz << "\n";
    GetOrInitHSAContext()->registerArrayMemory(p, sz);
}

} // namespace CLAMP
} // namespace Concurrency

///
/// memory allocator
///
namespace Concurrency {

class HSADevice;

class HSAView final : public AMPView
{
public:
    HSAView(AMPDevice* pMan) : AMPView(pMan) {}
private:
    void Push(void *kernel, int idx, void *device, bool isConst) override {
        PushArgImpl(kernel, idx, sizeof(void*), &device);
    }
};

class HSADevice final : public AMPDevice
{
    std::map<std::string, HSAContext::Kernel *> __mcw_hsa_kernels;
public:
    HSADevice() : AMPDevice(access_type_read_write) {}

    void* create(size_t count) override {
        void *data = aligned_alloc(0x1000, count);
        CLAMP::RegisterMemory(data, count);
        return data;
    }
    void release(void *data) override { ::operator delete(data); }

    std::wstring get_path() override { return L"hsa"; }
    std::wstring get_description() override { return L"HSA Device"; }
    size_t get_mem() override { return 0; }
    bool is_double() override { return true; }
    bool is_lim_double() override { return true; }
    bool is_unified() override { return true; }
    bool is_emulated() override { return false; }

    void* CreateKernel(const char* fun, void* size, void* source) override {
        std::string str(fun);
        HSAContext::Kernel *kernel = __mcw_hsa_kernels[str];
        if (!kernel) {
            size_t kernel_size = (size_t)((void *)size);
            char *kernel_source = (char*)malloc(kernel_size+1);
            memcpy(kernel_source, source, kernel_size);
            kernel_source[kernel_size] = '\0';
            std::string kname = std::string("&")+s;
            //std::cerr << "CLAMP::HSA::Creating kernel: " << kname << "\n";
            kernel = Concurrency::CLAMP::GetOrInitHSAContext()->
                createKernel(kernel_source, kernel_size, kname.c_str());
            free(kernel_source);
            if (!kernel) {
                std::cerr << "CLAMP::HSA: Unable to create kernel\n";
                abort();
            } else {
                //std::cerr << "CLAMP::HSA: Created kernel\n";
            }
            __mcw_hsa_kernels[str] = kernel;
        }

        HSAContext::Dispatch *dispatch = Concurrency::CLAMP::GetOrInitHSAContext()->createDispatch(kernel);
        dispatch->clearArgs();
#define HSAIL_HLC_DEVELOPMENT_COMPILER 1
#ifndef HSAIL_HLC_DEVELOPMENT_COMPILER
        //dispatch->pushLongArg(0);
        //dispatch->pushLongArg(0);
        //dispatch->pushLongArg(0);
        dispatch->pushLongArg(0);
        dispatch->pushLongArg(0);
        dispatch->pushLongArg(0);
#endif
        return dispatch;
    }

    void LaunchKernel(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        HSAContext::Dispatch *dispatch =
            reinterpret_cast<HSAContext::Dispatch*>(ker);
        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;
        dispatch->setLaunchAttributes(nr_dim, global, local);
        dispatch->dispatchKernelWaitComplete();
        delete(dispatch);
    }
    void LaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        HSAContext::Dispatch *dispatch =
            reinterpret_cast<HSAContext::Dispatch*>(ker);
        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;

        //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";
        //for (size_t i = 0; i < nr_dim; ++i) {
        //  std::cerr << "g: " << global[i] << " l: " << local[i] << "\n";
        //}
        dispatch->setLaunchAttributes(nr_dim, global, local);
        std::shared_future<void>* fut = dispatch->dispatchKernelAndGetFuture();
        return static_cast<void*>(fut);
    }
    std::shared_ptr<AMPView> createAloc() override {
        return std::shared_ptr<AMPView>(new HSAView(this));
    }
};

class HSAContext final : public AMPContext
{
public:
    HSAContext() {
        auto Man = std::shared_ptr<AMPDevice>(new HSADevice);
        default_map[Man] = Man->createAloc();
        Devices.push_back(Man);
        def = Man;
    }
};

static HSAContext ctx;

} // namespace Concurrency


///
/// kernel compilation / kernel launching
///

extern "C" void *GetContextImpl() {
  return &Concurrency::ctx;
}

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v) {
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

extern "C" void PushArgPtrImpl(void *ker, int idx, size_t sz, const void *v) {
  //std::cerr << "pushing:" << ker << " of size " << sz << "\n";
  HSAContext::Dispatch *dispatch =
      reinterpret_cast<HSAContext::Dispatch*>(ker);
  void *val = const_cast<void*>(v);
  dispatch->pushPointerArg(val);
}
