//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Kalmar Runtime implementation (HSA version)

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ext_amd.h>

#include <hcc/md5.h>
#include <hcc/kalmar_runtime.h>
#include <hcc/kalmar_aligned_alloc.h>

#include <time.h>
#include <iomanip>

#define KALMAR_DEBUG (0)

/////////////////////////////////////////////////
// kernel dispatch speed optimization flags
/////////////////////////////////////////////////

// size of default kernarg buffer in the kernarg pool in HSAContext
// default set as 128
#define KERNARG_BUFFER_SIZE (128)

// number of pre-allocated kernarg buffers in HSAContext
// default set as 64 (pre-allocating 64 of kernarg buffers in the pool)
#define KERNARG_POOL_SIZE (64)

// number of pre-allocated HSA signals in HSAContext
// default set as 64 (pre-allocating 64 HSA signals)
#define SIGNAL_POOL_SIZE (64)

// whether to use kernarg region found on the HSA agent
// default set as 1 (use karnarg region)
#define USE_KERNARG_REGION (1)

// whether to use hsa_memory_copy to copy prepared kernel arguments
// from host memory to kernarg region
// default set as 0 (NOT use hsa_memory_copy)
#define USE_HSA_MEMORY_COPY_FOR_KERNARG (0)

// whether to print out kernel dispatch time
// default set as 0 (NOT print out kernel dispatch time)
#define KALMAR_DISPATCH_TIME_PRINTOUT (0)

// threshold to clean up finished kernel in HSAQueue.asyncOps
// default set as 1024
#define ASYNCOPS_VECTOR_GC_SIZE (1024)


// whether to use MD5 as kernel indexing hash function
// default set as 0 (use faster FNV-1a hash instead)
#define USE_MD5_HASH (0)

// cutoff size used in FNV-1a hash function
// default set as 104, which is the larger value between HSA BrigModuleHeader
// and AMD GCN ISA header (Elf64_Ehdr) from Jack's research
#define FNV1A_CUTOFF_SIZE (104)

static const char* getHSAErrorString(hsa_status_t s) {

#define CASE_ERROR_STRING(X)  case X: error_string = #X ;break;

    const char* error_string;
    switch(s) {
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_ARGUMENT);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_ALLOCATION);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_AGENT);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_REGION);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_SIGNAL);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_QUEUE);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_OUT_OF_RESOURCES);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_PACKET_FORMAT);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_RESOURCE_FREE);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_NOT_INITIALIZED);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_REFCOUNT_OVERFLOW);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_INDEX);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_ISA);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_ISA_NAME);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_EXECUTABLE);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_FROZEN_EXECUTABLE);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_INVALID_SYMBOL_NAME);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_VARIABLE_UNDEFINED);
        CASE_ERROR_STRING(HSA_STATUS_ERROR_EXCEPTION);
        default: error_string = "Unknown Error Code";
    };
    return error_string;
}

#define STATUS_CHECK(s,line) if (s != HSA_STATUS_SUCCESS && s != HSA_STATUS_INFO_BREAK) {\
    const char* error_string = getHSAErrorString(s);\
		printf("### Error: %s (%d) at line:%d\n", error_string, s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

#define STATUS_CHECK_Q(s,q,line) if (s != HSA_STATUS_SUCCESS) {\
    const char* error_string = getHSAErrorString(s);\
		printf("### Error: %s (%d) at line:%d\n", error_string, s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(q));\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v);
extern "C" void PushArgPtrImpl(void *ker, int idx, size_t sz, const void *v);

// forward declaration
namespace Kalmar {
class HSAQueue;
class HSADevice;
} // namespace Kalmar

///
/// kernel compilation / kernel launching
///

/// modeling of HSA executable
class HSAExecutable {
private:
    hsa_code_object_t hsaCodeObject;
    hsa_executable_t hsaExecutable;
    friend class HSAKernel;
    friend class Kalmar::HSADevice;

public:
    HSAExecutable(hsa_executable_t _hsaExecutable,
                  hsa_code_object_t _hsaCodeObject) :
        hsaExecutable(_hsaExecutable),
        hsaCodeObject(_hsaCodeObject) {}

    ~HSAExecutable() {
      hsa_status_t status;

#if KALMAR_DEBUG
      std::cerr << "HSAExecutable::~HSAExecutable\n";
#endif

      status = hsa_executable_destroy(hsaExecutable);
      STATUS_CHECK(status, __LINE__);

      status = hsa_code_object_destroy(hsaCodeObject);
      STATUS_CHECK(status, __LINE__);
    }

    template<typename T>
    void setSymbolToValue(const char* symbolName, T value) {
        hsa_status_t status;

        // get symbol
        hsa_executable_symbol_t symbol;
        hsa_agent_t agent;
        status = hsa_executable_get_symbol(hsaExecutable, NULL, symbolName, agent, 0, &symbol);
        STATUS_CHECK(status, __LINE__);

        // get address of symbol
        uint64_t symbol_address;
        status = hsa_executable_symbol_get_info(symbol,
                                                HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                                                &symbol_address);
        STATUS_CHECK(status, __LINE__);

        // set the value of symbol
        T* symbol_ptr = (T*)symbol_address;
        *symbol_ptr = value;
    }
};

class HSAKernel {
private:
    HSAExecutable* executable;
    uint64_t kernelCodeHandle;
    hsa_executable_symbol_t hsaExecutableSymbol;
    friend class HSADispatch;

public:
    HSAKernel(HSAExecutable* _executable,
              hsa_executable_symbol_t _hsaExecutableSymbol,
              uint64_t _kernelCodeHandle) :
      executable(_executable),
      hsaExecutableSymbol(_hsaExecutableSymbol),
      kernelCodeHandle(_kernelCodeHandle) {}

    ~HSAKernel() {
#if KALMAR_DEBUG
      std::cerr << "HSAKernel::~HSAKernel\n";
#endif
    }
}; // end of HSAKernel

class HSABarrier : public Kalmar::KalmarAsyncOp {
private:
    hsa_signal_t signal;
    int signalIndex;
    bool isDispatched;
    hsa_wait_state_t waitMode;

    std::shared_future<void>* future;

    Kalmar::HSAQueue* hsaQueue;

public:
    std::shared_future<void>* getFuture() override { return future; }

    void* getNativeHandle() override { return &signal; }

    void setWaitMode(Kalmar::hcWaitMode mode) override {
        switch (mode) {
            case Kalmar::hcWaitModeBlocked:
                waitMode = HSA_WAIT_STATE_BLOCKED;
            break;
            case Kalmar::hcWaitModeActive:
                waitMode = HSA_WAIT_STATE_ACTIVE;
            break;
        }
    }

    bool isReady() override {
        return (hsa_signal_load_acquire(signal) == 0);
    }

    HSABarrier() : isDispatched(false), future(nullptr), hsaQueue(nullptr), waitMode(HSA_WAIT_STATE_BLOCKED) {}

    ~HSABarrier() {
#if KALMAR_DEBUG
        std::cerr << "HSABarrier::~HSABarrier()\n";
#endif
        if (isDispatched) {
            hsa_status_t status = HSA_STATUS_SUCCESS;
            status = waitComplete();
            STATUS_CHECK(status, __LINE__);
        }
        dispose();
    }

    hsa_status_t enqueueBarrier(hsa_queue_t* queue);

    hsa_status_t enqueueAsync(Kalmar::HSAQueue*);

    // wait for the barrier to complete
    hsa_status_t waitComplete();

    void dispose();

    uint64_t getTimestampFrequency() override {
        // get system tick frequency
        uint64_t timestamp_frequency_hz = 0L;
        hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_hz);
        return timestamp_frequency_hz;
    }

    uint64_t getBeginTimestamp() override;

    uint64_t getEndTimestamp() override;

}; // end of HSABarrier

class HSADispatch : public Kalmar::KalmarAsyncOp {
private:
    Kalmar::HSADevice* device;
    hsa_agent_t agent;
    HSAKernel* kernel;

    std::vector<uint8_t> arg_vec;
    uint32_t arg_count;
    size_t prevArgVecCapacity;
    void* kernargMemory;
    int kernargMemoryIndex;

    int launchDimensions;
    uint32_t workgroup_size[3];
    uint32_t global_size[3];

    hsa_signal_t signal;
    int signalIndex;
    hsa_kernel_dispatch_packet_t aql;
    bool isDispatched;
    hsa_wait_state_t waitMode;

    size_t dynamicGroupSize;

    std::shared_future<void>* future;

    Kalmar::HSAQueue* hsaQueue;

public:
    std::shared_future<void>* getFuture() override { return future; }

    void* getNativeHandle() override { return &signal; }

    void setWaitMode(Kalmar::hcWaitMode mode) override {
        switch (mode) {
            case Kalmar::hcWaitModeBlocked:
                waitMode = HSA_WAIT_STATE_BLOCKED;
            break;
            case Kalmar::hcWaitModeActive:
                waitMode = HSA_WAIT_STATE_ACTIVE;
            break;
        }
    }

    bool isReady() override {
        return (hsa_signal_load_acquire(signal) == 0);
    }

    ~HSADispatch() {
#if KALMAR_DEBUG
        std::cerr << "HSADispatch::~HSADispatch()\n";
#endif

        if (isDispatched) {
            hsa_status_t status = HSA_STATUS_SUCCESS;
            status = waitComplete();
            STATUS_CHECK(status, __LINE__);
        }
        dispose();
    }

    hsa_status_t setDynamicGroupSegment(size_t dynamicGroupSize) {
        this->dynamicGroupSize = dynamicGroupSize;
        return HSA_STATUS_SUCCESS;
    }

    HSADispatch(Kalmar::HSADevice* _device, HSAKernel* _kernel);

    hsa_status_t pushFloatArg(float f) { return pushArgPrivate(f); }
    hsa_status_t pushIntArg(int i) { return pushArgPrivate(i); }
    hsa_status_t pushBooleanArg(unsigned char z) { return pushArgPrivate(z); }
    hsa_status_t pushByteArg(char b) { return pushArgPrivate(b); }
    hsa_status_t pushLongArg(long j) { return pushArgPrivate(j); }
    hsa_status_t pushDoubleArg(double d) { return pushArgPrivate(d); }
    hsa_status_t pushPointerArg(void *addr) { return pushArgPrivate(addr); }

    hsa_status_t clearArgs() {
        arg_count = 0;
        arg_vec.clear();
        return HSA_STATUS_SUCCESS;
    }

    hsa_status_t setLaunchAttributes(int dims, size_t *globalDims, size_t *localDims);

    hsa_status_t dispatchKernelWaitComplete(Kalmar::HSAQueue*);

    hsa_status_t dispatchKernelAsync(Kalmar::HSAQueue*);

    uint32_t getGroupSegmentSize() {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        uint32_t group_segment_size = 0;
        status = hsa_executable_symbol_get_info(kernel->hsaExecutableSymbol,
                                                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                                &group_segment_size);
        STATUS_CHECK(status, __LINE__);
        return group_segment_size;
    }

    // dispatch a kernel asynchronously
    hsa_status_t dispatchKernel(hsa_queue_t* commandQueue);

    // wait for the kernel to finish execution
    hsa_status_t waitComplete();

    void dispose();

    uint64_t getTimestampFrequency() override {
        // get system tick frequency
        uint64_t timestamp_frequency_hz = 0L;
        hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_hz);
        return timestamp_frequency_hz;
    }

    uint64_t getBeginTimestamp() override;

    uint64_t getEndTimestamp() override;

private:
    template <typename T>
    hsa_status_t pushArgPrivate(T val) {
        /* add padding if necessary */
        int padding_size = (arg_vec.size() % sizeof(T)) ? (sizeof(T) - (arg_vec.size() % sizeof(T))) : 0;
#if KALMAR_DEBUG
        printf("push %lu bytes into kernarg: ", sizeof(T) + padding_size);
#endif
        for (size_t i = 0; i < padding_size; ++i) {
            arg_vec.push_back((uint8_t)0x00);
#if KALMAR_DEBUG
            printf("%02X ", (uint8_t)0x00);
#endif
        }
        uint8_t* ptr = static_cast<uint8_t*>(static_cast<void*>(&val));
        for (size_t i = 0; i < sizeof(T); ++i) {
            arg_vec.push_back(ptr[i]);
#if KALMAR_DEBUG
            printf("%02X ", ptr[i]);
#endif
        }
#if KALMAR_DEBUG
        printf("\n");
#endif
        arg_count++;
        return HSA_STATUS_SUCCESS;
    }

    void computeLaunchAttr(int level, int globalSize, int localSize, int recommendedSize) {
        // localSize of 0 means pick best
        if (localSize == 0) localSize = recommendedSize;
        localSize = std::min(localSize, recommendedSize);
        localSize = std::min(localSize, globalSize); // workgroup size shall not exceed grid size
  
        global_size[level] = globalSize;
        workgroup_size[level] = localSize;
        //std::cout << "level " << level << ", grid=" << global_size[level] 
        //          << ", group=" << workgroup_size[level] << std::endl;
    }

}; // end of HSADispatch

//-----
//Structure used to extract information from memory pool
struct pool_iterator
{
    hsa_amd_memory_pool_t _am_memory_pool;
    hsa_amd_memory_pool_t _am_host_memory_pool;

    hsa_amd_memory_pool_t _kernarg_memory_pool;
    hsa_amd_memory_pool_t _finegrained_system_memory_pool;
    hsa_amd_memory_pool_t _coarsegrained_system_memory_pool;
    hsa_amd_memory_pool_t _local_memory_pool;

    bool        _found_kernarg_memory_pool;
    bool        _found_finegrained_system_memory_pool;
    bool        _found_local_memory_pool;
    bool        _found_coarsegrained_system_memory_pool;

    pool_iterator() ;
};


pool_iterator::pool_iterator()
{
    _kernarg_memory_pool.handle=(uint64_t)-1;
    _finegrained_system_memory_pool.handle=(uint64_t)-1;
    _local_memory_pool.handle=(uint64_t)-1;
    _coarsegrained_system_memory_pool.handle=(uint64_t)-1;

    _found_kernarg_memory_pool = false;
    _found_finegrained_system_memory_pool = false;
    _found_local_memory_pool = false;
    _found_coarsegrained_system_memory_pool = false;
}
//-----


///
/// memory allocator
///
namespace Kalmar {


class HSAQueue final : public KalmarQueue
{
private:
    // HSA commmand queue associated with this HSAQueue instance
    hsa_queue_t* commandQueue;

    //
    // kernel dispatches and barriers associated with this HSAQueue instance
    //
    // When a kernel k is dispatched, we'll get a KalmarAsyncOp f.
    // This vector would hold f.  acccelerator_view::wait() would trigger
    // HSAQueue::wait(), and all future objects in the KalmarAsyncOp objects
    // will be waited on.
    //
    std::vector< std::shared_ptr<KalmarAsyncOp> > asyncOps;

    //
    // kernelBufferMap and bufferKernelMap forms the dependency graph of
    // kernel / kernel dispatches / buffers
    //
    // For a particular kernel k, kernelBufferMap[k] holds a vector of 
    // host buffers used by k. The vector is filled at HSAQueue::Push(),
    // when kernel arguments are prepared.
    //
    // When a kenrel k is to be dispatched, kernelBufferMap[k] will be traversed
    // to figure out if there is any previous kernel dispatch associated for
    // each buffer b used by k.  This is done by checking bufferKernelMap[b].
    // If there are previous kernel dispatches which use b, then we wait on
    // them before dispatch kernel k. bufferKernelMap[b] will be cleared then.
    //
    // After kernel k is dispatched, we'll get a KalmarAsync object f, we then
    // walk through each buffer b used by k and mark the association as:
    // bufferKernelMap[b] = f
    //
    // Finally kernelBufferMap[k] will be cleared.
    //

    // association between buffers and kernel dispatches
    // key: buffer address
    // value: a vector of kernel dispatches
    std::map<void*, std::vector< std::weak_ptr<KalmarAsyncOp> > > bufferKernelMap;

    // association between a kernel and buffers used by it
    // key: kernel
    // value: a vector of buffers used by the kernel
    std::map<void*, std::vector<void*> > kernelBufferMap;

public:
    HSAQueue(KalmarDevice* pDev, hsa_agent_t agent, execute_order order) : KalmarQueue(pDev, queuing_mode_automatic, order), commandQueue(nullptr), asyncOps(), bufferKernelMap(), kernelBufferMap() {
        hsa_status_t status;

        /// Query the maximum size of the queue.
        uint32_t queue_size = 0;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
        STATUS_CHECK(status, __LINE__);

        /// Create a queue using the maximum size.
        status = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, 
                                  UINT32_MAX, UINT32_MAX, &commandQueue);
#if KALMAR_DEBUG
        std::cerr << "HSAQueue::HSAQueue(): created an HSA command queue: " << commandQueue << "\n";
#endif
        STATUS_CHECK_Q(status, commandQueue, __LINE__);

        /// Enable profiling support for the queue.
        status = hsa_amd_profiling_set_profiler_enabled(commandQueue, 1);
    }

    void dispose() override {
        hsa_status_t status;

#if KALMAR_DEBUG
        std::cerr << "HSAQueue::dispose() in\n";
#endif

        // wait on all existing kernel dispatches and barriers to complete
        wait();

        // clear bufferKernelMap
        for (auto iter = bufferKernelMap.begin(); iter != bufferKernelMap.end(); ++iter) {
           iter->second.clear();
        }
        bufferKernelMap.clear();

        // clear kernelBufferMap
        for (auto iter = kernelBufferMap.begin(); iter != kernelBufferMap.end(); ++iter) {
           iter->second.clear();
        }
        kernelBufferMap.clear();

#if KALMAR_DEBUG
        std::cerr << "HSAQueue::dispose(): destroy an HSA command queue: " << commandQueue << "\n";
#endif
        status = hsa_queue_destroy(commandQueue);
        STATUS_CHECK(status, __LINE__);
        commandQueue = nullptr;

#if KALMAR_DEBUG
        std::cerr << "HSAQueue::dispose() out\n";
#endif
    }

    ~HSAQueue() {
#if KALMAR_DEBUG
        std::cerr << "HSAQueue::~HSAQueue() in\n";
#endif

        if (commandQueue != nullptr) {
            dispose();
        }

#if KALMAR_DEBUG
        std::cerr << "HSAQueue::~HSAQueue() out\n";
#endif
    }

    // FIXME: implement flush

    int getPendingAsyncOps() override {
        int count = 0;
        for (int i = 0; i < asyncOps.size(); ++i) {
            if (asyncOps[i] != nullptr) {
                ++count;
            }
        }
        return count;
    }

    void wait(hcWaitMode mode = hcWaitModeBlocked) override {
      // wait on all previous async operations to complete
      for (int i = 0; i < asyncOps.size(); ++i) {
        if (asyncOps[i] != nullptr) {
            auto asyncOp = asyncOps[i];
            // wait on valid futures only
            std::shared_future<void>* future = asyncOp->getFuture();
            if (future->valid()) {
                future->wait();
            }
        }
      }
      // clear async operations table
      asyncOps.clear();
    }

    void LaunchKernel(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        LaunchKernelWithDynamicGroupMemory(ker, nr_dim, global, local, 0);
    }

    void LaunchKernelWithDynamicGroupMemory(void *ker, size_t nr_dim, size_t *global, size_t *local, size_t dynamic_group_size) override {
        HSADispatch *dispatch =
            reinterpret_cast<HSADispatch*>(ker);
        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;
        dispatch->setLaunchAttributes(nr_dim, global, local);
        dispatch->setDynamicGroupSegment(dynamic_group_size);

        // wait for previous kernel dispatches be completed
        std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                      [&] (void* buffer) {
                        waitForDependentAsyncOps(buffer);
                      });

        // dispatch the kernel
        // and wait for its completion
        dispatch->dispatchKernelWaitComplete(this);

        // clear data in kernelBufferMap
        kernelBufferMap[ker].clear();

        delete(dispatch);
    }

    std::shared_ptr<KalmarAsyncOp> LaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        return LaunchKernelWithDynamicGroupMemoryAsync(ker, nr_dim, global, local, 0);
    }

    std::shared_ptr<KalmarAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(void *ker, size_t nr_dim, size_t *global, size_t *local, size_t dynamic_group_size) override {
        hsa_status_t status = HSA_STATUS_SUCCESS;      

        HSADispatch *dispatch =
            reinterpret_cast<HSADispatch*>(ker);

        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;
        dispatch->setLaunchAttributes(nr_dim, global, local);
        dispatch->setDynamicGroupSegment(dynamic_group_size);

        // wait for previous kernel dispatches be completed
        std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                      [&] (void* buffer) {
                        waitForDependentAsyncOps(buffer);
                      });

        // dispatch the kernel
        status = dispatch->dispatchKernelAsync(this);
        STATUS_CHECK(status, __LINE__);

        // create a shared_ptr instance
        std::shared_ptr<KalmarAsyncOp> sp_dispatch(dispatch);

        // associate the kernel dispatch with this queue
        asyncOps.push_back(sp_dispatch);

        // associate all buffers used by the kernel with the kernel dispatch instance
        std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                      [&] (void* buffer) {
                        bufferKernelMap[buffer].push_back(sp_dispatch);
                      });

        // clear data in kernelBufferMap
        kernelBufferMap[ker].clear();

        return sp_dispatch;
    }

    uint32_t GetGroupSegmentSize(void *ker) override {
        HSADispatch *dispatch = reinterpret_cast<HSADispatch*>(ker);
        return dispatch->getGroupSegmentSize();
    }

    // wait for dependent async operations to complete
    void waitForDependentAsyncOps(void* buffer) {
        auto dependentAsyncOpVector = bufferKernelMap[buffer];
        for (int i = 0; i < dependentAsyncOpVector.size(); ++i) {
          auto dependentAsyncOp = dependentAsyncOpVector[i];
          if (!dependentAsyncOp.expired()) {
            auto dependentAsyncOpPointer = dependentAsyncOp.lock();
            // wait on valid futures only
            std::shared_future<void>* future = dependentAsyncOpPointer->getFuture();
            if (future->valid()) {
              future->wait();
            }
          }
        }
        dependentAsyncOpVector.clear();
    }

    void read(void* device, void* dst, size_t count, size_t offset) override {
        waitForDependentAsyncOps(device);

        // do read
        if (dst != device) {
            if (!getDev()->is_unified()) {
#if KALMAR_DEBUG
                std::cerr << "read(" << device << "," << dst << "," << count << "," << offset << "): use HSA memory copy\n";
#endif
                hsa_status_t status = HSA_STATUS_SUCCESS;
                // Make sure host memory is accessible to gpu
                // FIXME: host memory is allocated through OS allocator, if not, correct it.
                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent()); 
                void* va = nullptr;
                status = hsa_amd_memory_lock(dst, count, agent, 1, &va);
                STATUS_CHECK(status, __LINE__);
                status = hsa_memory_copy(va, (char*)device + offset, count);
                STATUS_CHECK(status, __LINE__);
                // Unlock the host memory
                status = hsa_amd_memory_unlock(dst);
                STATUS_CHECK(status, __LINE__);
            } else {
#if KALMAR_DEBUG
                std::cerr << "read(" << device << "," << dst << "," << count << "," << offset << "): use host memory copy\n";
#endif
                memmove(dst, (char*)device + offset, count);
            }
        }
    }

    void write(void* device, const void* src, size_t count, size_t offset, bool blocking) override {
        waitForDependentAsyncOps(device);

        // do write
        if (src != device) {
            if (!getDev()->is_unified()) {
#if KALMAR_DEBUG
                std::cerr << "write(" << device << "," << src << "," << count << "," << offset << "," << blocking << "): use HSA memory copy\n";
#endif
                hsa_status_t status = HSA_STATUS_SUCCESS;
                // Make sure host memory is accessible to gpu
                // FIXME: host memory is allocated through OS allocator, if not, correct it.
                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent()); 
                void* va = nullptr;
                status = hsa_amd_memory_lock(const_cast<void*>(src), count, agent, 1, &va);
                STATUS_CHECK(status, __LINE__);
                status = hsa_memory_copy((char*)device + offset, va, count);
                STATUS_CHECK(status, __LINE__);
                // Unlock the host memory
                status = hsa_amd_memory_unlock(const_cast<void*>(src));
                STATUS_CHECK(status, __LINE__);
            } else {
#if KALMAR_DEBUG
                std::cerr << "write(" << device << "," << src << "," << count << "," << offset << "," << blocking << "): use host memory copy\n";
#endif
                memmove((char*)device + offset, src, count);
            }
        }
    }

    void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) override {
        waitForDependentAsyncOps(dst);
        waitForDependentAsyncOps(src);

        // do copy
        if (src != dst) {
            if (!getDev()->is_unified()) {
#if KALMAR_DEBUG
                std::cerr << "copy(" << src << "," << dst << "," << count << "," << src_offset << "," << dst_offset << "," << blocking << "): use HSA memory copy\n";
#endif
                hsa_status_t status = HSA_STATUS_SUCCESS;
                // FIXME: aftre p2p enabled, if this function is not expected to copy between two buffers from different device, then, delete allow_access API call.
                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
                status = hsa_amd_agents_allow_access(1, agent, NULL, src);
                STATUS_CHECK(status, __LINE__); 
                status = hsa_memory_copy((char*)dst + dst_offset, (char*)src + src_offset, count);
                STATUS_CHECK(status, __LINE__);
            } else {
#if KALMAR_DEBUG
                std::cerr << "copy(" << src << "," << dst << "," << count << "," << src_offset << "," << dst_offset << "," << blocking << "): use host memory copy\n";
#endif
                memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
            }
        }
    }

    void* map(void* device, size_t count, size_t offset, bool modify) override {
        waitForDependentAsyncOps(device);

        // do map

        // as HSA runtime doesn't have map/unmap facility at this moment,
        // we explicitly allocate a host memory buffer in this case 
        if (!getDev()->is_unified()) {
#if KALMAR_DEBUG
            std::cerr << "map(" << device << "," << count << "," << offset << "," << modify << "): use HSA memory map\n";
#endif
            hsa_status_t status = HSA_STATUS_SUCCESS;
            // allocate a host buffer
            // TODO: for safety, we copy to host, but we can map device memory to host through hsa_amd_agents_allow_access
            // withouth copying data.
            void* data = nullptr;
            hsa_amd_memory_pool_t* am_host_region = static_cast<hsa_amd_memory_pool_t*>(getHSAAMHostRegion());
            status = hsa_amd_memory_pool_allocate(*am_host_region, count, 0, &data);
            STATUS_CHECK(status, __LINE__);
            if (data != nullptr) {
              // copy data from device buffer to host buffer
              hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent()); 
              status = hsa_amd_agents_allow_access(1, agent, NULL, data);
              STATUS_CHECK(status, __LINE__);
              status = hsa_memory_copy(data, (char*)device + offset, count);
              STATUS_CHECK(status, __LINE__);
            } else {
#if KALMAR_DEBUG
              std::cerr << "host buffer allocation failed!\n";
#endif
              abort();
            }
#if KALMAR_DEBUG
            std::cerr << "map(): " << data << "\n";
#endif

            return data;
            
        } else {
#if KALMAR_DEBUG
            std::cerr << "map(" << device << "," << count << "," << offset << "," << modify << "): use host memory map\n";
#endif
            // for host memory we simply return the pointer plus offset
#if KALMAR_DEBUG
            std::cerr << "map(): " << ((char*)device+offset) << "\n";
#endif
            return (char*)device + offset;
        }
    }

    void unmap(void* device, void* addr, size_t count, size_t offset, bool modify) override {
        // do unmap

        // as HSA runtime doesn't have map/unmap facility at this moment,
        // we free the host memory buffer allocated in map()
        if (!getDev()->is_unified()) {
#if KALMAR_DEBUG
            std::cerr << "unmap(" << device << "," << addr << "," << count << "," << offset << "," << modify << "): use HSA memory unmap\n";
#endif
            if (modify) {
#if KALMAR_DEBUG
                std::cerr << "copy host buffer to device buffer\n";
#endif
                // copy data from host buffer to device buffer
                hsa_status_t status = HSA_STATUS_SUCCESS;
                status = hsa_memory_copy((char*)device + offset, addr, count);
                STATUS_CHECK(status, __LINE__);
            }

            // deallocate the host buffer
            hsa_amd_memory_pool_free(addr);
        } else {
#if KALMAR_DEBUG
            std::cerr << "unmap(" << device << "," << addr << "," << count << "," << offset << "," << modify << "): use host memory unmap\n";
#endif
            // for host memory there's nothing to be done
        }
    }

    void Push(void *kernel, int idx, void *device, bool modify) override {
        PushArgImpl(kernel, idx, sizeof(void*), &device);

        // register the buffer with the kernel
        // when the buffer may be read/written by the kernel
        // the buffer is not registered if it's only read by the kernel
        if (modify) {
          kernelBufferMap[kernel].push_back(device);
        }
    }

    void* getHSAQueue() override {
        return static_cast<void*>(commandQueue);
    }

    void* getHSAAgent() override;

    void* getHSAAMRegion() override;

    void* getHSAAMHostRegion() override;

    void* getHSAKernargRegion() override;

    bool hasHSAInterOp() override {
        return true;
    }

    // enqueue a barrier packet
    std::shared_ptr<KalmarAsyncOp> EnqueueMarker() {
        hsa_status_t status = HSA_STATUS_SUCCESS;

        // create shared_ptr instance
        std::shared_ptr<HSABarrier> barrier = std::make_shared<HSABarrier>();

        // enqueue the barrier
        status = barrier.get()->enqueueAsync(this);
        STATUS_CHECK(status, __LINE__);

        // associate the barrier with this queue
        asyncOps.push_back(barrier);

        return barrier;
    }

    // remove finished async operation from waiting list
    void removeAsyncOp(KalmarAsyncOp* asyncOp) {
        for (int i = 0; i < asyncOps.size(); ++i) {
            if (asyncOps[i].get() == asyncOp) {
                asyncOps[i] = nullptr;
            }
        }

        // GC for finished kernels
        if (asyncOps.size() > ASYNCOPS_VECTOR_GC_SIZE) {
          asyncOps.erase(std::remove(asyncOps.begin(), asyncOps.end(), nullptr),
                         asyncOps.end());
        }
    }
};

class HSADevice final : public KalmarDevice
{
private:
    /// memory pool for kernargs
    std::vector<void*> kernargPool;
    std::vector<bool> kernargPoolFlag;
    int kernargCursor;
    std::mutex kernargPoolMutex;


    std::map<std::string, HSAKernel *> programs;
    hsa_agent_t agent;
    size_t max_tile_static_size;

    std::mutex queues_mutex;
    std::vector< std::weak_ptr<KalmarQueue> > queues;

    pool_iterator ri;

    bool useCoarseGrainedRegion;

    uint32_t workgroup_max_size;
    uint16_t workgroup_max_dim[3];

    std::map<std::string, HSAExecutable*> executables;

    hsa_isa_t agentISA;

    hcAgentProfile profile;

    /*TODO: This is the first CPU which will provide system memory pool
    We might need to modify again in multiple CPU socket scenario. Because
    we must make sure there is pyshycial link between device and host. Currently,
    agent iterate function will push back all of the dGPU on the system, which might
    not be linked directly to the first cpu node, host */
    hsa_agent_t host_;

public:
 
    uint32_t getWorkgroupMaxSize() {
        return workgroup_max_size;
    }

    const uint16_t* getWorkgroupMaxDim() {
        return &workgroup_max_dim[0];
    }

    // Callback for hsa_amd_agent_iterate_memory_pools.
    // data is of type pool_iterator,
    // we save the pools we care about into this structure.
    static hsa_status_t get_memory_pools(hsa_amd_memory_pool_t region, void* data)
    {
    
        hsa_amd_segment_t segment;
        hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    
        if (segment == HSA_AMD_SEGMENT_GLOBAL) {
#if KALMAR_DEBUG
          size_t size = 0;
          hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
          size = size/(1024*1024);
          std::cerr << "found memory pool of GPU local memory, size(MB) = " << size << std::endl;
#endif 
          pool_iterator *ri = (pool_iterator*) (data);
          ri->_local_memory_pool = region;
          ri->_found_local_memory_pool = true;

          return HSA_STATUS_INFO_BREAK;
        }
    
        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t get_host_pools(hsa_amd_memory_pool_t region, void* data) {
        hsa_status_t status;
        hsa_amd_segment_t segment;
        status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        STATUS_CHECK(status, __LINE__);

        pool_iterator *ri = (pool_iterator*) (data);

        hsa_amd_memory_pool_global_flag_t flags;
        status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        STATUS_CHECK(status, __LINE__);

#if KALMAR_DEBUG
        size_t size = 0;
        status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
        STATUS_CHECK(status, __LINE__);
        size = size/(1024*1024);

#endif
        if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) && (!ri->_found_kernarg_memory_pool)) {
#if KALMAR_DEBUG
            std::cerr << "found kernarg memory pool on host memory, size(MB) = " << size << std::endl;
#endif 
            ri->_kernarg_memory_pool = region;
            ri->_found_kernarg_memory_pool = true;
        }
        
        if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) && (!ri->_found_finegrained_system_memory_pool)) {
#if KALMAR_DEBUG
            std::cerr << "found fine grained memory pool on host memory, size(MB) = " << size << std::endl;
#endif 
            ri->_finegrained_system_memory_pool = region;
            ri->_found_finegrained_system_memory_pool = true;
        }

        if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && (!ri->_found_coarsegrained_system_memory_pool)) {
#if KALMAR_DEBUG
            std::cerr << "found coarse-grain system memory pool, size(MB) = " << size << std::endl;
#endif 
            ri->_coarsegrained_system_memory_pool = region;
            ri->_found_coarsegrained_system_memory_pool = true;
        }
        return HSA_STATUS_SUCCESS;
    } 

    static hsa_status_t find_group_memory(hsa_amd_memory_pool_t region, void* data) {
      hsa_amd_segment_t segment;
      size_t size = 0;
      bool flag = false;

      hsa_status_t status = HSA_STATUS_SUCCESS;

      // get segment information
      status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
      STATUS_CHECK(status, __LINE__);

      if (segment == HSA_AMD_SEGMENT_GROUP) {
        // found group segment, get its size
        status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
        STATUS_CHECK(status, __LINE__);

        // save the result to data
        size_t* result = (size_t*)data;
        *result = size;

        return HSA_STATUS_INFO_BREAK;
      }

      // continue iteration
      return HSA_STATUS_SUCCESS;
    }

    hsa_agent_t& getAgent() {
        return agent;
    }

    HSADevice(hsa_agent_t a, hsa_agent_t host) : KalmarDevice(access_type_read_write),
                               agent(a), programs(), max_tile_static_size(0),
                               queues(), queues_mutex(),
                               ri(),
                               useCoarseGrainedRegion(false),
                               kernargPool(), kernargPoolFlag(), kernargCursor(0), kernargPoolMutex(),
                               executables(),
                               profile(hcAgentProfileNone),
                               path(), description(), host_(host) {
#if KALMAR_DEBUG
        std::cerr << "HSADevice::HSADevice()\n";
#endif

        hsa_status_t status = HSA_STATUS_SUCCESS;

        /// set up path and description
        {
            char name[64] {0};
            uint32_t node = 0;
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
            STATUS_CHECK(status, __LINE__);
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
            STATUS_CHECK(status, __LINE__);
    
            wchar_t path_wchar[128] {0};
            wchar_t description_wchar[128] {0};
            swprintf(path_wchar, 128, L"%s%u", name, node);
            swprintf(description_wchar, 128, L"AMD HSA Agent %s%u", name, node);
    
            path = std::wstring(path_wchar);
            description = std::wstring(description_wchar);

#if KALMAR_DEBUG
            std::wcerr << L"Path: " << path << L"\n";
            std::wcerr << L"Description: " << description << L"\n";
#endif
        }

        /// Iterate over memory pool of the device and its host
        status = hsa_amd_agent_iterate_memory_pools(agent, HSADevice::find_group_memory, &max_tile_static_size);
        STATUS_CHECK(status, __LINE__);

        status = hsa_amd_agent_iterate_memory_pools(agent, &HSADevice::get_memory_pools, &ri);
        STATUS_CHECK(status, __LINE__);

        status = hsa_amd_agent_iterate_memory_pools(host_, HSADevice::get_host_pools, &ri);
        STATUS_CHECK(status, __LINE__);

        /// after iterating memory regions, set if we can use coarse grained regions
        bool result = false;
        if (hasHSACoarsegrainedRegion()) {
            result = true;
            // environment variable HCC_HSA_USEHOSTMEMORY may be used to change
            // the default behavior
            char* hsa_behavior = getenv("HCC_HSA_USEHOSTMEMORY");
            if (hsa_behavior != nullptr) {
                if (std::string("ON") == hsa_behavior) {
                    result = false;
                }
            }
        }
        useCoarseGrainedRegion = result; 

        /// pre-allocate a pool of kernarg buffers in case:
        /// - kernarg region is available
        /// - compile-time macro USE_KERNARG_REGION is set
        /// - compile-time macro KERNARG_POOL_SIZE is larger than 0
        if (hasHSAKernargRegion() && USE_KERNARG_REGION) {
#if KERNARG_POOL_SIZE > 0
            hsa_amd_memory_pool_t kernarg_region = getHSAKernargRegion();

            // pre-allocate kernarg buffers
            void* kernargMemory = nullptr;
            for (int i = 0; i < KERNARG_POOL_SIZE; ++i) {
                status = hsa_amd_memory_pool_allocate(kernarg_region, KERNARG_BUFFER_SIZE, 0, &kernargMemory);
                STATUS_CHECK(status, __LINE__);

                // Allow device to access to it once it is allocated. Normally, this memory pool is on system memory.
                status = hsa_amd_agents_allow_access(1, &agent, NULL, kernargMemory);
                STATUS_CHECK(status, __LINE__);

                kernargPool.push_back(kernargMemory);
                kernargPoolFlag.push_back(false);
            }
#endif
        }

        // Setup AM pool.
        ri._am_memory_pool = (ri._found_local_memory_pool)
                                 ? ri._local_memory_pool
                                 : ri._finegrained_system_memory_pool;

        ri._am_host_memory_pool = (ri._found_coarsegrained_system_memory_pool)
                                      ? ri._coarsegrained_system_memory_pool
                                      : ri._finegrained_system_memory_pool;
        
        /// Query the maximum number of work-items in a workgroup
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &workgroup_max_size);
        STATUS_CHECK(status, __LINE__);

        /// Query the maximum number of work-items in each dimension of a workgroup
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &workgroup_max_dim);

        STATUS_CHECK(status, __LINE__);

        /// Get ISA associated with the agent
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &agentISA);
        STATUS_CHECK(status, __LINE__);

        /// Get the profile of the agent
        hsa_profile_t agentProfile;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agentProfile);
        STATUS_CHECK(status, __LINE__);

        if (agentProfile == HSA_PROFILE_BASE) {
            profile = hcAgentProfileBase;
        } else if (agentProfile == HSA_PROFILE_FULL) {
            profile = hcAgentProfileFull;
        }
    }

    ~HSADevice() {
#if KALMAR_DEBUG
        std::cerr << "HSADevice::~HSADevice() in\n";
#endif

        // release all queues
        queues_mutex.lock();
        for (auto queue_iterator : queues) {
            if (!queue_iterator.expired()) {
                auto queue = queue_iterator.lock();
                queue->dispose();
            }
        }
        queues.clear();
        queues_mutex.unlock();

        // deallocate kernarg buffers in the pool
        if (hasHSAKernargRegion() && USE_KERNARG_REGION) {
#if KERNARG_POOL_SIZE > 0
            kernargPoolMutex.lock();

            hsa_status_t status = HSA_STATUS_SUCCESS;

            for (int i = 0; i < kernargPool.size(); ++i) {
                hsa_amd_memory_pool_free(kernargPool[i]);
                STATUS_CHECK(status, __LINE__);
            }

            kernargPool.clear();
            kernargPoolFlag.clear();

            kernargPoolMutex.unlock();
#endif
        }

        // release all data in programs
        for (auto kernel_iterator : programs) {
            delete kernel_iterator.second;
        }
        programs.clear();

        // release executable
        for (auto executable_iterator : executables) {
            delete executable_iterator.second;
        }
        executables.clear();

#if KALMAR_DEBUG
        std::cerr << "HSADevice::~HSADevice() out\n";
#endif
    }

    std::wstring path;
    std::wstring description;

    std::wstring get_path() const override { return path; }
    std::wstring get_description() const override { return description; }
    size_t get_mem() const override { return 0; }
    bool is_double() const override { return true; }
    bool is_lim_double() const override { return true; }
    bool is_unified() const override {
        return (useCoarseGrainedRegion == false);
    }
    bool is_emulated() const override { return false; }

    void* create(size_t count, struct rw_info* key) override {
        void *data = nullptr;

        if (!is_unified()) {
#if KALMAR_DEBUG
            std::cerr << "create(" << count << "," << key << "): use HSA memory allocator\n";
#endif
            hsa_status_t status = HSA_STATUS_SUCCESS;
            auto am_region = getHSAAMRegion();
    
            status = hsa_amd_memory_pool_allocate(am_region, count, 0, &data);
            STATUS_CHECK(status, __LINE__);

            hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
            status = hsa_amd_agents_allow_access(1, agent, NULL, data);
            STATUS_CHECK(status, __LINE__); 
        } else {
#if KALMAR_DEBUG
            std::cerr << "create(" << count << "," << key << "): use host memory allocator\n";
#endif
            data = kalmar_aligned_alloc(0x1000, count);
        }

#if KALMAR_DEBUG
        std::cerr << "create(): " << data << "\n";
#endif

        return data;
    }
    
    void release(void *ptr, struct rw_info* key ) override {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (!is_unified()) {
#if KALMAR_DEBUG
            std::cerr << "release(" << ptr << "," << key << "): use HSA memory deallocator\n";
#endif
            status = hsa_amd_memory_pool_free(ptr);
            STATUS_CHECK(status, __LINE__);
        } else {
#if KALMAR_DEBUG
            std::cerr << "release(" << ptr << "," << key << "): use host memory deallocator\n";
#endif
            kalmar_aligned_free(ptr);
        }
    }

    // calculate MD5 checksum
    std::string kernel_checksum(size_t size, void* source) {
#if USE_MD5_HASH
        unsigned char md5_hash[16];
        memset(md5_hash, 0, sizeof(unsigned char) * 16);
        MD5_CTX md5ctx;
        MD5_Init(&md5ctx);
        MD5_Update(&md5ctx, source, size);
        MD5_Final(md5_hash, &md5ctx);

        std::stringstream checksum;
        checksum << std::setbase(16);
        for (int i = 0; i < 16; ++i) {
            checksum << static_cast<unsigned int>(md5_hash[i]);
        }

        return checksum.str();
#else
        // FNV-1a hashing, 64-bit version
        const uint64_t FNV_prime = 0x100000001b3;
        const uint64_t FNV_basis = 0xcbf29ce484222325;
        uint64_t hash = FNV_basis;

        const char *str = static_cast<const char *>(source);

        size = size > FNV1A_CUTOFF_SIZE ? FNV1A_CUTOFF_SIZE : size;
        for (auto i = 0; i < size; ++i) {
            hash ^= *str++;
            hash *= FNV_prime;
        }
        return std::to_string(hash);
#endif
    }

    void BuildProgram(void* size, void* source, bool needsCompilation = true) override {
        if (executables.find(kernel_checksum((size_t)size, source)) == executables.end()) {
            bool use_amdgpu = false;
#ifdef HSA_USE_AMDGPU_BACKEND
            const char *km_use_amdgpu = getenv("KM_USE_AMDGPU");
            use_amdgpu = !km_use_amdgpu || km_use_amdgpu[0] != '0';
#endif
            size_t kernel_size = (size_t)((void *)size);
            char *kernel_source = (char*)malloc(kernel_size+1);
            memcpy(kernel_source, source, kernel_size);
            kernel_source[kernel_size] = '\0';
            if (needsCompilation && !use_amdgpu) {
              BuildProgramImpl(kernel_source, kernel_size);
            } else {
              BuildOfflineFinalizedProgramImpl(kernel_source, kernel_size);
            }
            free(kernel_source);
        }
    }

    bool IsCompatibleKernel(void* size, void* source) override {
        hsa_status_t status;

        // Allocate memory for kernel source
        size_t kernel_size = (size_t)((void *)size);
        char *kernel_source = (char*)malloc(kernel_size+1);
        memcpy(kernel_source, source, kernel_size);
        kernel_source[kernel_size] = '\0';

        // Deserialize code object.
        hsa_code_object_t code_object = {0};
        status = hsa_code_object_deserialize(kernel_source, kernel_size, NULL, &code_object);
        STATUS_CHECK(status, __LINE__);
        assert(0 != code_object.handle);

        // Get ISA of the code object
        hsa_isa_t code_object_isa;
        status = hsa_code_object_get_info(code_object, HSA_CODE_OBJECT_INFO_ISA, &code_object_isa);
        STATUS_CHECK(status, __LINE__);

        // Check if the code object is compatible with ISA of the agent
        bool isCompatible = false;
        status = hsa_isa_compatible(code_object_isa, agentISA, &isCompatible);
        STATUS_CHECK(status, __LINE__);

        // Destroy code object
        status = hsa_code_object_destroy(code_object);
        STATUS_CHECK(status, __LINE__);

        // release allocated memory
        free(kernel_source);

        return isCompatible;
    }

    void* CreateKernel(const char* fun, void* size, void* source, bool needsCompilation = true) override {
        std::string str(fun);
        HSAKernel *kernel = programs[str];
        if (!kernel) {
            bool use_amdgpu = false;
#ifdef HSA_USE_AMDGPU_BACKEND
            const char *km_use_amdgpu = getenv("KM_USE_AMDGPU");
            use_amdgpu = !km_use_amdgpu || km_use_amdgpu[0] != '0';
#endif
            size_t kernel_size = (size_t)((void *)size);
            char *kernel_source = (char*)malloc(kernel_size+1);
            memcpy(kernel_source, source, kernel_size);
            kernel_source[kernel_size] = '\0';
            std::string kname;
            if (use_amdgpu) {
              kname = fun;
            } else {
              kname = std::string("&")+fun;
            }
            //std::cerr << "HSADevice::CreateKernel(): Creating kernel: " << kname << "\n";
            if (needsCompilation && !use_amdgpu) {
              kernel = CreateKernelImpl(kernel_source, kernel_size, kname.c_str());
            } else {
              kernel = CreateOfflineFinalizedKernelImpl(kernel_source, kernel_size, kname.c_str());
            }
            free(kernel_source);
            if (!kernel) {
                std::cerr << "HSADevice::CreateKernel(): Unable to create kernel\n";
                abort();
            } else {
                //std::cerr << "HSADevice::CreateKernel(): Created kernel\n";
            }
            programs[str] = kernel;
        }

        // HSADispatch instance will be deleted in:
        // HSAQueue::LaunchKernel()
        // or it will be created as a shared_ptr<KalmarAsyncOp> in:
        // HSAQueue::LaunchKernelAsync()
        HSADispatch *dispatch = new HSADispatch(this, kernel);
        dispatch->clearArgs();

        // HLC Stable would need 3 additional arguments
        // HLC Development would not need any additional arguments
#define HSAIL_HLC_DEVELOPMENT_COMPILER 1
#ifndef HSAIL_HLC_DEVELOPMENT_COMPILER
        dispatch->pushLongArg(0);
        dispatch->pushLongArg(0);
        dispatch->pushLongArg(0);
#endif
        return dispatch;
    }

    std::shared_ptr<KalmarQueue> createQueue(execute_order order = execute_in_order) override {
        std::shared_ptr<KalmarQueue> q =  std::shared_ptr<KalmarQueue>(new HSAQueue(this, agent, order));
        queues_mutex.lock();
        queues.push_back(q);
        queues_mutex.unlock();
        return q;
    }

    size_t GetMaxTileStaticSize() override {
        return max_tile_static_size;
    }

    std::vector< std::shared_ptr<KalmarQueue> > get_all_queues() override {
        std::vector< std::shared_ptr<KalmarQueue> > result;
        queues_mutex.lock();
        for (auto queue : queues) {
            if (!queue.expired()) {
                result.push_back(queue.lock());
            }
        }
        queues_mutex.unlock();
        return result;
    }

    hsa_amd_memory_pool_t& getHSAKernargRegion() {
        return ri._kernarg_memory_pool;
    }

    hsa_amd_memory_pool_t& getHSAAMHostRegion() {
        return ri._am_host_memory_pool;
    }

    hsa_amd_memory_pool_t& getHSAAMRegion() {
        return ri._am_memory_pool;
    }

    bool hasHSAKernargRegion() const { 
      return ri._found_kernarg_memory_pool;
    }

    bool hasHSAFinegrainedRegion() const {
      return ri._found_finegrained_system_memory_pool;
    }

    bool hasHSACoarsegrainedRegion() const {
      return ri. _found_local_memory_pool;
    }

    bool is_peer(const Kalmar::KalmarDevice* other) override {
      if(!hasHSACoarsegrainedRegion())
          return false;

      auto self_pool = getHSAAMRegion();
      hsa_amd_memory_pool_access_t access;

      hsa_agent_t* agent = static_cast<hsa_agent_t*>( const_cast<KalmarDevice *> (other)->getHSAAgent());

      //TODO: CPU acclerator will return NULL currently, return false.
      if(nullptr == agent)
          return false;

      hsa_status_t status = hsa_amd_agent_memory_pool_get_info(*agent, self_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);

      if(HSA_STATUS_SUCCESS != status)
          return false; 
    
      if ((HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT == access) || (HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT == access))
          return true;
  
      return false;
    }

    void releaseKernargBuffer(void* kernargBuffer, int kernargBufferIndex) {
        if (hasHSAKernargRegion() && USE_KERNARG_REGION) {
            if ( (KERNARG_POOL_SIZE > 0) && (kernargBufferIndex >= 0) ) {
                kernargPoolMutex.lock();

                // mark the kernarg buffer pointed by kernelBufferIndex as available
                kernargPoolFlag[kernargBufferIndex] = false;

                kernargPoolMutex.unlock();
             } else {
                if (kernargBuffer != nullptr) {
                    hsa_amd_memory_pool_free(kernargBuffer);
                }
             }
        }
    }

    std::pair<void*, int> getKernargBuffer(int size) {
        void* ret = nullptr;
        int cursor = 0;

        if (hasHSAKernargRegion() && USE_KERNARG_REGION) {

            // find an available buffer in the pool in case
            // - kernarg pool is available
            // - requested size is smaller than KERNARG_BUFFER_SIZE
            if ( (KERNARG_POOL_SIZE > 0) && (size <= KERNARG_BUFFER_SIZE) ) {
                kernargPoolMutex.lock();
                cursor = kernargCursor;
        
                if (kernargPoolFlag[cursor] == false) {
                    // the cursor is valid, use it
                    ret = kernargPool[cursor];
        
                    // set the kernarg buffer as used
                    kernargPoolFlag[cursor] = true;
        
                    // simply move the cursor to the next index
                    ++kernargCursor;
                    if (kernargCursor == kernargPool.size()) kernargCursor = 0;          
                } else {
                    // the cursor is not valid, sequentially find the next available slot
                    bool found = false;
        
                    int startingCursor = cursor;
                    do {
                        ++cursor;
                        if (cursor == kernargPool.size()) cursor = 0;
        
                        if (kernargPoolFlag[cursor] == false) {
                            // the cursor is valid, use it
                            ret = kernargPool[cursor];
        
                            // set the kernarg buffer as used
                            kernargPoolFlag[cursor] = true;
        
                            // simply move the cursor to the next index
                            kernargCursor = cursor + 1;
                            if (kernargCursor == kernargPool.size()) kernargCursor = 0;
        
                            // break from the loop
                            found = true;
                            break;
                        }
                    } while(cursor != startingCursor); // ensure we at most scan the vector once
        
                    if (found == false) {
                        hsa_status_t status = HSA_STATUS_SUCCESS;

                        // increase kernarg pool on demand by KERNARG_POOL_SIZE
                        hsa_amd_memory_pool_t kernarg_region = getHSAKernargRegion();
                       
                        // keep track of the size of kernarg pool before increasing it
                        int oldKernargPoolSize = kernargPool.size();
                        int oldKernargPoolFlagSize = kernargPoolFlag.size();
                        assert(oldKernargPoolSize == oldKernargPoolFlagSize);
            
                        // pre-allocate kernarg buffers
                        void* kernargMemory = nullptr;
                        for (int i = 0; i < KERNARG_POOL_SIZE; ++i) {
                            status = hsa_amd_memory_pool_allocate(kernarg_region, KERNARG_BUFFER_SIZE, 0, &kernargMemory);
                            STATUS_CHECK(status, __LINE__);

                            status = hsa_amd_agents_allow_access(1, &agent, NULL, kernargMemory);
                            STATUS_CHECK(status, __LINE__);
            
                            kernargPool.push_back(kernargMemory);
                            kernargPoolFlag.push_back(false);
                        }

                        assert(kernargPool.size() == oldKernargPoolSize + KERNARG_POOL_SIZE);
                        assert(kernargPoolFlag.size() == oldKernargPoolFlagSize + KERNARG_POOL_SIZE);

                        // set return values, after the pool has been increased

                        // use the first item in the newly allocated pool
                        cursor = oldKernargPoolSize;

                        // access the new item through the newly assigned cursor
                        ret = kernargPool[cursor];

                        // mark the item as used
                        kernargPoolFlag[cursor] = true;

                        // simply move the cursor to the next index
                        kernargCursor = cursor + 1;
                        if (kernargCursor == kernargPool.size()) kernargCursor = 0;

                        found = true;
                    }
        
                }
        
                kernargPoolMutex.unlock();
            } else {
                // allocate new buffers in case:
                // - the kernarg pool is set at compile-time
                // - requested kernarg buffer size is larger than KERNARG_BUFFER_SIZE

                hsa_status_t status = HSA_STATUS_SUCCESS;
                hsa_amd_memory_pool_t kernarg_region = getHSAKernargRegion();
    
                status = hsa_amd_memory_pool_allocate(kernarg_region, size, 0, &ret);
                STATUS_CHECK(status, __LINE__);

                status = hsa_amd_agents_allow_access(1, &agent, NULL, ret);
                STATUS_CHECK(status, __LINE__);
    
                // set cursor value as -1 to notice the buffer would be deallocated
                // instead of recycled back into the pool
                cursor = -1;
            }
        } else {
            // this function does nothing in case:
            // - kernarg region is not available on the agent
            // - or we choose not to use kernarg region by setting USE_KERNARG_REGION to 0
        }

        return std::make_pair(ret, cursor);
    }

    void* getSymbolAddress(const char* symbolName) override {
        hsa_status_t status;

        unsigned long* symbol_ptr = nullptr;
        if (executables.size() != 0) {
            // fix symbol name to match HSA rule
            std::string symbolString("&");
            symbolString += symbolName;

            // iterate through all HSA executables
            for (auto executable_iterator : executables) {
                HSAExecutable *executable = executable_iterator.second;

                // get symbol
                hsa_executable_symbol_t symbol;
                status = hsa_executable_get_symbol(executable->hsaExecutable, NULL, symbolString.c_str(), agent, 0, &symbol);
                if (status == HSA_STATUS_SUCCESS) {
                    // get address of symbol
                    uint64_t symbol_address;
                    status = hsa_executable_symbol_get_info(symbol,
                                                            HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS,
                                                            &symbol_address);
                    STATUS_CHECK(status, __LINE__);
        
                    symbol_ptr = (unsigned long*)symbol_address;
                    break;
                }
            }
        } else {
#if KALMAR_DEBUG
            std::cerr << "HSA executable NOT built yet!\n";
#endif
        }

        return symbol_ptr;
    }

    // FIXME: return values
    // TODO: Need more info about hostptr, is it OS allocated buffer or HSA allocator allocated buffer.
    // Or it might be the responsibility of caller? Because for OS allocated buffer, we need to call hsa_amd_memory_lock, otherwise, need to call
    // hsa_amd_agents_allow_access. Assume it is HSA allocated buffer.
    void memcpySymbol(void* symbolAddr, void* hostptr, size_t count, size_t offset = 0, enum hcMemcpyKind kind = hcMemcpyHostToDevice) override {
        hsa_status_t status;

        if (executables.size() != 0) {
            // copy data
            if (kind == hcMemcpyHostToDevice) {
                // host -> device
                status = hsa_memory_copy(symbolAddr, (char*)hostptr + offset, count);
                STATUS_CHECK(status, __LINE__);
            } else if (kind == hcMemcpyDeviceToHost) {
                // device -> host
                status = hsa_memory_copy(hostptr, (char*)symbolAddr + offset, count);
                STATUS_CHECK(status, __LINE__);
            }
        } else {
#if KALMAR_DEBUG
            std::cerr << "HSA executable NOT built yet!\n";
#endif
        }
    }

    // FIXME: return values
    void memcpySymbol(const char* symbolName, void* hostptr, size_t count, size_t offset = 0, enum hcMemcpyKind kind = hcMemcpyHostToDevice) override {
        hsa_status_t status;
    
        status = hsa_amd_agents_allow_access(1, &agent, NULL, hostptr);
        STATUS_CHECK(status, __LINE__);
        if (executables.size() != 0) {
            unsigned long* symbol_ptr = (unsigned long*)getSymbolAddress(symbolName);
            memcpySymbol(symbol_ptr, hostptr, count, offset, kind);
        } else {
#if KALMAR_DEBUG
            std::cerr << "HSA executable NOT built yet!\n";
#endif
        }
    }

    void* getHSAAgent() override; 

    hcAgentProfile getProfile() override { return profile; }

private:

    void BuildOfflineFinalizedProgramImpl(void* kernelBuffer, int kernelSize) {
        hsa_status_t status;

        std::string index = kernel_checksum((size_t)kernelSize, kernelBuffer);

        // load HSA program if we haven't done so
        if (executables.find(index) == executables.end()) {
            // Deserialize code object.
            hsa_code_object_t code_object = {0};
            status = hsa_code_object_deserialize(kernelBuffer, kernelSize, NULL, &code_object);
            STATUS_CHECK(status, __LINE__);
            assert(0 != code_object.handle);

            // Create the executable.
            hsa_executable_t hsaExecutable;
            status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                           NULL, &hsaExecutable);
            STATUS_CHECK(status, __LINE__);

            // Load the code object.
            status = hsa_executable_load_code_object(hsaExecutable, agent, code_object, NULL);
            STATUS_CHECK(status, __LINE__);

            // Freeze the executable.
            status = hsa_executable_freeze(hsaExecutable, NULL);
            STATUS_CHECK(status, __LINE__);

            // save everything as an HSAExecutable instance
            executables[index] = new HSAExecutable(hsaExecutable, code_object);
        }
    }

    HSAKernel* CreateOfflineFinalizedKernelImpl(void *kernelBuffer, int kernelSize, const char *entryName) {
        hsa_status_t status;

        std::string index = kernel_checksum((size_t)kernelSize, kernelBuffer);

        // load HSA program if we haven't done so
        if (executables.find(index) == executables.end()) {
            BuildOfflineFinalizedProgramImpl(kernelBuffer, kernelSize);
        }

        // fetch HSAExecutable*
        HSAExecutable* executable = executables[index];

        // Get symbol handle.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol(executable->hsaExecutable, NULL, entryName, agent, 0, &kernelSymbol);
        STATUS_CHECK(status, __LINE__);

        // Get code handle.
        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle);
        STATUS_CHECK(status, __LINE__);

        return new HSAKernel(executable, kernelSymbol, kernelCodeHandle);
    }

    void BuildProgramImpl(const char* hsailBuffer, int hsailSize) {
        hsa_status_t status;

        std::string index = kernel_checksum((size_t)hsailSize, (void*)hsailBuffer);

        // finalize HSA program if we haven't done so
        if (executables.find(index) == executables.end()) {
            /*
             * Load BRIG, encapsulated in an ELF container, into a BRIG module.
             */
            hsa_ext_module_t hsaModule = 0;
            hsaModule = (hsa_ext_module_t)hsailBuffer;

            /*
             * Create hsa program.
             */
            hsa_ext_program_t hsaProgram = {0};
            status = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL,
                                            HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO, NULL, &hsaProgram);
            STATUS_CHECK(status, __LINE__);

            /*
             * Add the BRIG module to hsa program.
             */
            status = hsa_ext_program_add_module(hsaProgram, hsaModule);
            STATUS_CHECK(status, __LINE__);

            /*
             * Finalize the hsa program.
             */
            hsa_isa_t isa = {0};
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
            STATUS_CHECK(status, __LINE__);

            hsa_ext_control_directives_t control_directives;
            memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));

            const char* extra_finalizer_opt = getenv("HCC_FINALIZE_OPT");
            hsa_code_object_t hsaCodeObject = {0};
            status = hsa_ext_program_finalize(hsaProgram, isa, 0, control_directives,
                                              extra_finalizer_opt, HSA_CODE_OBJECT_TYPE_PROGRAM, &hsaCodeObject);
            STATUS_CHECK(status, __LINE__);

            if (hsaProgram.handle != 0) {
                status = hsa_ext_program_destroy(hsaProgram);
                STATUS_CHECK(status, __LINE__);
            }

            // Create the executable.
            hsa_executable_t hsaExecutable;
            status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                           NULL, &hsaExecutable);
            STATUS_CHECK(status, __LINE__);

            // Load the code object.
            status = hsa_executable_load_code_object(hsaExecutable, agent, hsaCodeObject, NULL);
            STATUS_CHECK(status, __LINE__);

            // Freeze the executable.
            status = hsa_executable_freeze(hsaExecutable, NULL);
            STATUS_CHECK(status, __LINE__);

            // save everything as an HSAExecutable instance
            executables[index] = new HSAExecutable(hsaExecutable, hsaCodeObject);
        }
    }

    HSAKernel* CreateKernelImpl(const char *hsailBuffer, int hsailSize, const char *entryName) {
        hsa_status_t status;
  
        std::string index = kernel_checksum((size_t)hsailSize, (void*)hsailBuffer);

        // finalize HSA program if we haven't done so
        if (executables.find(index) == executables.end()) {
            BuildProgramImpl(hsailBuffer, hsailSize);
        }
  
        // fetch HSAExecutable*
        HSAExecutable* executable = executables[index];

        // Get symbol handle.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol(executable->hsaExecutable, NULL, entryName, agent, 0, &kernelSymbol);
        STATUS_CHECK(status, __LINE__);
  
        // Get code handle.
        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle);
        STATUS_CHECK(status, __LINE__);
  
        return new HSAKernel(executable, kernelSymbol, kernelCodeHandle);
    }

};

class HSAContext final : public KalmarContext
{
    /// memory pool for signals
    std::vector<hsa_signal_t> signalPool;
    std::vector<bool> signalPoolFlag;
    int signalCursor;
    std::mutex signalPoolMutex;
    /* TODO: Modify properly when supporing multi-gpu.
    When using memory pool api, each agent will only report memory pool
    which is attached with the agent itself physically, eg, GPU won't
    report system memory pool anymore. In order to change as little
    as possbile, will choose the first CPU as default host and hack the 
    HSADevice class to assign it the host memory pool to GPU agent. 
    */
    hsa_agent_t host;
    
    /// Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
    /// If so, cache to input data
    static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
        hsa_status_t status;
        hsa_device_type_t device_type;
        std::vector<hsa_agent_t>* pAgents = nullptr;

        if (data == nullptr) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        } else {
            pAgents = static_cast<std::vector<hsa_agent_t>*>(data);
        }

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if (stat != HSA_STATUS_SUCCESS) {
            return stat;
        }

#if KALMAR_DEBUG
        {
            char name[64];
            uint32_t node = 0;
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
            STATUS_CHECK(status, __LINE__);
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
            STATUS_CHECK(status, __LINE__);
            if (device_type == HSA_DEVICE_TYPE_GPU) {
                printf("GPU HSA agent: %s, Node ID: %u\n", name, node);
            } else if (device_type == HSA_DEVICE_TYPE_CPU) {
                printf("CPU HSA agent: %s, Node ID: %u\n", name, node);
            } else {
                printf("DSP HSA agent: %s, Node ID: %u\n", name, node);
            }
        }
#endif

        if (device_type == HSA_DEVICE_TYPE_GPU)  {
            pAgents->push_back(agent);
        }

        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t find_host(hsa_agent_t agent, void* data) {
        hsa_status_t status;
        hsa_device_type_t device_type;
        if(data == nullptr)
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        STATUS_CHECK(status, __LINE__);

        if(HSA_DEVICE_TYPE_CPU == device_type) {
            *(hsa_agent_t*)data = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    }


public:
    HSAContext() : KalmarContext(), signalPool(), signalPoolFlag(), signalCursor(0), signalPoolMutex() {
        host.handle = (uint64_t)-1;
        // initialize HSA runtime
#if KALMAR_DEBUG
        std::cerr << "HSAContext::HSAContext(): init HSA runtime\n";
#endif
        hsa_status_t status;
        status = hsa_init();
        STATUS_CHECK(status, __LINE__);

        // Iterate over the agents to find out gpu device
        std::vector<hsa_agent_t> agents;
        status = hsa_iterate_agents(&HSAContext::find_gpu, &agents);
        STATUS_CHECK(status, __LINE__);

        // Iterate over agents to find out the first cpu device as host
        status = hsa_iterate_agents(&HSAContext::find_host, &host);
        STATUS_CHECK(status, __LINE__);

        for (int i = 0; i < agents.size(); ++i) {
            hsa_agent_t agent = agents[i];
            auto Dev = new HSADevice(agent, host);
            // choose the first GPU device as the default device
            if (i == 0)
                def = Dev;
            Devices.push_back(Dev);
        }

        
#if SIGNAL_POOL_SIZE > 0
        signalPoolMutex.lock();

        // pre-allocate signals
        for (int i = 0; i < SIGNAL_POOL_SIZE; ++i) {
          hsa_signal_t signal;
          status = hsa_signal_create(1, 0, NULL, &signal);
          STATUS_CHECK(status, __LINE__);
          signalPool.push_back(signal);
          signalPoolFlag.push_back(false);
        }

        signalPoolMutex.unlock();
#endif
    }

    void releaseSignal(hsa_signal_t signal, int signalIndex) {
        hsa_status_t status = HSA_STATUS_SUCCESS;
#if SIGNAL_POOL_SIZE > 0
        signalPoolMutex.lock();

        // restore signal to the initial value 1
        hsa_signal_store_release(signal, 1);

        // mark the signal pointed by signalIndex as available
        signalPoolFlag[signalIndex] = false;

        signalPoolMutex.unlock();
#else
        status = hsa_signal_destroy(signal);
        STATUS_CHECK(status, __LINE__);
#endif
    }

    std::pair<hsa_signal_t, int> getSignal() {
        hsa_signal_t ret;

#if SIGNAL_POOL_SIZE > 0
        signalPoolMutex.lock();
        int cursor = signalCursor;

        if (signalPoolFlag[cursor] == false) {
            // the cursor is valid, use it
            ret = signalPool[cursor];

            // set the signal as used
            signalPoolFlag[cursor] = true;

            // simply move the cursor to the next index
            ++signalCursor;
            if (signalCursor == signalPool.size()) signalCursor = 0;
        } else {
            // the cursor is not valid, sequentially find the next available slot
            bool found = false;
            int startingCursor = cursor;
            do {
                ++cursor;
                if (cursor == signalPool.size()) cursor = 0;

                if (signalPoolFlag[cursor] == false) {
                    // the cursor is valid, use it
                    ret = signalPool[cursor];

                    // set the signal as used
                    signalPoolFlag[cursor] = true;

                    // simply move the cursor to the next index
                    signalCursor = cursor + 1;
                    if (signalCursor == signalPool.size()) signalCursor = 0;

                    // break from the loop
                    found = true;
                    break;
                }
            } while(cursor != startingCursor); // ensure we at most scan the vector once

            if (found == false) {
                hsa_status_t status = HSA_STATUS_SUCCESS;

                // increase signal pool on demand by SIGNAL_POOL_SIZE

                // keep track of the size of signal pool before increasing it
                int oldSignalPoolSize = signalPool.size();
                int oldSignalPoolFlagSize = signalPoolFlag.size();
                assert(oldSignalPoolSize == oldSignalPoolFlagSize);

                // increase signal pool on demand for another SIGNAL_POOL_SIZE
                for (int i = 0; i < SIGNAL_POOL_SIZE; ++i) {
                    hsa_signal_t signal;
                    status = hsa_signal_create(1, 0, NULL, &signal);
                    STATUS_CHECK(status, __LINE__);
                    signalPool.push_back(signal);
                    signalPoolFlag.push_back(false);
                }

                assert(signalPool.size() == oldSignalPoolSize + SIGNAL_POOL_SIZE);
                assert(signalPoolFlag.size() == oldSignalPoolFlagSize + SIGNAL_POOL_SIZE);

                // set return values, after the pool has been increased

                // use the first item in the newly allocated pool
                cursor = oldSignalPoolSize;

                // access the new item through the newly assigned cursor
                ret = signalPool[cursor];

                // mark the item as used
                signalPoolFlag[cursor] = true;

                // simply move the cursor to the next index
                signalCursor = cursor + 1;
                if (signalCursor == signalPool.size()) signalCursor = 0;

                found = true;
            } 
        }

        signalPoolMutex.unlock();
#else
        hsa_signal_create(1, 0, NULL, &ret);
        int cursor = 0;
#endif
        return std::make_pair(ret, cursor);
    }

    ~HSAContext() {
        hsa_status_t status = HSA_STATUS_SUCCESS;
#if KALMAR_DEBUG
        std::cerr << "HSAContext::~HSAContext() in\n";
#endif

        // destroy all KalmarDevices associated with this context
        for (auto dev : Devices)
            delete dev;
        Devices.clear();
        def = nullptr;

#if SIGNAL_POOL_SIZE > 0
        signalPoolMutex.lock();

        // deallocate signals in the pool
        for (int i = 0; i < signalPool.size(); ++i) {
            hsa_signal_t signal;
            status = hsa_signal_destroy(signalPool[i]);
            STATUS_CHECK(status, __LINE__);
        }

        signalPool.clear();
        signalPoolFlag.clear();

        signalPoolMutex.unlock();
#endif

        // shutdown HSA runtime
#if KALMAR_DEBUG
        std::cerr << "HSAContext::~HSAContext(): shut down HSA runtime\n";
#endif
        status = hsa_shut_down();
        STATUS_CHECK(status, __LINE__);

#if KALMAR_DEBUG
        std::cerr << "HSAContext::~HSAContext() out\n";
#endif
    }

    uint64_t getSystemTicks() override {
        // get system tick
        uint64_t timestamp = 0L;
        hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &timestamp);
        return timestamp;
    }

    uint64_t getSystemTickFrequency() override {
        // get system tick frequency
        uint64_t timestamp_frequency_hz = 0L;
        hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_frequency_hz);
        return timestamp_frequency_hz;
    }
};

static HSAContext ctx;

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSADevice
// ----------------------------------------------------------------------
namespace Kalmar {

inline void*
HSADevice::getHSAAgent() override {
    return static_cast<void*>(&getAgent());
}

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSAQueue
// ----------------------------------------------------------------------
namespace Kalmar {

inline void*
HSAQueue::getHSAAgent() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getAgent()));
}

inline void*
HSAQueue::getHSAAMRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAAMRegion()));
}

inline void*
HSAQueue::getHSAAMHostRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAAMHostRegion()));
}


inline void*
HSAQueue::getHSAKernargRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAKernargRegion()));
}

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSADispatch
// ----------------------------------------------------------------------

HSADispatch::HSADispatch(Kalmar::HSADevice* _device, HSAKernel* _kernel) :
    device(_device),
    agent(_device->getAgent()),
    kernel(_kernel),
    isDispatched(false),
    waitMode(HSA_WAIT_STATE_BLOCKED),
    dynamicGroupSize(0),
    future(nullptr),
    hsaQueue(nullptr),
    kernargMemory(nullptr) {

    clearArgs();
}


// dispatch a kernel asynchronously
hsa_status_t 
HSADispatch::dispatchKernel(hsa_queue_t* commandQueue) {
    struct timespec begin;
    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &begin);

    hsa_status_t status = HSA_STATUS_SUCCESS;
    if (isDispatched) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  
    /*
     * Create a signal to wait for the dispatch to finish.
     */
    std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
    signal = ret.first;
    signalIndex = ret.second;
  
    /*
     * Initialize the dispatch packet.
     */
    memset(&aql, 0, sizeof(aql));
  
    /*
     * Setup the dispatch information.
     */
    aql.completion_signal = signal;
    aql.setup = launchDimensions << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    aql.workgroup_size_x = workgroup_size[0];
    aql.workgroup_size_y = workgroup_size[1];
    aql.workgroup_size_z = workgroup_size[2];
    aql.grid_size_x = global_size[0];
    aql.grid_size_y = global_size[1];
    aql.grid_size_z = global_size[2];
  

    // set dispatch fences
    if (hsaQueue->get_execute_order() == Kalmar::execute_in_order) {
        //std::cout << "barrier bit on\n";
        // set AQL header with barrier bit on if execute in order
        aql.header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                     (1 << HSA_PACKET_HEADER_BARRIER) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    } else {
        //std::cout << "barrier bit off\n";
        // set AQL header with barrier bit off if execute in any order
        aql.header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    }
  
    // bind kernel code
    aql.kernel_object = kernel->kernelCodeHandle;
  
    // bind kernel arguments
    //printf("arg_vec size: %d in bytes: %d\n", arg_vec.size(), arg_vec.size());

    if (device->hasHSAKernargRegion() && USE_KERNARG_REGION) {
        hsa_amd_memory_pool_t kernarg_region = device->getHSAKernargRegion();

        if (arg_vec.size() > 0) {
            std::pair<void*, int> ret = device->getKernargBuffer(arg_vec.size());
            kernargMemory = ret.first;
            kernargMemoryIndex = ret.second;

#if USE_HSA_MEMORY_COPY_FOR_KERNARG
            // use hsa_memory_copy to copy kernel arguments from host to kernarg region
            status = hsa_memory_copy(kernargMemory, arg_vec.data(), arg_vec.size());
            STATUS_CHECK_Q(status, commandQueue, __LINE__);
#else
            // as kernarg buffers are fine-grained, we can directly use memcpy
            memcpy(kernargMemory, arg_vec.data(), arg_vec.size());
#endif

            aql.kernarg_address = kernargMemory;
        } else {
            aql.kernarg_address = nullptr;
        }
    }
    else {
        aql.kernarg_address = arg_vec.data();
    }


    //for (size_t i = 0; i < arg_vec.size(); ++i) {
    //  printf("%02X ", *(((uint8_t*)aql.kernarg_address)+i));
    //}
    //printf("\n");
 
    // Initialize memory resources needed to execute
    uint32_t group_segment_size;
    status = hsa_executable_symbol_get_info(kernel->hsaExecutableSymbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                            &group_segment_size);
    STATUS_CHECK_Q(status, commandQueue, __LINE__);

#ifndef HSA_USE_AMDGPU_BACKEND
    // let kernel know static group segment size
    kernel->executable->setSymbolToValue("&hcc_static_group_segment_size", group_segment_size);

    // let kernel know dynamic group segment size
    kernel->executable->setSymbolToValue("&hcc_dynamic_group_segment_size", this->dynamicGroupSize);
#endif

    // add dynamic group segment size
    group_segment_size += this->dynamicGroupSize;
    aql.group_segment_size = group_segment_size;

    uint32_t private_segment_size;
    status = hsa_executable_symbol_get_info(kernel->hsaExecutableSymbol,
                                            HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                            &private_segment_size);
    STATUS_CHECK_Q(status, commandQueue, __LINE__);
    aql.private_segment_size = private_segment_size;

    // write packet
    uint32_t queueMask = commandQueue->size - 1;
    // TODO: Need to check if package write is correct.
    uint64_t index = hsa_queue_load_write_index_relaxed(commandQueue);
    ((hsa_kernel_dispatch_packet_t*)(commandQueue->base_address))[index & queueMask] = aql;
    hsa_queue_store_write_index_relaxed(commandQueue, index + 1);
  
#if KALMAR_DEBUG
    std::cerr << "ring door bell to dispatch kernel\n";
#endif
  
    // Ring door bell
    hsa_signal_store_relaxed(commandQueue->doorbell_signal, index);
  
    isDispatched = true;

    clock_gettime(CLOCK_REALTIME, &end);

#if KALMAR_DISPATCH_TIME_PRINTOUT
    std::cerr << std::setprecision(6) << ((float)(end.tv_sec - begin.tv_sec) * 1000 * 1000 + (float)(end.tv_nsec - begin.tv_nsec) / 1000) << "\n";
#endif

    return status;
}



// wait for the kernel to finish execution
inline hsa_status_t
HSADispatch::waitComplete() {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    if (!isDispatched)  {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

#if KALMAR_DEBUG
    std::cerr << "wait for kernel dispatch completion with wait flag: " << waitMode << "\n";
#endif

    // wait for completion
    if (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), waitMode)!=0) {
        printf("Signal wait returned unexpected value\n");
        exit(0);
    }

#if KALMAR_DEBUG
    std::cerr << "complete!\n";
#endif

    if (kernargMemory != nullptr) {
      device->releaseKernargBuffer(kernargMemory, kernargMemoryIndex);
      kernargMemory = nullptr;
    }

    // unregister this async operation from HSAQueue
    if (this->hsaQueue != nullptr) {
        this->hsaQueue->removeAsyncOp(this);
    }

    isDispatched = false;
    return status;
}

inline hsa_status_t
HSADispatch::dispatchKernelWaitComplete(Kalmar::HSAQueue* hsaQueue) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (isDispatched) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // record HSAQueue association
    this->hsaQueue = hsaQueue;
    // extract hsa_queue_t from HSAQueue
    hsa_queue_t* queue = static_cast<hsa_queue_t*>(hsaQueue->getHSAQueue());

    // dispatch kernel
    status = dispatchKernel(queue);
    STATUS_CHECK_Q(status, queue, __LINE__);

    // wait for completion
    status = waitComplete();
    STATUS_CHECK_Q(status, queue, __LINE__);

    return status;
} 

inline hsa_status_t
HSADispatch::dispatchKernelAsync(Kalmar::HSAQueue* hsaQueue) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    // record HSAQueue association
    this->hsaQueue = hsaQueue;
    // extract hsa_queue_t from HSAQueue
    hsa_queue_t* queue = static_cast<hsa_queue_t*>(hsaQueue->getHSAQueue());

    // dispatch kernel
    status = dispatchKernel(queue);
    STATUS_CHECK_Q(status, queue, __LINE__);

    // dynamically allocate a std::shared_future<void> object
    future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
        waitComplete();
    }).share());

    return status;
}

inline void
HSADispatch::dispose() {
    hsa_status_t status;
    if (kernargMemory != nullptr) {
      device->releaseKernargBuffer(kernargMemory, kernargMemoryIndex);
      kernargMemory = nullptr;
    }

    clearArgs();
    std::vector<uint8_t>().swap(arg_vec);

    Kalmar::ctx.releaseSignal(signal, signalIndex);

    if (future != nullptr) {
      delete future;
      future = nullptr;
    }
}

inline uint64_t
HSADispatch::getBeginTimestamp() override {
    Kalmar::HSADevice* device = static_cast<Kalmar::HSADevice*>(hsaQueue->getDev());
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(device->getAgent(), signal, &time);
    return time.start;
}

inline uint64_t
HSADispatch::getEndTimestamp() override {
    Kalmar::HSADevice* device = static_cast<Kalmar::HSADevice*>(hsaQueue->getDev());
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(device->getAgent(), signal, &time);
    return time.end;
}

inline hsa_status_t
HSADispatch::setLaunchAttributes(int dims, size_t *globalDims, size_t *localDims) {
    assert((0 < dims) && (dims <= 3));

    // defaults
    launchDimensions = dims;
    workgroup_size[0] = workgroup_size[1] = workgroup_size[2] = 1;
    global_size[0] = global_size[1] = global_size[2] = 1;

    // for each workgroup dimension, make sure it does not exceed the maximum allowable limit
    const uint16_t* workgroup_max_dim = device->getWorkgroupMaxDim();
    for (int i = 0; i < dims; ++i) {
        computeLaunchAttr(i, globalDims[i], localDims[i], workgroup_max_dim[i]);
    }

    // reduce each dimension in case the overall workgroup limit is exceeded
    uint32_t workgroup_max_size = device->getWorkgroupMaxSize();
    int dim_iterator = 2;
    size_t workgroup_total_size = workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
    while(workgroup_total_size > workgroup_max_size) {
      // repeatedly cut each dimension into half until we are within the limit
      if (workgroup_size[dim_iterator] >= 2) {
        workgroup_size[dim_iterator] >>= 1;
      }
      if (--dim_iterator < 0) {
        dim_iterator = 2;
      }
      workgroup_total_size = workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
    }

    return HSA_STATUS_SUCCESS;
}


// ----------------------------------------------------------------------
// member function implementation of HSABarrier
// ----------------------------------------------------------------------

// wait for the barrier to complete
inline hsa_status_t
HSABarrier::waitComplete() {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    if (!isDispatched)  {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

#if KALMAR_DEBUG
    std::cerr << "wait for barrier completion with wait flag: " << waitMode << "\n";
#endif

    // Wait on completion signal until the barrier is finished
    hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, waitMode);

#if KALMAR_DEBUG
    std::cerr << "complete!\n";
#endif

    // unregister this async operation from HSAQueue
    if (this->hsaQueue != nullptr) {
        this->hsaQueue->removeAsyncOp(this);
    }

    isDispatched = false;

    return status;
}

inline hsa_status_t
HSABarrier::enqueueAsync(Kalmar::HSAQueue* hsaQueue) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    // record HSAQueue association
    this->hsaQueue = hsaQueue;
    // extract hsa_queue_t from HSAQueue
    hsa_queue_t* queue = static_cast<hsa_queue_t*>(hsaQueue->getHSAQueue());

    // enqueue barrier packet
    status = enqueueBarrier(queue);
    STATUS_CHECK_Q(status, queue, __LINE__);

    // dynamically allocate a std::shared_future<void> object
    future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
        waitComplete();
    }).share());

    return status;
}

inline hsa_status_t
HSABarrier::enqueueBarrier(hsa_queue_t* queue) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    if (isDispatched) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    // Create a signal to wait for the barrier to finish.
    std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
    signal = ret.first;
    signalIndex = ret.second;

    // Obtain the write index for the command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    const uint32_t queueMask = queue->size - 1;

    // Define the barrier packet to be at the calculated queue index address
    hsa_barrier_and_packet_t* barrier = &(((hsa_barrier_and_packet_t*)(queue->base_address))[index&queueMask]);
    memset(barrier, 0, sizeof(hsa_barrier_and_packet_t));

    // setup header
    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    barrier->header = header;

    barrier->completion_signal = signal;

#if KALMAR_DEBUG
    std::cerr << "ring door bell to dispatch barrier\n";
#endif

    // Increment write index and ring doorbell to dispatch the kernel
    hsa_queue_store_write_index_relaxed(queue, index+1);
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    isDispatched = true;

    return status;
}

inline void
HSABarrier::dispose() {
    Kalmar::ctx.releaseSignal(signal, signalIndex);

    if (future != nullptr) {
      delete future;
      future = nullptr;
    }
}

inline uint64_t
HSABarrier::getBeginTimestamp() override {
    Kalmar::HSADevice* device = static_cast<Kalmar::HSADevice*>(hsaQueue->getDev());
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(device->getAgent(), signal, &time);
    return time.start;
}

inline uint64_t
HSABarrier::getEndTimestamp() override {
    Kalmar::HSADevice* device = static_cast<Kalmar::HSADevice*>(hsaQueue->getDev());
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(device->getAgent(), signal, &time);
    return time.end;
}

// ----------------------------------------------------------------------
// extern "C" functions
// ----------------------------------------------------------------------

extern "C" void *GetContextImpl() {
  return &Kalmar::ctx;
}

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v) {
  //std::cerr << "pushing:" << ker << " of size " << sz << "\n";
  HSADispatch *dispatch =
      reinterpret_cast<HSADispatch*>(ker);
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
  HSADispatch *dispatch =
      reinterpret_cast<HSADispatch*>(ker);
  void *val = const_cast<void*>(v);
  dispatch->pushPointerArg(val);
}
