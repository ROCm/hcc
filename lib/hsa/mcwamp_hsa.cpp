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
#include <string>
#include <thread>
#include <vector>

#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <hsa_ext_amd.h>

#include <kalmar_runtime.h>

#define KALMAR_DEBUG (0)

#define STATUS_CHECK(s,line) if (s != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

#define STATUS_CHECK_Q(s,q,line) if (s != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(q));\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v);
extern "C" void PushArgPtrImpl(void *ker, int idx, size_t sz, const void *v);

// forward declaration
namespace Kalmar {
class HSAQueue;
} // namespace Kalmar

///
/// kernel compilation / kernel launching
///

class HSAKernel {
private:
    hsa_code_object_t hsaCodeObject;
    hsa_executable_t hsaExecutable;
    uint64_t kernelCodeHandle;
    hsa_executable_symbol_t hsaExecutableSymbol;
    friend class HSADispatch;

public:
    HSAKernel(hsa_executable_t _hsaExecutable,
              hsa_code_object_t _hsaCodeObject,
              hsa_executable_symbol_t _hsaExecutableSymbol,
              uint64_t _kernelCodeHandle) :
      hsaExecutable(_hsaExecutable),
      hsaCodeObject(_hsaCodeObject),
      hsaExecutableSymbol(_hsaExecutableSymbol),
      kernelCodeHandle(_kernelCodeHandle) {}

    ~HSAKernel() {
      hsa_status_t status;

#if KALMAR_DEBUG
      std::cerr << "HSAKernel::~HSAKernel\n";
#endif

      status = hsa_executable_destroy(hsaExecutable);
      STATUS_CHECK(status, __LINE__);

      status = hsa_code_object_destroy(hsaCodeObject);
      STATUS_CHECK(status, __LINE__);
    }
}; // end of HSAKernel

class HSABarrier : public Kalmar::KalmarAsyncOp {
private:
    hsa_signal_t signal;
    bool isDispatched;

    std::shared_future<void>* future;

    Kalmar::HSAQueue* hsaQueue;

public:
    std::shared_future<void>* getFuture() override { return future; }

    void* getNativeHandle() override { return &signal; }

    HSABarrier() : isDispatched(false), future(nullptr), hsaQueue(nullptr) {}

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

    hsa_status_t enqueueBarrier(hsa_queue_t* queue) {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (isDispatched) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }

        status = hsa_signal_create(1, 0, NULL, &signal);
        STATUS_CHECK_Q(status, queue, __LINE__);

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

    hsa_status_t enqueueAsync(Kalmar::HSAQueue*);

    // wait for the barrier to complete
    hsa_status_t waitComplete();

    void dispose() {
        hsa_status_t status;
        status = hsa_signal_destroy(signal);
        STATUS_CHECK(status, __LINE__);

        if (future != nullptr) {
          delete future;
          future = nullptr;
        }
    }

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
    hsa_agent_t agent;
    const HSAKernel* kernel;

    uint32_t workgroup_max_size;
    uint16_t workgroup_max_dim[3];

    std::vector<uint8_t> arg_vec;
    uint32_t arg_count;
    size_t prevArgVecCapacity;
    int launchDimensions;
    uint32_t workgroup_size[3];
    uint32_t global_size[3];
    static const int ARGS_VEC_INITIAL_CAPACITY = 256 * 8;   

    hsa_signal_t signal;
    hsa_kernel_dispatch_packet_t aql;
    bool isDispatched;

    size_t dynamicGroupSize;

    std::shared_future<void>* future;

    Kalmar::HSAQueue* hsaQueue;

public:
    std::shared_future<void>* getFuture() override { return future; }

    void* getNativeHandle() override { return &signal; }

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

    HSADispatch(hsa_agent_t _agent, const HSAKernel* _kernel) :
        agent(_agent),
        kernel(_kernel),
        isDispatched(false),
        dynamicGroupSize(0),
        future(nullptr),
        hsaQueue(nullptr) {

        // allocate the initial argument vector capacity
        arg_vec.reserve(ARGS_VEC_INITIAL_CAPACITY);
        registerArgVecMemory();

        clearArgs();

        hsa_status_t status;

        /// Query the maximum number of work-items in a workgroup
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &workgroup_max_size);
        STATUS_CHECK(status, __LINE__);

        /// Query the maximum number of work-items in each dimension of a workgroup
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &workgroup_max_dim);

        STATUS_CHECK(status, __LINE__);
    }

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

    uint32_t getWorkgroupMaxSize() {
        return workgroup_max_size;
    }

    const uint16_t* getWorkgroupMaxDim() {
        return &workgroup_max_dim[0];
    }

    hsa_status_t setLaunchAttributes(int dims, size_t *globalDims, size_t *localDims) {
        assert((0 < dims) && (dims <= 3));
  
        // defaults
        launchDimensions = dims;
        workgroup_size[0] = workgroup_size[1] = workgroup_size[2] = 1;
        global_size[0] = global_size[1] = global_size[2] = 1;
  
        // for each workgroup dimension, make sure it does not exceed the maximum allowable limit
        const uint16_t* workgroup_max_dim = getWorkgroupMaxDim();
        for (int i = 0; i < dims; ++i) {
            computeLaunchAttr(i, globalDims[i], localDims[i], workgroup_max_dim[i]);
        }
  
        // reduce each dimension in case the overall workgroup limit is exceeded
        uint32_t workgroup_max_size = getWorkgroupMaxSize();
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

    hsa_status_t dispatchKernelWaitComplete(hsa_queue_t* _queue) {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (isDispatched) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
        status = dispatchKernel(_queue);
        STATUS_CHECK_Q(status, _queue, __LINE__);

        status = waitComplete();
        STATUS_CHECK_Q(status, _queue, __LINE__);

        return status;
    } 

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
    hsa_status_t dispatchKernel(hsa_queue_t* commandQueue) {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (isDispatched) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
  
        // check if underlying arg_vec data might have changed, if so re-register
        if (arg_vec.capacity() > prevArgVecCapacity) {
             registerArgVecMemory();
        }
  
        /*
         * Create a signal to wait for the dispatch to finish.
         */
        status = hsa_signal_create(1, 0, NULL, &signal);
        STATUS_CHECK_Q(status, commandQueue, __LINE__);
  
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
        aql.header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                     (1 << HSA_PACKET_HEADER_BARRIER) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                     (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  
        // bind kernel code
        aql.kernel_object = kernel->kernelCodeHandle;
  
        // bind kernel arguments
        //printf("arg_vec size: %d in bytes: %d\n", arg_vec.size(), arg_vec.size());
        aql.kernarg_address = arg_vec.data();
        hsa_memory_register(arg_vec.data(), arg_vec.size());
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
        uint64_t index = hsa_queue_load_write_index_relaxed(commandQueue);
        ((hsa_kernel_dispatch_packet_t*)(commandQueue->base_address))[index & queueMask] = aql;
        hsa_queue_store_write_index_relaxed(commandQueue, index + 1);
  
#if KALMAR_DEBUG
        std::cerr << "ring door bell to dispatch kernel\n";
#endif
  
        // Ring door bell
        hsa_signal_store_relaxed(commandQueue->doorbell_signal, index);
  
        isDispatched = true;

        return status;
    }

    // wait for the kernel to finish execution
    hsa_status_t waitComplete();

    void dispose() {
        hsa_status_t status;
        status = hsa_memory_deregister(arg_vec.data(), arg_vec.capacity() * sizeof(uint8_t));
        assert(status == HSA_STATUS_SUCCESS);
        hsa_signal_destroy(aql.completion_signal);
        clearArgs();
        std::vector<uint8_t>().swap(arg_vec);

        if (future != nullptr) {
          delete future;
          future = nullptr;
        }
    }

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
        //printf("push %lu bytes into kernarg: ", sizeof(T) + padding_size);
        for (size_t i = 0; i < padding_size; ++i) {
            arg_vec.push_back((uint8_t)0x00);
            //printf("%02X ", (uint8_t)0x00);
        }
        uint8_t* ptr = static_cast<uint8_t*>(static_cast<void*>(&val));
        for (size_t i = 0; i < sizeof(T); ++i) {
            arg_vec.push_back(ptr[i]);
            //printf("%02X ", ptr[i]);
        }
        //printf("\n");
        arg_count++;
        return HSA_STATUS_SUCCESS;
    }

    void registerArgVecMemory() {
        // record current capacity to compare for changes
        prevArgVecCapacity = arg_vec.capacity();

        // register the memory behind the arg_vec
        hsa_status_t status = hsa_memory_register(arg_vec.data(), arg_vec.capacity() * sizeof(uint8_t));
        assert(status == HSA_STATUS_SUCCESS);
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
//Structure used to extract information from regions
struct region_iterator
{
    hsa_region_t _am_region;

    hsa_region_t _kernarg_region;
    hsa_region_t _finegrained_region;
    hsa_region_t _coarsegrained_region;

    bool        _found_kernarg_region;
    bool        _found_finegrained_region;
    bool        _found_coarsegrained_region;

    region_iterator() ;
};


region_iterator::region_iterator()
{
    _kernarg_region.handle=(uint64_t)-1;
    _finegrained_region.handle=(uint64_t)-1;
    _coarsegrained_region.handle=(uint64_t)-1;

    _found_kernarg_region = false;
    _found_finegrained_region = false;
    _found_coarsegrained_region = false;
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
    HSAQueue(KalmarDevice* pDev, hsa_agent_t agent) : KalmarQueue(pDev), commandQueue(nullptr), asyncOps(), bufferKernelMap(), kernelBufferMap() {
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

    ~HSAQueue() {
        hsa_status_t status;

#if KALMAR_DEBUG
        std::cerr << "HSAQueue::~HSAQueue()\n";
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
        std::cerr << "HSAQueue::~HSAQueue(): destroy an HSA command queue: " << commandQueue << "\n";
#endif
        status = hsa_queue_destroy(commandQueue);
        STATUS_CHECK(status, __LINE__);
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

    void wait() override {
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
        dispatch->dispatchKernelWaitComplete(commandQueue);

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
        if (dst != device)
            memmove(dst, (char*)device + offset, count);
    }

    void write(void* device, const void* src, size_t count, size_t offset, bool blocking) override {
        waitForDependentAsyncOps(device);

        // do write
        if (src != device)
            memmove((char*)device + offset, src, count);
    }

    void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) override {
        waitForDependentAsyncOps(dst);
        waitForDependentAsyncOps(src);

        // do copy
        if (src != dst)
            memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
    }

    void* map(void* device, size_t count, size_t offset, bool modify) override {
        waitForDependentAsyncOps(device);

        // do map
        return (char*)device + offset;
    }

    void unmap(void* device, void* addr) override {}

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
    }
};

class HSADevice final : public KalmarDevice
{
private:
    std::map<std::string, HSAKernel *> programs;
    hsa_agent_t agent;
    size_t max_tile_static_size;

    std::mutex queues_mutex;
    std::vector< std::weak_ptr<KalmarQueue> > queues;

    region_iterator ri;

public:

    // Callback for hsa_agent_iterate_regions.
    // data is of type region_iterator,
    // we save the regions we care about into this structure.
    static hsa_status_t get_memory_regions(hsa_region_t region, void* data)
    {
    
        hsa_region_segment_t segment;
        hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    
        region_iterator *ri = (region_iterator*) (data);
    
        hsa_region_global_flag_t flags;
        hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    
        if (segment == HSA_REGION_SEGMENT_GLOBAL) {
            if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
                ri->_kernarg_region = region;
                ri->_found_kernarg_region = true;
            }
    
            if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
                ri->_finegrained_region = region;
                ri->_found_finegrained_region = true;
            }
    
            if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
                ri->_coarsegrained_region = region;
                ri->_found_coarsegrained_region = true;
            }
        }
    
        return HSA_STATUS_SUCCESS;
    }

    static hsa_status_t find_group_memory(hsa_region_t region, void* data) {
      hsa_region_segment_t segment;
      size_t size = 0;
      bool flag = false;

      hsa_status_t status = HSA_STATUS_SUCCESS;

      // get segment information
      status = hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
      STATUS_CHECK(status, __LINE__);

      if (segment == HSA_REGION_SEGMENT_GROUP) {
        // found group segment, get its size
        status = hsa_region_get_info(region, HSA_REGION_INFO_SIZE, &size);
        STATUS_CHECK(status, __LINE__);

        // save the result to data
        size_t* result = (size_t*)data;
        *result = size;
      }

      // continue iteration
      return HSA_STATUS_SUCCESS;
    }

    hsa_agent_t& getAgent() {
        return agent;
    }

    HSADevice(hsa_agent_t a) : KalmarDevice(access_type_read_write),
                               agent(a), programs(), max_tile_static_size(0),
                               queues(), queues_mutex(),
                               ri() {
#if KALMAR_DEBUG
        std::cerr << "HSADevice::HSADevice()\n";
#endif

        /// iterate over memory regions of the agent
        hsa_status_t status = HSA_STATUS_SUCCESS;
        status = hsa_agent_iterate_regions(agent, HSADevice::find_group_memory, &max_tile_static_size);
        STATUS_CHECK(status, __LINE__);
    }

    ~HSADevice() {
#if KALMAR_DEBUG
        std::cerr << "HSADevice::~HSADevice()\n";
#endif
        // release all queues
        queues_mutex.lock();
        queues.clear();
        queues_mutex.unlock();

        // release all data in programs
        for (auto kernel_iterator : programs) {
            delete kernel_iterator.second;
        }
        programs.clear();
    }

    std::wstring get_path() const override { return L"hsa"; }
    std::wstring get_description() const override { return L"AMD HSA Agent"; }
    size_t get_mem() const override { return 0; }
    bool is_double() const override { return true; }
    bool is_lim_double() const override { return true; }
    bool is_unified() const override { return true; }
    bool is_emulated() const override { return false; }

    void* create(size_t count, struct rw_info* key) override {
        void *data = aligned_alloc(0x1000, count);
        hsa_memory_register(data, count);
        return data;
    }
    
    void release(void *ptr, struct rw_info* key ) override {
        hsa_memory_deregister(ptr, key->count);
        ::operator delete(ptr);
    }

    void* CreateKernel(const char* fun, void* size, void* source, bool needsCompilation = true) override {
        std::string str(fun);
        HSAKernel *kernel = programs[str];
        if (!kernel) {
            size_t kernel_size = (size_t)((void *)size);
            char *kernel_source = (char*)malloc(kernel_size+1);
            memcpy(kernel_source, source, kernel_size);
            kernel_source[kernel_size] = '\0';
            std::string kname = std::string("&")+fun;
            //std::cerr << "HSADevice::CreateKernel(): Creating kernel: " << kname << "\n";
            if (needsCompilation) {
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
        HSADispatch *dispatch = new HSADispatch(agent, kernel);
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

    std::shared_ptr<KalmarQueue> createQueue() override {
        std::shared_ptr<KalmarQueue> q =  std::shared_ptr<KalmarQueue>(new HSAQueue(this, agent));
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

    hsa_region_t& getHSAKernargRegion() {
    
        hsa_agent_iterate_regions(agent, &HSADevice::get_memory_regions, &ri);

        return ri._kernarg_region;
    }

    hsa_region_t& getHSAAMRegion() {
    
        hsa_agent_iterate_regions(agent, &HSADevice::get_memory_regions, &ri);

        // prefer coarse-grained over fine-grained
        if (ri._found_coarsegrained_region) {
            ri._am_region = ri._coarsegrained_region;
        } else if (ri._found_finegrained_region) {
            ri._am_region = ri._finegrained_region;
        } else {
            ri._am_region.handle = (uint64_t)(-1);
        }
    
        return ri._am_region;
    }

private:

    HSAKernel* CreateOfflineFinalizedKernelImpl(void *kernelBuffer, int kernelSize, const char *entryName) {
        hsa_status_t status;

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

        // Get symbol handle.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol(hsaExecutable, NULL, entryName, agent, 0, &kernelSymbol);
        STATUS_CHECK(status, __LINE__);

        // Get code handle.
        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle);
        STATUS_CHECK(status, __LINE__);

        return new HSAKernel(hsaExecutable, code_object, kernelSymbol, kernelCodeHandle);
    }

    HSAKernel* CreateKernelImpl(const char *hsailBuffer, int hsailSize, const char *entryName) {
        hsa_status_t status;
  
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
  
        hsa_code_object_t hsaCodeObject = {0};
        status = hsa_ext_program_finalize(hsaProgram, isa, 0, control_directives,
                                          NULL, HSA_CODE_OBJECT_TYPE_PROGRAM, &hsaCodeObject);
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
  
        // Get symbol handle.
        hsa_executable_symbol_t kernelSymbol;
        status = hsa_executable_get_symbol(hsaExecutable, NULL, entryName, agent, 0, &kernelSymbol);
        STATUS_CHECK(status, __LINE__);
  
        // Get code handle.
        uint64_t kernelCodeHandle;
        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle);
        STATUS_CHECK(status, __LINE__);
  
        return new HSAKernel(hsaExecutable, hsaCodeObject, kernelSymbol, kernelCodeHandle);
    }

};

class HSAContext final : public KalmarContext
{
    /// Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
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
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
            STATUS_CHECK(status, __LINE__);
            if (device_type == HSA_DEVICE_TYPE_GPU) {
                printf("GPU HSA agent: %s\n", name);
            } else if (device_type == HSA_DEVICE_TYPE_CPU) {
                printf("CPU HSA agent: %s\n", name);
            } else {
                printf("DSP HSA agent: %s\n", name);
            }
        }
#endif

        if (device_type == HSA_DEVICE_TYPE_GPU) {
            pAgents->push_back(agent);
        }

        return HSA_STATUS_SUCCESS;
    }

public:
    HSAContext() : KalmarContext() {
        // initialize HSA runtime
#if KALMAR_DEBUG
        std::cerr << "HSAContext::HSAContext(): init HSA runtime\n";
#endif
        hsa_status_t status;
        status = hsa_init();
        STATUS_CHECK(status, __LINE__);

        // Iterate over the agents
        std::vector<hsa_agent_t> agents;
        status = hsa_iterate_agents(&HSAContext::find_gpu, &agents);
        STATUS_CHECK(status, __LINE__);

        for (int i = 0; i < agents.size(); ++i) {
            hsa_agent_t agent = agents[i];
            auto Dev = new HSADevice(agent);
            if (i == 0)
                def = Dev;
            Devices.push_back(Dev);
        }
    }

    ~HSAContext() {
        // destroy all KalmarDevices associated with this context
        for (auto dev : Devices)
            delete dev;
        Devices.clear();
        def = nullptr;

        // shutdown HSA runtime
#if KALMAR_DEBUG
        std::cerr << "HSAContext::~HSAContext(): shut down HSA runtime\n";
#endif
        hsa_status_t status;
        status = hsa_shut_down();
        STATUS_CHECK(status, __LINE__);
    }
};

static HSAContext ctx;

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
HSAQueue::getHSAKernargRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAKernargRegion()));
}

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSADispatch
// ----------------------------------------------------------------------

// wait for the kernel to finish execution
inline hsa_status_t
HSADispatch::waitComplete() {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    if (!isDispatched)  {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

#if KALMAR_DEBUG
    std::cerr << "wait for kernel dispatch completion...\n";
#endif

    // wait for completion
    if (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)!=0) {
        printf("Signal wait returned unexpected value\n");
        exit(0);
    }

#if KALMAR_DEBUG
    std::cerr << "complete!\n";
#endif

    hsa_memory_deregister((void*)aql.kernarg_address, arg_vec.size());

    // unregister this async operation from HSAQueue
    if (this->hsaQueue != nullptr) {
        this->hsaQueue->removeAsyncOp(this);
    }

    isDispatched = false;
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
    std::cerr << "wait for barrier completion...\n";
#endif

    // Wait on completion signal until the barrier is finished
    hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

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

