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
#include <string>
#include <thread>
#include <vector>

#include <hsa.h>
#include <hsa_ext_finalize.h>

#include <kalmar_runtime.h>

#define KALMAR_DEBUG (0)

#define STATUS_CHECK(s,line) if (s != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

#define STATUS_CHECK_Q(s,line) if (s != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", s, line);\
                assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(commandQueue));\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
		exit(-1);\
	}

extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v);
extern "C" void PushArgPtrImpl(void *ker, int idx, size_t sz, const void *v);

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

class HSABarrier {
private:
    hsa_signal_t signal;
    bool isDispatched;

public:
    HSABarrier() : isDispatched(false) {}

    ~HSABarrier() {
#if KALMAR_DEBUG
        std::cerr << "HSABarrier::~HSABarrier()\n";
#endif
        if (isDispatched) {
            waitComplete();
        }
    }

    hsa_status_t enqueueBarrier(hsa_queue_t* queue) {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (isDispatched) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }

        status = hsa_signal_create(1, 0, NULL, &signal);
        STATUS_CHECK_Q(status, __LINE__);

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

    std::shared_future<void>* enqueueAndGetFuture(hsa_queue_t* queue) {
        enqueueBarrier(queue);

        // dynamically allocate a std::shared_future<void> object
        // it will be released in the private ctor of completion_future
        std::shared_future<void>* fut = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
            waitComplete();

#if KALMAR_DEBUG
          std::cerr << "destruct HSABarrier instance\n";
#endif

            delete(this); // destruct HSABarrier instance
        }).share());

        return fut;
    }

    // wait for the barrier to complete
    hsa_status_t waitComplete() {
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

        isDispatched = false;
        dispose();
        return status;
    }

    void dispose() {
        hsa_status_t status;
        status = hsa_signal_destroy(signal);
        STATUS_CHECK_Q(status, __LINE__);
    }

}; // end of HSABarrier

class HSADispatch {
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

public:
    ~HSADispatch() {
#if KALMAR_DEBUG
        std::cerr << "HSADispatch::~HSADispatch()\n";
#endif

        if (isDispatched) {
            waitComplete();
        }
    }

    hsa_status_t setDynamicGroupSegment(size_t dynamicGroupSize) {
        this->dynamicGroupSize = dynamicGroupSize;
        return HSA_STATUS_SUCCESS;
    }

    HSADispatch(hsa_agent_t _agent, const HSAKernel* _kernel) :
        agent(_agent),
        kernel(_kernel),
        isDispatched(false),
        dynamicGroupSize(0) {

        // allocate the initial argument vector capacity
        arg_vec.reserve(ARGS_VEC_INITIAL_CAPACITY);
        registerArgVecMemory();

        clearArgs();

        hsa_status_t status;

        /// Query the maximum number of work-items in a workgroup
        hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &workgroup_max_size);
        STATUS_CHECK_Q(status, __LINE__);

        /// Query the maximum number of work-items in each dimension of a workgroup
        hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &workgroup_max_dim);
        STATUS_CHECK_Q(status, __LINE__);
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
        while(workgroup_size[0] * workgroup_size[1] * workgroup_size[2] > workgroup_max_size) {
          // repeatedly cut each dimension into half until we are within the limit
          if (workgroup_size[dim_iterator] >= 2) {
            workgroup_size[dim_iterator] >>= 1;
          }
          if (--dim_iterator < 0) {
            dim_iterator = 2;
          }
        }
  
        return HSA_STATUS_SUCCESS;
    }

    hsa_status_t dispatchKernelWaitComplete(hsa_queue_t* _queue) {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (isDispatched) {
            return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        }
        dispatchKernel(_queue);
        waitComplete();
        return status;
    } 

    std::shared_future<void>* dispatchKernelAndGetFuture(hsa_queue_t* _queue) {
        dispatchKernel(_queue);

        // dynamically allocate a std::shared_future<void> object
        // it will be released in the private ctor of completion_future
        std::shared_future<void>* fut = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
          waitComplete();

#if KALMAR_DEBUG
          std::cerr << "destruct HSADispatch instance\n";
#endif

          delete(this);  // destruct HSADispatch instance
        }).share());

        return fut;
    }

    uint32_t getGroupSegmentSize() {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        uint32_t group_segment_size = 0;
        status = hsa_executable_symbol_get_info(kernel->hsaExecutableSymbol,
                                                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                                &group_segment_size);
        STATUS_CHECK_Q(status, __LINE__);
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
        STATUS_CHECK_Q(status, __LINE__);
  
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
        STATUS_CHECK_Q(status, __LINE__);

        // add dynamic group segment size
        group_segment_size += this->dynamicGroupSize;
        aql.group_segment_size = group_segment_size;
  
        uint32_t private_segment_size;
        status = hsa_executable_symbol_get_info(kernel->hsaExecutableSymbol,
                                                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                                &private_segment_size);
        STATUS_CHECK_Q(status, __LINE__);
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
    hsa_status_t waitComplete() {
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

        isDispatched = false;
        dispose();
        return status; 
    }

    void dispose() {
        hsa_status_t status;
        status = hsa_memory_deregister(arg_vec.data(), arg_vec.capacity() * sizeof(uint8_t));
        assert(status == HSA_STATUS_SUCCESS);
        hsa_signal_destroy(aql.completion_signal);
        clearArgs();
        std::vector<uint8_t>().swap(arg_vec);
    }

private:
    template <typename T>
    hsa_status_t pushArgPrivate(T val) {
        /* add padding if necessary */
        int padding_size = arg_vec.size() % sizeof(T);
        //printf("push %lu bytes into kernarg: ", sizeof(T) + padding_size);
        for (size_t i = 0; i < padding_size; ++i) {
            arg_vec.push_back((uint8_t)0x00);
            //printf("%02X ", (uint8_t)0x00);
        }
        uint8_t*ptr = static_cast<uint8_t*>(static_cast<void*>(&val));
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
        if (localSize == 0) localSize = recommendedSize;   // (globalSize > 16 ? 1 : globalSize);   
        localSize = std::min(localSize, recommendedSize);
  
        // Check if globalSize is a multiple of localSize
        // (this might be temporary until the runtime really does handle non-full last groups)
        int legalGroupSize = findLargestFactor(globalSize, localSize);
        //if (legalGroupSize != localSize) {
        //  std::cout << "WARNING: groupSize[" << level << "] reduced to " << legalGroupSize << std::endl;
        //}
  
        global_size[level] = globalSize;
        workgroup_size[level] = legalGroupSize;
        //std::cout << "level " << level << ", grid=" << global_size[level] 
        //          << ", group=" << workgroup_size[level] << std::endl;
    }

    // find largest factor less than or equal to start
    int findLargestFactor(int n, int start) {
        if (start > n) return n;
        for (int div = start; div >=1; div--) {
            if (n % div == 0) return div;
        }
        return 1;
    }

}; // end of HSADispatch

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
    // When a kernel k is dispatched, we'll get a future object f for k.
    // This vector would hold f.  acccelerator_view::wait() would trigger
    // HSAQueue::wait(), and all future objects in this vector will be waited
    // on.
    //
    std::vector< std::shared_future<void> > dispatches;

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
    // After kernel k is dispatched, we'll get a future object f for k, we then
    // walk through each buffer b used by k and mark the association as:
    // bufferKernelMap[b] = f
    //
    // Finally kernelBufferMap[k] will be cleared.
    //

    // association between buffers and kernel dispatches
    // key: buffer address
    // value: a vector of kernel dispatches
    std::map<void*, std::vector< std::shared_future<void> > > bufferKernelMap;

    // association between a kernel and buffers used by it
    // key: kernel
    // value: a vector of buffers used by the kernel
    std::map<void*, std::vector<void*> > kernelBufferMap;

public:
    HSAQueue(KalmarDevice* pDev, hsa_agent_t agent) : KalmarQueue(pDev), commandQueue(nullptr), dispatches(), bufferKernelMap(), kernelBufferMap() {
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
        STATUS_CHECK_Q(status, __LINE__);
    }

    ~HSAQueue() {
        hsa_status_t status;

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

    void wait() override {
      // wait on all previous dispatches to complete
      for (int i = 0; i < dispatches.size(); ++i) {
        // wait on valid futures only
        if (dispatches[i].valid()) {
          dispatches[i].wait();
        }
      }
      // clear previous dispatched kernel table
      dispatches.clear();
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
                        for (int i = 0; i < bufferKernelMap[buffer].size(); ++i) {
                          bufferKernelMap[buffer][i].wait();
                        }
                        bufferKernelMap[buffer].clear();
                      });

        // dispatch the kernel
        // and wait for its completion
        dispatch->dispatchKernelWaitComplete(commandQueue);

        // clear data in kernelBufferMap
        kernelBufferMap[ker].clear();

        delete(dispatch);
    }

    void* LaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        return LaunchKernelWithDynamicGroupMemoryAsync(ker, nr_dim, global, local, 0);
    }

    void* LaunchKernelWithDynamicGroupMemoryAsync(void *ker, size_t nr_dim, size_t *global, size_t *local, size_t dynamic_group_size) override {
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
                        for (int i = 0; i < bufferKernelMap[buffer].size(); ++i) {
                          bufferKernelMap[buffer][i].wait();
                        }
                        bufferKernelMap[buffer].clear();
                      });

        // dispatch the kernel
        std::shared_future<void>* fut = dispatch->dispatchKernelAndGetFuture(commandQueue);

        // associate the kernel dispatch with this queue
        dispatches.push_back(*fut);

        // associate all buffers used by the kernel with the kernel dispatch instance
        std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                      [&] (void* buffer) {
                        bufferKernelMap[buffer].push_back(*fut);
                      });

        // clear data in kernelBufferMap
        kernelBufferMap[ker].clear();

        return static_cast<void*>(fut);
    }

    uint32_t GetGroupSegmentSize(void *ker) override {
        HSADispatch *dispatch = reinterpret_cast<HSADispatch*>(ker);
        return dispatch->getGroupSegmentSize();
    }

    void read(void* device, void* dst, size_t count, size_t offset) override {
        // wait on previous kernel dispatches to complete
        for (int i = 0; i < bufferKernelMap[device].size(); ++i) {
          bufferKernelMap[device][i].wait();
        }
        bufferKernelMap[device].clear();

        // do read
        if (dst != device)
            memmove(dst, (char*)device + offset, count);
    }

    void write(void* device, const void* src, size_t count, size_t offset, bool blocking) override {
        // wait on previous kernel dispatches to complete
        for (int i = 0; i < bufferKernelMap[device].size(); ++i) {
          bufferKernelMap[device][i].wait();
        }
        bufferKernelMap[device].clear();

        // do write
        if (src != device)
            memmove((char*)device + offset, src, count);
    }

    void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) override {
        // wait on previous kernel dispatches to complete
        for (int i = 0; i < bufferKernelMap[dst].size(); ++i) {
          bufferKernelMap[dst][i].wait();
        }
        bufferKernelMap[dst].clear();

        // wait on previous kernel dispatches to complete
        for (int i = 0; i < bufferKernelMap[src].size(); ++i) {
          bufferKernelMap[src][i].wait();
        }
        bufferKernelMap[src].clear();

        // do copy
        if (src != dst)
            memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
    }

    void* map(void* device, size_t count, size_t offset, bool modify) override {
        // wait on previous kernel dispatches to complete
        for (int i = 0; i < bufferKernelMap[device].size(); ++i) {
          bufferKernelMap[device][i].wait();
        }
        bufferKernelMap[device].clear();

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

    bool hasHSAInterOp() override {
        return true;
    }

    // enqueue a barrier packet
    void* EnqueueMarker() {
        // HSABarrier instance will be deleted after the future object created is waited on
        HSABarrier *barrier = new HSABarrier();

        // enqueue the barrier
        std::shared_future<void>* fut = barrier->enqueueAndGetFuture(commandQueue);

        // associate the barrier with this queue
        dispatches.push_back(*fut);

        return static_cast<void*>(fut);
    }
};

class HSADevice final : public KalmarDevice
{
private:
    std::map<std::string, HSAKernel *> programs;
    hsa_agent_t agent;
    size_t max_tile_static_size;

public:
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

    hsa_agent_t getAgent() {
        return agent;
    }

    HSADevice(hsa_agent_t a) : KalmarDevice(access_type_read_write),
                               agent(a), programs(), max_tile_static_size(0) {
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
        return std::shared_ptr<KalmarQueue>(new HSAQueue(this, agent));
    }

    size_t GetMaxTileStaticSize() override {
        return max_tile_static_size;
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

