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

#include <amp_runtime.h>

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

      status = hsa_executable_destroy(hsaExecutable);
      STATUS_CHECK(status, __LINE__);

      status = hsa_code_object_destroy(hsaCodeObject);
      STATUS_CHECK(status, __LINE__);
   }
}; // end of HSAKernel

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

public:
    ~HSADispatch() {
        if (isDispatched) {
            waitComplete();
            dispose();
        }
    }

    HSADispatch(hsa_agent_t _agent, const HSAKernel* _kernel) :
        agent(_agent),
        kernel(_kernel),
        isDispatched(false) {

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
          delete(this);  // destruct HSADispatch instance
        }));

        return fut;
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
  
        //printf("ring door bell\n");
  
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

        //printf("wait for completion...");

        // wait for completion
        if (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), HSA_WAIT_STATE_ACTIVE)!=0) {
            printf("Signal wait returned unexpected value\n");
            exit(0);
        }

        //printf("complete!\n");

        hsa_memory_deregister((void*)aql.kernarg_address, arg_vec.size());

        hsa_signal_store_relaxed(signal, 1);
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
namespace Concurrency {


class HSAQueue final : public KalmarQueue
{
private:
    hsa_queue_t* commandQueue;

public:
    HSAQueue(KalmarDevice* pDev, hsa_agent_t agent) : KalmarQueue(pDev), commandQueue(nullptr) {
        hsa_status_t status;

        /// Query the maximum size of the queue.
        size_t queue_size = 0;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
        STATUS_CHECK(status, __LINE__);

        /// Create a queue using the maximum size.
        status = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 
                                  UINT32_MAX, UINT32_MAX, &commandQueue);
#if KALMAR_DEBUG
        std::cerr << "HSAQueue::HSAQueue(): created an HSA command queue: " << commandQueue << "\n";
#endif
        STATUS_CHECK_Q(status, __LINE__);
    }

    ~HSAQueue() {
        hsa_status_t status;

#if KALMAR_DEBUG
        std::cerr << "HSAQueue::~HSAQueue(): destroy an HSA command queue: " << commandQueue << "\n";
#endif
        status = hsa_queue_destroy(commandQueue);
        STATUS_CHECK(status, __LINE__);
    }

    void LaunchKernel(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        HSADispatch *dispatch =
            reinterpret_cast<HSADispatch*>(ker);
        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;
        dispatch->setLaunchAttributes(nr_dim, global, local);
        dispatch->dispatchKernelWaitComplete(commandQueue);
        delete(dispatch);
    }

    void* LaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        HSADispatch *dispatch =
            reinterpret_cast<HSADispatch*>(ker);
        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;

        //std::cerr<<"Launching: nr dim = " << nr_dim << "\n";
        //for (size_t i = 0; i < nr_dim; ++i) {
        //  std::cerr << "g: " << global[i] << " l: " << local[i] << "\n";
        //}
        dispatch->setLaunchAttributes(nr_dim, global, local);
        std::shared_future<void>* fut = dispatch->dispatchKernelAndGetFuture(commandQueue);
        return static_cast<void*>(fut);
    }

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

    void unmap(void* device, void* addr) override {}

    void Push(void *kernel, int idx, void *device, bool isConst) override {
        PushArgImpl(kernel, idx, sizeof(void*), &device);
    }
};

class HSADevice final : public KalmarDevice
{
private:
    std::map<std::string, HSAKernel *> programs;
    hsa_agent_t agent;

public:
    hsa_agent_t getAgent() {
        return agent;
    }

    HSADevice(hsa_agent_t a) : KalmarDevice(access_type_read_write),
                               agent(a), programs() {
#if KALMAR_DEBUG
        std::cerr << "HSADevice::HSADevice()\n";
#endif
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

} // namespace Concurrency


extern "C" void *GetContextImpl() {
  return &Concurrency::ctx;
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

