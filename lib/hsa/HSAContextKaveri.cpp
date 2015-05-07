#include <regex.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <future>
#include <thread>
#include <chrono>

#include "HSAContext.h"

#include "hsa.h"
#include "hsa_ext_finalize.h"

//#define KALMAR_DEBUG (1)

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

/*
 * Determines if the given agent is of type HSA_DEVICE_TYPE_GPU
 * and sets the value of data to the agent handle if it is.
 */
static hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
  // Find GPU device and use it.
  if (data == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  hsa_device_type_t device_type;
  hsa_status_t stat =
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }
  if (device_type == HSA_DEVICE_TYPE_GPU) {
    *((hsa_agent_t *)data) = agent;
  }
  return HSA_STATUS_SUCCESS;
}

// Levenshtein Distance to measure the difference of two sequences
// The shortest distance it returns the more likely the two sequences are equal
static inline int ldistance(const std::string source, const std::string target)
{
  int n = source.length();
  int m = target.length();
  if (m == 0)
    return n;
  if (n == 0)
    return m;

  //Construct a matrix
  typedef std::vector < std::vector < int >>Tmatrix;
  Tmatrix matrix(n + 1);

  for (int i = 0; i <= n; i++)
    matrix[i].resize(m + 1);
  for (int i = 1; i <= n; i++)
    matrix[i][0] = i;
  for (int i = 1; i <= m; i++)
    matrix[0][i] = i;

  for (int i = 1; i <= n; i++) {
    const char si = source[i - 1];
    for (int j = 1; j <= m; j++) {
      const char dj = target[j - 1];
      int cost;
      if (si == dj)
        cost = 0;
      else
        cost = 1;
      const int above = matrix[i - 1][j] + 1;
      const int left = matrix[i][j - 1] + 1;
      const int diag = matrix[i - 1][j - 1] + cost;
      matrix[i][j] = std::min(above, std::min(left, diag));
    }
  }
  return matrix[n][m];
}

size_t roundUp(size_t size) {
  size_t times = size / 0x1000;
  size_t rem = size % 0x1000;
  if (rem != 0) ++times;
  return times * 0x1000;
}

HSAContext* HSAContext::m_pContext = 0;

class HSAContextKaveriImpl : public HSAContext {
   friend HSAContext * HSAContext::Create(); 

private:

   class DispatchImpl;

   class KernelImpl : public HSAContext::Kernel {
   private:
      HSAContextKaveriImpl* context;
      hsa_code_object_t hsaCodeObject;
      hsa_executable_t hsaExecutable;
      uint64_t kernelCodeHandle;
      hsa_executable_symbol_t hsaExecutableSymbol;
      friend class DispatchImpl;

   public:
      KernelImpl(hsa_executable_t _hsaExecutable,
                 hsa_code_object_t _hsaCodeObject,
                 hsa_executable_symbol_t _hsaExecutableSymbol,
                 uint64_t _kernelCodeHandle,
                 HSAContextKaveriImpl* _context) {
         hsaExecutable = _hsaExecutable;
         hsaCodeObject = _hsaCodeObject;
         hsaExecutableSymbol = _hsaExecutableSymbol;
         kernelCodeHandle = _kernelCodeHandle;
         context = _context;
      }

      ~KernelImpl() {
         hsa_status_t status;

         status = hsa_executable_destroy(hsaExecutable);
         STATUS_CHECK(status, __LINE__);

         status = hsa_code_object_destroy(hsaCodeObject);
         STATUS_CHECK(status, __LINE__);
      }

   }; // end of KernelImpl


   class DispatchImpl : public HSAContext::Dispatch {
   private:
      HSAContextKaveriImpl* context;
      const KernelImpl* kernel;

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
      ~DispatchImpl() {
         if (isDispatched) {
           waitComplete();
           dispose();
         }
      }

      DispatchImpl(const KernelImpl* _kernel) : kernel(_kernel), isDispatched(false) {
         context = _kernel->context;
         
         // allocate the initial argument vector capacity
         arg_vec.reserve(ARGS_VEC_INITIAL_CAPACITY);
         registerArgVecMemory();

         clearArgs();
      }

      hsa_status_t pushFloatArg(float f) {
         return pushArgPrivate(f);
      }
      
      hsa_status_t pushIntArg(int i) {
         return pushArgPrivate(i);
      }
      
      hsa_status_t pushBooleanArg(unsigned char z) {
         return pushArgPrivate(z);
      }
      
      hsa_status_t pushByteArg(char b) {
         return pushArgPrivate(b);
      }
      
      hsa_status_t pushLongArg(long j) {
         return pushArgPrivate(j);
      }

      hsa_status_t pushDoubleArg(double d) {
         return pushArgPrivate(d);
      }
      
      
      hsa_status_t pushPointerArg(void *addr) {
         return pushArgPrivate(addr);
      }

      hsa_status_t clearArgs() {
         arg_count = 0;
         arg_vec.clear();
         return HSA_STATUS_SUCCESS;
      }

       hsa_status_t setLaunchAttributes(int dims, size_t *globalDims, size_t *localDims) {
         assert((0 < dims) && (dims <= 3));

         // defaults
         workgroup_size[1] = workgroup_size[2] = global_size[1] = global_size[2] = 1;
         launchDimensions = dims;

         switch (dims) {
           case 1:
             // according to the hsa folks, this is 256 for all current targets
             computeLaunchAttr(0, globalDims[0], localDims[0], 256);
             break;
           case 2:
             // according to some experiments, 64 * 32 (2048 workitems) is the best configuration
             computeLaunchAttr(0, globalDims[0], localDims[0], 64);
             computeLaunchAttr(1, globalDims[1], localDims[1], 32);
             break;
           case 3:
             // according to some experiments, 32 * 32 * 2 (2048 workitems) is the best configuration
             computeLaunchAttr(0, globalDims[0], localDims[0], 32);
             computeLaunchAttr(1, globalDims[1], localDims[1], 32);
             computeLaunchAttr(2, globalDims[2], localDims[2], 2);
             break;
         }

         return HSA_STATUS_SUCCESS;
      }

      hsa_status_t dispatchKernelWaitComplete() {
         hsa_status_t status = HSA_STATUS_SUCCESS;
         if (isDispatched) {
           return HSA_STATUS_ERROR_INVALID_ARGUMENT;
         }
         dispatchKernel();
         waitComplete();
         return status;
      } 


      std::shared_future<void>* dispatchKernelAndGetFuture() {
         dispatchKernel();
         auto waitFunc = [&]() {
           this->waitComplete();
           delete(this); // destruct DispatchImpl instance
         };
         std::packaged_task<void()> waitTask(waitFunc);

         // dynamically allocate a std::shared_future<void> object
         // it will be released in the private ctor of completion_future
         std::shared_future<void>* fut = new std::shared_future<void>(waitTask.get_future());

         std::thread waitThread(std::move(waitTask));
         waitThread.detach();         
         return fut;
      }

      // dispatch a kernel asynchronously
      hsa_status_t dispatchKernel() {
         hsa_status_t status = HSA_STATUS_SUCCESS;
         if (isDispatched) {
           return HSA_STATUS_ERROR_INVALID_ARGUMENT;
         }

         // check if underlying arg_vec data might have changed, if so re-register
         if (arg_vec.capacity() > prevArgVecCapacity) {
            registerArgVecMemory();
         }

         // get command queue from context
         hsa_queue_t* commandQueue = context->getQueue();

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

         hsa_memory_deregister((void*)aql.kernarg_address, roundUp(arg_vec.size()));

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


   }; // end of DispatchImpl

   private:
     hsa_agent_t device;
     hsa_queue_t* commandQueue;

   // constructor
   HSAContextKaveriImpl() {
     hsa_status_t status;

     // initialize HSA runtime
     status = hsa_init();
     STATUS_CHECK(status, __LINE__);

     /* 
      * Iterate over the agents and pick the gpu agent using 
      * the find_gpu callback.
      */
     device = {0};
     status = hsa_iterate_agents(find_gpu, &device);
     STATUS_CHECK(status, __LINE__);

#ifdef KALMAR_DEBUG
{
     char name[64];
     status = hsa_agent_get_info(device, HSA_AGENT_INFO_NAME, name);
     STATUS_CHECK(status, __LINE__);
     printf("using HSA agent %s\n", name);
}
#endif

     /*
      * Query the maximum size of the queue.
      */
     size_t queue_size = 0;
     status = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
     STATUS_CHECK(status, __LINE__);

     /*
      * Create a queue using the maximum size.
      */
     commandQueue = NULL;
     status = hsa_queue_create(device, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 
                               UINT32_MAX, UINT32_MAX, &commandQueue);
     STATUS_CHECK_Q(status, __LINE__);
   }

public:

    hsa_agent_t* getDevice() {
      return &device;
    }

    hsa_queue_t* getQueue() {
      return commandQueue;
    }

    Dispatch* createDispatch(const Kernel* kernel) {
      return new DispatchImpl((const KernelImpl*)kernel);
    }

    Kernel* createKernel(const char *hsailBuffer, int hsailSize, const char *entryName) {

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
      status = hsa_agent_get_info(device, HSA_AGENT_INFO_ISA, &isa);
      STATUS_CHECK(status, __LINE__);

      hsa_ext_control_directives_t control_directives;
      memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));

      hsa_code_object_t hsaCodeObject = {0};
      status = hsa_ext_program_finalize(hsaProgram, isa, 0, control_directives,
                                        "", HSA_CODE_OBJECT_TYPE_PROGRAM, &hsaCodeObject);
      STATUS_CHECK(status, __LINE__);

      if (hsaProgram.handle != 0) {
        status = hsa_ext_program_destroy(hsaProgram);
        STATUS_CHECK(status, __LINE__);
      }

      // Create the executable.
      hsa_executable_t hsaExecutable;
      status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                     "", &hsaExecutable);
      STATUS_CHECK(status, __LINE__);

      // Load the code object.
      status = hsa_executable_load_code_object(hsaExecutable, device, hsaCodeObject, "");
      STATUS_CHECK(status, __LINE__);

      // Freeze the executable.
      status = hsa_executable_freeze(hsaExecutable, "");
      STATUS_CHECK(status, __LINE__);

      // Get symbol handle.
      hsa_executable_symbol_t kernelSymbol;
      status = hsa_executable_get_symbol(hsaExecutable, "", entryName, device, 0, &kernelSymbol);
      STATUS_CHECK(status, __LINE__);

      // Get code handle.
      uint64_t kernelCodeHandle;
      status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle);
      STATUS_CHECK(status, __LINE__);

      return new KernelImpl(hsaExecutable, hsaCodeObject, kernelSymbol, kernelCodeHandle, this);
    }

   hsa_status_t dispose() {
      hsa_status_t status;

      status = hsa_queue_destroy(commandQueue);
      STATUS_CHECK(status, __LINE__);

      status = hsa_shut_down();

      return HSA_STATUS_SUCCESS;
   }

   hsa_status_t registerArrayMemory(void *addr, int lengthInBytes) {
      // std::cout << "HSA::registerArrayMemory: " << addr << ", " << lengthInBytes << std::endl;
      return hsa_memory_register(addr, lengthInBytes);
   }

   // destructor
   ~HSAContextKaveriImpl() {
     dispose();
   }

}; // end of HSAContextKaveriImpl

// Create an instance thru the HSAContext interface
HSAContext* HSAContext::Create() {
   if(!m_pContext)
      m_pContext = new HSAContextKaveriImpl();
  
   return m_pContext;
}

