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
#include "Brig_new.hpp"
#include "elf_utils.hpp"

#define STATUS_CHECK(s,line) if (status != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", status, line);\
                assert(HSA_STATUS_SUCCESS == hsa_close());\
		exit(-1);\
	}

#define STATUS_CHECK_Q(s,line) if (status != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", status, line);\
                assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(commandQueue));\
                assert(HSA_STATUS_SUCCESS == hsa_close());\
		exit(-1);\
	}

static hsa_status_t IterateAgent(hsa_agent_t agent, void *data) {
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
#if 0
                uint32_t d;
                hsa_dim3_t dim;
                uint32_t c;
                uint16_t a[3];
                uint32_t b;
                stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &d);
                stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_DIM, &dim);
                stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_SIZE, &c);
                stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &a);
                stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &b);

                printf("HSA_AGENT_INFO_WORKGROUP_MAX_DIM: (%u,%u,%u), HSA_AGENT_INFO_WORKGROUP_MAX_SIZE: %u\n", a[0], a[1], a[2], b);
                printf("HSA_AGENT_INFO_GRID_MAX_DIM: (%u,%u,%u), HSA_AGENT_INFO_GRID_MAX_SIZE: %u\n", dim.x, dim.y, dim.z, c);
                printf("HSA_AGENT_INFO_WAVEFRONT_SIZE: %u\n", d);
#endif

		return HSA_STATUS_SUCCESS;
}


bool FindSymbolOffset(hsa_ext_brig_module_t* brig_module, 
    const char* symbol_name,
    hsa_ext_brig_code_section_offset32_t& offset) {
    
    //Get the data section
     hsa_ext_brig_section_header_t* data_section_header = 
                brig_module->section[HSA_EXT_BRIG_SECTION_DATA];
    //Get the code section
     hsa_ext_brig_section_header_t* code_section_header =
             brig_module->section[HSA_EXT_BRIG_SECTION_CODE];

    //First entry into the BRIG code section
    BrigCodeOffset32_t code_offset = code_section_header->header_byte_count;
    BrigBase* code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
    size_t symbol_size = strlen(symbol_name);

    while (code_offset != code_section_header->byte_count) {
        if (code_entry->kind == BRIG_KIND_DIRECTIVE_KERNEL) {
            //Now find the data in the data section
            BrigDirectiveExecutable* directive_kernel = (BrigDirectiveExecutable*) (code_entry);
            BrigDataOffsetString32_t data_name_offset = directive_kernel->name;
            BrigData* data_entry = (BrigData*)((char*) data_section_header + data_name_offset);
            size_t data_entry_size = strlen((char*)data_entry->bytes);

            if (data_entry_size == symbol_size
                && strncmp(symbol_name, (char*)data_entry->bytes, symbol_size)==0) {
                offset = code_offset;
                return true;
            }

            // A HSAIL assembler has a bug that may generate symbol names 
            // with extra characters appended to the end, this is a workaround to that bug
            if (data_entry_size > symbol_size
                && strncmp(symbol_name, (char*)data_entry->bytes, symbol_size)==0) {
                offset = code_offset;
                return true;
            }

        }
        code_offset += code_entry->byteCount;
        code_entry = (BrigBase*) ((char*)code_section_header + code_offset);
    }
    return false;
}



hsa_status_t get_kernarg(hsa_region_t region, void* data) {
  hsa_region_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_FLAGS, &flags);
  if (flags & HSA_REGION_FLAG_KERNARG) {
    hsa_region_t * ret = (hsa_region_t *) data;
    *ret = region;
    //printf("found kernarg region: %08X\n", region);

    //size_t alloc_max_size;
    //hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &alloc_max_size);
    //printf("maximum allocate size: %lX\n", alloc_max_size);

    //size_t granule;
    //hsa_region_get_info(region, HSA_REGION_INFO_ALLOC_GRANULE,  &granule);
    //printf("minimum allocate size: %lX\n", granule);

    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
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
      hsa_ext_code_descriptor_t *hsaCodeDescriptor;
      friend class DispatchImpl;

   public:
      KernelImpl(hsa_ext_code_descriptor_t* _hsaCodeDescriptor, HSAContextKaveriImpl* _context) {
         context = _context;
         hsaCodeDescriptor =  _hsaCodeDescriptor;
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
      hsa_dispatch_packet_t aql;
      bool isDispatched;

   public:
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
         dispatchKernelAndGetFuture().wait();
         return status;
      } 


      std::shared_future<void> dispatchKernelAndGetFuture() {
         dispatchKernel();
         auto waitFunc = [&]() {
           this->waitComplete();
         };
         std::packaged_task<void()> waitTask(waitFunc);
         std::shared_future<void> fut = waitTask.get_future();
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

         // create a signal
         status = hsa_signal_create(1, 0, NULL, &signal);
         STATUS_CHECK_Q(status, __LINE__);

         // create a dispatch packet
         memset(&aql, 0, sizeof(aql));

         // setup dispatch sizes
         aql.completion_signal = signal;
         aql.dimensions = launchDimensions;
         aql.workgroup_size_x = workgroup_size[0];
         aql.workgroup_size_y = workgroup_size[1];
         aql.workgroup_size_z = workgroup_size[2];
         aql.grid_size_x = global_size[0];
         aql.grid_size_y = global_size[1];
         aql.grid_size_z = global_size[2];

         // set dispatch fences
         aql.header.type = HSA_PACKET_TYPE_DISPATCH;
         aql.header.acquire_fence_scope = 2;
         aql.header.release_fence_scope = 2;
         aql.header.barrier = 1;

         // bind kernel code
         aql.kernel_object_address = kernel->hsaCodeDescriptor->code.handle; 

         // bind kernel arguments
         //printf("arg_vec size: %d in bytes: %d\n", arg_vec.size(), arg_vec.size());
         hsa_region_t region;
         //printf("hsa_agent_iterate_regions\n");
         hsa_agent_iterate_regions(context->device, get_kernarg, &region);
         //printf("kernarg region: %08X\n", region);

         //printf("hsa_memory_allocate in region: %08X size: %d bytes\n", region, roundUp(arg_vec.size()));
         if ((status = hsa_memory_allocate(region, roundUp(arg_vec.size()), (void**) &aql.kernarg_address)) != HSA_STATUS_SUCCESS) {
           printf("hsa_memory_allocate error: %d\n", status);
           exit(1);
         }

         //printf("memcpy dst: %08X, src: %08X, %d kernargs, %d bytes\n", aql.kernarg_address, arg_vec.data(), arg_count, arg_vec.size());
         memcpy((void*)aql.kernarg_address, arg_vec.data(), arg_vec.size());
         //for (size_t i = 0; i < arg_vec.size(); ++i) {
         //  printf("%02X ", *(((uint8_t*)aql.kernarg_address)+i));
         //}
         //printf("\n");hsa_ext_brig_module_t

         // Initialize memory resources needed to execute
         aql.group_segment_size = kernel->hsaCodeDescriptor->workgroup_group_segment_byte_size;

         aql.private_segment_size = kernel->hsaCodeDescriptor->workitem_private_segment_byte_size;

         // write packet
         uint32_t queueMask = commandQueue->size - 1;
         uint64_t index = hsa_queue_load_write_index_relaxed(commandQueue);
         ((hsa_dispatch_packet_t*)(commandQueue->base_address))[index & queueMask] = aql;
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
         if (hsa_signal_wait_acquire(signal, HSA_LT, 1, uint64_t(-1), HSA_WAIT_EXPECTANCY_UNKNOWN)!=0) {
           printf("Signal wait returned unexpected value\n");
           exit(0);
         }

         //printf("complete!\n");

         hsa_memory_deregister((void*)aql.kernarg_address, roundUp(arg_vec.size()));
         free((void*)aql.kernarg_address);

         hsa_signal_store_relaxed(signal, 1);
         isDispatched = false;
         return status; 
      }

      void dispose() {
         hsa_status_t status;
         status = hsa_memory_deregister(arg_vec.data(), arg_vec.capacity() * sizeof(uint8_t));
         assert(status == HSA_STATUS_SUCCESS);
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
     hsa_ext_program_handle_t hsaProgram;

   // constructor
   HSAContextKaveriImpl() {
     hsa_status_t status;


     // initialize HSA runtime
     status = hsa_init();
     STATUS_CHECK(status, __LINE__);

     // device discovery
     device = 0;
	   status = hsa_iterate_agents(IterateAgent, &device);
     STATUS_CHECK(status, __LINE__);


     // create command queue
     size_t queue_size = 0;
     status = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
     STATUS_CHECK(status, __LINE__);

     commandQueue = NULL;
     status = hsa_queue_create(device, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, &commandQueue);
     STATUS_CHECK_Q(status, __LINE__);

     hsaProgram.handle = 0;
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

	    //Convert hsail kernel text to BRIG.
	    hsa_ext_brig_module_t* brigModule;
	    if (!CreateBrigModuleFromBrigMemory((char*)hsailBuffer, hsailSize, &brigModule)){
        STATUS_CHECK(status, __LINE__);
	    }

	    //Create hsa program.
	    status = hsa_ext_program_create(&device, 1, HSA_EXT_BRIG_MACHINE_LARGE, HSA_EXT_BRIG_PROFILE_FULL, &hsaProgram);
      STATUS_CHECK(status, __LINE__);

	    //Add BRIG module to hsa program.
	    hsa_ext_brig_module_handle_t module;
	    status = hsa_ext_add_module(hsaProgram, brigModule, &module);
      STATUS_CHECK(status, __LINE__);

	    // Construct finalization request list.
	    // @todo kzhuravl 6/16/2014 remove bare numbers, we actually need to find
	    // entry offset into the code section.
	    hsa_ext_finalization_request_t finalization_request_list;
      memset(&finalization_request_list, 0, sizeof(hsa_ext_finalization_request_t));
	    finalization_request_list.module = module;              // module handle.
	    finalization_request_list.program_call_convention = 0;  // program call convention. not supported.

	    if (!FindSymbolOffset(brigModule, entryName, finalization_request_list.symbol)){
        STATUS_CHECK(GENERIC_ERROR, __LINE__);
	    }

	    //Finalize hsa program.
	    status = hsa_ext_finalize_program(hsaProgram, device, 1, &finalization_request_list, NULL, NULL, 0, NULL, 0);
      STATUS_CHECK(status, __LINE__);

	    //Get hsa code descriptor address.
	    hsa_ext_code_descriptor_t *hsaCodeDescriptor;
	    status = hsa_ext_query_kernel_descriptor_address(hsaProgram, module, finalization_request_list.symbol, &hsaCodeDescriptor);
      STATUS_CHECK(status, __LINE__);

      return new KernelImpl(hsaCodeDescriptor, this); 
    }

   hsa_status_t dispose() {
      hsa_status_t status;

      if (hsaProgram.handle != 0) {
	      status = hsa_ext_program_destroy(hsaProgram);
        STATUS_CHECK(status, __LINE__);
      }

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

