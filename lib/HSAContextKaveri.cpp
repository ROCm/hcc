#include <regex.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

#include "HSAContext.h"
#include "fileUtils.h"

#include "hsa_ext_finalize.h"
#include "hsa_ext_private_amd.h"

#define STATUS_CHECK(s,line) if (status != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", status, line);\
		exit(-1);\
	}

#define STATUS_CHECK_Q(s,line) if (status != HSA_STATUS_SUCCESS) {\
		printf("### Error: %d at line:%d\n", status, line);\
                assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(commandQueue));\
		exit(-1);\
	}

/************************ From HSA example *********************************/ 

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

		return HSA_STATUS_SUCCESS;
	}

	static hsa_status_t IterateRegion(hsa_region_t region, void *data) {
		// Find system memory region.
		if (data == NULL) {
			return HSA_STATUS_ERROR_INVALID_ARGUMENT;
		}

		bool is_host = false;
		hsa_status_t stat =
			hsa_region_get_info(
			region, (hsa_region_info_t)HSA_EXT_REGION_INFO_HOST_ACCESS, &is_host);
		if (stat != HSA_STATUS_SUCCESS) {
			return stat;
		}

		if (is_host) {
			*((hsa_region_t *)data) = region;
		}
		return HSA_STATUS_SUCCESS;
	}

/**************************************************************************/


/*************************  From assemble.cpp *****************************/

// from assemble.cpp

#include <iostream>

//#include "assemble.h"
#include "hsail_c.h"
#include "hsa.h"
#include "hsa_ext_finalize.h"
#include <string>

enum SymbolType {
    GLOBAL_READ_SYMBOLS=0,
    KERNEL_SYMBOLS
};

#include "HSAILItems.h"
#include <assert.h>
using namespace Brig;
using namespace HSAIL_ASM;

int alignUp (int x, int number) {
    return x + (number - x%number);
}

void print_brig(hsa_ext_brig_module_t* brig_module){
    std::cout<<"Number of sections:"<<brig_module->section_count<<std::endl;
    for (int i=0; i<brig_module->section_count;i++) {
        hsa_ext_brig_section_header_t* section_header = brig_module->section[i];
        std::cout<<"Name:"<<(char*)section_header->name<<std::endl;
        std::cout<<"Header size:"<<section_header->header_byte_count<<std::endl;
        std::cout<<"Total size:"<<section_header->byte_count<<std::endl;
    }
}
bool CreateBrigModule(const char* kernel_source, hsa_ext_brig_module_t** brig_module_t){
    brig_container_t c = brig_container_create_empty();
    if (brig_container_assemble_from_memory(c, kernel_source, strlen(kernel_source))) { // or use brig_container_assemble_from_file
        printf("error assembling:%s\n", brig_container_get_error_text(c)); 
        brig_container_destroy(c);
        return false;
    }
    // \todo 1.0p: allow brig_container_t to manage the memory and just use brig_container_get_brig_module(c).
    uint32_t number_of_sections = brig_container_get_section_count(c);
    hsa_ext_brig_module_t* brig_module;
    brig_module = (hsa_ext_brig_module_t*)
                (malloc (sizeof(hsa_ext_brig_module_t) + sizeof(void*)*number_of_sections));
    brig_module->section_count = number_of_sections;
    for(int i=0; i < number_of_sections; ++i) {
        //create new section header
        uint64_t size_section_bytes = brig_container_get_section_size(c, i);
        void* section_copy = malloc(size_section_bytes);
        //copy the section data
        memcpy ((char*)section_copy,
            brig_container_get_section_bytes(c, i),
            size_section_bytes);
        brig_module->section[i] = (hsa_ext_brig_section_header_t*) section_copy;
    }
    //print_brig(brig_module);
    *brig_module_t = brig_module;
    brig_container_destroy(c);
    return true;
}

bool DestroyBrigModule(hsa_ext_brig_module_t* brig_module) {
     for (int i=0; i<brig_module->section_count;i++) {
        hsa_ext_brig_section_header_t* section_header = brig_module->section[i];
        free(section_header);
     }
     free (brig_module);
     return true;
}

char* GetSectionAndSize(hsa_ext_brig_module_t* brig_module, 
    int section_id, int* size) {
    hsa_ext_brig_section_header_t* section_header =
        brig_module->section[section_id];
    char* section_data = (char*)section_header + section_header->header_byte_count;
    int section_data_size = section_header->byte_count - 
        section_header->header_byte_count;
    *size = section_data_size;
    return section_data;
}

bool FindSymbolOffset(hsa_ext_brig_module_t* brig_module, 
    std::string symbol_name,SymbolType symbol_type, hsa_ext_brig_code_section_offset32_t& offset) {
        //Create a BRIG container
        BrigContainer c((Brig::BrigModule*) brig_module);
        Code first_d = c.code().begin();
        Code last_d = c.code().end();

        for (;first_d != last_d;first_d = first_d.next()) {
            switch (symbol_type) {
            case GLOBAL_READ_SYMBOLS :

                if (DirectiveVariable sym = first_d) {
                    if ((sym.segment() == BRIG_SEGMENT_GLOBAL) ||
                        (sym.segment() == BRIG_SEGMENT_READONLY)) {
                            std::string variable_name = (SRef)sym.name();
                            if (variable_name == symbol_name) {
                                offset = sym.brigOffset();
                                return true;
                            }
                    }
                }
                break;
            case KERNEL_SYMBOLS :
                if (DirectiveExecutable de = first_d) {
                    if (symbol_name == de.name()) {
                        offset = de.brigOffset();
                        return true;
                    }
                }
                break;
            default:
                return false;
            }
        }
        return false;
}


/**************************************************************************/


HSAContext* HSAContext::m_pContext = 0;

class HSAContextKaveriImpl : public HSAContext {
   friend HSAContext * HSAContext::Create(); 

private:
   class KernelImpl : public HSAContext::Kernel {
   private:
      HSAContextKaveriImpl* context;
      hsa_ext_code_descriptor_t *hsaCodeDescriptor;

      std::vector<uint64_t> arg_vec;
      uint32_t arg_count;
      size_t prevArgVecCapacity;
      int launchDimensions;
      uint32_t workgroup_size[3];
      uint32_t global_size[3];
      static const int ARGS_VEC_INITIAL_CAPACITY = 256;   

   public:
      KernelImpl(hsa_ext_code_descriptor_t* _hsaCodeDescriptor, HSAContextKaveriImpl* _context) {
         context = _context;
         hsaCodeDescriptor =  _hsaCodeDescriptor;
         
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

      // allow a previously pushed arg to be changed
      hsa_status_t setPointerArg(int idx, void *addr) {
         assert (idx < arg_count);
         return setArgPrivate(idx, addr);
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

         for (int k = 0; k < dims; ++k) {
            computeLaunchAttr(k, globalDims[k], localDims[k]);
         }
         return HSA_STATUS_SUCCESS;
      }

      hsa_status_t dispatchKernelWaitComplete() {
         hsa_status_t status = HSA_STATUS_SUCCESS;

         // check if underlying arg_vec data might have changed, if so re-register
         if (arg_vec.capacity() > prevArgVecCapacity) {
            registerArgVecMemory();
         }

         // get command queue from context
         hsa_queue_t* commandQueue = context->getQueue();

         // create a signal
         hsa_signal_t signal;
         status = hsa_signal_create(1, 1, context->getDevice(), &signal);
         STATUS_CHECK_Q(status, __LINE__);

         // create a dispatch packet
         hsa_dispatch_packet_t aql;
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
         aql.kernel_object_address = hsaCodeDescriptor->code.handle; 

         // bind kernel arguments
         aql.kernarg_address = (uint64_t)arg_vec.data();

         // Initialize memory resources needed to execute
         aql.group_segment_size = hsaCodeDescriptor->workgroup_group_segment_byte_size;
         aql.private_segment_size = hsaCodeDescriptor->workitem_private_segment_byte_size;

         // write packet
         uint32_t queueMask = commandQueue->size - 1;
         uint64_t index = hsa_queue_load_write_index_relaxed(commandQueue);
         ((hsa_dispatch_packet_t*)(commandQueue->base_address))[index & queueMask] = aql;
         hsa_queue_store_write_index_relaxed(commandQueue, index + 1);

         // Ring door bell
         hsa_signal_store_relaxed(commandQueue->doorbell_signal, index+1);

         // wait for completion
         signal = hsa_signal_wait_acquire(signal, HSA_LT, 1, -1, HSA_WAIT_EXPECTANCY_UNKNOWN);
         STATUS_CHECK_Q(status, __LINE__);

         hsa_signal_store_relaxed(signal, 1);

         return status; 
      }

      void dispose() {
         hsa_status_t status;
         status = hsa_memory_deregister(arg_vec.data(), arg_vec.capacity() * sizeof(uint64_t));
         assert(status == HSA_STATUS_SUCCESS);
      }

   private:
      template <typename T>
      hsa_status_t pushArgPrivate(T val) {
         // each arg takes up a 64-bit slot, no matter what its size
         const uint64_t  argAsU64 = 0;
         T* pt = (T *) &argAsU64;
         *pt = val;
         arg_vec.push_back(argAsU64);
         arg_count++;
         return HSA_STATUS_SUCCESS;
      }

      template <typename T>
      hsa_status_t setArgPrivate(int idx, T val) {
         // each arg takes up a 64-bit slot, no matter what its size
         uint64_t  argAsU64 = 0;
         T* pt = (T *) &argAsU64;
         *pt = val;
         arg_vec.at(idx) = argAsU64;    
         return HSA_STATUS_SUCCESS;
      }


      void registerArgVecMemory() {
         // record current capacity to compare for changes
         prevArgVecCapacity = arg_vec.capacity();

         // register the memory behind the arg_vec
         hsa_status_t status = hsa_memory_register(arg_vec.data(), arg_vec.capacity() * sizeof(uint64_t));
         assert(status == HSA_STATUS_SUCCESS);
      }

      void computeLaunchAttr(int level, int globalSize, int localSize) {
         // localSize of 0 means pick best
         // according to the hsa folks, this is 256 for all current targets
         if (localSize == 0) localSize = 256;   // (globalSize > 16 ? 1 : globalSize);   
         localSize = std::min(localSize, 256);

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


   }; // end of KernelImpl

   private:
     hsa_agent_t device;
     hsa_queue_t* commandQueue;
     hsa_ext_program_handle_t hsaProgram;
     KernelImpl *kernelImpl;

   // constructor
   HSAContextKaveriImpl() {
     hsa_status_t status;

     // initialize HSA runtime
     // device discovery
     device = 0;
	   status = hsa_iterate_agents(IterateAgent, &device);
     STATUS_CHECK(status, __LINE__);


     // create command queue
     size_t queue_size = 0;
     status = hsa_agent_get_info(device, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
     STATUS_CHECK(status, __LINE__);

     status = hsa_queue_create(device, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, &commandQueue);
     STATUS_CHECK_Q(status, __LINE__);

     kernelImpl = NULL;
   }

public:

    hsa_agent_t* getDevice() {
      return &device;
    }

    hsa_queue_t* getQueue() {
      return commandQueue;
    }

    Kernel * createKernel(const char *hsailBuffer, const char *entryName) {

      hsa_status_t status;

	    //Convert hsail kernel text to BRIG.
	    hsa_ext_brig_module_t* brigModule;
	    if (!CreateBrigModule(hsailBuffer, &brigModule)){
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
	    finalization_request_list.module = module;              // module handle.
	    finalization_request_list.symbol = 192;                 // entry offset into the code section.
	    finalization_request_list.program_call_convention = 0;  // program call convention. not supported.

	    if (!FindSymbolOffset(brigModule, entryName, KERNEL_SYMBOLS, finalization_request_list.symbol)){
        STATUS_CHECK(GENERIC_ERROR, __LINE__);
	    }

	    //Finalize hsa program.
	    status = hsa_ext_finalize_program(hsaProgram, &device, 1, &finalization_request_list, NULL, NULL, 0, NULL, 0);
      STATUS_CHECK(status, __LINE__);

	    //Get hsa code descriptor address.
	    hsa_ext_code_descriptor_t *hsaCodeDescriptor;
	    status = hsa_ext_query_kernel_descriptor_address(hsaProgram, module, finalization_request_list.symbol, &hsaCodeDescriptor);
      STATUS_CHECK(status, __LINE__);


      return new KernelImpl(hsaCodeDescriptor, this);
    }

   hsa_status_t dispose() {
      hsa_status_t status;

      if (kernelImpl) {
         kernelImpl->dispose();
      }

	    status = hsa_ext_program_destroy(hsaProgram);
      STATUS_CHECK(status, __LINE__);

      status = hsa_queue_destroy(commandQueue);
      STATUS_CHECK(status, __LINE__);

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

