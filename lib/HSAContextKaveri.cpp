#include <regex.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

#include "HSAContext.h"
#include "fileUtils.h"

#define HSAIL_STATUS_CHECK(s,line) if (status != HSA_STATUS_SUCCESS) {\
          printf("### Error: %d at line:%d\n", status, line);\
          printf("### HSAIL =\n%s\n", fixedHsail);\
          exit(-1);\
        }
    
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

HSAContext* HSAContext::m_pContext = 0;

class HSAContextKaveriImpl : public HSAContext {
   friend HSAContext * HSAContext::Create(); 

private:
   class KernelImpl : public HSAContext::Kernel {
   private:
      HSAContextKaveriImpl* context;
      hsa_kernel_code_t* kernel;

      std::vector<uint64_t> arg_vec;
      uint32_t arg_count;
      size_t prevArgVecCapacity;
      int launchDimensions;
      uint32_t workgroup_size[3];
      uint32_t global_size[3];
      static const int ARGS_VEC_INITIAL_CAPACITY = 256;   

   public:
      KernelImpl(hsa_kernel_code_t* _kernel, HSAContextKaveriImpl* _context) {
         context = _context;
         kernel = _kernel;
         
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
         hsa_signal_handle_t signal;
         hsa_signal_value_t value;
         value.value64 = 1;
         status = hsa_signal_create(value, &signal, context->getContext());
         STATUS_CHECK_Q(status, __LINE__);

         // create a dispatch packet
         hsa_aql_dispatch_packet_t aql;
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
         aql.header.format = HSA_AQL_FORMAT_DISPATCH;
         aql.header.acquire_fence_scope = 2;
         aql.header.release_fence_scope = 2;
         aql.header.barrier = 1;

         // bind kernel code
         aql.kernel_object_address = (uint64_t)kernel; 

         // bind kernel arguments
         aql.kernarg_address = (uint64_t)arg_vec.data();

         // Initialize memory resources needed to execute
         aql.group_segment_size_bytes = kernel->workgroup_group_segment_byte_size;
         aql.private_segment_size_bytes = kernel->workitem_private_segment_byte_size;

         // write packet
         uint32_t queueMask = commandQueue->size_packets - 1;
         uint64_t index = hsa_queue_get_write_index(commandQueue);
         ((hsa_aql_dispatch_packet_t*)(commandQueue->base_address))[index & queueMask] = aql;
         hsa_queue_set_write_index(commandQueue, index + 1);

         // Ring door bell
         value.value64 = index + 1;
         status = hsa_signal_send_relaxed(commandQueue->doorbell_signal, value);
         STATUS_CHECK_Q(status, __LINE__);

         // wait for completion
         value.value64 = 1;
         status = hsa_signal_wait_acquire(signal, -1, HSA_LT, value, NULL);
         STATUS_CHECK_Q(status, __LINE__);

         value.value64 = 1;
         hsa_signal_send_relaxed(signal, value);

         return status; 
      }

      void dispose() {
         hsa_status_t status;
         status = hsa_memory_deregister(arg_vec.data());
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
     hsa_runtime_context_t* context;
     hsa_topology_table_t* table;
     hsa_agent_t* device;
     hsa_queue_t* commandQueue;

     KernelImpl *kernelImpl;

   // constructor
   HSAContextKaveriImpl() {
     hsa_status_t status;

     // initialize HSA runtime
     status = hsa_open(&context);
     STATUS_CHECK(status, __LINE__);

     // device discovery
     status = hsa_topology_table_create(&table);
     STATUS_CHECK(status, __LINE__);

     assert(table->number_agents && "No HSA devices found!\n");

     // use the first HSA device
     device = ((hsa_agent_t*)((size_t)table->topology_table_base+table->agent_offset_list_bytes[0]));

     // create command queue
     status = hsa_queue_create(device, device->queue_size, HSA_QUEUE_TYPE_IN_ORDER, context, &commandQueue);
     STATUS_CHECK(status, __LINE__);

     kernelImpl = NULL;
   }

public:
    hsa_queue_t* getQueue() {
        return commandQueue;
    }

    hsa_runtime_context_t* getContext() {
        return context;
    }

    Kernel * createKernel(const char *hsailBuffer, const char *entryName) {
        hsa_status_t status;
        hsa_kernel_code_t* kernel;

        // "Fix" HSAIL
        //printf("Kernel entry is %s\n", entryName);
        //const char* fixedHsail = (fixHsail(hsailBuffer))->c_str();
        //printf("fixedHsail:\n");
        //printf("%s\n", fixedHsail);

        // Get program by hsail compile path.
        //status = hsa_finalize_hsail(device, fixedHsail, entryName, &kernel);
        status = hsa_finalize_hsail(device, hsailBuffer, entryName, &kernel);
        STATUS_CHECK(status, __LINE__);

        return new KernelImpl(kernel, this);
    }
   
   hsa_status_t dispose() {
      hsa_status_t status;

      if (kernelImpl) {
         kernelImpl->dispose();
      }

      status = hsa_queue_destroy(commandQueue);
      STATUS_CHECK(status, __LINE__);

      status = hsa_topology_table_destroy(table);
      STATUS_CHECK(status, __LINE__);

      status = hsa_close(context);
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

private:
#define MAX_MATCHES 5 //The maximum number of matches allowed in a single string
#define LINE 1024

    static char *getMatch(char *line, regmatch_t *matches, int i) {
        char value[256];
        int len = matches[i].rm_eo - matches[i].rm_so;
        strncpy(value, line + matches[i].rm_so, len);
        value[len] = '\0';
        return (strdup(value));
    }

   char *get_regerror (int errcode, regex_t *compiled) {
      size_t length = regerror (errcode, compiled, NULL, 0);
      char *buffer = (char*) malloc (length);
      (void) regerror (errcode, compiled, buffer, length);
      return buffer;
   }

    std::string* fixHsail(const char *hsailStr) {
        int rv;
        int origLength = strlen(hsailStr);
        int extraSize = 256;
        char* fixedFenceOpsString = new char[origLength + extraSize];
        memset(fixedFenceOpsString, 0, origLength + extraSize);

        // This part is to fix 1.0 spec fenced load/store until the 
        // public version simulator supports v1.0 instructions.
        // st_global_rel_u32 into atomicnoret_st_rel_sys_b32
        // ld_global_acq_u32 into atomic_ld_acq_sys_b32
        if (true) {
            char line[origLength];
            char* startPtr = (char*) hsailStr;
            int i = 0;
            int oldi = 0;
            while ((i < origLength) && (strlen(&(hsailStr[i])) != 0)) {
                // collect the instruction line including ';'
                //i = strcspn(&(hsailStr[i]), ";");
                i = strcspn(&(hsailStr[i]), "\n");
                memcpy(line, &(hsailStr[oldi]), ++i);
                line[i] = 0;
                oldi += i;
                i = oldi;

                // There might be trailing newlines etc after the 
                // final ';' so don't process them here and add one later
                char* semicolonPtr = strchr(line, ';');
                if (semicolonPtr == 0) {
                    //printf("##### No ; match in length=%ld: [[%s]]\n", strlen(line), line);
                    // Add whatever this is to the output, probably comments
                    strcat(fixedFenceOpsString, line);
                    continue;
                }

                //printf("### Next line:: %s[END]\n", line);

                // Find ptr to instruction
                if ((startPtr = strstr(line, "ld_global_acq_u")) != NULL) {
                    //printf("found: %s\n", line);

                    // print the first part before the match
                    long unmatchedLength = (long) startPtr - (long) line;
                    strncat(fixedFenceOpsString, line, unmatchedLength);

                    regex_t exp2;
                    //rv = regcomp(&exp2, "ld_global_acq_u([0-9]+) (.*), (.*);", REG_EXTENDED);
                    rv = regcomp(&exp2, "ld_global_acq_u([0-9]+) (.*), (.*);(.*)", REG_EXTENDED);
                    if (rv != 0) {
                        printf("regcomp 2 failed with %d\n", rv);
                        return new std::string("fixHsail FAILED");
                    }
                    regmatch_t matches[MAX_MATCHES]; //A list of the matches in the string (a list of 1)
                    if (regexec(&exp2, startPtr, MAX_MATCHES, matches, 0) == 0) {
                        char* size = getMatch(startPtr, matches, 1);
                        char* reg = getMatch(startPtr, matches, 2);
                        char* addr = getMatch(startPtr, matches, 3);
                        char* after = getMatch(startPtr, matches, 4);
                        char* newInstruction = new char[strlen(line) + 32];

                        sprintf(newInstruction, "atomic_ld_acq_sys_b%s %s, %s;%s", size, reg, addr, after);
                        strcat(fixedFenceOpsString, newInstruction);
                        //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                        delete newInstruction;
                    }
                    regfree(&exp2);
                } else if ((startPtr = strstr(line, "st_global_rel_u")) != NULL) {
                    //printf("found: [[%s]]\n", line);

                    // print the first part before the match
                    long unmatchedLength = (long) startPtr - (long) line;
                    strncat(fixedFenceOpsString, line, unmatchedLength);

                    regex_t exp2;
                    rv = regcomp(&exp2, "st_global_rel_u([0-9]+) (\\$[sd]+[0-9]*), (\\[\\$[A-Za-z0-9]* \\+ [A-Za-z0-9]*\\]);(.*)", REG_EXTENDED);
                    if (rv != 0) {
                        printf("regcomp 2 failed with %d\n", rv);
                        return new std::string("fixHsail FAILED");
                    }
                    regmatch_t matches[MAX_MATCHES]; //A list of the matches in the string (a list of 1)
                    if (regexec(&exp2, startPtr, MAX_MATCHES, matches, 0) == 0) {
                        char* size = getMatch(startPtr, matches, 1);
                        char* reg = getMatch(startPtr, matches, 2);
                        char* addr = getMatch(startPtr, matches, 3);
                        char* after = getMatch(startPtr, matches, 4);
                        char* newInstruction = new char[strlen(line) + 32];

                        sprintf(newInstruction, "atomicnoret_st_rel_sys_b%s %s, %s;%s", size, addr, reg, after);
                        strcat(fixedFenceOpsString, newInstruction);
                        //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                        delete newInstruction;
                    }
                    regfree(&exp2);
                } else if ((startPtr = strstr(line, "sync")) != NULL) { //touch up sync to memfence
                     long unmatchedLength = (long) startPtr - (long) line;
                     strncat(fixedFenceOpsString, line, unmatchedLength);
               
                     regex_t exp;
                     rv = regcomp(&exp, "sync;(.*)", REG_EXTENDED);
                     if(rv != 0) {
                        printf("regcomp failed while fixing sync with %d\n", rv);
                        return new std::string("fixHsail FAILED");
                     }
                     regmatch_t matches[MAX_MATCHES];
                     if (regexec(&exp, startPtr, MAX_MATCHES, matches, 0) == 0) {
                         char* comment = getMatch(startPtr, matches, 1);
                         char* newInstruction = new char[strlen(line) + 32];

                         sprintf(newInstruction, "memfence_ar_sys;%s", comment);
                         strcat(fixedFenceOpsString, newInstruction);
                         //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                         delete newInstruction;
                     }
                     else {
                         printf("regexec failed while fixing sync\n");
                         regfree(&exp);
                         return new std::string("fixHsail FAILED");
                     }
                     regfree(&exp);
                } else if ((startPtr = strstr(line, "atomic")) != NULL) { //touch up for atomic_*
                     //do necessary for each type of op
                     if((startPtr = strstr(line, "atomic_cas")) != NULL) {
                         //printf("found atomic_cas: [[%s]]\n", startPtr);
                         long unmatchedLength = (long) startPtr - (long) line;
                         strncat(fixedFenceOpsString, line, unmatchedLength);

                         regex_t exp;
                         rv = regcomp(&exp, "atomic_cas_global_b([0-9]+)(.*)", REG_EXTENDED);
                         if (rv != 0) {
                             printf("regcomp failed while fixing atomic_cas with %d\n", rv);
                             return new std::string("fixHsail FAILED");
                         }
                         regmatch_t matches[MAX_MATCHES]; //A list of the matches in the string (a list of 1)
                         if ((rv = regexec(&exp, startPtr, MAX_MATCHES, matches, 0)) == 0) {
                             char* size = getMatch(startPtr, matches, 1);
                             char* rest = getMatch(startPtr, matches, 2);
                             char* newInstruction = new char[strlen(line) + 32];

                             sprintf(newInstruction, "atomic_cas_global_ar_sys_b%s%s", size, rest);
                             strcat(fixedFenceOpsString, newInstruction);
                             //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                             delete newInstruction;
                         } 
						 else {
                             printf("regexec failed while fixing atomic_cas with %d\n", rv);
                             char *err = get_regerror (rv, &exp);
                             printf("error msg:%s\n", err);
                             regfree(&exp);
                             return new std::string("fixHsail FAILED");
                         } 
                         regfree(&exp);
                     }
                     else if((startPtr = strstr(line, "atomic_add")) != NULL) { //fix atomic_add
                         //printf("found atomic_add: [[%s]]\n", startPtr);
                         long unmatchedLength = (long) startPtr - (long) line;
                         strncat(fixedFenceOpsString, line, unmatchedLength);

                         regex_t exp;
                         rv = regcomp(&exp, "atomic_add_global_([bus][0-9]+)(.*)", REG_EXTENDED);
                         if (rv != 0) {
                             printf("regcomp failed while fixing atomic_add with %d\n", rv);
                             return new std::string("fixHsail FAILED");
                         }
                          regmatch_t matches[MAX_MATCHES]; //A list of the matches in the string (a list of 1)
                          if ((rv = regexec(&exp, startPtr, MAX_MATCHES, matches, 0)) == 0) {
                              char* size = getMatch(startPtr, matches, 1);
                              char* rest = getMatch(startPtr, matches, 2);
                              char* newInstruction = new char[strlen(line) + 32];
 
                              sprintf(newInstruction, "atomic_add_global_ar_sys_%s%s", size, rest);
                              strcat(fixedFenceOpsString, newInstruction);
                              //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                              delete newInstruction;
                          }
                          else {
                              printf("regexec failed while fixing atomic_add with %d\n", rv);
                              char *err = get_regerror (rv, &exp);
                              printf("error msg:%s\n", err);
                              regfree(&exp);
                              return new std::string("fixHsail FAILED");
                          }
                          regfree(&exp); 
                     }
                     else if((startPtr = strstr(line, "atomic_exch")) != NULL) { //fix atomic_exch
                         //printf("found atomic_exch: [[%s]]\n", startPtr);
                         long unmatchedLength = (long) startPtr - (long) line;
                         strncat(fixedFenceOpsString, line, unmatchedLength);

                         regex_t exp;
                         rv = regcomp(&exp, "atomic_exch_global_([bus][0-9]+)(.*)", REG_EXTENDED);
                         if (rv != 0) {
                             printf("regcomp failed while fixing atomic_exch with %d\n", rv);
                             return new std::string("fixHsail FAILED");
                         }
                          regmatch_t matches[MAX_MATCHES]; //A list of the matches in the string (a list of 1)
                          if ((rv = regexec(&exp, startPtr, MAX_MATCHES, matches, 0)) == 0) {
                              char* size = getMatch(startPtr, matches, 1);
                              char* rest = getMatch(startPtr, matches, 2);
                              char* newInstruction = new char[strlen(line) + 32];
 
                              sprintf(newInstruction, "atomic_exch_global_ar_sys_%s%s", size, rest);
                              strcat(fixedFenceOpsString, newInstruction);
                              //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                              delete newInstruction;
                          }
                          else {
                              printf("regexec failed while fixing atomic_exch with %d\n", rv);
                              char *err = get_regerror (rv, &exp);
                              printf("error msg:%s\n", err);
                              regfree(&exp);
                              return new std::string("fixHsail FAILED");
                          }
                          regfree(&exp); 
                     }
                     else {
                         strcat(fixedFenceOpsString, line);
					 }
                } else if ((startPtr = strstr(line, "barrier_fgroup")) != NULL) { //touch up barrier_fgroup to barrier
                     long unmatchedLength = (long) startPtr - (long) line;
                     strncat(fixedFenceOpsString, line, unmatchedLength);
  
                     regex_t exp;
                     rv = regcomp(&exp, "barrier_fgroup;(.*)", REG_EXTENDED);
                     if(rv != 0) {
                        printf("regcomp failed while fixing barrier_fgroup with %d\n", rv);
                        return new std::string("fixHsail FAILED");
                     }
                     regmatch_t matches[MAX_MATCHES];
                     if (regexec(&exp, startPtr, MAX_MATCHES, matches, 0) == 0) {
                        char* comment = getMatch(startPtr, matches, 1);
                        char* newInstruction = new char[strlen(line) + 32];
  
                        sprintf(newInstruction, "barrier;%s", comment);
                        strcat(fixedFenceOpsString, newInstruction);
                        //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                        delete newInstruction;
                     }
                     else {
                        printf("regexec failed while fixing barrier_fgroup\n");
                        regfree(&exp);
                        return new std::string("fixHsail FAILED");
                     }
                     regfree(&exp);
                } else if ((startPtr = strstr(line, "barrier_sys")) != NULL) { //touch up barrier_sys to barrier
                     long unmatchedLength = (long) startPtr - (long) line;
                     strncat(fixedFenceOpsString, line, unmatchedLength);
  
                     regex_t exp;
                     rv = regcomp(&exp, "barrier_sys;(.*)", REG_EXTENDED);
                     if(rv != 0) {
                        printf("regcomp failed while fixing barrier_sys with %d\n", rv);
                        return new std::string("fixHsail FAILED");
                     }
                     regmatch_t matches[MAX_MATCHES];
                     if (regexec(&exp, startPtr, MAX_MATCHES, matches, 0) == 0) {
                        char* comment = getMatch(startPtr, matches, 1);
                        char* newInstruction = new char[strlen(line) + 32];
  
                        sprintf(newInstruction, "barrier;%s", comment);
                        strcat(fixedFenceOpsString, newInstruction);
                        //printf("### Replaced %s[WITH]%s[END]\n", startPtr, newInstruction);
                        delete newInstruction;
                     }
                     else {
                        printf("regexec failed while fixing barrier_fgroup\n");
                        regfree(&exp);
                        return new std::string("fixHsail FAILED");
                     }
                     regfree(&exp);
               } else {
                    //printf("##### No regex match in length=%ld: [[%s]]\n", strlen(line), line);
                    strcat(fixedFenceOpsString, line);
                }
            }
        }

        // Add a trailing newline
        strcat(fixedFenceOpsString, "\n");
        //printf("original hsail string length = %ld\n", strlen(hsailStr));
        //printf("fixedFenceOpsString length = %ld\n", strlen(fixedFenceOpsString));
        //printf("fixed hsail = \n%s\n", fixedFenceOpsString);

        std::string *s = new std::string(fixedFenceOpsString);

        // the following conversions should no longer be needed because newer hsailasm is being used
        if (false) {
            // conversions are from 0.95 Spec format to MCW assembler format
            // version string, we don't really handle the non-$full models here 
            // also this should be made more regex based so we don't depend on whitespace
            replaceAll(*s, "0:95: $full : $large", "1:0");
            replaceAll(*s, "0:95: $full : $small", "1:0:$small");
            // workitemabsid mnemonic
            replaceAll(*s, "workitemabsid_u32", "workitemabsid");
            // mul_hi mnemonic
            replaceAll(*s, "mulhi", "mul_hi");

            // MCW assembler is pick about DOS line endings
            replaceAll(*s, "\r", "");
        }
        if (false) {
            // the following were the conversions from June 2012 format into MCW assembler format
            replaceAll(*s, "1:0:large", "1:0");
            replaceAll(*s, "1:0:small", "1:0:$small");
            replaceAll(*s, "workitemaid", "workitemabsid");
            replaceAll(*s, "cvt_near_f64_f32", "cvt_f64_f32");
        }
        return s;
    }
}; // end of HSAContextKaveriImpl

// Create an instance thru the HSAContext interface
HSAContext* HSAContext::Create() {
   if(!m_pContext)
      m_pContext = new HSAContextKaveriImpl();
  
   return m_pContext;
}

