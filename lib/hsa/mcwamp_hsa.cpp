//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Kalmar Runtime implementation (HSA version)

#include "../hc2/headers/types/program_state.hpp"

#include <algorithm>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>
#include <sys/syscall.h>

#ifndef USE_LIBCXX
#include <cxxabi.h>
#endif

#include <hsa/hsa.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa_ven_amd_loader.h>

#include "kalmar_runtime.h"
#include "kalmar_aligned_alloc.h"

#include "hc_am_internal.hpp"
#include "unpinned_copy_engine.h"
#include "hc_rt_debug.h"
#include "hc_printf.hpp"

#include "activity_prof.h"

#include <time.h>
#include <iomanip>

#define CHECK_OLDER_COMPLETE 0

// Used to mark pieces of HCC runtime which may be specific to AMD's HSA implementation.
// Intended to help identify this code when porting to another HSA impl.
// FIXME - The AMD_HSA path is experimental and should not be enabled in production
// #define AMD_HSA


/////////////////////////////////////////////////
// kernel dispatch speed optimization flags
/////////////////////////////////////////////////

// size of default kernarg buffer in the kernarg pool in HSAContext
#define KERNARG_BUFFER_SIZE (512)

// number of pre-allocated kernarg buffers in HSAContext
// Not required but typically should be greater than HCC_SIGNAL_POOL_SIZE 
// (some kernels don't allocate signals but nearly all need kernargs)
#define KERNARG_POOL_SIZE (1024)


// Maximum number of inflight commands sent to a single queue.
// If limit is exceeded, HCC will force a queue wait to reclaim
// resources (signals, kernarg)
// MUST be a power of 2.
#define MAX_INFLIGHT_COMMANDS_PER_QUEUE  (2*8192)

// threshold to clean up finished kernel in HSAQueue.asyncOps
int HCC_ASYNCOPS_SIZE = (2*8192);


//---
// Environment variables:
int HCC_PRINT_ENV=0;

int HCC_SIGNAL_POOL_SIZE=512;

int HCC_UNPINNED_COPY_MODE = UnpinnedCopyEngine::UseStaging;

int HCC_CHECK_COPY=0;

// Copy thresholds, in KB.  These are used for "choose-best" copy mode.
long int HCC_H2D_STAGING_THRESHOLD    = 64;
long int HCC_H2D_PININPLACE_THRESHOLD = 4096;
long int HCC_D2H_PININPLACE_THRESHOLD = 1024;

// Staging buffer size in KB for unpinned copy engines
int HCC_STAGING_BUFFER_SIZE = 4*1024;

// Default GPU device
unsigned int HCC_DEFAULT_GPU = 0;

unsigned int HCC_ENABLE_PRINTF = 0;

// Chicken bits:
int HCC_SERIALIZE_KERNEL = 0;
int HCC_SERIALIZE_COPY = 0;
int HCC_FORCE_COMPLETION_FUTURE = 0;
int HCC_FORCE_CROSS_QUEUE_FLUSH=0;

int HCC_OPT_FLUSH=1;


unsigned HCC_DB = 0;
unsigned HCC_DB_SYMBOL_FORMAT=0x10;

int HCC_MAX_QUEUES = 20;


#define HCC_PROFILE_SUMMARY (1<<0)
#define HCC_PROFILE_TRACE   (1<<1)
int HCC_PROFILE=0;


#define HCC_PROFILE_VERBOSE_BASIC                   (1 << 0)   // 0x1
#define HCC_PROFILE_VERBOSE_TIMESTAMP               (1 << 1)   // 0x2
#define HCC_PROFILE_VERBOSE_OPSEQNUM                (1 << 2)   // 0x4
#define HCC_PROFILE_VERBOSE_TID                     (1 << 3)   // 0x8
#define HCC_PROFILE_VERBOSE_BARRIER                 (1 << 4)   // 0x10
int HCC_PROFILE_VERBOSE=0x1F;



char * HCC_PROFILE_FILE=nullptr;

// Profiler:
// Use str::stream so output is atomic wrt other threads:
#define LOG_PROFILE(op, start, end, type, tag, msg) \
{\
    std::stringstream sstream;\
    sstream << "profile: " << std::setw(7) << type << ";\t" \
                         << std::setw(40) << tag\
                         << ";\t" << std::fixed << std::setw(6) << std::setprecision(1) << (end-start)/1000.0 << " us;";\
    if (HCC_PROFILE_VERBOSE & (HCC_PROFILE_VERBOSE_TIMESTAMP)) {\
            sstream << "\t" << start << ";\t" << end << ";";\
    }\
    if (HCC_PROFILE_VERBOSE & (HCC_PROFILE_VERBOSE_OPSEQNUM)) {\
            sstream << "\t" << *op << ";";\
    }\
   sstream <<  msg << "\n";\
   Kalmar::ctx.getHccProfileStream() << sstream.str();\
}

// Track a short thread-id, for debugging:
std::atomic<int> s_lastShortTid(1);

ShortTid::ShortTid() {
    _shortTid = s_lastShortTid.fetch_add(1);
}


thread_local ShortTid hcc_tlsShortTid;



#define HSA_BARRIER_DEP_SIGNAL_CNT (5)


// synchronization for copy commands in the same stream, regardless of command type.
// Add a signal dependencies between async copies -
// so completion signal from prev command used as input dep to next.
// If FORCE_SIGNAL_DEP_BETWEEN_COPIES=0 then data copies of the same kind (H2H, H2D, D2H, D2D)
// are assumed to be implicitly ordered.
// ROCR 1.2 runtime implementation currently provides this guarantee when using SDMA queues and compute shaders.
#define FORCE_SIGNAL_DEP_BETWEEN_COPIES (0)

#define CASE_STRING(X)  case X: case_string = #X ;break;

static const char* getHcCommandKindString(Kalmar::hcCommandKind k) {
    const char* case_string;

    switch(k) {
        using namespace Kalmar;
        CASE_STRING(hcCommandInvalid);
        CASE_STRING(hcMemcpyHostToHost);
        CASE_STRING(hcMemcpyHostToDevice);
        CASE_STRING(hcMemcpyDeviceToHost);
        CASE_STRING(hcMemcpyDeviceToDevice);
        CASE_STRING(hcCommandKernel);
        CASE_STRING(hcCommandMarker);
        default: case_string = "Unknown command type";
    };
    return case_string;
};



static const char* getHSAErrorString(hsa_status_t s) {

    const char* case_string;
    switch(s) {
        CASE_STRING(HSA_STATUS_ERROR);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_ARGUMENT);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_QUEUE_CREATION);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_ALLOCATION);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_AGENT);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_REGION);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_SIGNAL);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_QUEUE);
        CASE_STRING(HSA_STATUS_ERROR_OUT_OF_RESOURCES);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_PACKET_FORMAT);
        CASE_STRING(HSA_STATUS_ERROR_RESOURCE_FREE);
        CASE_STRING(HSA_STATUS_ERROR_NOT_INITIALIZED);
        CASE_STRING(HSA_STATUS_ERROR_REFCOUNT_OVERFLOW);
        CASE_STRING(HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_INDEX);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_ISA);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_ISA_NAME);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_EXECUTABLE);
        CASE_STRING(HSA_STATUS_ERROR_FROZEN_EXECUTABLE);
        CASE_STRING(HSA_STATUS_ERROR_INVALID_SYMBOL_NAME);
        CASE_STRING(HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED);
        CASE_STRING(HSA_STATUS_ERROR_VARIABLE_UNDEFINED);
        CASE_STRING(HSA_STATUS_ERROR_EXCEPTION);
        default: case_string = "Unknown Error Code";
    };
    return case_string;
}

#define STATUS_CHECK(s,line) if (s != HSA_STATUS_SUCCESS && s != HSA_STATUS_INFO_BREAK) {\
    hc::print_backtrace(); \
    const char* error_string = getHSAErrorString(s);\
        printf("### HCC STATUS_CHECK Error: %s (0x%x) at file:%s line:%d\n", error_string, s, __FILENAME__, line);\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
        abort();\
    }

#define STATUS_CHECK_SYMBOL(s,symbol,line) if (s != HSA_STATUS_SUCCESS && s != HSA_STATUS_INFO_BREAK) {\
    hc::print_backtrace(); \
    const char* error_string = getHSAErrorString(s);\
        printf("### HCC STATUS_CHECK_SYMBOL Error: %s (0x%x), symbol name:%s at file:%s line:%d\n", error_string, s, (symbol)!=nullptr?symbol:(const char*)"is a nullptr", __FILENAME__, line);\
                assert(HSA_STATUS_SUCCESS == hsa_shut_down());\
        abort();\
    }


// debug function to dump information on an HSA agent
static void dumpHSAAgentInfo(hsa_agent_t agent, const char* extra_string = (const char*)"") {
  hsa_status_t status;
  char name[64] = {0};
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  STATUS_CHECK(status, __LINE__);

  uint32_t node = 0;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
  STATUS_CHECK(status, __LINE__);

  wchar_t path_wchar[128] {0};
  swprintf(path_wchar, 128, L"%s%u", name, node);


  DBSTREAM << "Dump Agent Info (" << extra_string << ")" << std::endl;
  DBSTREAM << "\t Agent: ";
  DBWSTREAM  << path_wchar << L"\n";

  return;
}

static unsigned extractBits(unsigned v, unsigned pos, unsigned w)
{
    return (v >> pos) & ((1 << w) - 1);
};


namespace
{
    struct Symbol {
        std::string name;
        ELFIO::Elf64_Addr value = 0;
        ELFIO::Elf_Xword size = 0;
        ELFIO::Elf_Half sect_idx = 0;
        std::uint8_t bind = 0;
        std::uint8_t type = 0;
        std::uint8_t other = 0;
    };

    inline
    Symbol read_symbol(
        const ELFIO::symbol_section_accessor& section, unsigned int idx)
    {
        assert(idx < section.get_symbols_num());

        Symbol r;
        section.get_symbol(
            idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

        return r;
    }

    template<typename P>
    inline
    ELFIO::section* find_section_if(ELFIO::elfio& reader, P p)
    {
        using namespace std;

        const auto it = find_if(
            reader.sections.begin(), reader.sections.end(), move(p));

        return it != reader.sections.end() ? *it : nullptr;
    }

    inline
    std::vector<std::string> copy_names_of_undefined_symbols(
        const ELFIO::symbol_section_accessor& section)
    {
        using namespace ELFIO;
        using namespace std;

        vector<string> r;

        for (auto i = 0u; i != section.get_symbols_num(); ++i) {
            // TODO: this is boyscout code, caching the temporaries
            //       may be of worth.

            auto tmp = read_symbol(section, i);
            if (tmp.sect_idx == SHN_UNDEF && !tmp.name.empty()) {
                r.push_back(std::move(tmp.name));
            }
        }

        return r;
    }

    inline
    const std::unordered_map<
        std::string,
        std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>& symbol_addresses()
    {
        using namespace ELFIO;
        using namespace std;

        static unordered_map<string, pair<Elf64_Addr, Elf_Xword>> r;
        static once_flag f;

        call_once(f, []() {
            dl_iterate_phdr([](dl_phdr_info* info, size_t, void*) {
                static constexpr const char self[] = "/proc/self/exe";
                elfio reader;

                static unsigned int iter = 0u;
                if (reader.load(!iter++ ? self : info->dlpi_name)) {
                    auto it = find_section_if(
                        reader, [](const class section* x) {
                        return x->get_type() == SHT_SYMTAB;
                    });

                    if (it) {
                        const symbol_section_accessor symtab{reader, it};

                        for (auto i = 0u; i != symtab.get_symbols_num(); ++i) {
                            auto tmp = read_symbol(symtab, i);

                            if (tmp.type == STT_OBJECT &&
                                tmp.sect_idx != SHN_UNDEF) {
                                r.emplace(
                                    move(tmp.name),
                                    make_pair(tmp.value, tmp.size));
                            }
                        }
                    }
                }

                return 0;
            }, nullptr);
        });

        return r;
    }

    inline
    const std::vector<hsa_agent_t>& all_agents()
    {
        using namespace std;

        static vector<hsa_agent_t> r;
        static once_flag f;

        call_once(f, []() {
            for (auto&& acc : hc::accelerator::get_all()) {
                if (acc.is_hsa_accelerator()) {
                    r.push_back(
                        *static_cast<hsa_agent_t*>(acc.get_hsa_agent()));
                }
            }
        });

        return r;
    }

    inline
    void associate_code_object_symbols_with_host_allocation(
        const ELFIO::elfio& reader,
        const ELFIO::elfio& self_reader,
        ELFIO::section* code_object_dynsym,
        ELFIO::section* process_symtab,
        hsa_agent_t agent,
        hsa_executable_t executable)
    {
        using namespace ELFIO;
        using namespace std;

        if (!code_object_dynsym || !process_symtab) return;

        const auto undefined_symbols = copy_names_of_undefined_symbols(
            symbol_section_accessor{reader, code_object_dynsym});

        for (auto&& x : undefined_symbols) {
            using RAII_global =
                unique_ptr<void, decltype(hsa_amd_memory_unlock)*>;

            static unordered_map<string, RAII_global> globals;
            static once_flag f;
            call_once(f, [=]() { globals.reserve(symbol_addresses().size()); });

            if (globals.find(x) != globals.cend()) return;

            const auto it1 = symbol_addresses().find(x);

            if (it1 == symbol_addresses().cend()) {
                throw runtime_error{"Global symbol: " + x + " is undefined."};
            }

            static mutex mtx;
            lock_guard<mutex> lck{mtx};

            if (globals.find(x) != globals.cend()) return;

            void* host_ptr =
                reinterpret_cast<void*>(it1->second.first);
            void* agent_ptr = nullptr;
            hsa_amd_memory_lock(
                host_ptr,
                it1->second.second,
                // Awful cast because ROCr interface is misspecified.
                const_cast<hsa_agent_t*>(all_agents().data()),
                all_agents().size(),
                &agent_ptr);

            hsa_executable_agent_global_variable_define(
                executable, agent, x.c_str(), agent_ptr);

            globals.emplace(x, RAII_global{host_ptr, hsa_amd_memory_unlock});
        }
    }

    inline
    hsa_code_object_reader_t load_code_object_and_freeze_executable(
        void* elf,
        std::size_t byte_cnt,
        hsa_agent_t agent,
        hsa_executable_t executable)
    {   // TODO: the following sequence is inefficient, should be refactored
        //       into a single load of the file and subsequent ELFIO
        //       processing.
        using namespace std;

        hsa_code_object_reader_t r = {};
        hsa_code_object_reader_create_from_memory(elf, byte_cnt, &r);

        hsa_executable_load_agent_code_object(
            executable, agent, r, nullptr, nullptr);

        hsa_executable_freeze(executable, nullptr);

        return r;
    }
}



namespace hc {

// printf buffer size (in number of packets, see hc_printf.hpp)
constexpr unsigned int default_printf_buffer_size = 2048;

// address of the global printf buffer in host coherent system memory
PrintfPacket* printf_buffer = nullptr;

// store the address of agent accessible hc::printf_buffer;
PrintfPacket** printf_buffer_locked_va = nullptr;

} // namespace hc


namespace Kalmar {

enum class HCCRuntimeStatus{

  // No error
  HCCRT_STATUS_SUCCESS = 0x0,

  // A generic error
  HCCRT_STATUS_ERROR = 0x2000,

  // The maximum number of outstanding AQL packets in a queue has been reached
  HCCRT_STATUS_ERROR_COMMAND_QUEUE_OVERFLOW = 0x2001
};

const char* getHCCRuntimeStatusMessage(const HCCRuntimeStatus status) {
  const char* message = nullptr;
  switch(status) {
    //HCCRT_CASE_STATUS_STRING(HCCRT_STATUS_SUCCESS,"Success");
    case HCCRuntimeStatus::HCCRT_STATUS_SUCCESS:
      message = "Success"; break;
    case HCCRuntimeStatus::HCCRT_STATUS_ERROR:
      message = "Generic error"; break;
    case HCCRuntimeStatus::HCCRT_STATUS_ERROR_COMMAND_QUEUE_OVERFLOW:
      message = "Command queue overflow"; break;
    default:
      message = "Unknown error code"; break;
  };
  return message;
}

inline static void checkHCCRuntimeStatus(const HCCRuntimeStatus status, const unsigned int line, hsa_queue_t* q=nullptr) {
  if (status != HCCRuntimeStatus::HCCRT_STATUS_SUCCESS) {
    fprintf(stderr, "### HCC runtime error: %s at %s line:%d\n", getHCCRuntimeStatusMessage(status), __FILENAME__, line);
    std::string m("HCC Runtime Error - ");
    m += getHCCRuntimeStatusMessage(status);
    throw Kalmar::runtime_exception(m.c_str(), 0);
    //if (q != nullptr)
    //  assert(HSA_STATUS_SUCCESS == hsa_queue_destroy(q));
    //assert(HSA_STATUS_SUCCESS == hsa_shut_down());
    //exit(-1);
  }
}

} // namespace Kalmar



extern "C" void PushArgImpl(void *ker, int idx, size_t sz, const void *v);
extern "C" void PushArgPtrImpl(void *ker, int idx, size_t sz, const void *v);

// forward declaration
namespace Kalmar {
class HSAQueue;
class HSADevice;

namespace CLAMP {
  void LoadInMemoryProgram(KalmarQueue*);
} // namespace CLAMP
} // namespace Kalmar

///
/// kernel compilation / kernel launching
///

/// modeling of HSA executable
class HSAExecutable {
private:
    hsa_code_object_reader_t hsaCodeObjectReader;
    hsa_executable_t hsaExecutable;
    friend class HSAKernel;
    friend class Kalmar::HSADevice;

public:
    HSAExecutable(hsa_executable_t _hsaExecutable,
                  hsa_code_object_reader_t _hsaCodeObjectReader) :
        hsaExecutable(_hsaExecutable),
        hsaCodeObjectReader(_hsaCodeObjectReader) {}

    ~HSAExecutable() {
      hsa_status_t status;

      DBOUT(DB_INIT, "HSAExecutable::~HSAExecutable\n");

      status = hsa_executable_destroy(hsaExecutable);
      STATUS_CHECK(status, __LINE__);

      status = hsa_code_object_reader_destroy(hsaCodeObjectReader);
      STATUS_CHECK(status, __LINE__);
    }

    template<typename T>
    void setSymbolToValue(const char* symbolName, T value) {
        hsa_status_t status;

        // get symbol
        hsa_executable_symbol_t symbol;
        hsa_agent_t agent;
        status = hsa_executable_get_symbol_by_name(hsaExecutable, symbolName, const_cast<hsa_agent_t*>(&agent), &symbol);
        STATUS_CHECK_SYMBOL(status, symbolName, __LINE__);

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
    std::string kernelName;
    std::string shortKernelName; // short handle, format selectable with HCC_DB_KERNEL_NAME
    HSAExecutable* executable;
    uint64_t kernelCodeHandle;
    hsa_executable_symbol_t hsaExecutableSymbol;
    uint32_t static_group_segment_size;
    uint32_t private_segment_size;
    uint16_t workitem_vgpr_count;
    friend class HSADispatch;

public:
    HSAKernel(std::string &_kernelName, const std::string &x_shortKernelName, HSAExecutable* _executable,
              hsa_executable_symbol_t _hsaExecutableSymbol,
              uint64_t _kernelCodeHandle) :
        kernelName(_kernelName),
        shortKernelName(x_shortKernelName),
        executable(_executable),
        hsaExecutableSymbol(_hsaExecutableSymbol),
        kernelCodeHandle(_kernelCodeHandle) {

        if (shortKernelName.empty()) {
            shortKernelName = "<unknown_kernel>";
        }

        hsa_status_t status =
            hsa_executable_symbol_get_info(
                _hsaExecutableSymbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &this->static_group_segment_size);
        STATUS_CHECK(status, __LINE__);

        status =
            hsa_executable_symbol_get_info(
                _hsaExecutableSymbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &this->private_segment_size);
        STATUS_CHECK(status, __LINE__);

        workitem_vgpr_count = 0;

        uint16_t ext_version_major = 1;
        uint16_t ext_version_minor = 0;
        bool ext_supported = false;
        status =
            hsa_system_major_extension_supported(
                HSA_EXTENSION_AMD_LOADER,
                ext_version_major,
                &ext_version_minor,
                &ext_supported);
        STATUS_CHECK(status, __LINE__);
        if(!ext_supported)
            throw Kalmar::runtime_exception("HSA_EXTENSION_AMD_LOADER not supported.", 0);

        hsa_ven_amd_loader_1_01_pfn_t ext_table = {nullptr};
        status =
            hsa_system_get_major_extension_table(
                HSA_EXTENSION_AMD_LOADER,
                1,
                sizeof(ext_table),
                &ext_table);
        STATUS_CHECK(status, __LINE__);

        if (nullptr != ext_table.hsa_ven_amd_loader_query_host_address) {
            const amd_kernel_code_t* akc = nullptr;
            status =
                ext_table.hsa_ven_amd_loader_query_host_address(
                    reinterpret_cast<const void*>(kernelCodeHandle),
                    reinterpret_cast<const void**>(&akc));
            STATUS_CHECK(status, __LINE__);

            workitem_vgpr_count = akc->workitem_vgpr_count;
        }

        DBOUTL(DB_CODE, "Create kernel " << shortKernelName << " vpr_cnt=" << this->workitem_vgpr_count
                << " static_group_segment_size=" << this->static_group_segment_size
                << " private_segment_size=" << this->private_segment_size );

    }

    //TODO - fix this so all Kernels set the _kernelName to something sensible.
    const std::string &getKernelName() const { return shortKernelName; }
    const std::string &getLongKernelName() const { return kernelName; }

    ~HSAKernel() {
        DBOUT(DB_INIT, "HSAKernel::~HSAKernel\n");
    }
}; // end of HSAKernel

// Stores the device and queue for op coordinate:
struct HSAOpCoord
{
    HSAOpCoord(Kalmar::HSAQueue *queue);

    int         _deviceId;
    uint64_t    _queueId;
};

// Base class for the other HSA ops:
class HSAOp : public Kalmar::KalmarAsyncOp {
public:
    HSAOp(hc::HSAOpId id, Kalmar::KalmarQueue *queue, hc::hcCommandKind commandKind);

    const HSAOpCoord opCoord() const { return _opCoord; };

    void* getNativeHandle() override { return &_signal; }

    virtual bool barrierNextSyncNeedsSysRelease() const { return 0; };
    virtual bool barrierNextKernelNeedsSysAcquire() const { return 0; };

    Kalmar::HSAQueue *hsaQueue() const;
    bool isReady() override;

    static constexpr uint32_t _max_dependency_chain_length = 32;
    uint32_t           _dependency_chain_length;

protected:
    uint64_t     apiStartTick;
    HSAOpCoord   _opCoord;

    hsa_signal_t _signal;
    int          _signalIndex;

    hsa_agent_t  _agent;

    activity_prof::ActivityProf _activity_prof;

    hsa_status_t _wait_complete_status;
};
std::ostream& operator<<(std::ostream& os, const HSAOp & op);


class HSACopy : public HSAOp {
private:
    bool isSubmitted;
    bool isAsync;          // copy was performed asynchronously
    bool isSingleStepCopy;; // copy was performed on fast-path via a single call to the HSA copy routine
    bool isPeerToPeer;
    uint64_t apiStartTick;
    hsa_wait_state_t waitMode;

    std::shared_future<void>* future;


    // If copy is dependent on another operation, record reference here.
    // keep a reference which prevents those ops from being deleted until this op is deleted.
    std::shared_ptr<HSAOp> depAsyncOp;

    const Kalmar::HSADevice* copyDevice;  // Which device did the copy.

    // source pointer
    const void* src;


    // destination pointer
    void* dst;

    // bytes to be copied
    size_t sizeBytes;


public:
    std::shared_future<void>* getFuture() override { return future; }
    const Kalmar::HSADevice* getCopyDevice() const { return copyDevice; } ;  // Which device did the copy.


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


    std::string getCopyCommandString()
    {
        using namespace Kalmar;

        std::string s;
        switch (getCommandKind()) {
            case hcMemcpyHostToHost:
                s += "HostToHost";
                break;
            case hcMemcpyHostToDevice:
                s += "HostToDevice";
                break;
            case hcMemcpyDeviceToHost:
                s += "DeviceToHost";
                break;
            case hcMemcpyDeviceToDevice:
                if (isPeerToPeer) {
                    s += "PeerToPeer";
                } else {
                    s += "DeviceToDevice";
                }
                break;
            default:
                s += "UnknownCopy";
                break;
        };
        s += isAsync ? "_async" : "_sync";
        s += isSingleStepCopy ? "_fast" : "_slow";

        return s;

    }


    // Copy mode will be set later on.
    // HSA signals would be waited in HSA_WAIT_STATE_ACTIVE by default for HSACopy instances
    HSACopy(Kalmar::KalmarQueue *queue, const void* src_, void* dst_, size_t sizeBytes_);




    ~HSACopy() {
        if (future) {
            future->wait();
            STATUS_CHECK(_wait_complete_status, __LINE__);
        }
        dispose();
    }

    hsa_status_t enqueueAsyncCopyCommand(const Kalmar::HSADevice *copyDevice, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo);
    hsa_status_t enqueueAsyncCopy2dCommand(size_t width, size_t height, size_t srcPitch, size_t dstPitch, const Kalmar::HSADevice *copyDevice, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo);

    // wait for the async copy to complete
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

    uint64_t getStartTick();

    uint64_t getSystemTicks();

    // synchronous version of copy
    void syncCopy();
    void syncCopyExt(hc::hcCommandKind copyDir,
                     const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
                     const Kalmar::HSADevice *copyDevice, bool forceUnpinnedCopy);
    hsa_status_t syncCopy2DExt(hc::hcCommandKind copyDir,
                     const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,size_t width, size_t height, size_t srcPitch, size_t dstPitch,
                     const Kalmar::HSADevice *copyDevice, bool forceUnpinnedCopy);


private:
  hsa_status_t hcc_memory_async_copy(Kalmar::hcCommandKind copyKind, const Kalmar::HSADevice *copyDevice,
                                      const hc::AmPointerInfo &dstPtrInfo, const hc::AmPointerInfo &srcPtrInfo,
                                      size_t sizeBytes);

  hsa_status_t hcc_memory_async_copy_rect(Kalmar::hcCommandKind copyKind, const Kalmar::HSADevice *copyDevice,
                                      const hc::AmPointerInfo &dstPtrInfo, const hc::AmPointerInfo &srcPtrInfo,
                                      size_t width, size_t height, size_t srcPitch, size_t dstPitch);

}; // end of HSACopy

class HSABarrier : public HSAOp {
private:
    bool isDispatched;
    hsa_wait_state_t waitMode;


    std::shared_future<void>* future;

    // prior dependencies
    // maximum up to 5 prior dependencies could be associated with one
    // HSABarrier instance
    int depCount;
    hc::memory_scope _acquire_scope;

    // capture the state of _nextSyncNeedsSysRelease and _nextKernelNeedsSysAcquire after
    // the barrier is issued.  Cross-queue synchronziation commands which synchronize
    // with the barrier (create_blocking_marker) then can transer the correct "needs" flags.
    bool                                            _barrierNextSyncNeedsSysRelease;
    bool                                            _barrierNextKernelNeedsSysAcquire;

public:
    uint16_t  header;  // stores header of AQL packet.  Preserve so we can see flushes associated with this barrier.

    // array of all operations that this op depends on.
    // This array keeps a reference which prevents those ops from being deleted until this op is deleted.
    std::shared_ptr<HSAOp> depAsyncOps [HSA_BARRIER_DEP_SIGNAL_CNT];

public:
    std::shared_future<void>* getFuture() override { return future; }
    void acquire_scope(hc::memory_scope acquireScope) { _acquire_scope = acquireScope;};

    bool barrierNextSyncNeedsSysRelease() const override { return _barrierNextSyncNeedsSysRelease; };
    bool barrierNextKernelNeedsSysAcquire() const override { return _barrierNextKernelNeedsSysAcquire; };


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






    // constructor with 1 prior dependency
    HSABarrier(Kalmar::KalmarQueue *queue, std::shared_ptr <Kalmar::KalmarAsyncOp> dependent_op) :
        HSAOp(hc::HSA_OP_ID_BARRIER, queue, Kalmar::hcCommandMarker),
        isDispatched(false),
        future(nullptr),
        _acquire_scope(hc::no_scope),
        _barrierNextSyncNeedsSysRelease(false),
        _barrierNextKernelNeedsSysAcquire(false),
        waitMode(HSA_WAIT_STATE_BLOCKED)
    {

        if (dependent_op != nullptr) {
            assert (dependent_op->getCommandKind() == Kalmar::hcCommandMarker);

            depAsyncOps[0] = std::static_pointer_cast<HSAOp> (dependent_op);
            depCount = 1;
        } else {
            depCount = 0;
        }
    }

    // constructor with at most 5 prior dependencies
    HSABarrier(Kalmar::KalmarQueue *queue, int count, std::shared_ptr <Kalmar::KalmarAsyncOp> *dependent_op_array) :
        HSAOp(hc::HSA_OP_ID_BARRIER, queue, Kalmar::hcCommandMarker),
        isDispatched(false),
        future(nullptr),
        _acquire_scope(hc::no_scope),
        _barrierNextSyncNeedsSysRelease(false),
        _barrierNextKernelNeedsSysAcquire(false),
        waitMode(HSA_WAIT_STATE_BLOCKED),
        depCount(0)
    {
        if ((count >= 0) && (count <= 5)) {
            for (int i = 0; i < count; ++i) {
                if (dependent_op_array[i]) {
                    // squish null ops
                    depAsyncOps[depCount] = std::static_pointer_cast<HSAOp> (dependent_op_array[i]);
                    depCount++;
                }
            }
        } else {
            // throw an exception
            throw Kalmar::runtime_exception("Incorrect number of dependent signals passed to HSABarrier constructor", count);
        }
    }

    ~HSABarrier() {
        future->wait();
        STATUS_CHECK(_wait_complete_status, __LINE__);
        dispose();
    }


    hsa_status_t enqueueAsync(hc::memory_scope memory_scope);

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

class HSADispatch : public HSAOp {
private:
    Kalmar::HSADevice* device;

    const char *kernel_name;
    const HSAKernel* kernel;

    std::vector<uint8_t> arg_vec;
    uint32_t arg_count;
    size_t prevArgVecCapacity;
    void* kernargMemory;
    int kernargMemoryIndex;


    hsa_kernel_dispatch_packet_t aql;
    bool isDispatched;
    hsa_wait_state_t waitMode;


    std::shared_future<void>* future;

public:
    std::shared_future<void>* getFuture() override { return future; }

    void setKernelName(const char *x_kernel_name) { kernel_name = x_kernel_name;};
    const char *getKernelName() { return kernel_name ? kernel_name : (kernel ? kernel->shortKernelName.c_str() : "<unknown_kernel>"); };
    const char *getLongKernelName() { return (kernel ? kernel->getLongKernelName().c_str() : "<unknown_kernel>"); };


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


    ~HSADispatch() {
        if (future){
            future->wait();
            STATUS_CHECK(_wait_complete_status, __LINE__);
        }
        dispose();
    }

    HSADispatch(Kalmar::HSADevice* _device, Kalmar::KalmarQueue* _queue, HSAKernel* _kernel,
                const hsa_kernel_dispatch_packet_t *aql=nullptr);

    hsa_status_t pushFloatArg(float f) { return pushArgPrivate(f); }
    hsa_status_t pushIntArg(int i) { return pushArgPrivate(i); }
    hsa_status_t pushBooleanArg(unsigned char z) { return pushArgPrivate(z); }
    hsa_status_t pushByteArg(char b) { return pushArgPrivate(b); }
    hsa_status_t pushLongArg(long j) { return pushArgPrivate(j); }
    hsa_status_t pushDoubleArg(double d) { return pushArgPrivate(d); }
    hsa_status_t pushShortArg(short s) { return pushArgPrivate(s); }
    hsa_status_t pushPointerArg(void *addr) { return pushArgPrivate(addr); }

    hsa_status_t clearArgs() {
        arg_count = 0;
        arg_vec.clear();
        return HSA_STATUS_SUCCESS;
    }


    void overrideAcquireFenceIfNeeded();
    hsa_status_t setLaunchConfiguration(const int dims, size_t *globalDims, size_t *localDims,
                                        const int dynamicGroupSize);

    hsa_status_t dispatchKernelWaitComplete();

    hsa_status_t dispatchKernelAsyncFromOp();
    hsa_status_t dispatchKernelAsync(const void *hostKernarg, int hostKernargSize, bool allocSignal);

    // dispatch a kernel asynchronously
    hsa_status_t dispatchKernel(hsa_queue_t* lockedHsaQueue, const void *hostKernarg,
                               int hostKernargSize, bool allocSignal);

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

    const hsa_kernel_dispatch_packet_t &getAql() const { return aql; };

private:
    template <typename T>
    hsa_status_t pushArgPrivate(T val) {
        /* add padding if necessary */
        int padding_size = (arg_vec.size() % sizeof(T)) ? (sizeof(T) - (arg_vec.size() % sizeof(T))) : 0;
        DBOUT(DB_KERNARG, "push " << (sizeof(T) + padding_size) << " bytes into kernarg: ");

        for (size_t i = 0; i < padding_size; ++i) {
            arg_vec.push_back((uint8_t)0x00);
            DBOUT(DB_KERNARG, std::hex << std::setw(2) << std::setfill('0') << 0x00 << " ");
        }
        uint8_t* ptr = static_cast<uint8_t*>(static_cast<void*>(&val));
        for (size_t i = 0; i < sizeof(T); ++i) {
            arg_vec.push_back(ptr[i]);
            DBOUT(DB_KERNARG, std::hex << std::setw(2) << std::setfill('0') << +ptr[i] << " ");
        }
        DBOUT(DB_KERNARG, std::endl);

        arg_count++;
        return HSA_STATUS_SUCCESS;
    }

}; // end of HSADispatch

//-----
//Structure used to extract information from memory pool
struct pool_iterator
{
    hsa_amd_memory_pool_t _am_memory_pool;
    hsa_amd_memory_pool_t _am_host_memory_pool;
    hsa_amd_memory_pool_t _am_host_coherent_memory_pool;
    hsa_amd_memory_pool_t _am_finegrained_memory_pool;

    hsa_amd_memory_pool_t _kernarg_memory_pool;
    hsa_amd_memory_pool_t _finegrained_system_memory_pool;
    hsa_amd_memory_pool_t _coarsegrained_system_memory_pool;
    hsa_amd_memory_pool_t _local_memory_pool;
    hsa_amd_memory_pool_t _finegrained_local_memory_pool;

    bool        _found_kernarg_memory_pool;
    bool        _found_finegrained_system_memory_pool;
    bool        _found_local_memory_pool;
    bool        _found_coarsegrained_system_memory_pool;
    bool        _found_finegrained_local_memory_pool;

    size_t _local_memory_pool_size;

    pool_iterator() ;
};


pool_iterator::pool_iterator()
{
    _kernarg_memory_pool.handle=(uint64_t)-1;
    _finegrained_system_memory_pool.handle=(uint64_t)-1;
    _local_memory_pool.handle=(uint64_t)-1;
    _coarsegrained_system_memory_pool.handle=(uint64_t)-1;
    _finegrained_local_memory_pool.handle=(uint64_t)-1;

    _found_kernarg_memory_pool = false;
    _found_finegrained_system_memory_pool = false;
    _found_local_memory_pool = false;
    _found_coarsegrained_system_memory_pool = false;
    _found_finegrained_local_memory_pool = false;

    _local_memory_pool_size = 0;
}
//-----


///
/// memory allocator
///
namespace Kalmar {



// Small wrapper around the hsa hardware queue (ie returned from hsa_queue_create(...).
// This allows us to see which accelerator_view owns the hsa queue, and
// also tracks the state of the cu mask, profiling, priority of the HW queue.
// Rocr queues are shared by the allocated HSAQueues.  When an HSAQueue steals
// a rocrQueue, we ensure that the hw queue has the desired cu_mask and other state.
//
// HSAQueue is the implementation of accelerator_view for HSA back-and.  HSAQueue
// points to RocrQueue, or to nullptr if the HSAQueue is not currently attached to a RocrQueue.
struct RocrQueue {
    static void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void *data) {
        STATUS_CHECK(status, __LINE__);
    }

    RocrQueue(hsa_agent_t agent, size_t queue_size, HSAQueue *hccQueue, queue_priority priority)
        : _priority(priority)
    {
        // Map queue_priority to hsa_amd_queue_priority_t
        hsa_amd_queue_priority_t queue_priority;
        switch (priority) {
            case priority_low:
                queue_priority = HSA_AMD_QUEUE_PRIORITY_LOW;
                break;
            case priority_high:
                queue_priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
                break;
            case priority_normal:
            default:
                queue_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
                break;
        }

        assert(queue_size != 0);

        /// Create a queue using the maximum size.
        hsa_status_t status = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_SINGLE, callbackQueue, NULL,
                                  UINT32_MAX, UINT32_MAX, &_hwQueue);
        DBOUT(DB_QUEUE, "  " <<  __func__ << ": created an HSA command queue: " << _hwQueue << "\n");
        STATUS_CHECK(status, __LINE__);

        // Set queue priority
        status = hsa_amd_queue_set_priority(_hwQueue, queue_priority);
        DBOUT(DB_QUEUE, "  " <<  __func__ << ": set priority for HSA command queue: " << _hwQueue << " to " << queue_priority << "\n");
        STATUS_CHECK(status, __LINE__);

        // TODO - should we provide a mechanism to conditionally enable profiling as a performance optimization?
        status = hsa_amd_profiling_set_profiler_enabled(_hwQueue, 1);

        // Create the links between the queues:
        assignHccQueue(hccQueue);
    }

    ~RocrQueue() {

        DBOUT(DB_QUEUE, "  " <<  __func__ << ": destroy an HSA command queue: " << _hwQueue << "\n");

        hsa_status_t status = hsa_queue_destroy(_hwQueue);
        _hwQueue = 0;
        STATUS_CHECK(status, __LINE__);
    };

    void assignHccQueue(HSAQueue *hccQueue);

    hsa_status_t setCuMask(HSAQueue *hccQueue);


    hsa_queue_t *_hwQueue; // Pointer to the HSA queue this entry tracks.

    HSAQueue *_hccQueue;  // Pointe to the HCC "HSA" queue which is assigned to use the rocrQueue

    std::vector<uint32_t> cu_arrays;

    // Track profiling enabled state here. - no need now since all hw queues have profiling enabled.

    // Priority could be tracked here:
    queue_priority _priority;
};



class HSAQueue final : public KalmarQueue
{
private:
    friend class Kalmar::HSADevice;
    friend class RocrQueue;
    friend std::ostream& operator<<(std::ostream& os, const HSAQueue & hav);

    // ROCR queue associated with this HSAQueue instance.
    RocrQueue    *rocrQueue;


    // NOTE: Changed to recursive mutex since recursive locking may occur
    // within the same thread. In HSAQueue dtor, the call to dispose() will
    // lock the queue and then it will call wait().  In wait(), if it occurs
    // that a system scope release is needed, it would enqueue a system scope
    // marker, which will eventually turning it into enqueuing an HSABarrier
    // into the current queue. The recursive locking happens when HSABarrier
    // tries to lock the queue to insert a new packet.
    // Step through the runtime code with the unit test HC/execute_order.cpp
    // for details
    std::recursive_mutex   qmutex;  // Protect structures for this KalmarQueue.  Currently just the hsaQueue.


    bool         drainingQueue_;  // mode that we are draining queue, used to allow barrier ops to be enqueued.

    //
    // kernel dispatches and barriers associated with this HSAQueue instance
    //
    // When a kernel k is dispatched, we'll get a KalmarAsyncOp f.
    // This vector would hold f.  ops are added by replacing the op within the shared_ptr.
    // This could trigger the freeing of resources, one insertion at a time.
    //
    std::vector< std::shared_ptr<HSAOp> > asyncOps;

    // index where the next asyncOp should be inserted into the asyncOps container
    int asyncOpsIndex;

    uint64_t                                      queueSeqNum; // sequence-number of this queue.

    // Valid is used to prevent the fields of the HSAQueue from being disposed
    // multiple times.
    bool                                            valid;


    // Flag that is set when a kernel command is enqueued without system scope
    // Indicates queue needs a flush at the next queue::wait() call or copy to ensure
    // host data is valid.
    bool                                            _nextSyncNeedsSysRelease;

    // Flag that is set after a copy command is enqueued.
    // The next kernel command issued needs to add a system-scope acquire to
    // pick up any data that may have been written by the copy.
    bool                                            _nextKernelNeedsSysAcquire;


    // Kind of the youngest command in the queue.
    // Used to detect and enforce dependencies between commands.
    // Persists even after the youngest command has been removed.
    hcCommandKind youngestCommandKind;

    // Store current CU mask, if any.
    std::vector<uint32_t> cu_arrays;

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

    // signal used by sync copy only
    hsa_signal_t  sync_copy_signal;


public:
    HSAQueue(KalmarDevice* pDev, hsa_agent_t agent, execute_order order, queue_priority priority) ;

    bool nextKernelNeedsSysAcquire() const { return _nextKernelNeedsSysAcquire; };
    void setNextKernelNeedsSysAcquire(bool r) { _nextKernelNeedsSysAcquire = r; };

    bool nextSyncNeedsSysRelease() const { 
      DBOUT( DB_CMD2, "  HSAQueue::nextSyncNeedsSysRelease(): " <<  _nextSyncNeedsSysRelease << "\n");
      return _nextSyncNeedsSysRelease; 
    };
    void setNextSyncNeedsSysRelease(bool r) {
      DBOUT( DB_CMD2, "  HSAQueue::setNextSyncNeedsSysRelease(" <<  r << ")\n");
      _nextSyncNeedsSysRelease = r; 
    };

    uint64_t getSeqNum() const { return queueSeqNum; };

    Kalmar::HSADevice * getHSADev() const;

    void dispose() override;

    ~HSAQueue() {
        DBOUT(DB_INIT, "HSAQueue::~HSAQueue() in\n");
        if (valid) {
            dispose();
        }

        DBOUT(DB_INIT, "HSAQueue::~HSAQueue() " << this << "out\n");
    }

    int increment(int index) {
        // increment index, without using modulo
        return ((index+1) == asyncOps.size() ? 0 : (index+1));
    }

    int decrement(int index) {
        // decrement index, without using modulo
        return (index == 0 ? asyncOps.size()-1 : index-1);
    }

    std::shared_ptr<HSAOp> back() {
        // to emulate container.back(), but using the current insert index
        return asyncOps[decrement(asyncOpsIndex)];
    }

    // FIXME: implement flush
    //
    void printAsyncOps(std::ostream &s = std::cerr)
    {
        std::lock_guard<std::recursive_mutex> lg(qmutex);
        hsa_signal_value_t oldv=0;
        s << *this << " : " << asyncOps.size() << " op entries\n";
        for (int i=0; i<asyncOps.size(); i++) {
            const std::shared_ptr<HSAOp> &op = asyncOps[i];
            s << "index:" << std::setw(4) << i ;
            if (op != nullptr) {
                s << " op#"<< op->getSeqNum() ;
                hsa_signal_t signal = * (static_cast<hsa_signal_t*> (op->getNativeHandle()));
                hsa_signal_value_t v = 0;
                if (signal.handle) {
                    v = hsa_signal_load_scacquire(signal);
                }
                s  << " " << getHcCommandKindString(op->getCommandKind());
                // TODO - replace with virtual function
                if (op->getCommandKind() == hc::hcCommandMarker) {
                    auto b = static_cast<HSABarrier*> (op.get());
                    s << " acq=" << extractBits(b->header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
                    s << ",rel=" << extractBits(b->header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
                } else if (op->getCommandKind() == hc::hcCommandKernel) {
                    auto d = static_cast<HSADispatch*> (op.get());
                    s << " acq=" << extractBits(d->getAql().header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
                    s << ",rel=" << extractBits(d->getAql().header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
                }
                s  << " completion=0x" << std::hex << signal.handle << std::dec <<",value=" << v;

                if (v != oldv) {
                    s << " <--TRANSITION";
                    oldv = v;
                }
            } else {
                s << " op <nullptr>";
            }
            s  << "\n";

        }
    }

    // Save the command and type
    // TODO - can convert to reference?
    void pushAsyncOp(std::shared_ptr<HSAOp> op) {

        std::lock_guard<std::recursive_mutex> lg(qmutex);

        op->setSeqNumFromQueue();

        DBOUT(DB_CMD, "  pushing " << *op << " completion_signal="<< std::hex  << ((hsa_signal_t*)op->getNativeHandle())->handle << std::dec
                    << "  commandKind=" << getHcCommandKindString(op->getCommandKind())
                    << " "
                    << (op->getCommandKind() == hcCommandKernel ? ((static_cast<HSADispatch*> (op.get()))->getKernelName()) : "")  // change to getLongKernelName() for mangled name
                    << std::endl);

        youngestCommandKind = op->getCommandKind();
        asyncOps[asyncOpsIndex] = std::move(op);
        asyncOpsIndex = increment(asyncOpsIndex);

        if (DBFLAG(DB_QUEUE)) {
            printAsyncOps(std::cerr);
        }
    }



    // Check upcoming command that will be sent to this queue against the youngest async op
    // in the queue to detect if any command dependency is required.
    //
    // The function returns nullptr if no dependency is required. For example, back-to-back commands
    // of same type are often implicitly synchronized so no dependency is required.
    //
    // Also different modes and optimizations can control when dependencies are added.
    // TODO - return reference if possible to avoid shared ptr overhead.
    std::shared_ptr<KalmarAsyncOp> detectStreamDeps(hcCommandKind newCommandKind, KalmarAsyncOp *kNewOp) {

        std::lock_guard<std::recursive_mutex> lg(qmutex);

        const auto newOp = static_cast<const HSAOp*> (kNewOp);

        assert (newCommandKind != hcCommandInvalid);

        auto lastOp = back();

        if (lastOp.get()!=nullptr) {
            assert (youngestCommandKind != hcCommandInvalid);

            // Ensure we have not already added the op we are checking into asyncOps,
            // that must be done after we check for deps.
            if (newOp && (newOp == lastOp.get())) {
                throw Kalmar::runtime_exception("enqueued op before checking dependencies!", 0);
            }

            bool needDep = false;
            if  (newCommandKind != youngestCommandKind) {
                DBOUT(DB_CMD2, "Set NeedDep (command type changed) " 
                        << getHcCommandKindString(youngestCommandKind) 
                        << "  ->  " << getHcCommandKindString(newCommandKind) << "\n") ;
                needDep = true;
            };


            if (((newCommandKind == hcCommandKernel) && (youngestCommandKind == hcCommandMarker)) ||
                ((newCommandKind == hcCommandMarker) && (youngestCommandKind == hcCommandKernel))) {

                // No dependency required since Marker and Kernel share same queue and are ordered by AQL barrier bit.
                needDep = false;
            } else if (isCopyCommand(newCommandKind) && isCopyCommand(youngestCommandKind)) {
                assert (newOp);
                auto hsaCopyOp = static_cast<const HSACopy*> (newOp);
                auto youngestCopyOp = static_cast<const HSACopy*> (lastOp.get());

                if (hsaCopyOp->getCopyDevice() == nullptr ||
                     youngestCopyOp->getCopyDevice() == nullptr ||
                     hsaCopyOp->getCopyDevice() != youngestCopyOp->getCopyDevice()) {
                    // This covers cases where two copies are back-to-back in the queue but use different copy engines.
                    // In this case there is no implicit dependency between the ops so we need to add one
                    // here.
                    needDep = true;
                    DBOUT(DB_CMD2, "Set NeedDep for " << newOp << "(different copy engines) " );
                }
                if (FORCE_SIGNAL_DEP_BETWEEN_COPIES) {
                    DBOUT(DB_CMD2, "Set NeedDep for " << newOp << "(FORCE_SIGNAL_DEP_BETWEEN_COPIES) " );
                    needDep = true;
                }
            }

            if (needDep) {
                DBOUT(DB_CMD2, "command type changed " << getHcCommandKindString(youngestCommandKind) << "  ->  " << getHcCommandKindString(newCommandKind) << "\n") ;
                return lastOp;
            }
        }

        return nullptr;
    }


    void waitForStreamDeps (HSAOp *newOp) {
        std::shared_ptr<KalmarAsyncOp> depOp = detectStreamDeps(newOp->getCommandKind(), newOp);
        if (depOp != nullptr) {
            EnqueueMarkerWithDependency(1, &depOp, HCC_OPT_FLUSH ? hc::no_scope : hc::system_scope);
        }
    }


    int getPendingAsyncOps() override {
        std::lock_guard<std::recursive_mutex> lg(qmutex);
        int count = 0;
        for (int i = 0; i < asyncOps.size(); ++i) {
            auto &asyncOp = asyncOps[i];

            if (asyncOp != nullptr) {
                hsa_signal_t signal = *(static_cast <hsa_signal_t*> (asyncOp->getNativeHandle()));
                if (signal.handle) {
                    hsa_signal_value_t v = hsa_signal_load_scacquire(signal);
                    if (v != 0) {
                        ++count;
                    }
                } else {
                    ++count;
                }
            }
        }
        return count;
    }


    bool isEmpty() override {

        if (youngestCommandKind == hcCommandInvalid) {
            return true; // no op ever inserted 
        }

        // we need the lock because back() accesses asyncOps
        std::lock_guard<std::recursive_mutex> lg(qmutex);
        hsa_signal_t signal = *(static_cast <hsa_signal_t*> (back()->getNativeHandle()));
        if (signal.handle) {
            hsa_signal_value_t v = hsa_signal_load_scacquire(signal);
            if (v != 0) {
                return false;
            }
        } else {
            // youngest has no signal - enqueue a new one:
            auto marker = EnqueueMarker(hc::system_scope);
            DBOUTL(DB_CMD2, "Inside HSAQueue::isEmpty and queue contained only no-signal ops, enqueued marker " << marker << " into " << *this);
            return false;
        }

        return true;
    };


    // Must retain this exact function signature here even though mode not used since virtual interface in
    // runtime depends on this signature.
    void wait(hcWaitMode mode = hcWaitModeBlocked) override {
        // wait on all previous async operations to complete
        // Go in reverse order (from youngest to oldest).
        // Ensures younger ops have chance to complete before older ops reclaim their resources
        //

        std::lock_guard<std::recursive_mutex> lg(qmutex);

        if (HCC_OPT_FLUSH && nextSyncNeedsSysRelease()) {

            // In the loop below, this will be the first op waited on
            auto marker = EnqueueMarker(hc::system_scope);

            DBOUT(DB_CMD2, " Sys-release needed, enqueued marker into " << *this << " to release written data " << marker<<"\n");

        }

        DBOUT(DB_WAIT, *this << " wait, contents:\n");
        if (DBFLAG(DB_WAIT)) {
            printAsyncOps(std::cerr);
        }

        // if youngest op doesn't have a future, enqueue a marker to add the future
        bool need_marker = true;
        auto youngest_op = back();
        if (youngest_op != nullptr) {
            auto future = youngest_op->getFuture();
            if (future && future->valid()) {
                need_marker = false;
            }
        }
        if (need_marker) {
            auto marker = EnqueueMarker(hc::no_scope);
            DBOUT(DB_CMD2, "No future found in wait, enqueued marker into " << *this << "\n");
        }
        back()->getFuture()->wait();
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
        dispatch->setLaunchConfiguration(nr_dim, global, local, dynamic_group_size);

        // wait for previous kernel dispatches be completed
        std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                      [&] (void* buffer) {
                        waitForDependentAsyncOps(buffer);
                      });

        waitForStreamDeps(dispatch);



        // dispatch the kernel
        // and wait for its completion
        dispatch->dispatchKernelWaitComplete();

        // clear data in kernelBufferMap
        kernelBufferMap[ker].clear();
        kernelBufferMap.erase(ker);

        delete(dispatch);
    }

    std::shared_ptr<KalmarAsyncOp> LaunchKernelAsync(void *ker, size_t nr_dim, size_t *global, size_t *local) override {
        return LaunchKernelWithDynamicGroupMemoryAsync(ker, nr_dim, global, local, 0);
    }

    std::shared_ptr<KalmarAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(void *ker, size_t nr_dim, size_t *global, size_t *local, size_t dynamic_group_size) override {
        hsa_status_t status = HSA_STATUS_SUCCESS;

        HSADispatch *dispatch =
            reinterpret_cast<HSADispatch*>(ker);



        bool hasArrayViewBufferDeps = (kernelBufferMap.find(ker) != kernelBufferMap.end());


        if (hasArrayViewBufferDeps) {
            // wait for previous kernel dispatches be completed
            std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                      [&] (void* buffer) {
                        waitForDependentAsyncOps(buffer);
                     });
        }

        waitForStreamDeps(dispatch);


        // create a shared_ptr instance
        std::shared_ptr<KalmarAsyncOp> sp_dispatch(dispatch);
        // associate the kernel dispatch with this queue
        pushAsyncOp(std::static_pointer_cast<HSAOp> (sp_dispatch));

        size_t tmp_local[] = {0, 0, 0};
        if (!local)
            local = tmp_local;
        dispatch->setLaunchConfiguration(nr_dim, global, local, dynamic_group_size);

        // dispatch the kernel
        status = dispatch->dispatchKernelAsyncFromOp();
        STATUS_CHECK(status, __LINE__);


        if (hasArrayViewBufferDeps) {
            // associate all buffers used by the kernel with the kernel dispatch instance
            std::for_each(std::begin(kernelBufferMap[ker]), std::end(kernelBufferMap[ker]),
                          [&] (void* buffer) {
                            bufferKernelMap[buffer].push_back(sp_dispatch);
                          });

                // clear data in kernelBufferMap
                kernelBufferMap[ker].clear();
                kernelBufferMap.erase(ker);
        }

        return sp_dispatch;
    }


    void releaseToSystemIfNeeded()
    {
        if (HCC_OPT_FLUSH && nextSyncNeedsSysRelease()) {
            // In the loop below, this will be the first op waited on
            auto marker= EnqueueMarker(hc::system_scope);

            DBOUT(DB_CMD2, " In waitForDependentAsyncOps, sys-release needed: enqueued marker to release written data " << marker<<"\n");
        };
    }


    // wait for dependent async operations to complete
    void waitForDependentAsyncOps(void* buffer) {
        auto&& dependentAsyncOpVector = bufferKernelMap[buffer];
        for (int i = 0; i < dependentAsyncOpVector.size(); ++i) {
          auto dependentAsyncOp = dependentAsyncOpVector[i];
          auto dependentAsyncOpPointer = dependentAsyncOp.lock();
          if (dependentAsyncOpPointer) {
            // wait on valid futures only
            std::shared_future<void>* future = dependentAsyncOpPointer->getFuture();
            if (future->valid()) {
              future->wait();
            }
          }
        }
        dependentAsyncOpVector.clear();

    }


    void sync_copy(void* dst, hsa_agent_t dst_agent,
                   const void* src, hsa_agent_t src_agent,
                   size_t size) {

      if (DBFLAG(DB_COPY)) {
        dumpHSAAgentInfo(src_agent, "sync_copy source agent");
        dumpHSAAgentInfo(dst_agent, "sync_copy destination agent");
      }

      hsa_status_t status;
      hsa_signal_store_relaxed(sync_copy_signal, 1);
      status = hsa_amd_memory_async_copy(dst, dst_agent,
                                          src, src_agent,
                                          size, 0, nullptr, sync_copy_signal);
      STATUS_CHECK(status, __LINE__);
      hsa_signal_wait_scacquire(sync_copy_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
      return;
    }

    void read(void* device, void* dst, size_t count, size_t offset) override {
        waitForDependentAsyncOps(device);
        releaseToSystemIfNeeded();

        // do read
        if (dst != device) {
            if (!getDev()->is_unified()) {
                DBOUT(DB_COPY, "read(" << device << "," << dst << "," << count << "," << offset 
                                << "): use HSA memory copy\n");
                hsa_status_t status = HSA_STATUS_SUCCESS;
                // Make sure host memory is accessible to gpu
                // FIXME: host memory is allocated through OS allocator, if not, correct it.
                // dst--host buffer might be allocated through either OS allocator or hsa allocator.
                // Things become complicated, we may need some query API to query the pointer info, i.e.
                // allocator info. Same as write.
                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
                void* va = nullptr;
                status = hsa_amd_memory_lock(dst, count, agent, 1, &va);
                // TODO: If host buffer is not allocated through OS allocator, so far, lock
                // API will return nullptr to va, this is not specified in the spec, but will use it to
                // check if host buffer is allocated by hsa allocator
                if(va == NULL || status != HSA_STATUS_SUCCESS)
                {
                    status = hsa_amd_agents_allow_access(1, agent, NULL, dst);
                    STATUS_CHECK(status, __LINE__);
                    va = dst;
                }

                sync_copy(va, *static_cast<hsa_agent_t*>(getHostAgent()),  (char*)device + offset, *static_cast<hsa_agent_t*>(getHSAAgent()), count);

                // Unlock the host memory
                status = hsa_amd_memory_unlock(dst);
            } else {
                DBOUT(DB_COPY, "read(" << device << "," << dst << "," << count << "," << offset 
                                << "): use host memory copy\n");
                memmove(dst, (char*)device + offset, count);
            }
        }
    }

    void write(void* device, const void* src, size_t count, size_t offset, bool blocking) override {
        waitForDependentAsyncOps(device);
        releaseToSystemIfNeeded(); // may not be needed.

        // do write
        if (src != device) {
            if (!getDev()->is_unified()) {
                DBOUT(DB_COPY, "write(" << device << "," << src << "," << count << "," << offset 
                                << "," << blocking << "): use HSA memory copy\n");
                hsa_status_t status = HSA_STATUS_SUCCESS;
                // Make sure host memory is accessible to gpu
                // FIXME: host memory is allocated through OS allocator, if not, correct it.
                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
                const void* va = nullptr;
                status = hsa_amd_memory_lock(const_cast<void*>(src), count, agent, 1, (void**)&va);

                if(va == NULL || status != HSA_STATUS_SUCCESS)
                {
                    status = hsa_amd_agents_allow_access(1, agent, NULL, src);
                    STATUS_CHECK(status, __LINE__);
                    va = src;
                }
                sync_copy(((char*)device) + offset,  *agent, va,    *static_cast<hsa_agent_t*>(getHostAgent()), count);

                STATUS_CHECK(status, __LINE__);
                // Unlock the host memory
                status = hsa_amd_memory_unlock(const_cast<void*>(src));
            } else {
                DBOUT(DB_COPY, "write(" << device << "," << src << "," << count << "," << offset 
                                << "," << blocking << "): use host memory copy\n");
                memmove((char*)device + offset, src, count);
            }
        }
    }



    //FIXME: this API doesn't work in the P2P world because we don't who the source agent is!!!
    void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) override {
        waitForDependentAsyncOps(dst);
        waitForDependentAsyncOps(src);
        releaseToSystemIfNeeded();

        // do copy
        if (src != dst) {
            if (!getDev()->is_unified()) {
                DBOUT(DB_COPY, "copy(" << src << "," << dst << "," << count << "," << src_offset 
                               << "," << dst_offset << "," << blocking << "): use HSA memory copy\n");
                hsa_status_t status = HSA_STATUS_SUCCESS;
                // FIXME: aftre p2p enabled, if this function is not expected to copy between two buffers from different device, then, delete allow_access API call.
                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
                status = hsa_amd_agents_allow_access(1, agent, NULL, src);
                STATUS_CHECK(status, __LINE__);
                status = hsa_memory_copy((char*)dst + dst_offset, (char*)src + src_offset, count);
                STATUS_CHECK(status, __LINE__);
            } else {
                DBOUT(DB_COPY, "copy(" << src << "," << dst << "," << count << "," << src_offset 
                               << "," << dst_offset << "," << blocking << "): use host memory copy\n");
                memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
            }
        }
    }

    void* map(void* device, size_t count, size_t offset, bool modify) override {
        if (DBFLAG(DB_COPY)) {
            dumpHSAAgentInfo(*static_cast<hsa_agent_t*>(getHSAAgent()), "map(...)");
        }
        waitForDependentAsyncOps(device);
        releaseToSystemIfNeeded();

        // do map
        // as HSA runtime doesn't have map/unmap facility at this moment,
        // we explicitly allocate a host memory buffer in this case
        if (!getDev()->is_unified()) {
            if (DBFLAG(DB_COPY)) {
                DBWSTREAM << getDev()->get_path();
                DBSTREAM << ": map( <device> " << device << ", <count> " << count << ", <offset> " << offset 
                         << ", <modify> " << modify << "): use HSA memory map\n";
            }
            hsa_status_t status = HSA_STATUS_SUCCESS;
            // allocate a host buffer
            // TODO: for safety, we copy to host, but we can map device memory to host through hsa_amd_agents_allow_access
            // withouth copying data.  (Note: CPU only has WC access to data, which has very poor read perf)
            void* data = nullptr;
            hsa_amd_memory_pool_t* am_host_region = static_cast<hsa_amd_memory_pool_t*>(getHSAAMHostRegion());
            status = hsa_amd_memory_pool_allocate(*am_host_region, count, 0, &data);
            STATUS_CHECK(status, __LINE__);
            if (data != nullptr) {
              // copy data from device buffer to host buffer
              hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
              status = hsa_amd_agents_allow_access(1, agent, NULL, data);
              STATUS_CHECK(status, __LINE__);
              sync_copy(data, *static_cast<hsa_agent_t*>(getHostAgent()), ((char*)device) + offset, *agent, count);
            } else {
              throw Kalmar::runtime_exception("host buffer allocation failed!", 0);
            }
            return data;
        } else {
            if (DBFLAG(DB_COPY)) {
              DBWSTREAM << getDev()->get_path();
              DBSTREAM << ": map( <device> " << device << ", <count> " << count << ", <offset> " << offset 
                       << ", <modify> " << modify << "): use host memory map\n";
            }
            // for host memory we simply return the pointer plus offset
            return (char*)device + offset;
        }
    }

    void unmap(void* device, void* addr, size_t count, size_t offset, bool modify) override {
        // do unmap

        // as HSA runtime doesn't have map/unmap facility at this moment,
        // we free the host memory buffer allocated in map()
        if (!getDev()->is_unified()) {
            if (DBFLAG(DB_COPY)) {
                DBWSTREAM << getDev()->get_path();
                DBSTREAM << ": unmap( <device> " << device << ", <addr> " << addr << ", <count> " << count 
                         << ", <offset> " << offset << ", <modify> " << modify << "): use HSA memory unmap\n";
            }
            if (modify) {
                // copy data from host buffer to device buffer
                hsa_status_t status = HSA_STATUS_SUCCESS;

                hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
                sync_copy(((char*)device) + offset, *agent, addr, *static_cast<hsa_agent_t*>(getHostAgent()), count);
            }

            // deallocate the host buffer
            hsa_amd_memory_pool_free(addr);
        } else {
            if (DBFLAG(DB_COPY)) {
                DBWSTREAM << getDev()->get_path();
                DBSTREAM << ": unmap( <device> " << device << ", <addr> " << addr << ", <count> " << count 
                         << ", <offset> " << offset << ", <modify> " << modify <<"): use host memory unmap\n";
            }
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
        return static_cast<void*>(rocrQueue->_hwQueue);
    }

    hsa_queue_t *acquireLockedRocrQueue();

    void releaseLockedRocrQueue();


    void* getHSAAgent() override;

    void* getHostAgent() override;

    void* getHSAAMRegion() override;

    void* getHSACoherentAMHostRegion() override;

    void* getHSAAMHostRegion() override;

    void* getHSAKernargRegion() override;

    void* getHSAFinegrainedAMRegion() override;

    bool hasHSAInterOp() override {
        return true;
    }

    void dispatch_hsa_kernel(const hsa_kernel_dispatch_packet_t *aql,
                             const void * args, size_t argsize,
                             hc::completion_future *cf, const char *kernelName) override ;

    bool set_cu_mask(const std::vector<bool>& cu_mask) override {
        // get device's total compute unit count
        auto device = getDev();
        unsigned int physical_count = device->get_compute_unit_count();
        assert(physical_count > 0);

        uint32_t temp = 0;
        uint32_t bit_index = 0;

        // If cu_mask.size() is greater than physical_count, igore the rest.
        int iter = cu_mask.size() > physical_count ? physical_count : cu_mask.size();


        {
            std::lock_guard<std::recursive_mutex> l(this->qmutex);


            this->cu_arrays.clear();

            for(auto i = 0; i < iter; i++) {
                temp |= (uint32_t)(cu_mask[i]) << bit_index;

                if(++bit_index == 32) {
                    this->cu_arrays.push_back(temp);
                    bit_index = 0;
                    temp = 0;
                }
            }

            if(bit_index != 0) {
                this->cu_arrays.push_back(temp);
            }


            // Apply the new cu mask to the hw queue:
            return (rocrQueue->setCuMask(this) == HSA_STATUS_SUCCESS);

        }
    }

    // enqueue a barrier packet
    std::shared_ptr<KalmarAsyncOp> EnqueueMarker(memory_scope release_scope) override {

        hsa_status_t status = HSA_STATUS_SUCCESS;

        // create shared_ptr instance
        std::shared_ptr<HSABarrier> barrier = std::make_shared<HSABarrier>(this, 0, nullptr);
        // associate the barrier with this queue
        pushAsyncOp(barrier);

        // enqueue the barrier
        status = barrier.get()->enqueueAsync(release_scope);
        STATUS_CHECK(status, __LINE__);


        return barrier;
    }


    // enqueue a barrier packet with multiple prior dependencies
    // The marker will wait for all specified input dependencies to resolve and
    // also for all older commands in the queue to execute, and then will
    // signal completion by decrementing the associated signal.
    //
    // depOps specifies the other ops that this marker will depend on.  These
    // can be in any queue on any GPU .
    //
    // fenceScope specifies the scope of the acquire and release fence that will be
    // applied after the marker executes.  See hc::memory_scope
    std::shared_ptr<KalmarAsyncOp> EnqueueMarkerWithDependency(int count,
            std::shared_ptr <KalmarAsyncOp> *depOps,
            hc::memory_scope fenceScope) override {

        hsa_status_t status = HSA_STATUS_SUCCESS;

        if ((count >= 0) && (count <= HSA_BARRIER_DEP_SIGNAL_CNT)) {

            // create shared_ptr instance
            std::shared_ptr<HSABarrier> barrier = std::make_shared<HSABarrier>(this, count, depOps);
            // associate the barrier with this queue
            pushAsyncOp(barrier);

            for (int i=0; i<count; i++) {
                auto depOp = barrier->depAsyncOps[i];
                if (depOp != nullptr) {
                    auto depHSAQueue = static_cast<Kalmar::HSAQueue *> (depOp->getQueue());
                    // Same accelerator:
                    // Inherit system-acquire and system-release bits op we are dependent on.
                    //   - barriers
                    //
                    // _nextSyncNeedsSysRelease is set when a queue executes a kernel.
                    // It indicates the queue needs to execute a release-to-system
                    // before host can see the data - this is important for kernels which write
                    // non-coherent zero-copy host memory.
                    // If creating a dependency on a queue which needs_system_release, copy that
                    // state here.   If the host then waits on the freshly created marker,
                    // runtime will issue a system-release fence.
                    if (depOp->barrierNextKernelNeedsSysAcquire()) {
                        DBOUTL(DB_CMD2, *this << " setting NextKernelNeedsSysAcquire(true) due to dependency on barrier " << *depOp)
                        setNextKernelNeedsSysAcquire(true);
                    }
                    if (depOp->barrierNextSyncNeedsSysRelease()) {
                        DBOUTL(DB_CMD2, *this << " setting NextSyncNeedsSysRelease(true) due to dependency on barrier " << *depOp)
                        setNextSyncNeedsSysRelease(true);
                    }
                    if (HCC_FORCE_CROSS_QUEUE_FLUSH & 0x1) {
                        if (!depOp->barrierNextKernelNeedsSysAcquire()) {
                            DBOUTL(DB_RESOURCE, *this << " force setting NextSyncNeedsSysAcquire(true) even though barrier didn't require it " << *depOp)
                        }
                        setNextKernelNeedsSysAcquire(true);
                    }
                    if (HCC_FORCE_CROSS_QUEUE_FLUSH & 0x2) {
                        if (!depOp->barrierNextSyncNeedsSysRelease()) {
                            DBOUTL(DB_RESOURCE, *this << " force setting NextSyncNeedsSysRelease(true) even though barrier didn't require it " << *depOp)
                        }
                        setNextSyncNeedsSysRelease(true);
                    }

                    if (depHSAQueue->getHSADev() != this->getHSADev()) {
                        // Cross-accelerator dependency case.
                        // This requires system-scope acquire
                        // TODO - only needed if these are peer GPUs, could optimize with an extra check
                        DBOUT(DB_WAIT, "  Adding cross-accelerator system-scope acquire\n");
                        barrier->acquire_scope (hc::system_scope);
                    }

                } else {
                    break;
                }
            }

            // enqueue the barrier
            status = barrier.get()->enqueueAsync(fenceScope);
            STATUS_CHECK(status, __LINE__);


            return barrier;
        } else {
            // throw an exception
            throw Kalmar::runtime_exception("Incorrect number of dependent signals passed to EnqueueMarkerWithDependency", count);
        }
    }

    std::shared_ptr<KalmarAsyncOp> EnqueueAsyncCopyExt(const void* src, void* dst, size_t size_bytes,
                                                       hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
                                                       const Kalmar::KalmarDevice *copyDevice) override;

    std::shared_ptr<KalmarAsyncOp> EnqueueAsyncCopy2dExt(const void* src, void* dst, size_t width, size_t height, size_t srcPitch, size_t dstPitch,
                                                       hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
                                                       const Kalmar::KalmarDevice *copyDevice) override;

    std::shared_ptr<KalmarAsyncOp> EnqueueAsyncCopy(const void *src, void *dst, size_t size_bytes) override ;


    // synchronous copy
    void copy(const void *src, void *dst, size_t size_bytes) override {
        DBOUT(DB_COPY, "HSAQueue::copy(" << src << ", " << dst << ", " << size_bytes << ")\n");
        // wait for all previous async commands in this queue to finish
        this->wait();

        // create a HSACopy instance
        HSACopy* copyCommand = new HSACopy(this, src, dst, size_bytes);

        // synchronously do copy
        copyCommand->syncCopy();

        delete(copyCommand);
    }

    void copy_ext(const void *src, void *dst, size_t size_bytes, hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
                  const Kalmar::KalmarDevice *copyDevice, bool forceUnpinnedCopy) override ;

    bool copy2d_ext(const void *src, void *dst, size_t width, size_t height, size_t srcPitch, size_t dstPitch, hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo, const Kalmar::KalmarDevice *copyDevice, bool forceUnpinnedCopy);

    void copy_ext(const void *src, void *dst, size_t size_bytes, hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo, bool foo) override ;

};


void RocrQueue::assignHccQueue(HSAQueue *hccQueue) {
    assert (hccQueue->rocrQueue == nullptr);  // only needy should assign new queue
    hccQueue->rocrQueue = this;
    _hccQueue = hccQueue;

    setCuMask(hccQueue);
}

hsa_status_t RocrQueue::setCuMask(HSAQueue *hccQueue) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (this->cu_arrays != hccQueue->cu_arrays) {
        // Expensive operation:
        this->cu_arrays = hccQueue->cu_arrays;
        status = hsa_amd_queue_cu_set_mask(_hwQueue,  hccQueue->cu_arrays.size()*32, hccQueue->cu_arrays.data());
    }

    return status;
}


class HSADevice final : public KalmarDevice
{
    friend std::ostream& operator<<(std::ostream& os, const HSAQueue & hav);
private:
    /// memory pool for kernargs
    std::vector<void*> kernargPool;
    std::vector<bool> kernargPoolFlag;
    int kernargCursor;
    std::mutex kernargPoolMutex;


    std::map<std::string, HSAKernel *> programs;
    hsa_agent_t agent;
    size_t max_tile_static_size;

    size_t queue_size;
    std::mutex queues_mutex; // protects access to the queues vector:
    std::vector< std::weak_ptr<KalmarQueue> > queues;

    // TODO: Do we need to maintain a different mutex for each queue priority?
    // In which case HSAQueue::dispose needs to handle locking the appropriate mutex
    std::mutex                  rocrQueuesMutex; // protects rocrQueues
    std::vector< RocrQueue *>    rocrQueues[3];

    pool_iterator ri;

    bool useCoarseGrainedRegion;

    uint32_t workgroup_max_size;
    uint16_t workgroup_max_dim[3];

    std::map<std::string, HSAExecutable*> executables;

    hcAgentProfile profile;

    /*TODO: This is the first CPU which will provide system memory pool
    We might need to modify again in multiple CPU socket scenario. Because
    we must make sure there is pyshycial link between device and host. Currently,
    agent iterate function will push back all of the dGPU on the system, which might
    not be linked directly to the first cpu node, host */
    hsa_agent_t hostAgent;

    uint16_t versionMajor;
    uint16_t versionMinor;

    int      accSeqNum;     // unique accelerator seq num
    uint64_t queueSeqNums;  // used to assign queue seqnums.


public:
    // Structures to manage unpinnned memory copies
    class UnpinnedCopyEngine      *copy_engine[2]; // one for each direction.
    UnpinnedCopyEngine::CopyMode  copy_mode;

    // Creates or steals a rocrQueue and returns it in theif->rocrQueue
    void createOrstealRocrQueue(Kalmar::HSAQueue *thief, queue_priority priority = priority_normal) {
        RocrQueue *foundRQ = nullptr;

        // Allocate a new queue when we are below the HCC_MAX_QUEUES limit
        {
            std::lock_guard<std::mutex> lg(rocrQueuesMutex);
            auto rqSize = rocrQueues[0].size()+rocrQueues[1].size()+rocrQueues[2].size();
            if (rqSize < HCC_MAX_QUEUES) {
                foundRQ = new RocrQueue(agent, this->queue_size, thief, priority);
                rocrQueues[priority].push_back(foundRQ);
                DBOUT(DB_QUEUE, "Create new rocrQueue=" << foundRQ << " for thief=" << thief << "\n");
                return;
            }
        }

        // Steal an unused queue when we reaches the limit
        while (true) {

            // First make a pass to see if we can find an unused queue
            {
                std::lock_guard<std::mutex> lg(rocrQueuesMutex);
                for (auto rq : rocrQueues[priority]) {
                    if (rq->_hccQueue == nullptr) {
                        DBOUT(DB_QUEUE, "Found unused rocrQueue=" << rq << " for thief=" << thief << ".  hwQueue=" << rq->_hwQueue << "\n")
                        // update the queue pointers to indicate the theft
                        rq->assignHccQueue(thief);
                        return;
                    }
                }
            }

            // Second pass, try steal from a ROCR queue associated with an HCC queue, but with no active tasks
            {
                std::lock_guard<std::mutex> lg(rocrQueuesMutex);
                for (auto rq : rocrQueues[priority]) {

                    if (rq->_hccQueue == nullptr) {
                        DBOUT(DB_QUEUE, "Found unused rocrQueue=" << rq << " for thief=" << thief << ".  hwQueue=" << rq->_hwQueue << "\n")
                        // update the queue pointers to indicate the theft
                        rq->assignHccQueue(thief);
                        return;
                    } else if (rq->_hccQueue != thief)  {
                        auto victimHccQueue = rq->_hccQueue;
                        std::lock_guard<std::recursive_mutex> l(victimHccQueue->qmutex);
                        if (victimHccQueue->isEmpty()) {
                            DBOUT(DB_LOCK, " ptr:" << this << " lock_guard...\n");
                            assert (victimHccQueue->rocrQueue == rq);  // ensure the link is consistent.
                            victimHccQueue->rocrQueue = nullptr;

                            // update the queue pointers to indicate the theft:
                            rq->assignHccQueue(thief);
                            DBOUT(DB_QUEUE, "Stole existing rocrQueue=" << rq << " from victimHccQueue=" << victimHccQueue << " to hccQueue=" << thief << "\n")
                            return; // for
                        }
                    }
                }
            }
        }
    };


private:

    // NOTE: removeRocrQueue should only be called from HSAQueue::dispose
    // since there's an assumption on a specific locking sequence
    friend void HSAQueue::dispose();
    void removeRocrQueue(RocrQueue *rocrQueue) {

        // queues already locked:
        size_t hccSize = queues.size();

        // rocrQueuesMutex has already been acquired in HSAQueue::dispose

        // a perf optimization to keep the HSA queue if we have more HCC queues that might want it.
        // This defers expensive queue deallocation if an hccQueue that holds an hwQueue is destroyed -
        // keep the hwqueue around until the number of hccQueues drops below the number of hwQueues
        // we have already allocated.
        // TODO: Do we want to track HCC queues independently for each priority?
        // Or maybe modify queue stealing to steal unused HSA queue from other priority pools &
        // change priority as required?
        auto rqSize = rocrQueues[0].size()+rocrQueues[1].size()+rocrQueues[2].size();
        if (hccSize < rqSize) {
            queue_priority priority = rocrQueue->_priority;
            auto iter = std::find(rocrQueues[priority].begin(), rocrQueues[priority].end(), rocrQueue);
            assert(iter != rocrQueues[priority].end());
            // Remove the pointer from the list:
            rocrQueues[priority].erase(iter);
            DBOUT(DB_QUEUE, "removeRocrQueue-hard: rocrQueue=" << rocrQueue << " hccQueues/rocrQueues=" << hccSize << "/" << rqSize << "\n")
            delete rocrQueue; // this will delete the HSA HW queue.
        }
        else {
            DBOUT(DB_QUEUE, "removeRocrQueue-soft: rocrQueue=" << rocrQueue << " keep hwQUeue, set _hccQueue link to nullptr"
                                                               << " hccQueues/rocrQueues=" << hccSize << "/" << rqSize << "\n");
            rocrQueue->_hccQueue = nullptr; // mark it as available.
        }
    };

    access_type get_access_type(hsa_agent_t agent) {
        /// Get the profile of the agent
        hsa_status_t status = HSA_STATUS_SUCCESS;
        hsa_profile_t agentProfile;
        access_type at;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agentProfile);
        STATUS_CHECK(status, __LINE__);
        if (agentProfile == HSA_PROFILE_BASE) {
            profile = hcAgentProfileBase;
            at = access_type_none;
        } else if (agentProfile == HSA_PROFILE_FULL) {
            profile = hcAgentProfileFull;
            at = access_type_read_write;
        }
        return at;
    };

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
        hsa_status_t status;
        hsa_amd_segment_t segment;
        status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (status != HSA_STATUS_SUCCESS) {
          return status;
        }

        if (segment == HSA_AMD_SEGMENT_GLOBAL) {
          hsa_amd_memory_pool_global_flag_t flags;
          status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
          if (status != HSA_STATUS_SUCCESS) {
            return status;
          }

          size_t size = 0;
          status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
          if (status != HSA_STATUS_SUCCESS) {
            return status;
          }

          pool_iterator *ri = (pool_iterator*) (data);
          if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) && (!ri->_found_finegrained_local_memory_pool)) {
            DBOUT(DB_INIT, "  found fine grained memory pool of GPU local memory region=" << region.handle << ", size(MB) = " << (size/(1024*1024)) << std::endl);
            ri->_finegrained_local_memory_pool = region;
            ri->_found_finegrained_local_memory_pool = true;
          }
          if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && (!ri->_found_local_memory_pool)) {
            DBOUT(DB_INIT, "  found coarse grained memory pool of GPU local memory region=" << region.handle << ", size(MB) = " << (size/(1024*1024)) << std::endl);
            ri->_local_memory_pool = region;
            ri->_found_local_memory_pool = true;
            ri->_local_memory_pool_size = size;
          }
          if (ri->_found_finegrained_local_memory_pool && ri->_found_local_memory_pool) {
            return HSA_STATUS_INFO_BREAK;
          }
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

        size_t size = 0;
        status = hsa_amd_memory_pool_get_info(region, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
        STATUS_CHECK(status, __LINE__);
        size = size/(1024*1024);

        if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) && (!ri->_found_finegrained_system_memory_pool)) {
            DBOUT(DB_INIT, "found fine grained memory pool on host memory, size(MB) = " << size << std::endl);
            ri->_finegrained_system_memory_pool = region;
            ri->_found_finegrained_system_memory_pool = true;
        }

        if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && (!ri->_found_coarsegrained_system_memory_pool)) {
            DBOUT(DB_INIT, "found coarse-grain system memory pool=" << region.handle << " size(MB) = " << size << std::endl);
            ri->_coarsegrained_system_memory_pool = region;
            ri->_found_coarsegrained_system_memory_pool = true;
        }

        // choose coarse grained system for kernarg, if not available, fall back to fine grained system.
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
          if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
            DBOUT(DB_INIT, "using coarse grained system for kernarg memory, size(MB) = " << size << std::endl);
            ri->_kernarg_memory_pool = region;
            ri->_found_kernarg_memory_pool = true;
          }
          else if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED
                   && ri->_found_kernarg_memory_pool == false) {
            DBOUT(DB_INIT, "using fine grained system for kernarg memory, size(MB) = " << size << std::endl);
            ri->_kernarg_memory_pool = region;
            ri->_found_kernarg_memory_pool = true;
          }
          else {
            DBOUT(DB_INIT, "Unknown memory pool with kernarg_init flag set!!!, size(MB) = " << size << std::endl);
          }
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

    hsa_agent_t& getHostAgent() {
        return hostAgent;
    }

    // Returns true if specified agent has access to the specified pool.
    // Typically used to detect when a CPU agent has access to GPU device memory via large-bar:
    int hasAccess(hsa_agent_t agent, hsa_amd_memory_pool_t pool)
    {
        hsa_status_t err;
        hsa_amd_memory_pool_access_t access;
        err = hsa_amd_agent_memory_pool_get_info(agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
        STATUS_CHECK(err, __LINE__);
        return access;
    }





    HSADevice(hsa_agent_t a, hsa_agent_t host, int x_accSeqNum);

    ~HSADevice() {
        DBOUT(DB_INIT, "HSADevice::~HSADevice() in\n");

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
#if KERNARG_POOL_SIZE > 0
        kernargPoolMutex.lock();

        hsa_status_t status = HSA_STATUS_SUCCESS;

        // kernargPool is allocated in batch, KERNARG_POOL_SIZE for each
        // allocation. it is therefore be released also in batch.
        for (int i = 0; i < kernargPool.size() / KERNARG_POOL_SIZE; ++i) {
            hsa_amd_memory_pool_free(kernargPool[i * KERNARG_POOL_SIZE]);
            STATUS_CHECK(status, __LINE__);
        }

        kernargPool.clear();
        kernargPoolFlag.clear();

        kernargPoolMutex.unlock();
#endif

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


        for (int i=0; i<2; i++) {
            if (copy_engine[i]) {
                delete copy_engine[i];
                copy_engine[i] = NULL;
            }
        }


        DBOUT(DB_INIT, "HSADevice::~HSADevice() out\n");
    }

    std::wstring path;
    std::wstring description;
    uint32_t node;

    std::wstring get_path() const override { return path; }
    std::wstring get_description() const override { return description; }
    size_t get_mem() const override { return ri._local_memory_pool_size; }
    bool is_double() const override { return true; }
    bool is_lim_double() const override { return true; }
    bool is_unified() const override {
        return (useCoarseGrainedRegion == false);
    }
    bool is_emulated() const override { return false; }
    uint32_t get_version() const { return ((static_cast<unsigned int>(versionMajor) << 16) | versionMinor); }

    bool has_cpu_accessible_am() const override { return cpu_accessible_am; }

    void* create(size_t count, struct rw_info* key) override {
        void *data = nullptr;

        if (!is_unified()) {
            DBOUT(DB_INIT, "create( <count> " << count << ", <key> " << key << "): use HSA memory allocator\n");
            hsa_status_t status = HSA_STATUS_SUCCESS;
            auto am_region = getHSAAMRegion();

            status = hsa_amd_memory_pool_allocate(am_region, count, 0, &data);
            STATUS_CHECK(status, __LINE__);

            hsa_agent_t* agent = static_cast<hsa_agent_t*>(getHSAAgent());
            status = hsa_amd_agents_allow_access(1, agent, NULL, data);
            STATUS_CHECK(status, __LINE__);
        } else {
            DBOUT(DB_INIT, "create( <count> " << count << ", <key> " << key << "): use host memory allocator\n");
            data = kalmar_aligned_alloc(0x1000, count);
        }
        return data;
    }

    void release(void *ptr, struct rw_info* key ) override {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        if (!is_unified()) {
            DBOUT(DB_INIT, "release(" << ptr << "," << key << "): use HSA memory deallocator\n");
            status = hsa_amd_memory_pool_free(ptr);
            STATUS_CHECK(status, __LINE__);
        } else {
            DBOUT(DB_INIT, "release(" << ptr << "," << key << "): use host memory deallocator\n");
            kalmar_aligned_free(ptr);
        }
    }

    // calculate MD5 checksum
    std::string kernel_checksum(size_t size, void* source) {
        // FNV-1a hashing, 64-bit version
        const uint64_t FNV_prime = 0x100000001b3;
        const uint64_t FNV_basis = 0xcbf29ce484222325;
        uint64_t hash = FNV_basis;

        const char *str = static_cast<const char *>(source);
        for (auto i = 0; i < size; ++i) {
            hash ^= *str++;
            hash *= FNV_prime;
        }
        return std::to_string(hash);
    }

    void BuildProgram(void* size, void* source) override {
        if (executables.find(kernel_checksum((size_t)size, source)) == executables.end()) {
            size_t kernel_size = (size_t)((void *)size);
            char *kernel_source = (char*)malloc(kernel_size+1);
            memcpy(kernel_source, source, kernel_size);
            kernel_source[kernel_size] = '\0';
            BuildOfflineFinalizedProgramImpl(kernel_source, kernel_size);
            free(kernel_source);
        }
    }

    inline
    std::string get_isa_name_from_triple(std::string triple)
    {
        static hsa_isa_t tmp{};
        static const bool is_old_rocr{
            hsa_isa_from_name(triple.c_str(), &tmp) != HSA_STATUS_SUCCESS};

        if (is_old_rocr) {
            auto tmp {triple.substr(triple.rfind('x') + 1)};
            triple.replace(0, std::string::npos, "AMD:AMDGPU");

            for (auto&& x : tmp) {
                triple.push_back(':');
                triple.push_back(x);
            }
        }

        return triple;
    }

    bool IsCompatibleKernel(void* size, void* source) override {
        using namespace ELFIO;
        using namespace std;

        hsa_status_t status;

        // Allocate memory for kernel source
        size_t kernel_size = (size_t)((void *)size);
        char *kernel_source = (char*)malloc(kernel_size+1);
        memcpy(kernel_source, source, kernel_size);
        kernel_source[kernel_size] = '\0';

        // Set up ELF header reader
        elfio reader;
        istringstream kern_stream{string{
            kernel_source,
            kernel_source + kernel_size}};
        reader.load(kern_stream);

        // Get ISA from ELF header
        std::string triple = "amdgcn-amd-amdhsa--gfx";
        unsigned MACH = reader.get_flags() & hc::EF_AMDGPU_MACH;

        switch(MACH) {
            case hc::EF_AMDGPU_MACH_AMDGCN_GFX701 : triple.append("701"); break;
            case hc::EF_AMDGPU_MACH_AMDGCN_GFX803 : triple.append("803"); break;
            case hc::EF_AMDGPU_MACH_AMDGCN_GFX900 : triple.append("900"); break;
            case hc::EF_AMDGPU_MACH_AMDGCN_GFX906 : triple.append("906"); break;
        }

        const auto isa{get_isa_name_from_triple(std::move(triple))};

        struct isa_comp_data {
            hsa_isa_t isa_type;
            bool is_compatible;
        } co_data;

        status = hsa_isa_from_name(isa.c_str(), &co_data.isa_type);
        STATUS_CHECK(status, __LINE__);

        // Check if the code object is compatible with ISA of the agent
        co_data.is_compatible = false;
        status =
            hsa_agent_iterate_isas(
                agent,
                [](hsa_isa_t agent_isa, void* data) {
                    isa_comp_data* co_data = static_cast<isa_comp_data*>(data);
                    if (agent_isa == co_data->isa_type) {
                        co_data->is_compatible = true;
                        return HSA_STATUS_INFO_BREAK;
                    }
                    return HSA_STATUS_SUCCESS;
                },
                &co_data);
        STATUS_CHECK(status, __LINE__);

        // release allocated memory
        free(kernel_source);

        return co_data.is_compatible;
    }

    void* CreateKernel(const char* fun, Kalmar::KalmarQueue *queue) override {
        // try load kernels lazily in case it was not done so at bootstrap
        // due to HCC_LAZYINIT env var
        if (executables.size() == 0) {
          CLAMP::LoadInMemoryProgram(queue);
        }

        std::string str(fun);
        HSAKernel *kernel = programs[str];

        const char *demangled = "<demangle_error>";
        std::string shortName;

        if (!kernel) {
            int demangleStatus = 0;
            int kernelNameFormat = HCC_DB_SYMBOL_FORMAT & 0xf;
            if (kernelNameFormat == 1) {
                shortName = fun; // mangled name
            } else {
#ifndef USE_LIBCXX
                demangled = abi::__cxa_demangle(fun, nullptr, nullptr, &demangleStatus);
#endif
                shortName = demangleStatus ? fun : std::string(demangled);
                try {
                    if (demangleStatus == 0) {

                        if (kernelNameFormat == 2) {
                            shortName = demangled;
                        } else {
                            // kernelNameFormat == 0 or unspecified:

                            // Example: HIP_kernel_functor_name_begin_unnamed_HIP_kernel_functor_name_end_5::__cxxamp_trampoline(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float*, long long)"

                            std::string hip_begin_str  ("::HIP_kernel_functor_name_begin_");
                            std::string hip_end_str    ("_HIP_kernel_functor_name_end");
                            int hip_begin = shortName.find(hip_begin_str);
                            int hip_end   = shortName.find(hip_end_str);

                            if ((hip_begin != -1) && (hip_end != -1) && (hip_end > hip_begin)) {
                                // HIP kernel with markers
                                int start_pos = hip_begin + hip_begin_str.length();
                                std::string hipname = shortName.substr(start_pos, hip_end - start_pos) ;
                                DBOUTL(DB_CODE, "hipname=" << hipname);
                                if (hipname == "unnamed") {
                                    shortName = shortName.substr(0, hip_begin);
                                } else {
                                    shortName = hipname;
                                }

                            } else {
                                // PFE not from HIP:

                                // strip off hip launch template wrapper:
                                std::string hipImplString ("void hip_impl::grid_launch_hip_impl_<");
                                int begin = shortName.find(hipImplString);
                                if ((begin != std::string::npos)) {
                                    begin += hipImplString.length() ;
                                } else {
                                    begin = 0;
                                }

                                shortName = shortName.substr(begin);

                                // Strip off any leading return type:
                                begin = shortName.find(" ", 0);
                                if (begin == std::string::npos) {
                                    begin = 0;
                                } else {
                                    begin +=1; // skip the space
                                }
                                shortName = shortName.substr(begin);

                                DBOUTL(DB_CODE, "shortKernel processing demangled non-hip.  beginChar=" << begin << " shortName=" << shortName);
                            }

                        }

                        if (HCC_DB_SYMBOL_FORMAT & 0x10) {
                            // trim everything after first (
                            int begin = shortName.find("(");
                            shortName = shortName.substr(0, begin);
                        }
                    }
                } catch (std::out_of_range& exception) {
                    // Do something sensible if string pattern is not what we expect
                    shortName = fun;
                };
            }
            DBOUT (DB_CODE, "CreateKernel_short=      " << shortName << "\n");
            DBOUT (DB_CODE, "CreateKernel_demangled=  " << demangled << "\n");
            DBOUT (DB_CODE, "CreateKernel_raw=       " << fun << "\n");

            if (executables.size() != 0) {
                for (auto executable_iterator : executables) {
                    HSAExecutable *executable = executable_iterator.second;

                    // Get symbol handle.
                    hsa_status_t status;
                    hsa_executable_symbol_t kernelSymbol;
                    status = hsa_executable_get_symbol_by_name(executable->hsaExecutable, fun, const_cast<hsa_agent_t*>(&agent), &kernelSymbol);
                    if (status == HSA_STATUS_SUCCESS) {
                        // Get code handle.
                        uint64_t kernelCodeHandle;
                        status = hsa_executable_symbol_get_info(kernelSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelCodeHandle);
                        if (status == HSA_STATUS_SUCCESS) {
                            kernel =  new HSAKernel(str, shortName, executable, kernelSymbol, kernelCodeHandle);
                            break;
                        }
                    }
                }
            }

            if (!kernel) {
                const auto it =
                    shared_object_kernels(hc2::program_state()).find(agent);
                if (it != shared_object_kernels(hc2::program_state()).cend()) {
                    const auto k = std::find_if(
                        it->second.cbegin(),
                        it->second.cend(),
                        [&](hsa_executable_symbol_t x) {
                            return hc2::hsa_symbol_name(x) == str;
                        });
                    if (k != it->second.cend()) {
                        uint64_t h = 0u;
                        if (hsa_executable_symbol_get_info(
                                *k, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &h) ==
                            HSA_STATUS_SUCCESS) {

                            kernel = new HSAKernel(str, shortName, nullptr, *k, h);
                        }
                    }
                }
                if (!kernel) {
                    hc::print_backtrace();
                    std::cerr << "HSADevice::CreateKernel(): Unable to create kernel " << shortName << " \n";
                    std::cerr << "  CreateKernel_raw=  " << fun << "\n";
                    std::cerr << "  CreateKernel_demangled=  " << demangled << "\n";

                    if (demangled) {
                        free((void*)demangled); // cxa_dmangle mallocs memory.
                    }
                    abort();
                }
            } else {
                //std::cerr << "HSADevice::CreateKernel(): Created kernel\n";
            }
            programs[str] = kernel;
        }

        // HSADispatch instance will be deleted in:
        // HSAQueue::LaunchKernel()
        // or it will be created as a shared_ptr<KalmarAsyncOp> in:
        // HSAQueue::LaunchKernelAsync()
        HSADispatch *dispatch = new HSADispatch(this, queue, kernel);
        return dispatch;
    }

    std::shared_ptr<KalmarQueue> createQueue(execute_order order = execute_in_order, queue_priority priority= priority_normal) override {
        auto hsaAv = new HSAQueue(this, agent, order, priority);
        std::shared_ptr<KalmarQueue> q =  std::shared_ptr<KalmarQueue>(hsaAv);
        queues_mutex.lock();
        queues.push_back(q);
        hsaAv->queueSeqNum = this->queueSeqNums++;
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

    hsa_amd_memory_pool_t& getHSACoherentAMHostRegion() {
        return ri._am_host_coherent_memory_pool;
    }

    hsa_amd_memory_pool_t& getHSAAMRegion() {
        return ri._am_memory_pool;
    }
    
    hsa_amd_memory_pool_t& getHSAFinegrainedAMRegion() {
        return ri._am_finegrained_memory_pool;
    }

    bool hasHSAKernargRegion() const {
      return ri._found_kernarg_memory_pool;
    }

    bool hasHSAFinegrainedRegion() const {
      return ri._found_finegrained_system_memory_pool;
    }

    bool hasHSACoarsegrainedRegion() const {
      return ri._found_local_memory_pool;
    }
    
    bool hasHSAFinegrainedVRAMRegion() const {
      return ri._found_finegrained_local_memory_pool;
    }

    bool is_peer(const Kalmar::KalmarDevice* other) override {
      hsa_status_t status;

      if(!hasHSACoarsegrainedRegion())
          return false;

      auto self_pool = getHSAAMRegion();
      hsa_amd_memory_pool_access_t access;

      hsa_agent_t* agent = static_cast<hsa_agent_t*>( const_cast<KalmarDevice *> (other)->getHSAAgent());

      //TODO: CPU acclerator will return NULL currently, return false.
      if(nullptr == agent)
          return false;

      // If the agent's node is the same as the current device then
      // it's the same HSA agent and therefore not a peer
      uint32_t node = 0;
      status = hsa_agent_get_info(*agent, HSA_AGENT_INFO_NODE, &node);
      if (status != HSA_STATUS_SUCCESS)
        return false;
      if (node == this->node)
        return false;


      status = hsa_amd_agent_memory_pool_get_info(*agent, self_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);

      if(HSA_STATUS_SUCCESS != status)
          return false;

      if ((HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT == access) || (HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT == access))
          return true;

      return false;
    }

    unsigned int get_compute_unit_count() override {
        hsa_agent_t agent = getAgent();

        uint32_t compute_unit_count = 0;
        hsa_status_t status = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &compute_unit_count);
        if(status == HSA_STATUS_SUCCESS)
            return compute_unit_count;
        else
            return 0;
    }


    int get_seqnum() const override {
        return this->accSeqNum;
    }


    bool has_cpu_accessible_am() override {
        return cpu_accessible_am;
    };

    void releaseKernargBuffer(void* kernargBuffer, int kernargBufferIndex) {
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

    void growKernargBuffer()
    {
        uint8_t * kernargMemory = nullptr;
        // increase kernarg pool on demand by KERNARG_POOL_SIZE
        hsa_amd_memory_pool_t kernarg_region = getHSAKernargRegion();

        hsa_status_t status = hsa_amd_memory_pool_allocate(kernarg_region, KERNARG_POOL_SIZE * KERNARG_BUFFER_SIZE, 0, (void**)(&kernargMemory));
        STATUS_CHECK(status, __LINE__);

        status = hsa_amd_agents_allow_access(1, &agent, NULL, kernargMemory);
        STATUS_CHECK(status, __LINE__);

        for (size_t i = 0; i < KERNARG_POOL_SIZE * KERNARG_BUFFER_SIZE; i+=KERNARG_BUFFER_SIZE) {
            kernargPool.push_back(kernargMemory+i);
            kernargPoolFlag.push_back(false);
        };
    }

    std::pair<void*, int> getKernargBuffer(int size) {
        void* ret = nullptr;
        int cursor = 0;

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


                    growKernargBuffer();
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
            memset (ret, 0x00, KERNARG_BUFFER_SIZE);
        } else {
            // allocate new buffers in case:
            // - the kernarg pool is set at compile-time
            // - requested kernarg buffer size is larger than KERNARG_BUFFER_SIZE
            //

            hsa_status_t status = HSA_STATUS_SUCCESS;
            hsa_amd_memory_pool_t kernarg_region = getHSAKernargRegion();

            status = hsa_amd_memory_pool_allocate(kernarg_region, size, 0, &ret);
            STATUS_CHECK(status, __LINE__);

            status = hsa_amd_agents_allow_access(1, &agent, NULL, ret);
            STATUS_CHECK(status, __LINE__);

            DBOUTL(DB_RESOURCE, "Allocating non-pool kernarg buffer size=" << size );

            // set cursor value as -1 to notice the buffer would be deallocated
            // instead of recycled back into the pool
            cursor = -1;
            memset (ret, 0x00, size);
        }



        return std::make_pair(ret, cursor);
    }

    void* getSymbolAddress(const char* symbolName) override {
        hsa_status_t status;

        unsigned long* symbol_ptr = nullptr;
        if (executables.size() != 0) {
            // iterate through all HSA executables
            for (auto executable_iterator : executables) {
                HSAExecutable *executable = executable_iterator.second;

                // get symbol
                hsa_executable_symbol_t symbol;
                status = hsa_executable_get_symbol_by_name(executable->hsaExecutable, symbolName, const_cast<hsa_agent_t*>(&agent), &symbol);
                //STATUS_CHECK_SYMBOL(status, symbolName, __LINE__);

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
            throw Kalmar::runtime_exception("HSA executable NOT built yet!", 0);
        }

        return symbol_ptr;
    }

    // FIXME: return values
    // TODO: Need more info about hostptr, is it OS allocated buffer or HSA allocator allocated buffer.
    // Or it might be the responsibility of caller? Because for OS allocated buffer, we need to call hsa_amd_memory_lock, otherwise, need to call
    // hsa_amd_agents_allow_access. Assume it is HSA allocated buffer.
    void memcpySymbol(void* symbolAddr, void* hostptr, size_t count, size_t offset = 0, enum hcCommandKind kind = hcMemcpyHostToDevice) override {
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
            throw Kalmar::runtime_exception("HSA executable NOT built yet!", 0);
        }
    }

    // FIXME: return values
    void memcpySymbol(const char* symbolName, void* hostptr, size_t count, size_t offset = 0, enum hcCommandKind kind = hcMemcpyHostToDevice) override {
        if (executables.size() != 0) {
            unsigned long* symbol_ptr = (unsigned long*)getSymbolAddress(symbolName);
            memcpySymbol(symbol_ptr, hostptr, count, offset, kind);
        } else {
            throw Kalmar::runtime_exception("HSA executable NOT built yet!", 0);
        }
    }

    void* getHSAAgent() override;

    hcAgentProfile getProfile() override { return profile; }

private:

    void BuildOfflineFinalizedProgramImpl(void* kernelBuffer, int kernelSize) {
        using namespace ELFIO;
        using namespace std;

        hsa_status_t status;

        string index = kernel_checksum((size_t)kernelSize, kernelBuffer);

        // load HSA program if we haven't done so
        if (executables.find(index) == executables.end()) {
            // Create the executable.
            hsa_executable_t hsaExecutable;
            status = hsa_executable_create_alt(
                HSA_PROFILE_FULL,
                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                nullptr,
                &hsaExecutable);
            STATUS_CHECK(status, __LINE__);

            // Define the global symbol hc::printf_buffer with the actual address
            status = hsa_executable_agent_global_variable_define(hsaExecutable, agent
                                                              , "_ZN2hc13printf_bufferE"
                                                              , hc::printf_buffer_locked_va);
            STATUS_CHECK(status, __LINE__);

            elfio reader;
            istringstream tmp{string{
                static_cast<char*>(kernelBuffer),
                static_cast<char*>(kernelBuffer) + kernelSize}};
            reader.load(tmp);

            elfio self_reader;
            self_reader.load("/proc/self/exe");

            const auto symtab =
                find_section_if(self_reader, [](const ELFIO::section* x) {
                    return x->get_type() == SHT_SYMTAB;
            });

            const auto code_object_dynsym =
                find_section_if(reader, [](const ELFIO::section* x) {
                    return x->get_type() == SHT_DYNSYM;
            });

            associate_code_object_symbols_with_host_allocation(
                reader,
                self_reader,
                code_object_dynsym,
                symtab,
                agent,
                hsaExecutable);

            auto code_object_reader = load_code_object_and_freeze_executable(
                kernelBuffer, kernelSize, agent, hsaExecutable);

            if (DBFLAG(DB_INIT)) {
                dumpHSAAgentInfo(agent, "Loading code object ");
            }

            // save everything as an HSAExecutable instance
            executables[index] = new HSAExecutable(
                hsaExecutable, code_object_reader);
        }
    }


    static int get_seqnum_from_agent(hsa_agent_t hsaAgent) ;
};

} // end namespace Kalmar

namespace hc {
// regex for finding format string specifiers
static const std::regex specifierPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([hl]*)([diuoxXfFeEgGaAcsp]){1}");
static const std::regex signedIntegerPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([hl]*)([cdi]){1}");
static const std::regex unsignedIntegerPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([hl]*)([uoxX]){1}");
static const std::regex floatPattern("(%){1}[-+#0]*[0-9]*((.)[0-9]+){0,1}([fFeEgGaA]){1}");
static const std::regex pointerPattern("(%){1}[ps]");
static const std::regex doubleAmpersandPattern("(%){2}");
static const std::string ampersand("%");

static inline void processPrintfPackets(PrintfPacket* packets, const unsigned int numPackets) {

  for (unsigned int i = 0; i < numPackets; ) {

    unsigned int numPrintfArgs = packets[i++].data.ui;
    if (numPrintfArgs == 0)
      continue;

    // get the format
    unsigned int formatStringIndex = i++;
    assert(packets[formatStringIndex].type == PRINTF_CHAR_PTR
           || packets[formatStringIndex].type == PRINTF_CONST_CHAR_PTR);
    std::string formatString((const char*)packets[formatStringIndex].data.cptr);
    std::smatch specifierMatches;

#if HC_PRINTF_DEBUG
    std::printf("%s:%d \t number of matches = %d\n", __FUNCTION__, __LINE__, (int)specifierMatches.size());
#endif

    for (unsigned int j = 1; j < numPrintfArgs; ++j, ++i) {

      if (!std::regex_search(formatString, specifierMatches, specifierPattern)) {
        // More printf argument than format specifier??
        // Just skip to the next printf request
        i+=(numPrintfArgs - j);
        break;
      }

      std::string specifier = specifierMatches.str();
#if HC_PRINTF_DEBUG
      std::cout << " (specifier found: " << specifier << ") ";
#endif

      // print the substring before the specifier
      // clean up all the double ampersands
      std::string prefix = specifierMatches.prefix();
      prefix = std::regex_replace(prefix,doubleAmpersandPattern,ampersand);
      std::printf("%s",prefix.c_str());

      std::smatch specifierTypeMatch;
      if (std::regex_search(specifier, specifierTypeMatch, unsignedIntegerPattern)) {
        std::printf(specifier.c_str(), packets[i].data.ui);
      } else if (std::regex_search(specifier, specifierTypeMatch, signedIntegerPattern)) {
        std::printf(specifier.c_str(), packets[i].data.i);
      } else if (std::regex_search(specifier, specifierTypeMatch, floatPattern)) {
        if (packets[i].type == PRINTF_HALF)
          std::cout << static_cast<float>(packets[i].data.h);
        else if (packets[i].type == PRINTF_FLOAT)
          std::printf(specifier.c_str(), packets[i].data.f);
        else
          std::printf(specifier.c_str(), packets[i].data.d);
      } else if (std::regex_search(specifier, specifierTypeMatch, pointerPattern)) {
        std::printf(specifier.c_str(), packets[i].data.cptr);
      }
      else {
        assert(false);
      }
      formatString = specifierMatches.suffix();
    }
    // print the substring after the last specifier
    // clean up all the double ampersands before printing
    formatString = std::regex_replace(formatString,doubleAmpersandPattern,ampersand);
    std::printf("%s",formatString.c_str());
  }
  std::flush(std::cout);
}

static inline void processPrintfBuffer(PrintfPacket* gpuBuffer) {

  if (gpuBuffer == nullptr) return;

  unsigned int cursor = gpuBuffer[PRINTF_OFFSETS].data.uia[0];

  // check whether the printf buffer is non-empty
  if (cursor !=  PRINTF_HEADER_SIZE) {
    unsigned int bufferSize = gpuBuffer[PRINTF_BUFFER_SIZE].data.ui;
    unsigned int numPackets = ((bufferSize<cursor)?bufferSize:cursor) - PRINTF_HEADER_SIZE;

    processPrintfPackets(gpuBuffer+PRINTF_HEADER_SIZE, numPackets);

    // reset the printf buffer and string buffer
    gpuBuffer[PRINTF_OFFSETS].data.uia[0] = PRINTF_HEADER_SIZE;
    gpuBuffer[PRINTF_OFFSETS].data.uia[1] = 0;
  }
}
} // end namespace hc

namespace Kalmar {

template <typename T>
static void hccgetenv(const char *var_name, T *var, const char *usage)
{
    char * env = getenv(var_name);

    if (env != NULL) {
        long int t = strtol(env, NULL, 0);
        *var = t;
    }

    if (HCC_PRINT_ENV) {
        std::cout << std::left << std::setw(30) << var_name << " = " << *var << " : " << usage << std::endl;
    };
}


// specialize for char*
template <>
void hccgetenv(const char *var_name, char **var, const char *usage)
{
    char * env = getenv(var_name);

    if (env != NULL) {
        *var = env;
    }

    if (HCC_PRINT_ENV) {
        std::cout << std::left << std::setw(30) << var_name << " = " << *var << " : " << usage << std::endl;
    };
}

// Helper function to return environment var:
// Handles signed int or long int types, note call to strol above:
#define GET_ENV_INT(envVar, usage)  hccgetenv (#envVar, &envVar, usage)
#define GET_ENV_STRING(envVar, usage)  hccgetenv (#envVar, &envVar, usage)



class HSAContext final : public KalmarContext
{
public:
    std::map<uint64_t, HSADevice *> agentToDeviceMap_;
private:
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

    // GPU devices
    std::vector<hsa_agent_t> agents;

    std::ofstream hccProfileFile; // if using a file open it here
    std::ostream *hccProfileStream = nullptr; // point at file or default stream

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

        if (DBFLAG(DB_INIT)) {
            char name[64];
            uint32_t node = 0;
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
            STATUS_CHECK(status, __LINE__);
            status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
            STATUS_CHECK(status, __LINE__);
            if (device_type == HSA_DEVICE_TYPE_GPU) {
                DBOUTL(DB_INIT,"GPU HSA agent: " << name << " Node ID: " << node );
            } else if (device_type == HSA_DEVICE_TYPE_CPU) {
                DBOUTL(DB_INIT,"CPU HSA agent: " << name << " Node ID: " << node );
            } else {
                DBOUTL(DB_INIT,"Other HSA agent: " << name << " Node ID: " << node );
            }
        }

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
    void ReadHccEnv() ;
    std::ostream &getHccProfileStream() const { return *hccProfileStream; };

    HSAContext() : KalmarContext(), signalPool(), signalPoolFlag(), signalCursor(0), signalPoolMutex() {
        host.handle = (uint64_t)-1;

        ReadHccEnv();

        // initialize HSA runtime

        DBOUT(DB_INIT,"HSAContext::HSAContext(): init HSA runtime");

        hsa_status_t status;
        status = hsa_init();
        if (status != HSA_STATUS_SUCCESS)
          return;

        STATUS_CHECK(status, __LINE__);

        // Iterate over the agents to find out gpu device
        status = hsa_iterate_agents(&HSAContext::find_gpu, &agents);
        STATUS_CHECK(status, __LINE__);

        // Iterate over agents to find out the first cpu device as host
        status = hsa_iterate_agents(&HSAContext::find_host, &host);
        STATUS_CHECK(status, __LINE__);

        // The Devices vector is not empty here since CPU devices have
        // been added to this vector already.  This provides the index
        // to first GPU device that will be added to Devices vector
        int first_gpu_index = Devices.size();

        Devices.resize(Devices.size() + agents.size());
        for (int i = 0; i < agents.size(); ++i) {
            hsa_agent_t agent = agents[i];
            Devices[first_gpu_index + i] = new HSADevice(agent, host, i);
        }

        DBOUT(DB_INIT, "Setting GPU " << HCC_DEFAULT_GPU << " as the default accelerator\n");
        if (first_gpu_index + HCC_DEFAULT_GPU >= Devices.size()) {
            hc::print_backtrace();
            std::cerr << "GPU device " << HCC_DEFAULT_GPU << " doesn't not exist\n" << std::endl;
            abort();
        }
        def = Devices[first_gpu_index + HCC_DEFAULT_GPU];

        signalPoolMutex.lock();

        // pre-allocate signals
        DBOUT(DB_SIG,  " pre-allocate " << HCC_SIGNAL_POOL_SIZE << " signals\n");
        for (int i = 0; i < HCC_SIGNAL_POOL_SIZE; ++i) {
          hsa_signal_t signal;
          status = hsa_signal_create(1, 0, NULL, &signal);
          STATUS_CHECK(status, __LINE__);
          signalPool.push_back(signal);
          signalPoolFlag.push_back(false);
        }

        signalPoolMutex.unlock();

        initPrintfBuffer();

        init_success = true;
    }

    void releaseSignal(hsa_signal_t signal, int signalIndex) {

        if (signal.handle) {

            DBOUT(DB_SIG, "  releaseSignal: 0x" << std::hex << signal.handle << std::dec << " and restored value to 1\n");
            hsa_status_t status = HSA_STATUS_SUCCESS;
            signalPoolMutex.lock();

            // restore signal to the initial value 1
            hsa_signal_store_screlease(signal, 1);

            // mark the signal pointed by signalIndex as available
            signalPoolFlag[signalIndex] = false;

            signalPoolMutex.unlock();
        }
    }

    std::pair<hsa_signal_t, int> getSignal() {
        hsa_signal_t ret;

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

                // increase signal pool on demand by HCC_SIGNAL_POOL_SIZE

                // keep track of the size of signal pool before increasing it
                int oldSignalPoolSize = signalPool.size();
                int oldSignalPoolFlagSize = signalPoolFlag.size();
                assert(oldSignalPoolSize == oldSignalPoolFlagSize);

                DBOUTL(DB_RESOURCE, "Growing signal pool from " << signalPool.size() << " to " << signalPool.size() + HCC_SIGNAL_POOL_SIZE);

                // increase signal pool on demand for another HCC_SIGNAL_POOL_SIZE
                for (int i = 0; i < HCC_SIGNAL_POOL_SIZE; ++i) {
                    hsa_signal_t signal;
                    status = hsa_signal_create(1, 0, NULL, &signal);
                    STATUS_CHECK(status, __LINE__);
                    signalPool.push_back(signal);
                    signalPoolFlag.push_back(false);
                }

                DBOUT(DB_SIG,  "grew signal pool to size=" << signalPool.size() << "\n");

                assert(signalPool.size() == oldSignalPoolSize + HCC_SIGNAL_POOL_SIZE);
                assert(signalPoolFlag.size() == oldSignalPoolFlagSize + HCC_SIGNAL_POOL_SIZE);

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
        return std::make_pair(ret, cursor);
    }

    ~HSAContext() {
        hsa_status_t status = HSA_STATUS_SUCCESS;
        DBOUT(DB_INIT, "HSAContext::~HSAContext() in\n");

        if (!init_success)
          return;

        // deallocate the printf buffer
        if (HCC_ENABLE_PRINTF &&
            hc::printf_buffer != nullptr) {
           // do a final flush
           flushPrintfBuffer();

           hc::deletePrintfBuffer(hc::printf_buffer);
        }
        status = hsa_amd_memory_unlock(&hc::printf_buffer);
        STATUS_CHECK(status, __LINE__);
        hc::printf_buffer_locked_va = nullptr;

        // destroy all KalmarDevices associated with this context
        for (auto dev : Devices)
            delete dev;
        Devices.clear();
        def = nullptr;

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

        // shutdown HSA runtime
        status = hsa_shut_down();
        STATUS_CHECK(status, __LINE__);

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

    void initPrintfBuffer() override {

        if (HCC_ENABLE_PRINTF) {
          if (hc::printf_buffer != nullptr) {
            // Check whether the printf buffer is still valid
            // because it may have been annihilated by HIP's hipDeviceReset().
            // Re-allocate the printf buffer if that happens.
            hc::AmPointerInfo info;
            am_status_t status = am_memtracker_getinfo(&info, hc::printf_buffer);
            if (status != AM_SUCCESS) {
              hc::printf_buffer = nullptr;
            }
          }
          if (hc::printf_buffer == nullptr) {
            hc::printf_buffer = hc::createPrintfBuffer(hc::default_printf_buffer_size);
          }
        }

        // pinned hc::printf_buffer so that the GPUs could access it
        if (hc::printf_buffer_locked_va == nullptr) {
          hsa_status_t status = HSA_STATUS_SUCCESS;
          hsa_agent_t* hsa_agents = agents.data();
          status = hsa_amd_memory_lock(&hc::printf_buffer, sizeof(hc::printf_buffer),
                                       hsa_agents, agents.size(), (void**)&hc::printf_buffer_locked_va);
          STATUS_CHECK(status, __LINE__);
        }
    }

    void flushPrintfBuffer() override {

      if (!HCC_ENABLE_PRINTF)  return;

      hc::processPrintfBuffer(hc::printf_buffer);
    }

    void* getPrintfBufferPointerVA() override {
      return hc::printf_buffer_locked_va;
    }
};

static HSAContext ctx;

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSADevice
// ----------------------------------------------------------------------
namespace Kalmar {


// Global free function to read HCC_ENV vars.  Really this should be called once per process not once-per-event.
// Global so HCC clients or debuggers can force a re-read of the environment variables.
void HSAContext::ReadHccEnv()
{
    GET_ENV_INT(HCC_PRINT_ENV, "Print values of HCC environment variables");

   // 0x1=pre-serialize, 0x2=post-serialize , 0x3= pre- and post- serialize.
   // HCC_SERIALIZE_KERNEL serializes PFE, GL, and dispatch_hsa_kernel calls.
   // HCC_SERIALIZE_COPY serializes av::copy_async operations.  (array_view copies are not currently impacted))
    GET_ENV_INT(HCC_SERIALIZE_KERNEL,
                 "0x1=pre-serialize before each kernel launch, 0x2=post-serialize after each kernel launch, 0x3=both");
    GET_ENV_INT(HCC_SERIALIZE_COPY,
                 "0x1=pre-serialize before each data copy, 0x2=post-serialize after each data copy, 0x3=both");

    GET_ENV_INT(HCC_FORCE_COMPLETION_FUTURE, "Force all kernel commands to allocate a completion signal.");


    GET_ENV_INT(HCC_DB, "Enable HCC trace debug");
    GET_ENV_INT(HCC_DB_SYMBOL_FORMAT, "Select format of symbol (kernel) name used in debug.  0=short,1=mangled,1=demangled.  Bit 0x10 removes arguments.");

    GET_ENV_INT(HCC_OPT_FLUSH, "Perform system-scope acquire/release only at CPU sync boundaries (rather than after each kernel)");
    GET_ENV_INT(HCC_FORCE_CROSS_QUEUE_FLUSH, "create_blocking_marker will force need for sys acquire (0x1) and release (0x2) queue where the marker is created. 0x3 sets need for both flags.");
    GET_ENV_INT(HCC_MAX_QUEUES, "Set max number of HSA queues this process will use.  accelerator_views will share the allotted queues and steal from each other as necessary");

    GET_ENV_INT(HCC_SIGNAL_POOL_SIZE, "Number of pre-allocated HSA signals.  Signals are precious resource so manage carefully");

    GET_ENV_INT(HCC_ASYNCOPS_SIZE, "Number of HSA operations to allow prior to resource cleanup.");

    GET_ENV_INT(HCC_UNPINNED_COPY_MODE, "Select algorithm for unpinned copies. 0=ChooseBest(see thresholds), 1=PinInPlace, 2=StagingBuffer, 3=Memcpy");

    GET_ENV_INT(HCC_CHECK_COPY, "Check dst == src after each copy operation.  Only works on large-bar systems.");


    // Select thresholds to use for unpinned copies
    GET_ENV_INT (HCC_H2D_STAGING_THRESHOLD,    "Min size (in KB) to use staging buffer algorithm for H2D copy if ChooseBest algorithm selected");
    GET_ENV_INT (HCC_H2D_PININPLACE_THRESHOLD, "Min size (in KB) to use pin-in-place algorithm for H2D copy if ChooseBest algorithm selected");
    GET_ENV_INT (HCC_D2H_PININPLACE_THRESHOLD, "Min size (in KB) to use pin-in-place for D2H copy if ChooseBest algorithm selected");

    GET_ENV_INT (HCC_STAGING_BUFFER_SIZE, "Unpinned copy engine staging buffer size in KB");
  
    // Change the default GPU
    GET_ENV_INT (HCC_DEFAULT_GPU, "Change the default GPU (Default is device 0)");

    // Enable printf support
    GET_ENV_INT (HCC_ENABLE_PRINTF, "Enable hc::printf");

    GET_ENV_INT    (HCC_PROFILE,         "Enable HCC kernel and data profiling.  1=summary, 2=trace");
    GET_ENV_INT    (HCC_PROFILE_VERBOSE, "Bitmark to control profile verbosity and format. 0x1=default, 0x2=show begin/end, 0x4=show barrier");
    GET_ENV_STRING (HCC_PROFILE_FILE,    "Set file name for HCC_PROFILE mode.  Default=stderr");

    if (HCC_PROFILE) {
        if (HCC_PROFILE_FILE==nullptr || !strcmp(HCC_PROFILE_FILE, "stderr")) {
            ctx.hccProfileStream = &std::cerr;
        } else if (!strcmp(HCC_PROFILE_FILE, "stdout")) {
            ctx.hccProfileStream = &std::cout;
        } else {
            ctx.hccProfileFile.open(HCC_PROFILE_FILE, std::ios::out);
            assert (!ctx.hccProfileFile.fail());

            ctx.hccProfileStream = &ctx.hccProfileFile;
        }
    }

};


HSADevice::HSADevice(hsa_agent_t a, hsa_agent_t host, int x_accSeqNum) : 
                               KalmarDevice(get_access_type(a)),
                               agent(a), programs(), max_tile_static_size(0),
                               queue_size(0), queues(), queues_mutex(),
                               rocrQueues(/*empty*/), rocrQueuesMutex(),
                               ri(),
                               useCoarseGrainedRegion(false),
                               kernargPool(), kernargPoolFlag(), kernargCursor(0), kernargPoolMutex(),
                               executables(),
                               path(), description(), hostAgent(host),
                               versionMajor(0), versionMinor(0), accSeqNum(x_accSeqNum), queueSeqNums(0) {
    DBOUT(DB_INIT, "HSADevice::HSADevice()\n");

    hsa_status_t status = HSA_STATUS_SUCCESS;

    /// set up path and description
    /// and version information
    {
        char name[64] {0};
        node = 0;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
        STATUS_CHECK(status, __LINE__);
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node);
        STATUS_CHECK(status, __LINE__);

        wchar_t path_wchar[128] {0};
        wchar_t description_wchar[128] {0};
        swprintf(path_wchar, 128, L"%s%u", name, node);
        swprintf(description_wchar, 128, L"AMD HSA Agent %s, Node %u", name, node);

        path = std::wstring(path_wchar);
        description = std::wstring(description_wchar);

        if (DBFLAG(DB_INIT)) {
          DBWSTREAM << L"Path: " << path << L"\n";
          DBWSTREAM << L"Description: " << description << L"\n";
        }

        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_VERSION_MAJOR, &versionMajor);
        STATUS_CHECK(status, __LINE__);
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_VERSION_MINOR, &versionMinor);
        STATUS_CHECK(status, __LINE__);

        DBOUT(DB_INIT,"  Version Major: " << versionMajor << " Minor: " << versionMinor << "\n");
    }


    {
        /// Set the queue size to use when creating hsa queues:
        this->queue_size = 0;
        status = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &this->queue_size);
        STATUS_CHECK(status, __LINE__);

        // MAX_INFLIGHT_COMMANDS_PER_QUEUE throttles the number of commands that can be in the queue, so no reason
        // to allocate a huge HSA queue - size it to it is large enough to handle the inflight commands.
        this->queue_size = 2*MAX_INFLIGHT_COMMANDS_PER_QUEUE;

        // Check that the queue size is valid, these assumptions are used in hsa_queue_create.
        assert (__builtin_popcount(MAX_INFLIGHT_COMMANDS_PER_QUEUE) == 1); // make sure this is power of 2.
    }

    status = hsa_amd_profiling_async_copy_enable(1);
    STATUS_CHECK(status, __LINE__);



    /// Iterate over memory pool of the device and its host
    status = hsa_amd_agent_iterate_memory_pools(agent, HSADevice::find_group_memory, &max_tile_static_size);
    STATUS_CHECK(status, __LINE__);

    status = hsa_amd_agent_iterate_memory_pools(agent, &HSADevice::get_memory_pools, &ri);
    STATUS_CHECK(status, __LINE__);

    status = hsa_amd_agent_iterate_memory_pools(hostAgent, HSADevice::get_host_pools, &ri);
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
    /// - compile-time macro KERNARG_POOL_SIZE is larger than 0
#if KERNARG_POOL_SIZE > 0
    growKernargBuffer();
#endif

    // Setup AM pool.
    ri._am_memory_pool = (ri._found_local_memory_pool)
                             ? ri._local_memory_pool
                             : ri._finegrained_system_memory_pool;

    ri._am_host_memory_pool = (ri._found_coarsegrained_system_memory_pool)
                                  ? ri._coarsegrained_system_memory_pool
                                  : ri._finegrained_system_memory_pool;

    ri._am_host_coherent_memory_pool = (ri._found_finegrained_system_memory_pool)
                                  ? ri._finegrained_system_memory_pool
                                  : ri._coarsegrained_system_memory_pool;

    // No fallback for finegrained GPU memory. Applications might rely on memory type.
    ri._am_finegrained_memory_pool = ri._finegrained_local_memory_pool;

    /// Query the maximum number of work-items in a workgroup
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &workgroup_max_size);
    STATUS_CHECK(status, __LINE__);

    /// Query the maximum number of work-items in each dimension of a workgroup
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &workgroup_max_dim);

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

    //---
    this->copy_mode = static_cast<UnpinnedCopyEngine::CopyMode> (HCC_UNPINNED_COPY_MODE);
    //Provide an environment variable to select the mode used to perform the copy operaton
    switch (this->copy_mode) {
        case UnpinnedCopyEngine::ChooseBest:    //0
        case UnpinnedCopyEngine::UsePinInPlace: //1
        case UnpinnedCopyEngine::UseStaging:    //2
        case UnpinnedCopyEngine::UseMemcpy:     //3
            break;
        default:
            this->copy_mode = UnpinnedCopyEngine::ChooseBest;
    };

    HCC_H2D_STAGING_THRESHOLD    *= 1024;
    HCC_H2D_PININPLACE_THRESHOLD *= 1024;
    HCC_D2H_PININPLACE_THRESHOLD *= 1024;

    static const size_t stagingSize = HCC_STAGING_BUFFER_SIZE * 1024;

    // FIXME: Disable optimizated data copies on large bar system for now due to stability issues
    //this->cpu_accessible_am = hasAccess(hostAgent, ri._am_memory_pool);
    this->cpu_accessible_am = false;

    hsa_amd_memory_pool_t hostPool = (getHSAAMHostRegion());
    copy_engine[0] = new UnpinnedCopyEngine(agent, hostAgent, stagingSize, 2/*staging buffers*/,
                                            this->cpu_accessible_am,
                                            HCC_H2D_STAGING_THRESHOLD,
                                            HCC_H2D_PININPLACE_THRESHOLD,
                                            HCC_D2H_PININPLACE_THRESHOLD);

    copy_engine[1] = new UnpinnedCopyEngine(agent, hostAgent, stagingSize, 2/*staging Buffers*/,
                                            this->cpu_accessible_am,
                                            HCC_H2D_STAGING_THRESHOLD,
                                            HCC_H2D_PININPLACE_THRESHOLD,
                                            HCC_D2H_PININPLACE_THRESHOLD);


    if (HCC_CHECK_COPY && !this->cpu_accessible_am) {
        throw Kalmar::runtime_exception("HCC_CHECK_COPY can only be used on machines where accelerator memory is visible to CPU (ie large-bar systems)", 0);
    }


    ctx.agentToDeviceMap_.insert(std::pair<uint64_t, HSADevice*> (agent.handle, this));

}

inline void*
HSADevice::getHSAAgent() override {
    return static_cast<void*>(&getAgent());
}

static int get_seqnum_from_agent(hsa_agent_t hsaAgent)
{
    auto i = ctx.agentToDeviceMap_.find(hsaAgent.handle);
    if (i != ctx.agentToDeviceMap_.end()) {
        return i->second->get_seqnum();
    } else {
        return -1;
    }
}

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSAQueue
// ----------------------------------------------------------------------
namespace Kalmar  {


std::ostream& operator<<(std::ostream& os, const HSAQueue & hav)
{
    auto device = static_cast<Kalmar::HSADevice*>(hav.getDev());
    os << "queue#" << device->accSeqNum << "." << hav.queueSeqNum;
    return os;
}




HSAQueue::HSAQueue(KalmarDevice* pDev, hsa_agent_t agent, execute_order order, queue_priority priority) :
    KalmarQueue(pDev, queuing_mode_automatic, order, priority),
    rocrQueue(nullptr),
    asyncOps(HCC_ASYNCOPS_SIZE), asyncOpsIndex(0),
    valid(true), _nextSyncNeedsSysRelease(false), _nextKernelNeedsSysAcquire(false), bufferKernelMap(), kernelBufferMap()
{
    {
        // Protect the HSA queue we can steal it.
        DBOUT(DB_LOCK, " ptr:" << this << " create lock_guard...\n");

        std::lock_guard<std::recursive_mutex> l(this->qmutex);

        auto device = static_cast<Kalmar::HSADevice*>(this->getDev());
        device->createOrstealRocrQueue(this, priority);
    }


    youngestCommandKind = hcCommandInvalid;

    hsa_status_t status= hsa_signal_create(1, 1, &agent, &sync_copy_signal);
    STATUS_CHECK(status, __LINE__);
}


void HSAQueue::dispose() override {
    hsa_status_t status;

    DBOUT(DB_INIT, "HSAQueue::dispose() " << this << "in\n");
    {
        DBOUT(DB_LOCK, " ptr:" << this << " dispose lock_guard...\n");

        Kalmar::HSADevice* device = static_cast<Kalmar::HSADevice*>(getDev());

        // NOTE: needs to acquire rocrQueuesMutex and then the qumtex in this
        // sequence in order to avoid potential deadlock with other threads
        // executing createOrstealRocrQueue at the same time
        std::lock_guard<std::mutex> rl(device->rocrQueuesMutex);
        std::lock_guard<std::recursive_mutex> l(this->qmutex);

        if (HCC_OPT_FLUSH && nextSyncNeedsSysRelease()) {
            auto marker = EnqueueMarker(hc::system_scope);
            DBOUTL(DB_CMD2, "HSAQueue::dispose() Sys-release needed, enqueued marker into " << *this << " to release written data " << marker);
        }

        // clear asyncOps to trigger any lingering resource cleanup while we still hold the locks
        // this implicitly waits on all remaining async ops as they destruct, including the
        // possible marker from above
        asyncOps.clear();

        this->valid = false;

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

        if (this->rocrQueue != nullptr) {
            device->removeRocrQueue(rocrQueue);
            rocrQueue = nullptr;
        }
    }

    status = hsa_signal_destroy(sync_copy_signal);

    STATUS_CHECK(status, __LINE__);

    DBOUT(DB_INIT, "HSAQueue::dispose() " << this <<  " out\n");
}

Kalmar::HSADevice * HSAQueue::getHSADev() const {
    return static_cast<Kalmar::HSADevice*>(this->getDev());
};

hsa_queue_t *HSAQueue::acquireLockedRocrQueue() {
    DBOUT(DB_LOCK, " ptr:" << this << " lock...\n");
    this->qmutex.lock();
    if (this->rocrQueue == nullptr) {
        auto device = static_cast<Kalmar::HSADevice*>(this->getDev());
        device->createOrstealRocrQueue(this);
    }

    DBOUT (DB_QUEUE, "acquireLockedRocrQueue returned hwQueue=" << this->rocrQueue->_hwQueue << "\n");
    assert (this->rocrQueue->_hwQueue != 0);
    return this->rocrQueue->_hwQueue;
}

void HSAQueue::releaseLockedRocrQueue()
{

    DBOUT(DB_LOCK, " ptr:" << this << " unlock...\n");
    this->qmutex.unlock();
}

inline void*
HSAQueue::getHSAAgent() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getAgent()));
}
inline void*
HSAQueue::getHostAgent() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHostAgent()));
}
inline void*
HSAQueue::getHSAAMRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAAMRegion()));
}
inline void*
HSAQueue::getHSAFinegrainedAMRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAFinegrainedAMRegion()));
}
inline void*
HSAQueue::getHSACoherentAMHostRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSACoherentAMHostRegion()));
}
inline void*
HSAQueue::getHSAAMHostRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAAMHostRegion()));
}


inline void*
HSAQueue::getHSAKernargRegion() override {
    return static_cast<void*>(&(static_cast<HSADevice*>(getDev())->getHSAKernargRegion()));
}

void HSAQueue::copy_ext(const void *src, void *dst, size_t size_bytes, hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
              const Kalmar::KalmarDevice *copyDevice, bool forceUnpinnedCopy) override {
    // wait for all previous async commands in this queue to finish
    // TODO - can remove this synchronization, copy is tail-synchronous not required on front end.
    this->wait();


    const Kalmar::HSADevice *copyDeviceHsa = static_cast<const Kalmar::HSADevice*> (copyDevice);

    // create a HSACopy instance
    HSACopy* copyCommand = new HSACopy(this, src, dst, size_bytes);
    copyCommand->setCommandKind(copyDir);

    // synchronously do copy
    // FIX me, pull from constructor.
    copyCommand->syncCopyExt(copyDir, srcPtrInfo, dstPtrInfo, copyDeviceHsa, forceUnpinnedCopy);

    // TODO - should remove from queue instead?
    delete(copyCommand);

};


// TODO - remove me
void HSAQueue::copy_ext(const void *src, void *dst, size_t size_bytes, hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo, bool foo) override {

    const Kalmar::KalmarDevice *copyDevice;
    if (srcPtrInfo._isInDeviceMem) {
        copyDevice = (srcPtrInfo._acc.get_dev_ptr());
    } else if (dstPtrInfo._isInDeviceMem) {
        copyDevice = (dstPtrInfo._acc.get_dev_ptr());
    } else {
        copyDevice = nullptr;
    }

    copy_ext(src, dst, size_bytes, copyDir, srcPtrInfo, dstPtrInfo, copyDevice);
}

bool HSAQueue::copy2d_ext(const void *src, void *dst, size_t width, size_t height, size_t srcPitch, size_t dstPitch, hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo, const Kalmar::KalmarDevice *copyDevice, bool forceUnpinnedCopy) { 
    bool retStatus = true;
    this->wait();


    const Kalmar::HSADevice *copyDeviceHsa = static_cast<const Kalmar::HSADevice*> (copyDevice);

    HSACopy* copyCommand = new HSACopy(this, src, dst, width*height);
    copyCommand->setCommandKind(copyDir);

    hsa_status_t status = copyCommand->syncCopy2DExt(copyDir, srcPtrInfo, dstPtrInfo, width, height, srcPitch, dstPitch,copyDeviceHsa, forceUnpinnedCopy);

    delete(copyCommand);
    if(status != HSA_STATUS_SUCCESS)
        retStatus = false;

    return retStatus;
};

std::shared_ptr<KalmarAsyncOp> HSAQueue::EnqueueAsyncCopyExt(const void* src, void* dst, size_t size_bytes,
                                                   hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
                                                   const Kalmar::KalmarDevice *copyDevice) override {

    hsa_status_t status = HSA_STATUS_SUCCESS;

    // create shared_ptr instance
    const Kalmar::HSADevice *copyDeviceHsa = static_cast<const Kalmar::HSADevice*> (copyDevice);
    std::shared_ptr<HSACopy> copyCommand = std::make_shared<HSACopy>(this, src, dst, size_bytes);

    // euqueue the async copy command
    status = copyCommand.get()->enqueueAsyncCopyCommand(copyDeviceHsa, srcPtrInfo, dstPtrInfo);
    STATUS_CHECK(status, __LINE__);

    // associate the async copy command with this queue
    pushAsyncOp(copyCommand);

    return copyCommand;
};

std::shared_ptr<KalmarAsyncOp> HSAQueue::EnqueueAsyncCopy2dExt(const void* src, void* dst, size_t width, size_t height, size_t srcPitch, size_t dstPitch,
                                                   hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,
                                                   const Kalmar::KalmarDevice *copyDevice) override {


    hsa_status_t status = HSA_STATUS_SUCCESS;

    //create shared_ptr instance
    const Kalmar::HSADevice *copy2dDeviceHsa = static_cast<const Kalmar::HSADevice*> (copyDevice);
    std::shared_ptr<HSACopy> copy2dCommand = std::make_shared<HSACopy>(this, src, dst, width*height);

    //euqueue the async copy command
    status = copy2dCommand.get()->enqueueAsyncCopy2dCommand(width, height, srcPitch, dstPitch, copy2dDeviceHsa, srcPtrInfo, dstPtrInfo);
    if(status != HSA_STATUS_SUCCESS)
        return nullptr;
    //associate the async copy command with this queue
    pushAsyncOp(copy2dCommand);

    return copy2dCommand;
};

// enqueue an async copy command
std::shared_ptr<KalmarAsyncOp> HSAQueue::EnqueueAsyncCopy(const void *src, void *dst, size_t size_bytes) override {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    // create shared_ptr instance
    std::shared_ptr<HSACopy> copyCommand = std::make_shared<HSACopy>(this, src, dst, size_bytes);


    hc::accelerator acc;
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);

    bool srcInTracker = (hc::am_memtracker_getinfo(&srcPtrInfo, src) == AM_SUCCESS);
    bool dstInTracker = (hc::am_memtracker_getinfo(&dstPtrInfo, dst) == AM_SUCCESS);

    if (!srcInTracker) {
        // throw an exception
        throw Kalmar::runtime_exception("trying to copy from unpinned src pointer", 0);
    } else if (!dstInTracker) {
        // throw an exception
        throw Kalmar::runtime_exception("trying to copy from unpinned dst pointer", 0);
    };


    // Select optimal copy agent:
    // Prefer source SDMA engine if possible since this is typically the fastest, unless the source data is in host mem.
    //
    // If the src agent cannot see both src and dest pointers, then the async copy will fault.
    // The caller of this function is responsible for avoiding this situation, by examining the
    // host and device allow-access mappings and using a CPU staging copy BEFORE calling
    // this routine.
    const Kalmar::HSADevice *copyDevice;
    if (srcPtrInfo._isInDeviceMem) {  // D2H or D2D
        copyDevice = static_cast<Kalmar::HSADevice*>(srcPtrInfo._acc.get_dev_ptr());
    } else if (dstPtrInfo._isInDeviceMem) { // H2D
        copyDevice = static_cast<Kalmar::HSADevice*>(dstPtrInfo._acc.get_dev_ptr());
    } else {
        copyDevice = nullptr; // H2H
    }

    // enqueue the async copy command
    status = copyCommand.get()->enqueueAsyncCopyCommand(copyDevice, srcPtrInfo, dstPtrInfo);
    STATUS_CHECK(status, __LINE__);

    // associate the async copy command with this queue
    pushAsyncOp(copyCommand);

    return copyCommand;
}


void
HSAQueue::dispatch_hsa_kernel(const hsa_kernel_dispatch_packet_t *aql,
                         const void * args, size_t argSize,
                         hc::completion_future *cf, const char *kernelName) override
{
    uint16_t dims = (aql->setup >> HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) &
                    ((1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS) - 1);

    if (dims == 0) {
        throw Kalmar::runtime_exception("dispatch_hsa_kernel: must set dims in aql.header", 0);
    }

    uint16_t packetType = (aql->header >> HSA_PACKET_HEADER_TYPE) &
                          ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);


    if (packetType != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
        throw Kalmar::runtime_exception("dispatch_hsa_kernel: must set packetType and fence bits in aql.header", 0);
    }


    Kalmar::HSADevice* device = static_cast<Kalmar::HSADevice*>(this->getDev());

    std::shared_ptr<HSADispatch> sp_dispatch = std::make_shared<HSADispatch>(device, this/*queue*/, nullptr, aql);

    HSADispatch *dispatch = sp_dispatch.get();
    waitForStreamDeps(dispatch);

    pushAsyncOp(sp_dispatch);
    dispatch->setKernelName(kernelName);

    // We used to skip signal creation as part of HCC_OPT_FLUSH being true.
    // However, the new asyncOps resource cleanup logic requires all async ops to have a signal.
    dispatch->dispatchKernelAsync(args, argSize, true);

    if (cf) {
        *cf = hc::completion_future(sp_dispatch);
    }
};

} // namespace Kalmar

// ----------------------------------------------------------------------
// member function implementation of HSADispatch
// ----------------------------------------------------------------------

HSADispatch::HSADispatch(Kalmar::HSADevice* _device, Kalmar::KalmarQueue *queue, HSAKernel* _kernel,
                         const hsa_kernel_dispatch_packet_t *aql) :
    HSAOp(hc::HSA_OP_ID_DISPATCH, queue, Kalmar::hcCommandKernel),
    device(_device),
    kernel_name(nullptr),
    kernel(_kernel),
    isDispatched(false),
    waitMode(HSA_WAIT_STATE_BLOCKED),
    future(nullptr),
    kernargMemory(nullptr)
{
    if (aql) {
        this->aql = *aql;
    }
    clearArgs();
}




static std::ostream& PrintHeader(std::ostream& os, uint16_t h)
{
    os << "header=" << std::hex << h << "("
    //os << std::hex << "("
       << "type=" << extractBits(h, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE)
       << ",barrier=" << extractBits (h, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_WIDTH_BARRIER)
       << ",acquire=" << extractBits(h, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE)
       << ",release=" << extractBits(h, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE)
       << ")";


    return os;
}


static std::ostream& operator<<(std::ostream& os, const hsa_kernel_dispatch_packet_t &aql)
{
    PrintHeader(os, aql.header);
    os << " setup=" << std::hex <<  aql.setup
       << " grid=[" << std::dec << aql.grid_size_x << "." <<  aql.grid_size_y << "." <<  aql.grid_size_z << "]"
       << " group=[" << std::dec << aql.workgroup_size_x << "." <<  aql.workgroup_size_y << "." <<  aql.workgroup_size_z << "]"
       << " private_seg_size=" <<  aql.private_segment_size
       << " group_seg_size=" <<  aql.group_segment_size
       << " kernel_object=0x" << std::hex <<  aql.kernel_object
       << " kernarg_address=0x" <<  aql.kernarg_address
       << " completion_signal=0x" <<  aql.completion_signal.handle
       << std::dec;


    return os;
}

static std::string rawAql(const hsa_kernel_dispatch_packet_t &aql)
{
    std::stringstream ss;
    const unsigned *aqlBytes = (unsigned*)&aql;
     ss << "    raw_aql=[" << std::hex << std::setfill('0');
     for (int i=0; i<sizeof(aql)/sizeof(unsigned); i++) {
         ss << " 0x" << std::setw(8) << aqlBytes[i];
     }
     ss << " ]" ;
     return ss.str();
}


static std::ostream& operator<<(std::ostream& os, const hsa_barrier_and_packet_t &aql)
{
    PrintHeader(os, aql.header);
    os << " dep_signal[0]=0x" <<  aql.dep_signal[0].handle
       << " dep_signal[1]=0x" <<  aql.dep_signal[1].handle
       << " dep_signal[2]=0x" <<  aql.dep_signal[2].handle
       << " dep_signal[3]=0x" <<  aql.dep_signal[3].handle
       << " dep_signal[4]=0x" <<  aql.dep_signal[4].handle
       << " completion_signal=0x" <<  aql.completion_signal.handle
       << std::dec;


   return os;
}

//static std::ostream& rawAql(std::ostream& os, const hsa_barrier_and_packet_t &aql)
static std::string rawAql(const hsa_barrier_and_packet_t &aql)
{
    std::stringstream ss;
    const unsigned *aqlBytes = (unsigned*)&aql;
     ss << "    raw_aql=[" << std::hex << std::setfill('0');
     for (int i=0; i<sizeof(aql)/sizeof(unsigned); i++) {
         ss << " 0x" << std::setw(8) << aqlBytes[i];
     }
     ss << " ]" ;
     return ss.str();
}


static void printKernarg(const void *kernarg_address, int bytesToPrint)
{
    const unsigned int *ck = static_cast<const unsigned int*> (kernarg_address);


    std::stringstream ks;
    ks << "kernarg_address: 0x" << kernarg_address << ", total of " << bytesToPrint << " bytes:";
    for (int i=0; i<bytesToPrint/sizeof(unsigned int); i++) {
        bool newLine = ((i % 4) ==0);

        if (newLine) {
            ks << "\n      ";
            ks << "0x" << std::setw(16) << std::setfill('0') << &(ck[i]) <<  ": " ;
        }

        ks << "0x" << std::hex << std::setfill('0') << std::setw(8) << ck[i] << "  ";
    };
    ks << "\n";


    DBOUT(DB_KERNARG, ks.str());

}


// dispatch a kernel asynchronously
// -  allocates signal, copies arguments into kernarg buffer, and places aql packet into queue.
hsa_status_t
HSADispatch::dispatchKernel(hsa_queue_t* lockedHsaQueue, const void *hostKernarg,
                            int hostKernargSize, bool allocSignal) {

    hsa_status_t status = HSA_STATUS_SUCCESS;
    if (isDispatched) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }


    /*
     * Setup the dispatch information.
     */
    // set dispatch fences
    // The fence bits must be set on entry into this function.
    uint16_t header = aql.header;
    if (hsaQueue()->get_execute_order() == Kalmar::execute_in_order) {
        //std::cout << "barrier bit on\n";
        // set AQL header with barrier bit on if execute in order
        header |= ((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                     (1 << HSA_PACKET_HEADER_BARRIER));
    } else {
        //std::cout << "barrier bit off\n";
        // set AQL header with barrier bit off if execute in any order
        header |= (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    }


    // bind kernel arguments
    //printf("hostKernargSize size: %d in bytesn", hostKernargSize);

    if (hostKernargSize > 0) {
        hsa_amd_memory_pool_t kernarg_region = device->getHSAKernargRegion();
        std::pair<void*, int> ret = device->getKernargBuffer(hostKernargSize);
        kernargMemory = ret.first;
        kernargMemoryIndex = ret.second;
        //std::cerr << "op #" << getSeqNum() << " allocated kernarg cursor=" << kernargMemoryIndex << "\n";

        // as kernarg buffers are fine-grained, we can directly use memcpy
        memcpy(kernargMemory, hostKernarg, hostKernargSize);

        aql.kernarg_address = kernargMemory;
    } else {
        aql.kernarg_address = nullptr;
    }


    // write packet
    uint32_t queueMask = lockedHsaQueue->size - 1;
    // TODO: Need to check if package write is correct.
    uint64_t index = hsa_queue_load_write_index_relaxed(lockedHsaQueue);
    uint64_t nextIndex = index + 1;
    if (nextIndex - hsa_queue_load_read_index_scacquire(lockedHsaQueue) >= lockedHsaQueue->size) {
      checkHCCRuntimeStatus(Kalmar::HCCRuntimeStatus::HCCRT_STATUS_ERROR_COMMAND_QUEUE_OVERFLOW, __LINE__, lockedHsaQueue);
    }


    hsa_kernel_dispatch_packet_t* q_aql =
        &(((hsa_kernel_dispatch_packet_t*)(lockedHsaQueue->base_address))[index & queueMask]);

    // Copy mostly-finished AQL packet into the queue
    *q_aql = aql;

    // Set some specific fields:
    if (allocSignal) {
        /*
         * Create a signal to wait for the dispatch to finish.
         */
        std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
        _signal = ret.first;
        _signalIndex = ret.second;
        q_aql->completion_signal = _signal;
    } else {
        _signal.handle = 0;
        _signalIndex = -1;
    }

    // Lastly copy in the header:
    q_aql->header = header;

    hsa_queue_store_write_index_relaxed(lockedHsaQueue, index + 1);
    DBOUTL(DB_AQL, " dispatch_aql " << *this << "(hwq=" << lockedHsaQueue << ") kernargs=" << hostKernargSize << " " << *q_aql );
    DBOUTL(DB_AQL2, rawAql(*q_aql));

    if (DBFLAG(DB_KERNARG)) {
        printKernarg(q_aql->kernarg_address, hostKernargSize);
    }


    // Ring door bell
    hsa_signal_store_relaxed(lockedHsaQueue->doorbell_signal, index);

    isDispatched = true;

    return status;
}



// wait for the kernel to finish execution
inline hsa_status_t
HSADispatch::waitComplete() {
    _wait_complete_status = HSA_STATUS_SUCCESS;
    if (!isDispatched)  {
        _wait_complete_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
        return _wait_complete_status;
    }

    if (_signal.handle) {
        DBOUT(DB_MISC, "wait for kernel dispatch op#" << *this  << " completion with wait flag: " << waitMode << "  signal="<< std::hex  << _signal.handle << std::dec << "\n");

        // wait for completion
        if (hsa_signal_wait_scacquire(_signal, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1), waitMode)!=0) {
            throw Kalmar::runtime_exception("Signal wait returned unexpected value\n", 0);
        }

        DBOUT (DB_MISC, "complete!\n");
    } else {
        // Some commands may have null signal - in this case we can't actually
        // track their status so assume they are complete.
        // In practice, apps would need to use another form of synchronization for
        // these such as waiting on a younger command or using a queue sync.
        DBOUT (DB_MISC, "null signal, considered complete\n");
    }

    isDispatched = false;

    return _wait_complete_status;
}

inline hsa_status_t
HSADispatch::dispatchKernelWaitComplete() {
    _wait_complete_status = HSA_STATUS_SUCCESS;

    if (isDispatched) {
        _wait_complete_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
        return _wait_complete_status;
    }

    // WaitComplete dispatches need to ensure all data is released to system scope
    // This ensures the op is trule "complete" before continuing.
    // This WaitComplete path is used for AMP-style dispatches and may merit future review&optimization.
    aql.header =
        ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    hsaQueue()->setNextKernelNeedsSysAcquire(false);
    hsaQueue()->setNextSyncNeedsSysRelease(false);

    {
        // extract hsa_queue_t from HSAQueue
        hsa_queue_t* rocrQueue = hsaQueue()->acquireLockedRocrQueue();

        // dispatch kernel
        _wait_complete_status = dispatchKernel(rocrQueue, arg_vec.data(), arg_vec.size(), true);
        STATUS_CHECK(_wait_complete_status, __LINE__);

        hsaQueue()->releaseLockedRocrQueue();
    }

    // wait for completion
    waitComplete();
    STATUS_CHECK(_wait_complete_status, __LINE__);

    return _wait_complete_status;
}


// Flavor used when launching dispatch with args and signal created by HCC
// (As opposed to the dispatch_hsa_kernel path)
inline hsa_status_t
HSADispatch::dispatchKernelAsyncFromOp()
{
    return dispatchKernelAsync(arg_vec.data(), arg_vec.size(), true);
}

inline hsa_status_t
HSADispatch::dispatchKernelAsync(const void *hostKernarg, int hostKernargSize, bool allocSignal) {
    if (_activity_prof.is_enabled()) {
        allocSignal = true;
    }

    if (HCC_SERIALIZE_KERNEL & 0x1) {
        hsaQueue()->wait();
    }

    hsa_status_t status = HSA_STATUS_SUCCESS;

    {
        // extract hsa_queue_t from HSAQueue
        hsa_queue_t* rocrQueue = hsaQueue()->acquireLockedRocrQueue();

        // If HCC_OPT_FLUSH=1, we are not flushing to system scope after each command.
        // Set the flag so we remember to do so at next queue::wait() call.
        hsaQueue()->setNextSyncNeedsSysRelease(true);

        if (HCC_OPT_FLUSH) {
          overrideAcquireFenceIfNeeded();
        }

        // dispatch kernel
        status = dispatchKernel(rocrQueue, hostKernarg, hostKernargSize, allocSignal);
        STATUS_CHECK(status, __LINE__);

        hsaQueue()->releaseLockedRocrQueue();
    }


    // dynamically allocate a std::shared_future<void> object
    future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
        waitComplete();
    }).share());

    if (HCC_SERIALIZE_KERNEL & 0x2) {
        future->wait();
        STATUS_CHECK(_wait_complete_status, __LINE__);
    };


    return status;
}

inline void
HSADispatch::dispose() {
    hsa_status_t status;
    if (kernargMemory != nullptr) {
      //std::cerr << "op#" << getSeqNum() << " releasing kernal arg buffer index=" << kernargMemoryIndex<< "\n";
      device->releaseKernargBuffer(kernargMemory, kernargMemoryIndex);
      kernargMemory = nullptr;
    }

    clearArgs();
    std::vector<uint8_t>().swap(arg_vec);

    if (HCC_PROFILE & HCC_PROFILE_TRACE) {
        uint64_t start = getBeginTimestamp();
        uint64_t end   = getEndTimestamp();
        //std::string kname = kernel ? (kernel->kernelName + "+++" + kernel->shortKernelName) : "hmm";
        //LOG_PROFILE(this, start, end, "kernel", kname.c_str(), std::hex << "kernel="<< kernel << " " << (kernel? kernel->kernelCodeHandle:0x0) << " aql.kernel_object=" << aql.kernel_object << std::dec);
        LOG_PROFILE(this, start, end, "kernel", getKernelName(), "");
    }

    _activity_prof.report_gpu_timestamps<HSADispatch>(this);

    if (future != nullptr) {
      delete future;
      future = nullptr;
    }

    Kalmar::ctx.releaseSignal(_signal, _signalIndex);
}

inline uint64_t
HSADispatch::getBeginTimestamp() override {
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(_agent, _signal, &time);
    return time.start;
}

inline uint64_t
HSADispatch::getEndTimestamp() override {
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(_agent, _signal, &time);
    return time.end;
}

void HSADispatch::overrideAcquireFenceIfNeeded()
{
    if (hsaQueue()->nextKernelNeedsSysAcquire())  {
       DBOUT( DB_CMD2, "  kernel AQL packet adding system-scope acquire\n");
       // Pick up system acquire if needed.
       aql.header |= ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) ;
       hsaQueue()->setNextKernelNeedsSysAcquire(false);
    }
}

inline hsa_status_t
HSADispatch::setLaunchConfiguration(const int dims, size_t *globalDims, size_t *localDims,
                                    const int dynamicGroupSize) {
    assert((0 < dims) && (dims <= 3));
    DBOUT(DB_MISC, "static group segment size: " << kernel->static_group_segment_size
                   << " dynamic group segment size: " << dynamicGroupSize << "\n");

    // Set group dims
    // for each workgroup dimension, make sure it does not exceed the maximum allowable limit
    const uint16_t* workgroup_max_dim = device->getWorkgroupMaxDim();

    unsigned int workgroup_size[3] = { 1, 1, 1};

    // Check whether the user specified a workgroup size
    if (localDims[0] != 0) {
      for (int i = 0; i < dims; i++) {
        // If user specify a group size that exceeds the device limit
        // throw an error
        if (localDims[i] > workgroup_max_dim[i]) {
          std::stringstream msg;
          msg << "The extent of the tile (" << localDims[i] 
              << ") exceeds the device limit (" << workgroup_max_dim[i] << ").";
          throw Kalmar::runtime_exception(msg.str().c_str(), -1);
        } else if (localDims[i] > globalDims[i]) {
          std::stringstream msg;
          msg << "The extent of the tile (" << localDims[i] 
              << ") exceeds the compute grid extent (" << globalDims[i] << ").";
          throw Kalmar::runtime_exception(msg.str().c_str(), -1);
        }
        workgroup_size[i] = localDims[i];
      }
    }
    else {

      constexpr unsigned int recommended_flat_workgroup_size = 64;

      // user didn't specify a workgroup size
      if (dims == 1) {
        workgroup_size[0] = recommended_flat_workgroup_size;
      }
      else if (dims == 2) {

        // compute the group size for the 1st dimension
        for (unsigned int i = 1; ; i<<=1) {
          if (i == recommended_flat_workgroup_size
              || i >= globalDims[0]) {
            workgroup_size[0] = 
              std::min(i, static_cast<unsigned int>(globalDims[0]));
            break;
          }
        }

        // compute the group size for the 2nd dimension
        workgroup_size[1] = recommended_flat_workgroup_size / workgroup_size[0];
      }
      else if (dims == 3) {

        // compute the group size for the 1st dimension
        for (unsigned int i = 1; ; i<<=1) {
          if (i == recommended_flat_workgroup_size
              || i >= globalDims[0]) {
            workgroup_size[0] = 
              std::min(i, static_cast<unsigned int>(globalDims[0]));
            break;
          }
        }

        // compute the group size for the 2nd dimension
        for (unsigned int j = 1; ; j<<=1) {
          unsigned int flat_group_size = workgroup_size[0] * j;
          if (flat_group_size > recommended_flat_workgroup_size) {
            workgroup_size[1] = j >> 1;
            break;
          }
          else if (flat_group_size == recommended_flat_workgroup_size
              || j >= globalDims[1]) {
            workgroup_size[1] = 
              std::min(j, static_cast<unsigned int>(globalDims[1]));
            break;
          }
        }

        // compute the group size for the 3rd dimension
        workgroup_size[2] = recommended_flat_workgroup_size / 
                              (workgroup_size[0] * workgroup_size[1]);
      }
    }

    auto kernel = this->kernel;

    auto calculate_kernel_max_flat_workgroup_size = [&] {
      constexpr unsigned int max_num_vgprs_per_work_item = 256;
      constexpr unsigned int num_work_items_per_simd = 64;
      constexpr unsigned int num_simds_per_cu = 4;
      const unsigned int workitem_vgpr_count = std::max((unsigned int)kernel->workitem_vgpr_count, 1u);
      unsigned int max_flat_group_size = (max_num_vgprs_per_work_item / workitem_vgpr_count) 
                                           * num_work_items_per_simd * num_simds_per_cu;
      return max_flat_group_size;
    };

    auto validate_kernel_flat_group_size = [&] {
      const unsigned int actual_flat_group_size = workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
      const unsigned int max_num_work_items_per_cu = calculate_kernel_max_flat_workgroup_size();
      if (actual_flat_group_size > max_num_work_items_per_cu) {
        std::stringstream msg;
        msg << "The number of work items (" << actual_flat_group_size 
            << ") per work group exceeds the limit (" << max_num_work_items_per_cu << ") of kernel "
            << kernel->kernelName << " .";
        throw Kalmar::runtime_exception(msg.str().c_str(), -1);
      }
    };
    validate_kernel_flat_group_size();

    memset(&aql, 0, sizeof(aql));

    // Copy info from kernel into AQL packet:
    // bind kernel code
    aql.kernel_object = kernel->kernelCodeHandle;

    aql.group_segment_size   = kernel->static_group_segment_size + dynamicGroupSize;
    aql.private_segment_size = kernel->private_segment_size;

    // Set global dims:
    aql.grid_size_x = globalDims[0];
    aql.grid_size_y = (dims > 1 ) ? globalDims[1] : 1;
    aql.grid_size_z = (dims > 2 ) ? globalDims[2] : 1;

    aql.workgroup_size_x = workgroup_size[0];
    aql.workgroup_size_y = workgroup_size[1];
    aql.workgroup_size_z = workgroup_size[2];

    aql.setup = dims << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    aql.header = 0;
    if (HCC_OPT_FLUSH) {
        aql.header = ((HSA_FENCE_SCOPE_AGENT) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                     ((HSA_FENCE_SCOPE_AGENT) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
        overrideAcquireFenceIfNeeded();
    } else {
        aql.header = ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                     ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    }

    return HSA_STATUS_SUCCESS;
}


// ----------------------------------------------------------------------
// member function implementation of HSABarrier
// ----------------------------------------------------------------------

// wait for the barrier to complete
inline hsa_status_t
HSABarrier::waitComplete() {
    _wait_complete_status = HSA_STATUS_SUCCESS;
    if (!isDispatched)  {
        _wait_complete_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
        return _wait_complete_status;
    }

    DBOUT(DB_WAIT,  "  wait for barrier " << *this << " completion with wait flag: " << waitMode << "  signal="<< std::hex  << _signal.handle << std::dec <<"...\n");

    // Wait on completion signal until the barrier is finished
    hsa_signal_wait_scacquire(_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, waitMode);


    isDispatched = false;

    return _wait_complete_status;
}


// TODO - remove hsaQueue parm.
inline hsa_status_t
HSABarrier::enqueueAsync(hc::memory_scope fenceScope) {

    hc::memory_scope toDoReleaseFenceScope = fenceScope;
    hc::memory_scope toDoAcquireFenceScope = _acquire_scope;

    hsa_queue_t* rocrQueue = hsaQueue()->acquireLockedRocrQueue();

    if ( hsaQueue()->nextSyncNeedsSysRelease() ) {
         toDoReleaseFenceScope = hc::system_scope;
         hsaQueue()->setNextSyncNeedsSysRelease(false);
         DBOUTL( DB_CMD2, "Fuse System Scope Release Fencing request from the HSAQueue") ;
    };

    if ( hsaQueue()->nextKernelNeedsSysAcquire() ) {
         toDoAcquireFenceScope = hc::system_scope;
         hsaQueue()->setNextKernelNeedsSysAcquire(false);
         DBOUTL( DB_CMD2, "Fuse System Scope Acquire Fencing request from the HSAQueue") ;
    };

    if (toDoReleaseFenceScope > toDoAcquireFenceScope) {
        DBOUTL( DB_CMD2, "  marker overriding acquireScope to " << fenceScope);
        toDoAcquireFenceScope = toDoReleaseFenceScope;
        _acquire_scope = toDoAcquireFenceScope;
    }

    // set acquire scope:
    unsigned fenceBits = 0;

    switch (toDoAcquireFenceScope) {
        case hc::no_scope:
            fenceBits |= ((HSA_FENCE_SCOPE_NONE) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
            break;
        case hc::accelerator_scope:
            fenceBits |= ((HSA_FENCE_SCOPE_AGENT) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
            break;
        case hc::system_scope:
            fenceBits |= ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
            break;
        default:
            STATUS_CHECK(HSA_STATUS_ERROR_INVALID_ARGUMENT, __LINE__);
    }

    switch (toDoReleaseFenceScope) {
        case hc::no_scope:
            fenceBits |= ((HSA_FENCE_SCOPE_NONE) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
            break;
        case hc::accelerator_scope:
            fenceBits |= ((HSA_FENCE_SCOPE_AGENT) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
            break;
        case hc::system_scope:
            fenceBits |= ((HSA_FENCE_SCOPE_SYSTEM) << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
            break;
        default:
            STATUS_CHECK(HSA_STATUS_ERROR_INVALID_ARGUMENT, __LINE__);
    };

    if (isDispatched) {
        STATUS_CHECK(HSA_STATUS_ERROR_INVALID_ARGUMENT, __LINE__);
    }

    // Create a signal to wait for the barrier to finish.
    std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
    _signal = ret.first;
    _signalIndex = ret.second;


    // setup header
    header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
#ifndef AMD_HSA
    // AMD implementation does not require barrier bit on barrier packet and executes a little faster without it set.
    header |= (1 << HSA_PACKET_HEADER_BARRIER);
#endif
    header |= fenceBits;

    // Obtain the write index for the command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(rocrQueue);
    const uint32_t queueMask = rocrQueue->size - 1;
    uint64_t nextIndex = index + 1;
    if (nextIndex - hsa_queue_load_read_index_scacquire(rocrQueue) >= rocrQueue->size) {
      checkHCCRuntimeStatus(Kalmar::HCCRuntimeStatus::HCCRT_STATUS_ERROR_COMMAND_QUEUE_OVERFLOW, __LINE__, rocrQueue);
    }

    // Define the barrier packet to be at the calculated queue index address
    hsa_barrier_and_packet_t* barrier = &(((hsa_barrier_and_packet_t*)(rocrQueue->base_address))[index&queueMask]);
    memset(barrier, 0, sizeof(hsa_barrier_and_packet_t));

    // setup dependent signals
    if ((depCount > 0) && (depCount <= 5)) {
      for (int i = 0; i < depCount; ++i) {
        barrier->dep_signal[i] = *(static_cast <hsa_signal_t*> (depAsyncOps[i]->getNativeHandle()));
      }
    }

    barrier->completion_signal = _signal;

    // Set header last:
    barrier->header = header;

    DBOUTL(DB_AQL, " barrier_aql " << *this << " "<< *barrier );
    DBOUTL(DB_AQL2, rawAql(*barrier));

    // Increment write index and ring doorbell to dispatch the kernel
    hsa_queue_store_write_index_relaxed(rocrQueue, nextIndex);
    hsa_signal_store_relaxed(rocrQueue->doorbell_signal, index);

    isDispatched = true;

    // capture the state of these flags after the barrier executes.
    _barrierNextKernelNeedsSysAcquire = hsaQueue()->nextKernelNeedsSysAcquire();
    _barrierNextSyncNeedsSysRelease   = hsaQueue()->nextSyncNeedsSysRelease();

    hsaQueue()->releaseLockedRocrQueue();

    // dynamically allocate a std::shared_future<void> object
    future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
        waitComplete();
    }).share());

    return HSA_STATUS_SUCCESS;
}


static std::string fenceToString(int fenceBits)
{
    switch (fenceBits) {
        case 0: return "none";
        case 1: return "acc";
        case 2: return "sys";
        case 3: return "sys";
        default: return "???";
    };
}


inline void
HSABarrier::dispose() {
    if ((HCC_PROFILE & HCC_PROFILE_TRACE) && (HCC_PROFILE_VERBOSE & HCC_PROFILE_VERBOSE_BARRIER)) {
        uint64_t start = getBeginTimestamp();
        uint64_t end   = getEndTimestamp();
        int acqBits = extractBits(header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
        int relBits = extractBits(header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);

        std::stringstream depss;
        for (int i=0; i<depCount; i++) {
            if (i==0) {
                depss << " deps=";
            } else {
                depss << ",";
            }
            depss << *depAsyncOps[i];
        };
        LOG_PROFILE(this, start, end, "barrier", "depcnt=" + std::to_string(depCount) + ",acq=" + fenceToString(acqBits) + ",rel=" + fenceToString(relBits), depss.str())
    }

    // Release references to our dependent ops:
    for (int i=0; i<depCount; i++) {
        depAsyncOps[i] = nullptr;
    }

    _activity_prof.report_gpu_timestamps<HSABarrier>(this);

    if (future != nullptr) {
      delete future;
      future = nullptr;
    }

    Kalmar::ctx.releaseSignal(_signal, _signalIndex);
}

inline uint64_t
HSABarrier::getBeginTimestamp() override {
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(_agent, _signal, &time);
    return time.start;
}

inline uint64_t
HSABarrier::getEndTimestamp() override {
    hsa_amd_profiling_dispatch_time_t time;
    hsa_amd_profiling_get_dispatch_time(_agent, _signal, &time);
    return time.end;
}

// ----------------------------------------------------------------------
// member function implementation of HSAOp
// ----------------------------------------------------------------------
HSAOpCoord::HSAOpCoord(Kalmar::HSAQueue *queue) :
        _deviceId(queue->getDev()->get_seqnum()),
        _queueId(queue->getSeqNum())
        {}

HSAOp::HSAOp(hc::HSAOpId id, Kalmar::KalmarQueue *queue, hc::hcCommandKind commandKind) :
    KalmarAsyncOp(queue, commandKind),
    _opCoord(static_cast<Kalmar::HSAQueue*> (queue)),

    _signalIndex(-1),
    _agent(static_cast<Kalmar::HSADevice*>(hsaQueue()->getDev())->getAgent()),
    _dependency_chain_length(0),
    _activity_prof(id, _opCoord._queueId, _opCoord._deviceId)
{
    _signal.handle=0;
    apiStartTick = Kalmar::ctx.getSystemTicks();

    _activity_prof.initialize();
};

Kalmar::HSAQueue *HSAOp::hsaQueue() const 
{ 
    return static_cast<Kalmar::HSAQueue *> (this->getQueue()); 
};

bool HSAOp::isReady() override {
    return (hsa_signal_load_scacquire(_signal) == 0);
}


// ----------------------------------------------------------------------
// member function implementation of HSACopy
// ----------------------------------------------------------------------
//
// Copy mode will be set later on.
// HSA signals would be waited in HSA_WAIT_STATE_ACTIVE by default for HSACopy instances
HSACopy::HSACopy(Kalmar::KalmarQueue *queue, const void* src_, void* dst_, size_t sizeBytes_) : HSAOp(hc::HSA_OP_ID_COPY, queue, Kalmar::hcCommandInvalid),
    isSubmitted(false), isAsync(false), isSingleStepCopy(false), isPeerToPeer(false), future(nullptr), depAsyncOp(nullptr), copyDevice(nullptr), waitMode(HSA_WAIT_STATE_ACTIVE),
    src(src_), dst(dst_),
    sizeBytes(sizeBytes_)
{


    apiStartTick = Kalmar::ctx.getSystemTicks();
}

// wait for the async copy to complete
inline hsa_status_t
HSACopy::waitComplete() {
    _wait_complete_status = HSA_STATUS_SUCCESS;
    if (!isSubmitted)  {
        _wait_complete_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;
        return _wait_complete_status;
    }

    // Wait on completion signal until the async copy is finishedS
    if (DBFLAG(DB_WAIT)) {
        hsa_signal_value_t v = -1000;
        if (_signal.handle) {
            hsa_signal_load_scacquire(_signal);
        }
        DBOUT(DB_WAIT, "  wait for copy op#" << getSeqNum() << " completion with wait flag: " << waitMode << "signal="<< std::hex  << _signal.handle << std::dec <<" currentVal=" << v << "...\n");
    }

    // Wait on completion signal until the async copy is finished
    hsa_signal_wait_scacquire(_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, waitMode);

    isSubmitted = false;

    return _wait_complete_status;
}


void checkCopy(const void *s1, const void *s2, size_t sizeBytes)
{
    if (memcmp(s1, s2, sizeBytes) != 0) {
        throw Kalmar::runtime_exception("HCC_CHECK_COPY mismatch detected", 0);
    }
}



// Small wrapper that calls hsa_amd_memory_async_copy.
// HCC knows exactly which copy-engine it wants to perfom the copy and has already made.
hsa_status_t HSACopy::hcc_memory_async_copy(Kalmar::hcCommandKind copyKind, const Kalmar::HSADevice *copyDeviceArg,
                      const hc::AmPointerInfo &dstPtrInfo, const hc::AmPointerInfo &srcPtrInfo, size_t sizeBytes)
{
    this->isSingleStepCopy = true;

    // beautiful...:
    hsa_agent_t copyAgent = * static_cast<hsa_agent_t*>(const_cast<Kalmar::HSADevice*>(copyDeviceArg)->getHSAAgent());
    hsa_status_t status;
    hsa_device_type_t device_type;
    status = hsa_agent_get_info(copyAgent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != HSA_STATUS_SUCCESS) {
        throw Kalmar::runtime_exception("invalid copy agent used for hcc_memory_async_copy", status);
    }
    if (device_type != HSA_DEVICE_TYPE_GPU) {
        throw Kalmar::runtime_exception("copy agent must be GPU hcc_memory_async_copy", -1);
    }

    hsa_agent_t hostAgent = const_cast<Kalmar::HSADevice *> (copyDeviceArg)->getHostAgent();

    /* Determine src and dst pointer passed to ROCR runtime.
     *
     * Pre-condition:
     * - this->dst and this->src must be tracked by AM API implemantation.
     * - dstPtrInfo and srcPtrInfo must be valid.
     */
    void *dstPtr = nullptr;
    void *srcPtr = nullptr;

    hsa_agent_t srcAgent, dstAgent;
    switch (copyKind) {
        case Kalmar::hcMemcpyHostToHost:
            srcAgent=hostAgent; dstAgent=hostAgent;

            /* H2H case
             * We expect ROCR runtime to continue use the CPU for host to host
             * copies, and thus must pass host pointers here.
             */
            dstPtr = this->dst;
            srcPtr = const_cast<void*>(this->src);
            this->copyDevice = nullptr;
            break;
        case Kalmar::hcMemcpyHostToDevice:
            srcAgent=hostAgent; dstAgent=copyAgent;

            /* H2D case
             * Destination is simply this->dst.
             * Source has to be calculated by adding the offset to the pinned
             * host pointer.
             */
            dstPtr = this->dst;
            srcPtr = reinterpret_cast<unsigned char*>(srcPtrInfo._devicePointer) +
                     (reinterpret_cast<unsigned char*>(const_cast<void*>(this->src)) -
                      reinterpret_cast<unsigned char*>(srcPtrInfo._hostPointer));
            this->copyDevice = copyDeviceArg;
            break;
        case Kalmar::hcMemcpyDeviceToHost:
            srcAgent=copyAgent; dstAgent=hostAgent;

            /* D2H case
             * Source is simply this->src.
             * Desination has to be calculated by adding the offset to the
             * pinned host pointer.
             */
            dstPtr = reinterpret_cast<unsigned char*>(dstPtrInfo._devicePointer) +
                     (reinterpret_cast<unsigned char*>(this->dst) -
                      reinterpret_cast<unsigned char*>(dstPtrInfo._hostPointer));
            srcPtr = const_cast<void*>(this->src);
            this->copyDevice = copyDeviceArg;
            break;
        case Kalmar::hcMemcpyDeviceToDevice:
            this->isPeerToPeer = (dstPtrInfo._acc != srcPtrInfo._acc);
            srcAgent = *reinterpret_cast<hsa_agent_t*>(srcPtrInfo._acc.get_hsa_agent());
            dstAgent = *reinterpret_cast<hsa_agent_t*>(dstPtrInfo._acc.get_hsa_agent());
            /* D2D case
             * Simply pass this->src and this->dst to ROCR runtime.
             */
            dstPtr = this->dst;
            srcPtr = const_cast<void*>(this->src);

            // ROCR will decide which device to make the copy
            this->copyDevice = nullptr;
            break;
        default:
            throw Kalmar::runtime_exception("bad copyKind in hcc_memory_async_copy", copyKind);
    };



    int depSignalCnt = 0;
    hsa_signal_t depSignal = { .handle = 0x0 };
    {
        // Create a signal to wait for the async copy command to finish.
        std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
        _signal = ret.first;
        _signalIndex = ret.second;

        if (!hsaQueue()->nextSyncNeedsSysRelease()) {
            DBOUT( DB_CMD2, "  copy launching without adding system release\n");
        }

        auto fenceScope = (hsaQueue()->nextSyncNeedsSysRelease()) ? hc::system_scope : hc::no_scope;

        depAsyncOp = std::static_pointer_cast<HSAOp> (hsaQueue()->detectStreamDeps(this->getCommandKind(), this));
        if (depAsyncOp) {
            depSignal = * (static_cast <hsa_signal_t*> (depAsyncOp->getNativeHandle()));
        }

        // We need to ensure the copy waits for preceding commands the HCC queue to complete, if those commands exist.
        // The copy has to be set so that it depends on the completion_signal of the youngest command in the queue.
        if (depAsyncOp || fenceScope != hc::no_scope) {
        
            // Normally we can use the input signal to hsa_amd_memory_async_copy to ensure the copy waits for youngest op.
            // However, two cases require special handling:
            //    - the youngest op may not have a completion signal - this is optional for kernel launch commands.
            //    - we may need a system-scope fence. This is true if any kernels have been executed in this queue, or
            //      in streams that we depend on.
            // For both of these cases, we create an additional barrier packet in the source, and attach the desired fence.
            // Then we make the copy depend on the signal written by this command.
            if ((depAsyncOp && depSignal.handle == 0x0) || 
                  (fenceScope != hc::no_scope) ||
                  (depAsyncOp && depAsyncOp->_dependency_chain_length > _max_dependency_chain_length)) {
                DBOUT( DB_CMD2, "  asyncCopy adding marker for needed dependency or release\n");

                // Set depAsyncOp for use by the async copy below:
                depAsyncOp = std::static_pointer_cast<HSAOp> (hsaQueue()->EnqueueMarkerWithDependency(0, nullptr, fenceScope));
                depSignal = * (static_cast <hsa_signal_t*> (depAsyncOp->getNativeHandle()));
            }
            else if (depAsyncOp) {
              this->_dependency_chain_length = depAsyncOp->_dependency_chain_length + 1;
            }

            depSignalCnt = 1;

            DBOUT( DB_CMD2, "  asyncCopy sent with dependency on op#" << depAsyncOp->getSeqNum() << " depSignal="<< std::hex  << depSignal.handle << std::dec <<"\n");
        }


        if (DBFLAG(DB_CMD)) {
            hsa_signal_value_t v = hsa_signal_load_scacquire(_signal);
            DBOUT(DB_CMD,  "  hsa_amd_memory_async_copy launched " << " completionSignal="<< std::hex  << _signal.handle
                      << "  InitSignalValue=" << v << " depSignalCnt=" << depSignalCnt
                      << "  copyAgent=" << copyDevice
                      << "\n");
        }
    }

    /* ROCR logic to select the copy agent:
     *
     *  Decide which copy agent to use :
     *
     *   1. Pick source agent if src agent is a GPU (regardless of the dst agent).
     *   2. Pick destination agent if src argent is not a GPU, and the dst agent is a GPU.
     *   3. If both src and dst agents are CPUs, launch a CPU thread to perform memcpy. Will wait on host for dependent signals to resolve.
     *
     *    Decide which DMA engine on the copy agent to use :
     *
     *     1.   Use SDMA, if the src agent is a CPU AND dst agent is a GPU.
     *     2.   Use SDMA, if the src agent is a GPU AND dst agent is a CPU.
     *     3.   if both src and dst agents are GPU, let ROCR to decide the copy strategy.
     */

    DBOUT(DB_AQL, "hsa_amd_memory_async_copy("
                   <<  "dstPtr=" << dstPtr << ",0x" << std::hex << dstAgent.handle
                   << ",srcPtr=" << srcPtr << ",0x" << std::hex << srcAgent.handle
                   << ",sizeBytes=" << std::dec << sizeBytes
                   << ",depSignalCnt=" << depSignalCnt << "," << &depSignal << ","
                   << std::hex << _signal.handle << "\n" << std::dec);

    status = hsa_amd_memory_async_copy(dstPtr, dstAgent, srcPtr, srcAgent, sizeBytes,
                                         depSignalCnt, depSignalCnt?&depSignal:nullptr, _signal);
    if (status != HSA_STATUS_SUCCESS) {
        throw Kalmar::runtime_exception("hsa_amd_memory_async_copy error", status);
    }

    if (HCC_CHECK_COPY) {
        hsa_signal_wait_scacquire(_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        checkCopy(dstPtr, srcPtr, sizeBytes);
    }

    // Next kernel needs to acquire the result of the copy.
    // This holds true for any copy direction, since host memory can also be cached on this GPU.
    DBOUT( DB_CMD2, "  copy setNextKernelNeedsSysAcquire(true)\n");
    // HSA memory copy requires a system-scope acquire before the next kernel command - set flag here so we remember:
    hsaQueue()->setNextKernelNeedsSysAcquire(true);

    isSubmitted = true;
    return status;
}

hsa_status_t HSACopy::hcc_memory_async_copy_rect(Kalmar::hcCommandKind copyKind, const Kalmar::HSADevice *copyDeviceArg,
                      const hc::AmPointerInfo &dstPtrInfo, const hc::AmPointerInfo &srcPtrInfo, size_t width,
                      size_t height, size_t srcPitch, size_t dstPitch) {
                      
    this->isSingleStepCopy = true;

    hsa_agent_t copyAgent = * static_cast<hsa_agent_t*>(const_cast<Kalmar::HSADevice*>(copyDeviceArg)->getHSAAgent());
    hsa_status_t status;
    hsa_device_type_t device_type;
    status = hsa_agent_get_info(copyAgent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (status != HSA_STATUS_SUCCESS) {
        throw Kalmar::runtime_exception("invalid copy agent used for hcc_memory_async_copy", status);
    }
    if (device_type != HSA_DEVICE_TYPE_GPU) {
        throw Kalmar::runtime_exception("copy agent must be GPU hcc_memory_async_copy", -1);
    }

    hsa_agent_t hostAgent = const_cast<Kalmar::HSADevice *> (copyDeviceArg)->getHostAgent();
    void *dstPtr = nullptr;
    void *srcPtr = nullptr;

    switch (copyKind) {
        case Kalmar::hcMemcpyHostToHost:
            dstPtr = this->dst;
            srcPtr = const_cast<void*>(this->src);
            this->copyDevice = nullptr;
            break;
        case Kalmar::hcMemcpyHostToDevice:
            dstPtr = this->dst;
            srcPtr = reinterpret_cast<unsigned char*>(srcPtrInfo._devicePointer) +
                     (reinterpret_cast<unsigned char*>(const_cast<void*>(this->src)) -
                      reinterpret_cast<unsigned char*>(srcPtrInfo._hostPointer));
            this->copyDevice = copyDeviceArg;
            break;
        case Kalmar::hcMemcpyDeviceToHost:
            dstPtr = reinterpret_cast<unsigned char*>(dstPtrInfo._devicePointer) +
                     (reinterpret_cast<unsigned char*>(this->dst) -
                      reinterpret_cast<unsigned char*>(dstPtrInfo._hostPointer));
            srcPtr = const_cast<void*>(this->src);
            this->copyDevice = copyDeviceArg;
            break;
        case Kalmar::hcMemcpyDeviceToDevice:
            this->isPeerToPeer = (dstPtrInfo._acc != srcPtrInfo._acc);
            dstPtr = this->dst;
            srcPtr = const_cast<void*>(this->src);
            this->copyDevice = nullptr;
            break;
        default:
            throw Kalmar::runtime_exception("bad copyKind in hcc_memory_async_copy", copyKind);
    };
    hsa_pitched_ptr_t src, dst;
    hsa_dim3_t srcOff, dstOff, range;
    src.slice=0;
    dst.slice=0;
    range.z=1;
    srcOff.x=srcOff.y=srcOff.z=0;
    dstOff.x=dstOff.y=dstOff.z=0;
    src.base = srcPtr;
    src.pitch = srcPitch;
    dst.base = dstPtr;
    dst.pitch=dstPitch;
    range.x=width;
    range.y=height;

    int depSignalCnt = 0;
    hsa_signal_t depSignal = { .handle = 0x0 };
    {
        //Create a signal to wait for the async copy command to finish.
        std::pair<hsa_signal_t, int> ret = Kalmar::ctx.getSignal();
        _signal = ret.first;
        _signalIndex = ret.second; 

        if (!hsaQueue()->nextSyncNeedsSysRelease()) {
            DBOUT( DB_CMD2, "  copy launching without adding system release\n");
        }

        auto fenceScope = (hsaQueue()->nextSyncNeedsSysRelease()) ? hc::system_scope : hc::no_scope;

        depAsyncOp = std::static_pointer_cast<HSAOp> (hsaQueue()->detectStreamDeps(this->getCommandKind(), this));
        if (depAsyncOp) {
            depSignal = * (static_cast <hsa_signal_t*> (depAsyncOp->getNativeHandle()));
        }

        if (depAsyncOp || fenceScope != hc::no_scope) {     
            if ((depAsyncOp && depSignal.handle == 0x0) || (fenceScope != hc::no_scope)) {
                DBOUT( DB_CMD2, "  asyncCopy adding marker for needed dependency or release\n");

                depAsyncOp = std::static_pointer_cast<HSAOp> (hsaQueue()->EnqueueMarkerWithDependency(0, nullptr, fenceScope));
                depSignal = * (static_cast <hsa_signal_t*> (depAsyncOp->getNativeHandle()));
            };

            depSignalCnt = 1;

            DBOUT( DB_CMD2, "  asyncCopy sent with dependency on op#" << depAsyncOp->getSeqNum() << " depSignal="<< std::hex  << depSignal.handle << std::dec <<"\n");
        }
        if (DBFLAG(DB_CMD)) {
            hsa_signal_value_t v = hsa_signal_load_scacquire(_signal);
            DBOUT(DB_CMD,  "  hcc_memory_async_copy_rect launched " << " completionSignal="<< std::hex  << _signal.handle
                << "  InitSignalValue=" << v << " depSignalCnt=" << depSignalCnt
                << "  copyAgent=" << copyDevice
                << "\n");
        }
    }

    status = hsa_amd_memory_async_copy_rect(&dst, &dstOff, &src, &srcOff, &range, copyAgent, hsa_amd_copy_direction_t(copyKind),depSignalCnt, depSignalCnt ? &depSignal:NULL,_signal);

    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    DBOUT( DB_CMD2, "  copy setNextKernelNeedsSysAcquire(true)\n");
    hsaQueue()->setNextKernelNeedsSysAcquire(true);

    isSubmitted = true;
    return status;
}

static Kalmar::hcCommandKind resolveMemcpyDirection(bool srcInDeviceMem, bool dstInDeviceMem)
{
    if (!srcInDeviceMem && !dstInDeviceMem) {
        return Kalmar::hcMemcpyHostToHost;
    } else if (!srcInDeviceMem && dstInDeviceMem) {
        return Kalmar::hcMemcpyHostToDevice;
    } else if (srcInDeviceMem && !dstInDeviceMem) {
        return Kalmar::hcMemcpyDeviceToHost;
    } else if (srcInDeviceMem &&  dstInDeviceMem) {
        return Kalmar::hcMemcpyDeviceToDevice;
    } else {
        // Invalid copy copyDir - should never reach here since we cover all 4 possible options above.
        throw Kalmar::runtime_exception("invalid copy copyDir", 0);
    }
}

inline hsa_status_t
HSACopy::enqueueAsyncCopyCommand(const Kalmar::HSADevice *copyDevice, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo) {

    if (HCC_SERIALIZE_COPY & 0x1) {
        hsaQueue()->wait();
    }

    // Performs an async copy.
    // This routine deals only with "mapped" pointers - see syncCopy for an explanation.

    // enqueue async copy command
    if (isSubmitted) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    isAsync = true;
    setCommandKind (resolveMemcpyDirection(srcPtrInfo._isInDeviceMem, dstPtrInfo._isInDeviceMem));
    hsa_status_t status = hcc_memory_async_copy(getCommandKind(), copyDevice, dstPtrInfo, srcPtrInfo, sizeBytes);
    STATUS_CHECK(status, __LINE__);

    // dynamically allocate a std::shared_future<void> object
    future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
        waitComplete();
    }).share());

    if (HCC_SERIALIZE_COPY & 0x2) {
        future->wait();
        STATUS_CHECK(_wait_complete_status, __LINE__);
    };

    return status;
}

inline hsa_status_t
HSACopy::enqueueAsyncCopy2dCommand(size_t width, size_t height, size_t srcPitch, size_t dstPitch, 
                                    const Kalmar::HSADevice *copyDevice, 
                                    const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo) {

    if (HCC_SERIALIZE_COPY & 0x1) {
        hsaQueue()->wait();
    }
    // enqueue async copy command
    if (isSubmitted) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    isAsync = true;
    setCommandKind (resolveMemcpyDirection(srcPtrInfo._isInDeviceMem, dstPtrInfo._isInDeviceMem));
    hsa_status_t status = hcc_memory_async_copy_rect(getCommandKind(), copyDevice, 
                                                      dstPtrInfo, srcPtrInfo, 
                                                      width, height, srcPitch, dstPitch);
    if(status != HSA_STATUS_SUCCESS)
        return status;

    //dynamically allocate a std::shared_future<void> object
    future = new std::shared_future<void>(std::async(std::launch::deferred, [&] {
             waitComplete();
             }).share());

    if (HCC_SERIALIZE_COPY & 0x2) {
        status = waitComplete();
        STATUS_CHECK(status, __LINE__);
    };

    return status;
}

inline void
HSACopy::dispose() {

    if (future != nullptr) {
        delete future;
        future = nullptr;
    }

    // HSA signal may not necessarily be allocated by HSACopy instance
    // only release the signal if it was really allocated (signalIndex >= 0)
    if (_signalIndex >= 0) {
        if (HCC_PROFILE & HCC_PROFILE_TRACE) {
            uint64_t start = getBeginTimestamp();
            uint64_t end   = getEndTimestamp();

            double bw = (double)(sizeBytes)/(end-start) * (1000.0/1024.0) * (1000.0/1024.0);

            LOG_PROFILE(this, start, end, "copy", getCopyCommandString(),  "\t" << sizeBytes << " bytes;\t" << sizeBytes/1024.0/1024 << " MB;\t" << bw << " GB/s;");
        }
        _activity_prof.report_gpu_timestamps<HSACopy>(this, sizeBytes);
        Kalmar::ctx.releaseSignal(_signal, _signalIndex);
    } else {
        if (HCC_PROFILE & HCC_PROFILE_TRACE) {
            uint64_t start = apiStartTick;
            uint64_t end   = Kalmar::ctx.getSystemTicks();
            double bw = (double)(sizeBytes)/(end-start) * (1000.0/1024.0) * (1000.0/1024.0);
            LOG_PROFILE(this, start, end, "copyslo", getCopyCommandString(),  "\t" << sizeBytes << " bytes;\t" << sizeBytes/1024.0/1024 << " MB;\t" << bw << " GB/s;");
        }
        _activity_prof.report_system_ticks<HSACopy>(this, sizeBytes);
    }

    // clear reference counts for dependent ops.
    depAsyncOp = nullptr;
}

inline uint64_t
HSACopy::getBeginTimestamp() override {
    hsa_amd_profiling_async_copy_time_t time;
    hsa_amd_profiling_get_async_copy_time(_signal, &time);
    return time.start;
}

inline uint64_t
HSACopy::getEndTimestamp() override {
    hsa_amd_profiling_async_copy_time_t time;
    hsa_amd_profiling_get_async_copy_time(_signal, &time);
    return time.end;
}

inline uint64_t
HSACopy::getStartTick() {
    return apiStartTick;
}

inline uint64_t
HSACopy::getSystemTicks() {
    return Kalmar::ctx.getSystemTicks();
}

void
HSACopy::syncCopyExt(hc::hcCommandKind copyDir, const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo, const Kalmar::HSADevice *copyDevice, bool forceUnpinnedCopy)
{
    bool srcInTracker = (srcPtrInfo._sizeBytes != 0);
    bool dstInTracker = (dstPtrInfo._sizeBytes != 0);


// TODO - Clean up code below.
    // Copy already called queue.wait() so there are no dependent signals.
    hsa_signal_t depSignal;
    int depSignalCnt = 0;


    if ((copyDevice == nullptr) && (copyDir != Kalmar::hcMemcpyHostToHost) && (copyDir != Kalmar::hcMemcpyDeviceToDevice)) {
        throw Kalmar::runtime_exception("Null copyDevice can only be used with HostToHost or DeviceToDevice copy", -1);
    }


    DBOUT(DB_COPY, "hcCommandKind: " << getHcCommandKindString(copyDir) << "\n");

    bool useFastCopy = true;
    switch (copyDir) {
        case Kalmar::hcMemcpyHostToDevice:
            if (!srcInTracker || forceUnpinnedCopy) {
                DBOUT(DB_COPY,"HSACopy::syncCopyExt(), invoke UnpinnedCopyEngine::CopyHostToDevice()\n");

                copyDevice->copy_engine[0]->CopyHostToDevice(copyDevice->copy_mode, dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                useFastCopy = false;
            }
            break;


        case Kalmar::hcMemcpyDeviceToHost:
            if (!dstInTracker || forceUnpinnedCopy) {
                DBOUT(DB_COPY,"HSACopy::syncCopyExt(), invoke UnpinnedCopyEngine::CopyDeviceToHost()\n");
                UnpinnedCopyEngine::CopyMode d2hCopyMode = copyDevice->copy_mode;
                if (d2hCopyMode == UnpinnedCopyEngine::UseMemcpy) {
                    // override since D2H does not support Memcpy
                    d2hCopyMode = UnpinnedCopyEngine::ChooseBest;
                }
                copyDevice->copy_engine[1]->CopyDeviceToHost(d2hCopyMode, dst, src, sizeBytes, depSignalCnt ? &depSignal : NULL);
                useFastCopy = false;
            };
            break;

        case Kalmar::hcMemcpyHostToHost:
            DBOUT(DB_COPY,"HSACopy::syncCopyExt(), invoke memcpy\n");
            // Since this is sync copy, we assume here that the GPU has already drained younger commands.

            // This works for both mapped and unmapped memory:
            memcpy(dst, src, sizeBytes);
            useFastCopy = false;
            break;

        case Kalmar::hcMemcpyDeviceToDevice:
            if (forceUnpinnedCopy) {
                // TODO - is this a same-device copy or a P2P?
                hsa_agent_t dstAgent = * (static_cast<hsa_agent_t*> (dstPtrInfo._acc.get_hsa_agent()));
                hsa_agent_t srcAgent = * (static_cast<hsa_agent_t*> (srcPtrInfo._acc.get_hsa_agent()));
                DBOUT(DB_COPY, "HSACopy::syncCopyExt() P2P copy by engine forcing use of staging buffers.  copyEngine=" << copyDevice << "\n");

                isPeerToPeer = true;

                // TODO, which staging buffer should we use for this to be optimal?
                copyDevice->copy_engine[1]->CopyPeerToPeer(dst, dstAgent, src, srcAgent, sizeBytes, depSignalCnt ? &depSignal : NULL);

                useFastCopy = false;
            }
            break;

        default:
            throw Kalmar::runtime_exception("unexpected copy type", HSA_STATUS_SUCCESS);

    };


    if (useFastCopy) {
        // Didn't already handle copy with one of special (slow) cases above, use the standard runtime copy path.

        DBOUT(DB_COPY, "HSACopy::syncCopyExt(), useFastCopy=1\n");
        DBOUT(DB_CMD, "HSACopy::syncCopyExt(), invoke hsa_amd_memory_async_copy()\n");

        if (copyDevice == nullptr) {
            throw Kalmar::runtime_exception("Null copyDevice reached call to hcc_memory_async_copy", -1);
        }
        hsa_status_t hsa_status = hcc_memory_async_copy(copyDir, copyDevice, dstPtrInfo, srcPtrInfo, sizeBytes);

        if (hsa_status == HSA_STATUS_SUCCESS) {
            DBOUT(DB_COPY, "HSACopy::syncCopyExt(), wait for completion...");
            hsa_signal_wait_relaxed(_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, waitMode);
            DBOUT(DB_COPY,"done!\n");
        } else {
            DBOUT(DB_COPY, "HSACopy::syncCopyExt(), hsa_amd_memory_async_copy() returns: 0x" << std::hex << hsa_status << std::dec <<"\n");
            throw Kalmar::runtime_exception("hsa_amd_memory_async_copy error", hsa_status);
        }
    }
    if (HCC_CHECK_COPY) {
        checkCopy(dst, src, sizeBytes);
    }
}

hsa_status_t HSACopy::syncCopy2DExt(hc::hcCommandKind copyDir,
                     const hc::AmPointerInfo &srcPtrInfo, const hc::AmPointerInfo &dstPtrInfo,size_t width, size_t height, size_t srcPitch, size_t dstPitch,
                     const Kalmar::HSADevice *copyDevice, bool forceUnpinnedCopy)
{
    bool srcInTracker = (srcPtrInfo._sizeBytes != 0);
    bool dstInTracker = (dstPtrInfo._sizeBytes != 0);

    if ((copyDevice == nullptr) && (copyDir != Kalmar::hcMemcpyHostToHost) && (copyDir != Kalmar::hcMemcpyDeviceToDevice)) {
        throw Kalmar::runtime_exception("Null copyDevice can only be used with HostToHost or DeviceToDevice copy", -1);
    }

    hsa_signal_store_relaxed(_signal, 1);
    hsa_status_t hsa_status = hcc_memory_async_copy_rect(copyDir, copyDevice, dstPtrInfo, srcPtrInfo, width, height, srcPitch, dstPitch);
    if (hsa_status == HSA_STATUS_SUCCESS) {
        DBOUT(DB_COPY, "HSACopy::syncCopy2DExt(), wait for completion...");
        hsa_signal_wait_relaxed(_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, waitMode);
        DBOUT(DB_COPY,"done!\n");
    } else {
        DBOUT(DB_COPY, "HSACopy::syncCopy2DExt(), hcc_amd_memory_async_copy_rect() returns: 0x" << std::hex << hsa_status << std::dec <<"\n");
    }
    return hsa_status;
}

// Performs a copy, potentially through a staging buffer .
// This routine can take mapped or unmapped src and dst pointers.
//    "Mapped" means the pointers are mapped into the address space of the device associated with this HSAQueue.
//     Mapped memory may be physically located on this device, or pinned in the CPU, or on another device (for P2P access).
//     If the memory is not mapped, it can still be copied usign an intermediate staging buffer approach.
//
//     In some cases (ie for array or array_view) we already know the src or dst are mapped, and the *IsMapped parameters
//     allow communicating that information to this function.  *IsMapped=False indicates the map state is unknown,
//     so the functions uses the memory tracker to determine mapped or unmapped and *IsInDeviceMem
//
// The copies are performed host-synchronously - the routine waits until the copy completes before returning.
void
HSACopy::syncCopy() {

    DBOUT(DB_COPY, "HSACopy::syncCopy(" << hsaQueue() << "), src = " << src << ", dst = " << dst << ", sizeBytes = " << sizeBytes << "\n");

    // The tracker stores information on all device memory allocations and all pinned host memory, for the specified device
    // If the memory is not found in the tracker, then it is assumed to be unpinned host memory.
    bool srcInTracker = false;
    bool srcInDeviceMem = false;
    bool dstInTracker = false;
    bool dstInDeviceMem = false;

    hc::accelerator acc;
    hc::AmPointerInfo srcPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);
    hc::AmPointerInfo dstPtrInfo(NULL, NULL, NULL, 0, acc, 0, 0);

    if (hc::am_memtracker_getinfo(&srcPtrInfo, src) == AM_SUCCESS) {
        srcInTracker = true;
        srcInDeviceMem = (srcPtrInfo._isInDeviceMem);
    }  // Else - srcNotMapped=srcInDeviceMem=false

    if (hc::am_memtracker_getinfo(&dstPtrInfo, dst) == AM_SUCCESS) {
        dstInTracker = true;
        dstInDeviceMem = (dstPtrInfo._isInDeviceMem);
    } // Else - dstNotMapped=dstInDeviceMem=false


    DBOUTL(DB_COPY,  " srcInTracker: " << srcInTracker
                  << " srcInDeviceMem: " << srcInDeviceMem
                  << " dstInTracker: " << dstInTracker
                  << " dstInDeviceMem: " << dstInDeviceMem);

    // Resolve default to a specific Kind so we know which algorithm to use:
    setCommandKind (resolveMemcpyDirection(srcInDeviceMem, dstInDeviceMem));

    Kalmar::HSADevice *copyDevice;
    if (srcInDeviceMem) {  // D2D, H2D
        copyDevice = static_cast<Kalmar::HSADevice*> (srcPtrInfo._acc.get_dev_ptr());
    }else if (dstInDeviceMem) {  // D2H
        copyDevice = static_cast<Kalmar::HSADevice*> (dstPtrInfo._acc.get_dev_ptr());
    } else {
        copyDevice = nullptr;  // H2D
    }

    syncCopyExt(getCommandKind(), srcPtrInfo, dstPtrInfo, copyDevice, false);
};


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
    case sizeof(short):
      dispatch->pushShortArg(*reinterpret_cast<short*>(val));
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


// op printer
std::ostream& operator<<(std::ostream& os, const HSAOp & op)
{
     os << "#" << op.opCoord()._deviceId << "." ;
     os << op.opCoord()._queueId << "." ;
     os << op.getSeqNum();
    return os;
}

// Profiling routines

extern "C" void InitActivityCallbackImpl(void* id_callback, void* op_callback, void* arg) {
    activity_prof::CallbacksTable::init(reinterpret_cast<activity_prof::id_callback_fun_t>(id_callback),
                                        reinterpret_cast<activity_prof::callback_fun_t>(op_callback),
                                        arg);
}

extern "C" bool EnableActivityCallbackImpl(unsigned op, bool enable) {
    return activity_prof::CallbacksTable::set_enabled(op, enable);
}

extern "C" const char* GetCmdNameImpl(unsigned op) {
    return getHcCommandKindString(static_cast<Kalmar::hcCommandKind>(op));
}

// TODO;
// - add common HSAAsyncOp for barrier, etc.  '
//   - store queue, completion signal, other common info.

//   - remove hsaqueeu
