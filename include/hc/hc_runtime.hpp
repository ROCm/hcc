//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_aligned_alloc.hpp"
#include "hc_defines.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <cstring>
#include <future>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace hc
{
    namespace detail
    {
        namespace enums
        {
            /// access_type is used for accelerator that supports unified memory
            /// Such accelerator can use access_type to control whether can
            /// access data on it or not
            enum access_type {
                access_type_none = 0,
                access_type_read = (1 << 0),
                access_type_write = (1 << 1),
                access_type_read_write = access_type_read | access_type_write,
                access_type_auto = (1 << 31)
            };

            enum queuing_mode {
                queuing_mode_immediate,
                queuing_mode_automatic
            };

            enum execute_order {
                execute_in_order,
                execute_any_order
            };

            // Flags to specify visibility of previous commands after a marker
            // is executed.
            enum memory_scope {
                no_scope=0,           // No release operation applied
                accelerator_scope=1,  // Release to current accelerator
                system_scope=2,       // Release to system (CPU + all
                                      // accelerators)
            };

            static
            inline
            memory_scope greater_scope(memory_scope scope1, memory_scope scope2)
            {
                if ((scope1==system_scope) || (scope2 == system_scope)) {
                    return system_scope;
                }
                if ((scope1==accelerator_scope) ||
                    (scope2 == accelerator_scope)) {
                    return accelerator_scope;
                }
                return no_scope;
            }

            enum hcCommandKind {
                hcCommandInvalid= -1,

                hcMemcpyHostToHost = 0,
                hcMemcpyHostToDevice = 1,
                hcMemcpyDeviceToHost = 2,
                hcMemcpyDeviceToDevice = 3,
                hcCommandKernel = 4,
                hcCommandMarker = 5,
            };

            // Commands sent to copy queues:
            static
            inline
            bool isCopyCommand(hcCommandKind k)
            {
                switch (k) {
                    case hcMemcpyHostToHost:
                    case hcMemcpyHostToDevice:
                    case hcMemcpyDeviceToHost:
                    case hcMemcpyDeviceToDevice:
                        return true;
                    default:
                        return false;
                }
            }

            // Commands sent to compute queue:
            static
            inline
            bool isComputeQueueCommand(hcCommandKind k)
            {
                return (k == hcCommandKernel) || (k == hcCommandMarker);
            }

            enum hcWaitMode {
                hcWaitModeBlocked = 0,
                hcWaitModeActive = 1
            };

            enum accelerator_profile {
                accelerator_profile_none = 0,
                accelerator_profile_base = 1,
                accelerator_profile_full = 2
            };
        } // namespace hc::detail::enums

        inline
        void throwing_hsa_result_check(
            hsa_status_t s,
            const std::string& file,
            const std::string& fn,
            int line)
        {
            if (s == HSA_STATUS_SUCCESS || s == HSA_STATUS_INFO_BREAK) return;

            const char* p{};
            auto r = hsa_status_string(s, &p);

            throw std::system_error{
                (r == HSA_STATUS_SUCCESS) ? s : r,
                std::system_category(),
                "In " + file +
                    ", in function " + fn +
                    ", on line " + std::to_string(line) +
                    ", HSA RT failed: " + p
            };
        }
    } // Namespace hc::detail.
    class AmPointerInfo;
    class completion_future;
} // Namespace hc.

/** \cond HIDDEN_SYMBOLS */
// namespace detail {

// using namespace hc::detail::enums;

// /// forward declaration
// class HCCDevice;
// class HCCQueue;
// struct rw_info;

// /// HCCAsyncOp
// ///
// /// This is an abstraction of all asynchronous operations within detail
// class HCCAsyncOp {
// public:
//   HCCAsyncOp(HCCQueue *xqueue, hcCommandKind xCommandKind) : queue(xqueue), commandKind(xCommandKind), seqNum(0) {}

//   virtual ~HCCAsyncOp() {}
//   virtual const std::shared_future<void>& getFuture() const = 0;
//   virtual void* getNativeHandle() { return nullptr;}

//   /**
//    * Get the timestamp when the asynchronous operation begins.
//    *
//    * @return An implementation-defined timestamp.
//    */
//   virtual uint64_t getBeginTimestamp() { return 0L; }

//   /**
//    * Get the timestamp when the asynchronous operation completes.
//    *
//    * @return An implementation-defined timestamp.
//    */
//   virtual uint64_t getEndTimestamp() { return 0L; }

//   /**
//    * Get the frequency of timestamp.
//    *
//    * @return An implementation-defined frequency for the asynchronous operation.
//    */
//   virtual uint64_t getTimestampFrequency() { return 0L; }

//   /**
//    * Get if the async operations has been completed.
//    *
//    * @return True if the async operation has been completed, false if not.
//    */
//   virtual bool isReady() { return false; }

//   /**
//    * Set the wait mode of the async operation.
//    *
//    * @param mode[in] wait mode, must be one of the value in hcWaitMode enum.
//    */
//   virtual void setWaitMode(hcWaitMode mode) = 0;

//   void setSeqNumFromQueue();
//   uint64_t getSeqNum () const { return seqNum;};

//   hcCommandKind getCommandKind() const { return commandKind; };
//   void          setCommandKind(hcCommandKind xCommandKind) { commandKind = xCommandKind; };

//   HCCQueue  *getQueue() const { return queue; };

// private:
//   HCCQueue    *queue;

//   // Kind of this command - copy, kernel, barrier, etc:
//   hcCommandKind  commandKind;


//   // Sequence number of this op in the queue it is dispatched into.
//   uint64_t       seqNum;

// };

// /// HCCQueue
// /// This is the implementation of accelerator_view
// /// HCCQueue is responsible for data operations and launch kernel
// class HCCQueue
// {
// public:

//   HCCQueue(HCCDevice* pDev, queuing_mode mode = queuing_mode_automatic, execute_order order = execute_in_order)
//       : pDev(pDev), mode(mode), order(order), opSeqNums(0) {}

//   virtual ~HCCQueue() {}

//   virtual void flush() {}
//   virtual void wait(hcWaitMode mode = hcWaitModeBlocked) = 0;

//   // sync kernel launch with dynamic group memory
//   virtual
//   void LaunchKernelWithDynamicGroupMemory(
//     void* kernel,
//     size_t dim_ext,
//     const size_t* ext,
//     const size_t* local_size,
//     size_t dynamic_group_size) = 0;

//   // async kernel launch with dynamic group memory
//   virtual
//   std::shared_ptr<HCCAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(
//     void* kernel,
//     std::size_t dim_ext,
//     const std::size_t* ext,
//     const std::size_t* local_size,
//     std::size_t dynamic_group_size) = 0;

//   // sync kernel launch
//   virtual
//   void LaunchKernel(
//     void* kernel,
//     size_t dim_ext,
//     const size_t* ext,
//     const size_t* local_size) = 0;

//   // async kernel launch
//   virtual
//   std::shared_ptr<HCCAsyncOp> LaunchKernelAsync(
//     void* kernel,
//     std::size_t dim_ext,
//     const std::size_t* ext,
//     const std::size_t* local_size) = 0;

//   /// read data from device to host
//   virtual void read(void* device, void* dst, size_t count, size_t offset) = 0;

//   /// write data from host to device
//   virtual void write(void* device, const void* src, size_t count, size_t offset, bool blocking) = 0;

//   /// copy data between two device pointers
//   virtual void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) = 0;



//   /// map host accessible pointer from device
//   virtual void* map(void* device, size_t count, size_t offset, bool modify) = 0;

//   /// unmap host accessible pointer
//   virtual void unmap(void* device, void* addr, size_t count, size_t offset, bool modify) = 0;

//   /// push device pointer to kernel argument list
//   virtual void Push(void *kernel, int idx, void* device, bool modify) = 0;

//   virtual uint32_t GetGroupSegmentSize(void*) = 0;

//   HCCDevice* getDev() const { return pDev; }
//   queuing_mode get_mode() const { return mode; }
//   void set_mode(queuing_mode mod) { mode = mod; }

//   execute_order get_execute_order() const { return order; }

//   /// get number of pending async operations in the queue
//   virtual int getPendingAsyncOps() { return 0; }

//   /// Is the queue empty?  Same as getPendingAsyncOps but may be faster.
//   virtual bool isEmpty() { return 0; }

//   /// get underlying native queue handle
//   virtual void* getHSAQueue() { return nullptr; }

//   /// get underlying native agent handle
//   virtual void* getHSAAgent() { return nullptr; }

//   /// get AM region handle
//   virtual void* getHSAAMRegion() { return nullptr; }

//   virtual void* getHSAAMHostRegion() { return nullptr; }

//   virtual void* getHSACoherentAMHostRegion() { return nullptr; }

//   /// get kernarg region handle
//   virtual void* getHSAKernargRegion() { return nullptr; }

//   /// check if the queue is an HSA queue
//   virtual bool hasHSAInterOp() { return false; }

//   /// enqueue marker
//   virtual std::shared_ptr<HCCAsyncOp> EnqueueMarker(memory_scope) { return nullptr; }

//   /// enqueue marker with prior dependency
//   virtual
//   std::shared_ptr<HCCAsyncOp> EnqueueMarkerWithDependency(
//       int count, std::shared_ptr<HCCAsyncOp>* depOps, memory_scope scope) = 0;

//   virtual
//   std::shared_ptr<HCCAsyncOp> detectStreamDeps(
//       hcCommandKind commandKind, HCCAsyncOp *newCopyOp) = 0;


//   /// copy src to dst asynchronously
//   virtual
//   std::shared_ptr<HCCAsyncOp> EnqueueAsyncCopy(
//       const void* src, void* dst, size_t size_bytes) = 0;
//   virtual
//   std::shared_ptr<HCCAsyncOp> EnqueueAsyncCopyExt(
//       const void* src,
//       void* dst,
//       size_t size_bytes,
//       hcCommandKind copyDir,
//       const hc::AmPointerInfo& srcInfo,
//       const hc::AmPointerInfo& dstInfo,
//       const detail::HCCDevice *copyDevice) = 0;

//   // Copy src to dst synchronously
//   virtual
//   void copy(const void *src, void *dst, size_t size_bytes) = 0;

//   /// copy src to dst, with caller providing extended information about the pointers.
//   //// TODO - remove me, this form is deprecated.
//   virtual
//   void copy_ext(
//       const void* src,
//       void* dst,
//       size_t size_bytes,
//       hcCommandKind copyDir,
//       const hc::AmPointerInfo& srcInfo,
//       const hc::AmPointerInfo& dstInfo,
//       bool forceUnpinnedCopy) = 0;
//   virtual
//   void copy_ext(
//       const void* src,
//       void* dst,
//       size_t size_bytes,
//       hcCommandKind copyDir,
//       const hc::AmPointerInfo& srcInfo,
//       const hc::AmPointerInfo& dstInfo,
//       const detail::HCCDevice* copyDev,
//       bool forceUnpinnedCopy) = 0;

//   /// cleanup internal resource
//   /// this function is usually called by dtor of the implementation classes
//   /// in rare occasions it may be called by other functions to ensure proper
//   /// resource clean up sequence
//   virtual void dispose() {}

//   virtual
//   void dispatch_hsa_kernel(
//       const hsa_kernel_dispatch_packet_t* aql,
//       void* args,
//       size_t argsize,
//       hc::completion_future* cf,
//       const char* kernel_name) = 0;

//   /// set CU affinity of this queue.
//   /// the setting is permanent until the queue is destroyed or another setting
//   /// is called.
//   virtual
//   bool set_cu_mask(const std::vector<bool>&) = 0;


//   uint64_t assign_op_seq_num() { return ++opSeqNums; };

// private:
//   HCCDevice* pDev;
//   queuing_mode mode;
//   execute_order order;

//   uint64_t      opSeqNums; // last seqnum assigned to an op in this queue
// };

// /// HCCDevice
// /// This is the base implementation of accelerator
// /// HCCDevice is responsible for create/release memory on device
// class HCCDevice
// {
// private:
//     access_type cpu_type;

//     // Set true if the device has large bar

// #if !TLS_QUEUE
//     /// default HCCQueue
//     std::shared_ptr<HCCQueue> def;
//     /// make sure HCCQueue is created only once
//     std::once_flag flag;
// #else
//     /// default HCCQueue for each calling thread
//     std::map< std::thread::id, std::shared_ptr<HCCQueue> > tlsDefaultQueueMap;
//     /// mutex for tlsDefaultQueueMap
//     std::mutex tlsDefaultQueueMap_mutex;
// #endif

// protected:
//     // True if the device memory is mapped into CPU address space and can be
//     // directly accessed with CPU memory operations.
//     bool cpu_accessible_am;


//     HCCDevice(access_type type = access_type_none)
//         : cpu_type(type),
// #if !TLS_QUEUE
//           def(), flag()
// #else
//           tlsDefaultQueueMap(), tlsDefaultQueueMap_mutex()
// #endif
//           {}
// public:
//     access_type get_access() const { return cpu_type; }
//     void set_access(access_type type) { cpu_type = type; }

//     virtual std::wstring get_path() const = 0;
//     virtual std::wstring get_description() const = 0;
//     virtual size_t get_mem() const = 0;
//     virtual bool is_double() const = 0;
//     virtual bool is_lim_double() const = 0;
//     virtual bool is_unified() const = 0;
//     virtual bool is_emulated() const = 0;
//     virtual uint32_t get_version() const = 0;

//     /// create buffer
//     /// @key on device that supports shared memory
//     //       key can used to avoid duplicate allocation
//     virtual void* create(size_t count, struct rw_info* key) = 0;

//     /// release buffer
//     /// @key: used to avoid duplicate release
//     virtual void release(void* ptr, struct rw_info* key) = 0;

//     /// build program
//     virtual
//     void BuildProgram(void* size, void* source) = 0;

//     /// create kernel
//     virtual
//     void* CreateKernel(
//         const char* fun,
//         HCCQueue *queue,
//         std::unique_ptr<void, void (*)(void*)> callable,
//         std::size_t callable_size = 0u) = 0;

//     /// check if a given kernel is compatible with the device
//     virtual
//     bool IsCompatibleKernel(void* size, void* source) = 0;

//     /// check the dimension information is correct
//     virtual
//     bool check(size_t* size, size_t dim_ext) = 0;

//     /// create HCCQueue from current device
//     virtual
//     std::shared_ptr<HCCQueue> createQueue(
//         execute_order order = execute_in_order) = 0;
//     virtual ~HCCDevice() = default;

//     std::shared_ptr<HCCQueue> get_default_queue() {
// #if !TLS_QUEUE
//         std::call_once(flag, [&]() {
//             def = createQueue();
//         });
//         return def;
// #else
//         std::thread::id tid = std::this_thread::get_id();
//         tlsDefaultQueueMap_mutex.lock();
//         if (tlsDefaultQueueMap.find(tid) == tlsDefaultQueueMap.end()) {
//             tlsDefaultQueueMap[tid] = createQueue();
//         }
//         std::shared_ptr<HCCQueue> result = tlsDefaultQueueMap[tid];
//         tlsDefaultQueueMap_mutex.unlock();
//         return result;
// #endif
//     }

//     /// get max tile static area size
//     virtual size_t GetMaxTileStaticSize() { return 0; }

//     /// get all queues associated with this device
//     virtual
//     std::vector<std::shared_ptr<HCCQueue>> get_all_queues()
//     {
//         return std::vector< std::shared_ptr<HCCQueue> >();
//     }

//     virtual
//     void memcpySymbol(
//         const char* symbolName,
//         void* hostptr,
//         size_t count,
//         size_t offset = 0,
//         hcCommandKind kind = hcMemcpyHostToDevice) = 0;

//     virtual
//     void memcpySymbol(
//         void* symbolAddr,
//         void* hostptr,
//         size_t count,
//         size_t offset = 0,
//         hcCommandKind kind = hcMemcpyHostToDevice) = 0;

//     virtual
//     void* getSymbolAddress(const char* symbolName) = 0;

//     /// get underlying native agent handle
//     virtual void* getHSAAgent() { return nullptr; }

//     /// get the profile of the agent
//     virtual hcAgentProfile getProfile() { return hcAgentProfileNone; }

//     /// check if @p other can access to this device's device memory, return true
//     /// if so, false otherwise
//     virtual
//     bool is_peer(const HCCDevice* other) = 0;

//     /// get device's compute unit count
//     virtual unsigned int get_compute_unit_count() {return 0;}

//     virtual int get_seqnum() const {return -1;}

//     virtual bool has_cpu_accessible_am() const { return false; }

// };

// class CPUQueue final : public HCCQueue
// {
// public:

//   CPUQueue(HCCDevice* pDev) : HCCQueue(pDev) {}

//   void read(void* device, void* dst, size_t count, size_t offset) override {
//       if (dst != device)
//           memmove(dst, (char*)device + offset, count);
//   }

//   void write(
//       void* device,
//       const void* src,
//       size_t count,
//       size_t offset,
//       bool) override
//   {
//       if (src != device)
//           memmove((char*)device + offset, src, count);
//   }

//   void copy(
//       void* src,
//       void* dst,
//       size_t count,
//       size_t src_offset,
//       size_t dst_offset,
//       bool) override {
//       if (src != dst)
//           memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
//   }

//   void* map(void* device, size_t, size_t offset, bool) override
//   {
//       return (char*)device + offset;
//   }

//   void unmap(void*, void*, size_t, size_t, bool) override {}

//   void Push(void*, int, void*, bool) override {}

//   void wait(hcWaitMode = hcWaitModeBlocked) override {}

//     void copy(const void*, void*, size_t) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   void copy_ext(
//       const void*,
//       void*,
//       size_t,
//       hcCommandKind,
//       const hc::AmPointerInfo&,
//       const hc::AmPointerInfo&,
//       bool) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   void copy_ext(
//       const void*,
//       void*,
//       size_t,
//       hcCommandKind,
//       const hc::AmPointerInfo&,
//       const hc::AmPointerInfo&,
//       const detail::HCCDevice*,
//       bool) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   [[noreturn]]
//   std::shared_ptr<HCCAsyncOp> detectStreamDeps(hcCommandKind, HCCAsyncOp*) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   void dispatch_hsa_kernel(
//     const hsa_kernel_dispatch_packet_t*,
//     void*,
//     size_t,
//     hc::completion_future*,
//     const char*) override
//   {
//     throw std::runtime_error{"Unimplemented."};
//   }
//   [[noreturn]]
//   std::shared_ptr<HCCAsyncOp> EnqueueAsyncCopy(
//       const void*, void*, std::size_t) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   [[noreturn]]
//   std::shared_ptr<HCCAsyncOp> EnqueueAsyncCopyExt(
//       const void*,
//       void*,
//       size_t,
//       hcCommandKind,
//       const hc::AmPointerInfo&,
//       const hc::AmPointerInfo&,
//       const detail::HCCDevice*) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   [[noreturn]]
//   std::shared_ptr<HCCAsyncOp> EnqueueMarkerWithDependency(
//       int, std::shared_ptr<HCCAsyncOp>*, memory_scope) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   [[noreturn]]
//   std::uint32_t GetGroupSegmentSize(void*) override
//   {
//       throw std::runtime_error{"Unsupported."};
//   }
//   void LaunchKernel(
//       void*,
//       std::size_t,
//       const std::size_t*,
//       const std::size_t*) override
//   {
//     throw std::runtime_error{"Unsupported."};
//   }
//   [[noreturn]]
//   std::shared_ptr<HCCAsyncOp> LaunchKernelAsync(
//       void*,
//       std::size_t,
//       const std::size_t*,
//       const std::size_t*) override
//   {
//     throw std::runtime_error{"Unsupported."};
//   }
//   void LaunchKernelWithDynamicGroupMemory(
//     void*,
//     std::size_t,
//     const std::size_t*,
//     const std::size_t*,
//     std::size_t) override
//   {
//     throw std::runtime_error{"Unsupported."};
//   }
//   [[noreturn]]
//   std::shared_ptr<HCCAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(
//     void*,
//     std::size_t,
//     const std::size_t*,
//     const std::size_t*,
//     std::size_t) override
//   {
//     throw std::runtime_error{"Unimplemented."};
//   }
//   [[noreturn]]
//   bool set_cu_mask(const std::vector<bool>&) override
//   {
//       throw std::runtime_error{"Unimplemented."};
//   }
// };

// /// cpu accelerator
// class CPUDevice final : public HCCDevice
// {
// public:
//     std::wstring get_path() const override { return L"cpu"; }
//     std::wstring get_description() const override { return L"CPU Device"; }
//     size_t get_mem() const override { return 0; }
//     bool is_double() const override { return true; }
//     bool is_lim_double() const override { return true; }
//     bool is_unified() const override { return true; }
//     bool is_emulated() const override { return true; }
//     uint32_t get_version() const override { return 0; }

//     std::shared_ptr<HCCQueue> createQueue(
//         execute_order = execute_in_order) override
//     {
//         return std::shared_ptr<HCCQueue>(new CPUQueue(this));
//     }
//     void* create(size_t count, struct rw_info* /* not used */ ) override { return hc_aligned_alloc(0x1000, count); }
//     void release(void* ptr, struct rw_info* /* not used */) override { hc_aligned_free(ptr); }

//     void BuildProgram(void*, void*) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
//     [[noreturn]]
//     bool check(std::size_t*, std::size_t) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
//     [[noreturn]]
//     void* CreateKernel(
//         const char*,
//         HCCQueue*,
//         std::unique_ptr<void, void (*)(void*)>,
//         std::size_t = 0u) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
//     [[noreturn]]
//     void* getSymbolAddress(const char*) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
//     [[noreturn]]
//     bool IsCompatibleKernel(void*, void*) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
//     bool is_peer(const HCCDevice*) override
//     {
//         return false; // CPU is not a peer.
//     }
//     void memcpySymbol(
//         const char*,
//         void*,
//         size_t,
//         size_t = 0,
//         hcCommandKind = hcMemcpyHostToDevice) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
//     void memcpySymbol(
//         void*,
//         void*,
//         size_t,
//         size_t = 0,
//         hcCommandKind = hcMemcpyHostToDevice) override
//     {
//         throw std::runtime_error{"Unsupported."};
//     }
// };

// /// HCCContext
// /// This is responsible for managing all devices
// /// User will need to add their customize devices
// class HCCContext
// {
// private:
//     //TODO: Think about a system which has multiple CPU socket, e.g. server. In this case,
//     //We might be able to assume that only the first device is CPU, or we only mimic one cpu
//     //device when constructing HCCContext.
//     HCCDevice* get_default_dev() {
//         if (!def) {
//             if (Devices.size() <= 1) {
//                 fprintf(stderr, "There is no device can be used to do the computation\n");
//                 exit(-1);
//             }
//             def = Devices[1];
//         }
//         return def;
//     }
// protected:
//     /// default device
//     HCCDevice* def;
//     std::vector<HCCDevice*> Devices;
//     HCCContext() : def(nullptr), Devices() { Devices.push_back(new CPUDevice); }

//     bool init_success = false;

// public:
//     virtual ~HCCContext() {}

//     std::vector<HCCDevice*> getDevices() { return Devices; }

//     /// set default device by path
//     bool set_default(const std::wstring& path)
//     {
//         for (auto&& Device : Devices) {
//             if (Device->get_path() != path) continue;

//             def = Device;

//             return true;
//         }

//         return false;
//     }

//     /// get auto selection queue
//     std::shared_ptr<HCCQueue> auto_select() {
//         return get_default_dev()->get_default_queue();
//     }

//     /// get device from path
//     HCCDevice* getDevice(std::wstring path = L"") {
//         if (path == L"default" || path == L"") return get_default_dev();

//         for (auto&& Device : Devices) {
//             if (Device->get_path() != path) continue;

//             return Device;
//         }

//         return get_default_dev();
//     }

//     /// get system ticks
//     virtual uint64_t getSystemTicks() { return 0L; };

//     /// get tick frequency
//     virtual uint64_t getSystemTickFrequency() { return 0L; };

//     // initialize the printf buffer
//     virtual void initPrintfBuffer() {};

//     // flush the device printf buffer
//     virtual void flushPrintfBuffer() {};

//     // get the locked printf buffer VA
//     virtual void* getPrintfBufferPointerVA() { return nullptr; };
// };

// HCCContext *getContext();

// namespace CLAMP {
// void* CreateKernel(
//     const char*,
//     HCCQueue*,
//     std::unique_ptr<void, void (*)(void*)>,
//     std::size_t = 0u);
// } // namespace CLAMP

// inline
// const std::shared_ptr<HCCQueue> get_cpu_queue()
// {
//     static auto cpu_queue =
//         getContext()->getDevice(L"cpu")->get_default_queue();
//     return cpu_queue;
// }

// inline
// bool is_cpu_queue(const std::shared_ptr<HCCQueue>& Queue)
// {
//     return Queue->getDev()->get_path() == L"cpu";
// }

// //--- Implementation:
// //
// inline void HCCAsyncOp::setSeqNumFromQueue()  { seqNum = queue->assign_op_seq_num(); };

// } // namespace detail

// /** \endcond */
