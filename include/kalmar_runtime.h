#pragma once

#include "hc_defines.h"
#include "kalmar_aligned_alloc.h"

namespace hc {
class AmPointerInfo;
class completion_future;
}; // end namespace hc

typedef struct hsa_kernel_dispatch_packet_s hsa_kernel_dispatch_packet_t;

namespace Kalmar {
namespace enums {

/// access_type is used for accelerator that supports unified memory
/// Such accelerator can use access_type to control whether can access data on
/// it or not
enum access_type
{
    access_type_none = 0,
    access_type_read = (1 << 0),
    access_type_write = (1 << 1),
    access_type_read_write = access_type_read | access_type_write,
    access_type_auto = (1 << 31)
};

enum queuing_mode
{
    queuing_mode_immediate,
    queuing_mode_automatic
};

enum execute_order
{
    execute_in_order,
    execute_any_order
};


// Flags to specify visibility of previous commands after a marker is executed.
enum memory_scope
{
    no_scope=0,           // No release operation applied
    accelerator_scope=1,  // Release to current accelerator
    system_scope=2,       // Release to system (CPU + all accelerators)
};

static inline memory_scope greater_scope(memory_scope scope1, memory_scope scope2)
{
    if ((scope1==system_scope) || (scope2 == system_scope)) {
        return system_scope;
    } else if ((scope1==accelerator_scope) || (scope2 == accelerator_scope)) {
        return accelerator_scope;
    } else {
        return no_scope;
    }
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
static inline bool isCopyCommand(hcCommandKind k) 
{
    switch (k) {
        case hcMemcpyHostToHost:
        case hcMemcpyHostToDevice:
        case hcMemcpyDeviceToHost:
        case hcMemcpyDeviceToDevice:
            return true;
        default:
            return false;
    };
};


// Commands sent to compute queue:
static inline bool isComputeQueueCommand(hcCommandKind k) {
    return (k == hcCommandKernel) || (k == hcCommandMarker);
};




enum hcWaitMode {
    hcWaitModeBlocked = 0,
    hcWaitModeActive = 1
};

enum hcAgentProfile {
    hcAgentProfileNone = 0,
    hcAgentProfileBase = 1,
    hcAgentProfileFull = 2
};

} // namespace enums
} // namespace Kalmar

 
/** \cond HIDDEN_SYMBOLS */
namespace Kalmar {

using namespace Kalmar::enums;

/// forward declaration
class KalmarDevice;
class KalmarQueue;
struct rw_info;

/// KalmarAsyncOp
///
/// This is an abstraction of all asynchronous operations within Kalmar
class KalmarAsyncOp {
public:
  KalmarAsyncOp(KalmarQueue *xqueue, hcCommandKind xCommandKind) : queue(xqueue), commandKind(xCommandKind), seqNum(0) {} 

  virtual ~KalmarAsyncOp() {} 
  virtual std::shared_future<void>* getFuture() { return nullptr; }
  virtual void* getNativeHandle() { return nullptr;}

  /**
   * Get the timestamp when the asynchronous operation begins.
   *
   * @return An implementaion-defined timestamp.
   */
  virtual uint64_t getBeginTimestamp() { return 0L; }

  /**
   * Get the timestamp when the asynchronous operation completes.
   *
   * @return An implementation-defined timestamp.
   */
  virtual uint64_t getEndTimestamp() { return 0L; }

  /**
   * Get the frequency of timestamp.
   *
   * @return An implementation-defined frequency for the asynchronous operation.
   */
  virtual uint64_t getTimestampFrequency() { return 0L; }

  /**
   * Get if the async operations has been completed.
   *
   * @return True if the async operation has been completed, false if not.
   */
  virtual bool isReady() { return false; }

  /**
   * Set the wait mode of the async operation.
   *
   * @param mode[in] wait mode, must be one of the value in hcWaitMode enum.
   */
  virtual void setWaitMode(hcWaitMode mode) {}

  uint64_t getSeqNum () const { return seqNum;};
  void     setSeqNum (uint64_t s) {seqNum = s;};

  hcCommandKind getCommandKind() const { return commandKind; };
  void          setCommandKind(hcCommandKind xCommandKind) { commandKind = xCommandKind; };

  KalmarQueue  *getQueue() const { return queue; };

private:
  KalmarQueue    *queue;

  // Kind of this command - copy, kernel, barrier, etc:
  hcCommandKind  commandKind;


  // Sequence number of this op in the queue it is dispatched into.
  uint64_t       seqNum;

};

/// KalmarQueue
/// This is the implementation of accelerator_view
/// KalamrQueue is responsible for data operations and launch kernel
class KalmarQueue
{
public:

  KalmarQueue(KalmarDevice* pDev, queuing_mode mode = queuing_mode_automatic, execute_order order = execute_in_order)
      : pDev(pDev), mode(mode), order(order) {}

  virtual ~KalmarQueue() {}

  virtual void flush() {}
  virtual void wait(hcWaitMode mode = hcWaitModeBlocked) {}

  // sync kernel launch with dynamic group memory
  virtual void LaunchKernelWithDynamicGroupMemory(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size, size_t dynamic_group_size) {}

  // async kernel launch with dynamic group memory
  virtual std::shared_ptr<KalmarAsyncOp> LaunchKernelWithDynamicGroupMemoryAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size, size_t dynamic_group_size) { return nullptr; }

  // sync kernel launch
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {}

  // async kernel launch
  virtual std::shared_ptr<KalmarAsyncOp> LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) { return LaunchKernelWithDynamicGroupMemoryAsync(kernel, dim_ext, ext, local_size, 0); }

  /// read data from device to host
  virtual void read(void* device, void* dst, size_t count, size_t offset) = 0;

  /// wrtie data from host to device
  virtual void write(void* device, const void* src, size_t count, size_t offset, bool blocking) = 0;

  /// copy data between two device pointers
  virtual void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) = 0;



  /// map host accessible pointer from device
  virtual void* map(void* device, size_t count, size_t offset, bool modify) = 0;

  /// unmap host accessible pointer
  virtual void unmap(void* device, void* addr, size_t count, size_t offset, bool modify) = 0;

  /// push device pointer to kernel argument list
  virtual void Push(void *kernel, int idx, void* device, bool modify) = 0;

  virtual uint32_t GetGroupSegmentSize(void *kernel) { return 0; }

  KalmarDevice* getDev() const { return pDev; }
  queuing_mode get_mode() const { return mode; }
  void set_mode(queuing_mode mod) { mode = mod; }

  execute_order get_execute_order() const { return order; }

  /// get number of pending async operations in the queue
  virtual int getPendingAsyncOps() { return 0; }

  /// Is the queue empty?  Same as getPendingAsyncOps but may be faster.
  virtual bool isEmpty() { return 0; }

  /// get underlying native queue handle
  virtual void* getHSAQueue() { return nullptr; }

  /// get underlying native agent handle
  virtual void* getHSAAgent() { return nullptr; }

  /// get AM region handle
  virtual void* getHSAAMRegion() { return nullptr; }

  virtual void* getHSAAMHostRegion() { return nullptr; }
  
  virtual void* getHSACoherentAMHostRegion() { return nullptr; }

  /// get kernarg region handle
  virtual void* getHSAKernargRegion() { return nullptr; }

  /// check if the queue is an HSA queue
  virtual bool hasHSAInterOp() { return false; }

  /// enqueue marker
  virtual std::shared_ptr<KalmarAsyncOp> EnqueueMarker(memory_scope) { return nullptr; }

  /// enqueue marker with prior dependency
  virtual std::shared_ptr<KalmarAsyncOp> EnqueueMarkerWithDependency(int count, std::shared_ptr <KalmarAsyncOp> *depOps, memory_scope scope) { return nullptr; }

  virtual std::shared_ptr<KalmarAsyncOp> detectStreamDeps(hcCommandKind commandKind, KalmarAsyncOp *newCopyOp) { return nullptr; };


  /// copy src to dst asynchronously
  virtual std::shared_ptr<KalmarAsyncOp> EnqueueAsyncCopy(const void* src, void* dst, size_t size_bytes) { return nullptr; }
  virtual std::shared_ptr<KalmarAsyncOp> EnqueueAsyncCopyExt(const void* src, void* dst, size_t size_bytes, 
                                                             hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, 
                                                             const Kalmar::KalmarDevice *copyDevice) { return nullptr; };

  // Copy src to dst synchronously
  virtual void copy(const void *src, void *dst, size_t size_bytes) { }

  /// copy src to dst, with caller providing extended information about the pointers.
  //// TODO - remove me, this form is deprecated.
  virtual void copy_ext(const void *src, void *dst, size_t size_bytes, hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, bool forceUnpinnedCopy) { };
  virtual void copy_ext(const void *src, void *dst, size_t size_bytes, hcCommandKind copyDir, const hc::AmPointerInfo &srcInfo, const hc::AmPointerInfo &dstInfo, 
                        const Kalmar::KalmarDevice *copyDev, bool forceUnpinnedCopy) { };

  /// cleanup internal resource
  /// this function is usually called by dtor of the implementation classes
  /// in rare occasions it may be called by other functions to ensure proper
  /// resource clean up sequence
  virtual void dispose() {}

  virtual void dispatch_hsa_kernel(const hsa_kernel_dispatch_packet_t *aql, 
                                   const void * args, size_t argsize,
                                   hc::completion_future *cf)  { };
 
  /// set CU affinity of this queue.
  /// the setting is permanent until the queue is destroyed or another setting
  /// is called.
  virtual bool set_cu_mask(const std::vector<bool>& cu_mask) { return false; };

private:
  KalmarDevice* pDev;
  queuing_mode mode;
  execute_order order;
};

/// KalmarDevice
/// This is the base implementation of accelerator
/// KalmarDevice is responsible for create/release memory on device
class KalmarDevice
{
private:
    access_type cpu_type;

    // Set true if the device has large bar

#if !TLS_QUEUE
    /// default KalmarQueue
    std::shared_ptr<KalmarQueue> def;
    /// make sure KalamrQueue is created only once
    std::once_flag flag;
#else
    /// default KalmarQueue for each calling thread
    std::map< std::thread::id, std::shared_ptr<KalmarQueue> > tlsDefaultQueueMap;
    /// mutex for tlsDefaultQueueMap
    std::mutex tlsDefaultQueueMap_mutex;
#endif

protected:
    // True if the device memory is mapped into CPU address space and can be
    // directly accessed with CPU memory operations.
    bool cpu_accessible_am;


    KalmarDevice(access_type type = access_type_read_write)
        : cpu_type(type),
#if !TLS_QUEUE
          def(), flag()
#else
          tlsDefaultQueueMap(), tlsDefaultQueueMap_mutex()
#endif
          {}
public:
    access_type get_access() const { return cpu_type; }
    void set_access(access_type type) { cpu_type = type; }

    virtual std::wstring get_path() const = 0;
    virtual std::wstring get_description() const = 0;
    virtual size_t get_mem() const = 0;
    virtual bool is_double() const = 0;
    virtual bool is_lim_double() const = 0;
    virtual bool is_unified() const = 0;
    virtual bool is_emulated() const = 0;
    virtual uint32_t get_version() const = 0;

    /// create buffer
    /// @key on device that supports shared memory
    //       key can used to avoid duplicate allocation
    virtual void* create(size_t count, struct rw_info* key) = 0;

    /// release buffer
    /// @key: used to avoid duplicate release
    virtual void release(void* ptr, struct rw_info* key) = 0;

    /// build program
    virtual void BuildProgram(void* size, void* source) {}

    /// create kernel
    virtual void* CreateKernel(const char* fun, KalmarQueue *queue) { return nullptr; }

    /// check if a given kernel is compatible with the device
    virtual bool IsCompatibleKernel(void* size, void* source) { return true; }

    /// check the dimension information is correct
    virtual bool check(size_t* size, size_t dim_ext) { return true; }

    /// create KalmarQueue from current device
    virtual std::shared_ptr<KalmarQueue> createQueue(execute_order order = execute_in_order) = 0;
    virtual ~KalmarDevice() {}

    std::shared_ptr<KalmarQueue> get_default_queue() {
#if !TLS_QUEUE
        std::call_once(flag, [&]() { 
            def = createQueue();
        });
        return def;
#else
        std::thread::id tid = std::this_thread::get_id();
        tlsDefaultQueueMap_mutex.lock();
        if (tlsDefaultQueueMap.find(tid) == tlsDefaultQueueMap.end()) {
            tlsDefaultQueueMap[tid] = createQueue();
        }
        std::shared_ptr<KalmarQueue> result = tlsDefaultQueueMap[tid];
        tlsDefaultQueueMap_mutex.unlock();
        return result;
#endif
    }

    /// get max tile static area size
    virtual size_t GetMaxTileStaticSize() { return 0; }

    /// get all queues associated with this device
    virtual std::vector< std::shared_ptr<KalmarQueue> > get_all_queues() { return std::vector< std::shared_ptr<KalmarQueue> >(); }

    virtual void memcpySymbol(const char* symbolName, void* hostptr, size_t count, size_t offset = 0, hcCommandKind kind = hcMemcpyHostToDevice) {}

    virtual void memcpySymbol(void* symbolAddr, void* hostptr, size_t count, size_t offset = 0, hcCommandKind kind = hcMemcpyHostToDevice) {}

    virtual void* getSymbolAddress(const char* symbolName) { return nullptr; }

    /// get underlying native agent handle
    virtual void* getHSAAgent() { return nullptr; }

    /// get the profile of the agent
    virtual hcAgentProfile getProfile() { return hcAgentProfileNone; }

    /// check if @p other can access to this device's device memory, return true if so, false otherwise
    virtual bool is_peer(const KalmarDevice* other) {return false;}

    /// get device's compute unit count
    virtual unsigned int get_compute_unit_count() {return 0;}

    virtual int get_seqnum() const {return -1;}

    virtual bool has_cpu_accessible_am() {return false;}

};

class CPUQueue final : public KalmarQueue
{
public:

  CPUQueue(KalmarDevice* pDev) : KalmarQueue(pDev) {}

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

  void unmap(void* device, void* addr, size_t count, size_t offset, bool modify) override {}

  void Push(void *kernel, int idx, void* device, bool modify) override {}
};

/// cpu accelerator
class CPUDevice final : public KalmarDevice
{
public:
    std::wstring get_path() const override { return L"cpu"; }
    std::wstring get_description() const override { return L"CPU Device"; }
    size_t get_mem() const override { return 0; }
    bool is_double() const override { return true; }
    bool is_lim_double() const override { return true; }
    bool is_unified() const override { return true; }
    bool is_emulated() const override { return true; }
    uint32_t get_version() const override { return 0; }

    std::shared_ptr<KalmarQueue> createQueue(execute_order order = execute_in_order) override { return std::shared_ptr<KalmarQueue>(new CPUQueue(this)); }
    void* create(size_t count, struct rw_info* /* not used */ ) override { return kalmar_aligned_alloc(0x1000, count); }
    void release(void* ptr, struct rw_info* /* nout used */) override { kalmar_aligned_free(ptr); }
    void* CreateKernel(const char* fun, KalmarQueue *queue) { return nullptr; }
};

/// KalmarContext
/// This is responsible for managing all devices
/// User will need to add their customize devices
class KalmarContext
{
private:
    //TODO: Think about a system which has multiple CPU socket, e.g. server. In this case,
    //We might be able to assume that only the first device is CPU, or we only mimic one cpu
    //device when constructing KalmarContext.
    KalmarDevice* get_default_dev() {
        if (!def) {
            if (Devices.size() <= 1) {
                fprintf(stderr, "There is no device can be used to do the computation\n");
                exit(-1);
            }
            def = Devices[1];
        }
        return def;
    }
protected:
    /// default device
    KalmarDevice* def;
    std::vector<KalmarDevice*> Devices;
    KalmarContext() : def(nullptr), Devices() { Devices.push_back(new CPUDevice); }
public:
    virtual ~KalmarContext() {}

    std::vector<KalmarDevice*> getDevices() { return Devices; }

    /// set default device by path
    bool set_default(const std::wstring& path) {
        auto result = std::find_if(std::begin(Devices), std::end(Devices),
                                   [&] (const KalmarDevice* pDev)
                                   { return pDev->get_path() == path; });
        if (result == std::end(Devices))
            return false;
        else {
            def = *result;
            return true;
        }
    }

    /// get auto selection queue
    std::shared_ptr<KalmarQueue> auto_select() {
        return get_default_dev()->get_default_queue();
    }

    /// get device from path
    KalmarDevice* getDevice(std::wstring path = L"") {
        if (path == L"default" || path == L"")
            return get_default_dev();
        auto result = std::find_if(std::begin(Devices), std::end(Devices),
                                   [&] (const KalmarDevice* dev)
                                   { return dev->get_path() == path; });
        if (result != std::end(Devices))
            return *result;
        else
            return get_default_dev();
    }

    /// get system ticks
    virtual uint64_t getSystemTicks() { return 0L; };

    /// get tick frequency
    virtual uint64_t getSystemTickFrequency() { return 0L; };
};

KalmarContext *getContext();

namespace CLAMP {
// used in parallel_for_each.h
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
extern bool is_cpu();
extern bool in_cpu_kernel();
extern void enter_kernel();
extern void leave_kernel();
#endif

extern void *CreateKernel(std::string, KalmarQueue*);

extern void PushArg(void *, int, size_t, const void *);
extern void PushArgPtr(void *, int, size_t, const void *);

} // namespace CLAMP

static inline const std::shared_ptr<KalmarQueue> get_cpu_queue() {
    static auto cpu_queue = getContext()->getDevice(L"cpu")->get_default_queue();
    return cpu_queue;
}

static inline bool is_cpu_queue(const std::shared_ptr<KalmarQueue>& Queue) {
    return Queue->getDev()->get_path() == L"cpu";
}

static inline void copy_helper(std::shared_ptr<KalmarQueue>& srcQueue, void* src,
                               std::shared_ptr<KalmarQueue>& dstQueue, void* dst,
                               size_t cnt, bool block,
                               size_t src_offset = 0, size_t dst_offset = 0) {
    /// In shared memory architecture, src and dst may points to the same buffer
    /// avoid unnecessary copy
    if (src == dst)
        return ;
    /// If device pointer comes from cpu, let the device queue to handle the copy
    /// For example, if src is on cpu and dst is on device,
    /// in OpenCL, clEnqueueWrtieBuffer to write data from src to device
    
    if (is_cpu_queue(dstQueue))
        srcQueue->read(src, (char*)dst + dst_offset, cnt, src_offset);
    else
        dstQueue->write(dst, (char*)src + src_offset, cnt, dst_offset, block);
}

/// software MSI protocol
/// https://en.wikipedia.org/wiki/MSI_protocol
/// Used to avoid unnecessary copy when array_view<const, T> is used
enum states
{
    /// exclusive owned data, safe to read and wrtie
    modified,
    /// shared on multiple devices, the content are all the same, cannot modify
    shared,
    // not able to read and write
    invalid
};

/// buffer information
/// Used in rw_info, represent cached data for each device
/// Whenever rw_info is going to be used on device, it will create a buffer at
/// that device.
/// @data: device data pointer
/// @state: used to implement MSI protocol
struct dev_info
{
    void* data; /// pointer to device data
    states state; /// state of the data on current device
};

/// rw_info is modeled as multiprocessor without shared cache
/// each accelerator represents a processor in the system
///
/// +---+  +----+  +----+
/// |cpu|  |acc1|  |acc2|
/// +---+  +----+  +----+
///
/// Whenever rw_info is going to be used on device, it will allocate memory on
/// targeting device and do the computation
struct rw_info
{
    /// host accessible pointer, it will be set if
    /// 1. rw_info constructed by cpu accelerator
    /// 2. rw_info constructed by accelerator supports
    ///    unified memory and access_type is not none
    void *data;
    const size_t count;
    /// This pointer pointes to the latest queue that manages the data
    std::shared_ptr<KalmarQueue> curr;
    /// This pointer pointes to the queue that used to construct this rw_info
    /// This will be null if the constructor is constructed by size only
    std::shared_ptr<KalmarQueue> master;
    /// staged queue
    std::shared_ptr<KalmarQueue> stage;
    /// This is used as cache for device buffer
    /// When this rw_info is going to be used(computed) on device,
    /// rw_info will allocate buffer for the device
    std::map<KalmarDevice*, dev_info> devs;
    access_type mode;
    /// This will be set if this rw_info is constructed with host pointer
    /// because rw_info cannot free host pointer
    unsigned int HostPtr : 1;

    /// A flag to mark whether to call release() to explicitly deallocate
    /// device memory.  The flag should be set as false when rw_info is
    /// constructed with a given device pointer.
    bool toReleaseDevPointer;


    /// consruct array_view
    /// According to standard, array_view will be constructed by size, or size with
    /// host pointer.
    /// If it is constructed with host pointer, treat it is constructed on cpu
    /// device, set the HostPtr flag to prevent destructor to release it
    rw_info(const size_t count, void* ptr)
        : data(ptr), count(count), curr(nullptr), master(nullptr), stage(nullptr),
        devs(), mode(access_type_none), HostPtr(ptr != nullptr), toReleaseDevPointer(true) {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
            /// if array_view is constructed in cpu path kernel
            /// allocate memory for it and do nothing
            if (CLAMP::in_cpu_kernel() && ptr == nullptr) {
                data = kalmar_aligned_alloc(0x1000, count);
                return;
            }
#endif
            if (ptr) {
                mode = access_type_read_write;
                curr = master = get_cpu_queue();
                devs[curr->getDev()] = {ptr, modified};
            }
        }

    /// construct array
    /// According to AMP standard, array should be constructed with
    /// 1. one accelerator_view
    /// 2. one acceleratir_view, with another staged one
    ///    In this case, master should be cpu device
    ///    If it is not, ignore the stage one, fallback to case 1.
    rw_info(const std::shared_ptr<KalmarQueue>& Queue, const std::shared_ptr<KalmarQueue>& Stage,
            const size_t count, access_type mode_) : data(nullptr), count(count),
    curr(Queue), master(Queue), stage(nullptr), devs(), mode(mode_), HostPtr(false), toReleaseDevPointer(true) {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
        if (CLAMP::in_cpu_kernel() && data == nullptr) {
            data = kalmar_aligned_alloc(0x1000, count);
            return;
        }
#endif
        if (mode == access_type_auto)
            mode = curr->getDev()->get_access();
        devs[curr->getDev()] = {curr->getDev()->create(count, this), modified};

        /// set data pointer, if it is accessible from cpu
        if (is_cpu_queue(curr) || (curr->getDev()->is_unified() && mode != access_type_none))
            data = devs[curr->getDev()].data;
        if (is_cpu_queue(curr)) {
            stage = Stage;
            if (Stage != curr)
                devs[stage->getDev()] = {stage->getDev()->create(count, this), invalid};
        } else
            /// if curr is not cpu, ignore the stage one
            stage = curr;
    }

    /// construct array with given device pointer
    /// most of the logic are the same as the constructor above, except that
    /// toReleaseDevPointer is now set as false, so when this instance goes
    /// into destruction, device memory associated with it will NOT be
    /// released
    rw_info(const std::shared_ptr<KalmarQueue>& Queue, const std::shared_ptr<KalmarQueue>& Stage,
            const size_t count,
            void* device_pointer,
            access_type mode_) : data(nullptr), count(count), curr(Queue), master(Queue), stage(nullptr), devs(), mode(mode_), HostPtr(false), toReleaseDevPointer(false) {
         if (mode == access_type_auto)
             mode = curr->getDev()->get_access();
         devs[curr->getDev()] = { device_pointer, modified };

         /// set data pointer, if it is accessible from cpu
         if (is_cpu_queue(curr) || (curr->getDev()->is_unified() && mode != access_type_none))
             data = devs[curr->getDev()].data;
         if (is_cpu_queue(curr)) {
             stage = Stage;
             if (Stage != curr)
                 devs[stage->getDev()] = {stage->getDev()->create(count, this), invalid};
         } else
             /// if curr is not cpu, ignore the stage one
             stage = curr;
    }

    void* get_device_pointer() {
        return devs[curr->getDev()].data;
    }

    void construct(std::shared_ptr<KalmarQueue> pQueue) {
        curr = pQueue;
        devs[pQueue->getDev()] = {pQueue->getDev()->create(count, this), invalid};
        if (is_cpu_queue(pQueue))
            data = devs[pQueue->getDev()].data;
    }

    void disc() {
        for (auto& it : devs)
            it.second.state = invalid;
    }

    /// optimization: Before performing copy, if the state of cpu accelerator is
    /// shared, it implies that the data on cpu is the same on device where
    /// curr located, use data on cpu to perform the later operation
    /// For example, if data on device a is going to be copied to device b
    /// and the data on device a and cpu is the same, it is okay to copy data 
    /// from cpu to device b
    void try_switch_to_cpu() {
        if (is_cpu_queue(curr))
            return;
        auto cpu_queue = get_cpu_queue();
        if (devs.find(cpu_queue->getDev()) != std::end(devs))
            if (devs[cpu_queue->getDev()].state == shared)
                curr = cpu_queue;
    }

    /// synchronize data to device pQueue belongs to by using pQuquq
    /// @pQueue: queue that used to synchronize
    /// @modify: the data will be modified or not
    /// @blcok: this call will be blocking or not
    ///         none blocking occurs in serialization stage
    void sync(std::shared_ptr<KalmarQueue> pQueue, bool modify, bool block = true) {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        if (!curr) {
            /// This can only happen if array_view is constructed with size and
            /// is not accessed before
            dev_info dev = {pQueue->getDev()->create(count, this),
                modify ? modified : shared};
            devs[pQueue->getDev()] = dev;
            if (is_cpu_queue(pQueue))
                data = dev.data;
            curr = pQueue;
            return;
        }

        if (curr == pQueue)
            return;

        /// If both queues are from the same device, upadte state only
        if (curr->getDev() == pQueue->getDev()) {
            // curr->wait();
            curr = pQueue;
            if (modify) {
                disc();
                devs[curr->getDev()].state = modified;
            }
            return;
        }

        /// If the buffer on device is not allocated, allocate space for it
        if (devs.find(pQueue->getDev()) == std::end(devs)) {
            dev_info dev = {pQueue->getDev()->create(count, this), invalid};
            devs[pQueue->getDev()] = dev;
            if (is_cpu_queue(pQueue))
                data = dev.data;
        }

        try_switch_to_cpu();
        dev_info& dst = devs[pQueue->getDev()];
        dev_info& src = devs[curr->getDev()];
        if (dst.state == invalid && src.state != invalid)
            copy_helper(curr, src.data, pQueue, dst.data, count, block);
        /// if the data on current device is going to be modified
        /// changed the state of current device as modified
        curr = pQueue;
        if (modify) {
            disc();
            dst.state = modified;
        } else {
            dst.state = shared;
            if (src.state == modified)
                src.state = shared;
        }
    }

    /// return a host accessible pointer from device
    /// @cnt: size to map
    /// @offset: offset to map
    /// @modify: change state if it is going to be modified
    void* map(size_t cnt, size_t offset, bool modify) {
        if (cnt == 0)
            cnt = count;
        /// This can only happen if this rw_info is constructed only with size
        /// and not accessed on any device
        if (!curr) {
            curr = getContext()->auto_select();
            devs[curr->getDev()] = {curr->getDev()->create(count, this), modify ? modified : shared};
            return curr->map(data, cnt, offset, modify);
        }
        try_switch_to_cpu();
        dev_info& info = devs[curr->getDev()];
        if (info.state == shared && modify) {
            disc();
            info.state = modified;
        }
        return curr->map(info.data, cnt, offset, modify);
    }

    void unmap(void* addr, size_t cnt, size_t offset, bool modify) { curr->unmap(devs[curr->getDev()].data, addr, cnt, offset, modify); }

    /// synchronize data to master accelerator
    /// used in array
    /// master is not necessary to be cpu device
    void synchronize(bool modify) { sync(master, modify); }

    /// synchronize data to cpu accelerator
    /// used in array_view
    void get_cpu_access(bool modify) { sync(get_cpu_queue(), modify); }

    /// Write data from host source pointer to device
    /// Change state to modified, because the device has exclusive copy of data
    void write(const void* src, int cnt, int offset, bool blocking) {
        curr->write(devs[curr->getDev()].data, src, cnt, offset, blocking);
        dev_info& dev = devs[curr->getDev()];
        if (dev.state != modified) {
            disc();
            dev.state = modified;
        }
    }

    /// Read data to host pointer from device
    void read(void* dst, int cnt, int offset) {
        curr->read(devs[curr->getDev()].data, dst, cnt, offset);
    }

    /// copy data from "this" to other
    void copy(rw_info* other, int src_offset, int dst_offset, int cnt) {
        if (cnt == 0)
            cnt = count;
        if (!curr) {
            if (!other->curr)
                return;
            else
                construct(other->curr);
        } else {
            if (!other->curr)
                other->construct(curr);
        }
        dev_info& dst = other->devs[other->curr->getDev()];
        dev_info& src = devs[curr->getDev()];
        /// If src.state is invalid, zero the data on it
        if (src.state == invalid) {
            src.state = shared;
            if (is_cpu_queue(curr))
                memset((char*)src.data + src_offset, 0, cnt);
            else {
                void *ptr = kalmar_aligned_alloc(0x1000, cnt);
                memset(ptr, 0, cnt);
                curr->write(src.data, ptr, cnt, src_offset, true);
                kalmar_aligned_free(ptr);
            }
        }
        copy_helper(curr, src.data, other->curr, dst.data, cnt, true, src_offset, dst_offset);
        other->disc();
        dst.state = modified;
    }

    ~rw_info() {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
        if (CLAMP::in_cpu_kernel()) {
            if (data && !HostPtr)
                kalmar_aligned_free(data);
            return;
        }
#endif
        /// If this rw_info is constructed by host pointer
        /// 1. synchronize latest data to host pointer
        /// 2. Because the data pointer cannout be released, erase itself from devs
        if (HostPtr)
            synchronize(false);
        if (curr) {
            // Wait issues a system-scope release:
            // Need to make sure we write-back cache contents before deallocating the memory those writes might eventually touch
            curr->wait();
        }
        auto cpu_dev = get_cpu_queue()->getDev();
        if (devs.find(cpu_dev) != std::end(devs)) {
            if (!HostPtr)
                cpu_dev->release(devs[cpu_dev].data, this);
            devs.erase(cpu_dev);
        }
        KalmarDevice* pDev;
        dev_info info;
        for (const auto it : devs) {
            std::tie(pDev, info) = it;
            if (toReleaseDevPointer)
                pDev->release(info.data, this);
        }
    }
};

} // namespace Kalmar

/** \endcond */
