#ifndef __CLAMP_AMP_RUNTIME
#define __CLAMP_AMP_RUNTIME

#include <map>
#include <mutex>

namespace Concurrency {

#ifndef E_FAIL
#define E_FAIL 0x80004005
#endif

static const char *__errorMsg_UnsupportedAccelerator = "concurrency::parallel_for_each is not supported on the selected accelerator \"CPU accelerator\".";

typedef int HRESULT;
class runtime_exception : public std::exception
{
public:
  runtime_exception(const char * message, HRESULT hresult) throw() : _M_msg(message), err_code(hresult) {}
  explicit runtime_exception(HRESULT hresult) throw() : err_code(hresult) {}
  runtime_exception(const runtime_exception& other) throw() : _M_msg(other.what()), err_code(other.err_code) {}
  runtime_exception& operator=(const runtime_exception& other) throw() {
    _M_msg = *(other.what());
    err_code = other.err_code;
    return *this;
  }
  virtual ~runtime_exception() throw() {}
  virtual const char* what() const throw() {return _M_msg.c_str();}
  HRESULT get_error_code() const {return err_code;}

private:
  std::string _M_msg;
  HRESULT err_code;
};



/// forward declaration
class KalmarDevice;
struct rw_info;

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

/// KalmarQueue
/// This is the implementation of accelerator_view
/// KalamrQueue is responsible for data operations and launch kernel
class KalmarQueue
{
public:

  KalmarQueue(KalmarDevice* pDev, queuing_mode mode = queuing_mode_automatic)
      : pDev(pDev), mode(mode) {}

  virtual ~KalmarQueue() {}

  virtual void flush() {}
  virtual void wait() {}
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {}
  virtual void* LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) { return nullptr; }

  /// read data from device to host
  virtual void read(void* device, void* dst, size_t count, size_t offset) = 0;

  /// wrtie data from host to device
  virtual void write(void* device, const void* src, size_t count, size_t offset, bool blocking) = 0;

  /// copy data between two device pointers
  virtual void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) = 0;

  /// map host accessible pointer from device
  virtual void* map(void* device, size_t count, size_t offset, bool modify) = 0;

  /// unmap host accessible pointer
  virtual void unmap(void* device, void* addr) = 0;

  /// push device pointer to kernel argument list
  virtual void Push(void *kernel, int idx, void* device, bool isConst) = 0;

  KalmarDevice* getDev() { return pDev; }
  queuing_mode get_mode() const { return mode; }
  void set_mode(queuing_mode mod) { mode = mod; }

private:
  KalmarDevice* pDev;
  queuing_mode mode;
};

/// KalmarDevice
/// This is the base implementation of accelerator
/// KalmarDevice is responsible for create/release memory on device
class KalmarDevice
{
private:
    access_type cpu_type;
    /// default KalmarQueue
    std::shared_ptr<KalmarQueue> def;
    /// make sure KalamrQueue is created only once
    std::once_flag flag;
protected:
    KalmarDevice(access_type type = access_type_read_write) : cpu_type(type), def(), flag() {}
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


    /// create buffer
    /// @key on device that supports shared memory
    //       key can used to avoid duplicate allocation
    virtual void* create(size_t count, struct rw_info* key) = 0;

    /// release buffer
    /// @key: used to avoid duplicate release
    virtual void release(void* ptr, struct rw_info* key) = 0;

    /// create kernel
    virtual void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }

    /// check the dimension information is correct
    virtual bool check(size_t* size, size_t dim_ext) { return true; }

    /// create KalmarQueue from current device
    virtual std::shared_ptr<KalmarQueue> createQueue() = 0;
    virtual ~KalmarDevice() {}

    std::shared_ptr<KalmarQueue> get_default_queue() {
        std::call_once(flag, [&]() { def = createQueue(); });
        return def;
    }
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

  void unmap(void* device, void* addr) override {}

  void Push(void *kernel, int idx, void* device, bool isConst) override {}
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


    std::shared_ptr<KalmarQueue> createQueue() { return std::shared_ptr<KalmarQueue>(new CPUQueue(this)); }
    void* create(size_t count, struct rw_info* /* not used */ ) override { return aligned_alloc(0x1000, count); }
    void release(void* ptr, struct rw_info* /* nout used */) override { ::operator delete(ptr); }
    void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }
};

/// KalmarContext
/// This is responsible for managing all devices
/// User will need to add their customize devices
class KalmarContext
{
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
    if (src == dst)
        return ;
    if (is_cpu_queue(srcQueue))
        dstQueue->write(dst, (char*)src + src_offset, cnt, dst_offset, block);
    else if (is_cpu_queue(dstQueue))
        srcQueue->read(src, (char*)dst + dst_offset, cnt, src_offset);
    else
        dstQueue->copy(src, dst, cnt, src_offset, dst_offset, block);
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
/// Whenever rw_info is going to be used on device, it will create a buffer at
/// that device. data pointer points to device data pointer
/// state info is stored to implement MSI protocol in rw_info
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
/// Whenever rw_info is going to be used on device, it will allocate space on
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


    /// consruct array_view
    /// According to standard, array_view will be constructed by size, or size with
    /// host pointer.
    /// If it is constructed with host pointer, treat it is constructed on cpu
    /// device, set the HostPtr flag to prevent destructor to release it
    rw_info(const size_t count, void* ptr)
        : data(ptr), count(count), curr(nullptr), master(nullptr), stage(nullptr),
        devs(), mode(access_type_none), HostPtr(ptr != nullptr) {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
            /// if array_view is constructed in cpu path kernel
            /// allocate memory for it and do nothing
            if (CLAMP::in_cpu_kernel() && ptr == nullptr) {
                data = aligned_alloc(0x1000, count);
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
    rw_info(const std::shared_ptr<KalmarQueue> Queue, const std::shared_ptr<KalmarQueue> Stage,
            const size_t count, access_type mode_) : data(nullptr), count(count),
    curr(Queue), master(Queue), stage(nullptr), devs(), mode(mode_), HostPtr(false) {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
        if (CLAMP::in_cpu_kernel() && data == nullptr) {
            data = aligned_alloc(0x1000, count);
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
    /// @modify:
    /// @blcok: this call should act blocking or not
    //          non-blocking call will occur in serilization stage
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

        /// If both queues are from the same device, change state only
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

    void unmap(void* addr) { curr->unmap(devs[curr->getDev()].data, addr); }

    /// synchronize data to master accelerator
    /// used in array
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
                void *ptr = aligned_alloc(0x1000, cnt);
                memset(ptr, 0, cnt);
                curr->write(src.data, ptr, cnt, src_offset, true);
                ::operator delete(ptr);
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
                ::operator delete(data);
            return;
        }
#endif
        /// If this rw_info is constructed by host pointer
        /// 1. synchronize latest data to host pointer
        /// 2. Because the data pointer cannout be released, erase itself from devs
        if (HostPtr)
            synchronize(false);
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
            pDev->release(info.data, this);
        }
    }
};

} // namespace Concurrency

#endif // __CLAMP_AMP_RUNTIME
