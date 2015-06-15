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



class KalmarDevice;
struct rw_info;

enum access_type
{
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

class KalmarQueue
{
public:
  virtual ~KalmarQueue() {}
  virtual void flush() {}
  virtual void wait() {}
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {}
  virtual void* LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) { return nullptr; }

  /// read data from device to host
  virtual void read(void* device, void* dst, size_t count, size_t offset) {
      if (dst != device)
          memmove(dst, (char*)device + offset, count);
  }

  /// wrtie data from host to device
  virtual void write(void* device, const void* src, size_t count, size_t offset, bool blocking, bool free) {
      if (src != device)
          memmove((char*)device + offset, src, count);
      if (free)
          ::operator delete(const_cast<void*>(src));
  }

  /// copy data between device pointer
  virtual void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) {
      if (src != dst)
          memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
  }

  /// map host accessible pointer from device
  virtual void* map(void* device, size_t count, size_t offset, bool modify) {
      return (char*)device + offset;
  }

  /// unmap host accessible pointer
  virtual void unmap(void* device, void* addr) {}
  
  /// push device pointer to kernel argument list
  virtual void Push(void *kernel, int idx, void* device, bool isConst) {}

  KalmarDevice* getDev() { return pDev; }
  queuing_mode get_mode() const { return mode; }
  void set_mode(queuing_mode mod) { mode = mod; }
  KalmarQueue(KalmarDevice* pDev, queuing_mode mode = queuing_mode_automatic)
      : pDev(pDev), mode(mode) {}
private:
  KalmarDevice* pDev;
  queuing_mode mode;
};

class KalmarDevice
{
private:
    access_type cpu_type;
    std::shared_ptr<KalmarQueue> def;
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


    /// create buffer on device
    /// @key on accelerator that supports shared memory
    //       key can used to avoid duplicate allocation
    virtual void* create(size_t count, struct rw_info* key) = 0;

    /// release buffer on device
    virtual void release(void* ptr) = 0;

    /// create kernel for current device
    virtual void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }

    /// check the dimension information is correct
    virtual bool check(size_t* size, size_t dim_ext) { return true; }

    /// create work queue from current device
    virtual std::shared_ptr<KalmarQueue> createQueue() = 0;
    virtual ~KalmarDevice() {}

    std::shared_ptr<KalmarQueue> get_default() {
        std::call_once(flag, [&]() { def = createQueue(); });
        return def;
    }
};

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


    std::shared_ptr<KalmarQueue> createQueue() { return std::shared_ptr<KalmarQueue>(new KalmarQueue(this)); }
    void* create(size_t count, struct rw_info* /* not used */ ) override { return aligned_alloc(0x1000, count); }
    void release(void* ptr) override { ::operator delete(ptr); }
    void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }
};

class AMPContext
{
protected:
    /// default device
    KalmarDevice* def;
    std::vector<KalmarDevice*> Devices;
    AMPContext() : def(nullptr), Devices() { Devices.push_back(new CPUDevice); }
public:
    virtual ~AMPContext() {
        for (auto dev : Devices)
            delete dev;
    }

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
        if (!def)
            def = Devices[1];
        return def->get_default();
    }

    /// get device from path
    KalmarDevice* getDevice(std::wstring path = L"") {
        if (path == L"default" || path == L"") {
            if (!def)
                def = Devices[1];
            return def;
        }
        auto result = std::find_if(std::begin(Devices), std::end(Devices),
                                   [&] (const KalmarDevice* dev)
                                   { return dev->get_path() == path; });
        if (result != std::end(Devices))
            return *result;
        else
            return Devices[1];
    }
};

AMPContext *getContext();

namespace CLAMP {
// used in parallel_for_each.h
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
extern bool is_cpu();
extern bool in_cpu_kernel();
extern void enter_kernel();
extern void leave_kernel();
#endif

extern void *CreateKernel(std::string, KalmarQueue*);
extern void MatchKernelNames(std::string &);

extern void PushArg(void *, int, size_t, const void *);
extern void PushArgPtr(void *, int, size_t, const void *);

} // namespace CLAMP

static inline const std::shared_ptr<KalmarQueue> get_cpu_queue() {
    static auto cpu_queue = getContext()->getDevice(L"cpu")->get_default();
    return cpu_queue;
}

/// software MSI protocol
/// Used to avoid unnecessary copy when array_view<const, T> is used
enum states
{
    modified, // exclusive owned data, can read and write to it
    shared, // shared on multiple device, the content on it is the same
    invalid // not able to read and write on the data, need to read from other device
};

/// store information of each device
struct dev_info
{
    void* data; /// pointer to device data
    states state; /// state of the current data
};

static inline bool is_cpu_dev(const std::shared_ptr<KalmarQueue>& Queue) {
    return Queue->getDev()->get_path() == L"cpu";
}

static inline void copy_helper(std::shared_ptr<KalmarQueue>& srcQueue, dev_info& src,
                               std::shared_ptr<KalmarQueue>& dstQueue, dev_info& dst,
                               size_t cnt, bool block,
                               size_t src_offset = 0, size_t dst_offset = 0) {
    if (is_cpu_dev(srcQueue))
        dstQueue->write(dst.data, (char*)src.data + src_offset, cnt, dst_offset, block, false);
    else if (is_cpu_dev(dstQueue))
        srcQueue->read(src.data, (char*)dst.data + dst_offset, cnt, src_offset);
    else {
        if (dstQueue->getDev() == srcQueue->getDev())
            dstQueue->copy(src.data, dst.data, cnt, src_offset, dst_offset, block);
        else {
            if (src.data != dst.data) {
                void* temp = ::operator new(cnt);
                srcQueue->read(src.data, temp, cnt, src_offset);
                dstQueue->write(dst.data, temp, cnt, dst_offset, block, true);
            }
        }
    }
}

struct rw_info
{
    /// host accessible pointer
    /// it will be set if
    /// 1. rw_info constructed on cpu accelerator
    /// 2. rw_info constructed on accelerator supports
    ///    unified memory and access_type is not none
    void *data;
    const size_t count;
    std::shared_ptr<KalmarQueue> curr;
    std::shared_ptr<KalmarQueue> master;
    std::shared_ptr<KalmarQueue> stage;
    std::map<KalmarDevice*, dev_info> devs;
    access_type mode;
    /// This will be set if this rw_info is constructed with host pointer
    /// because rw_info cannot free host pointer
    unsigned int HostPtr : 1;


    // consruct array_view
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

    // construct array
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

        // set data pointer
        if (is_cpu_dev(curr) || (curr->getDev()->is_unified() && mode != access_type_none))
            data = devs[curr->getDev()].data;
        if (is_cpu_dev(curr)) {
            stage = Stage;
            if (Stage != curr)
                devs[stage->getDev()] = {stage->getDev()->create(count, this), invalid};
        } else
            stage = curr;
    }

    void construct(std::shared_ptr<KalmarQueue> pQueue) {
        curr = pQueue;
        devs[pQueue->getDev()] = {pQueue->getDev()->create(count, this), invalid};
        if (is_cpu_dev(pQueue))
            data = devs[pQueue->getDev()].data;
    }

    void disc() {
        for (auto& it : devs)
            it.second.state = invalid;
    }

    /// optimization: Before performing copy, if the state of cpu accelerator is shared,
    /// it means the data on cpu is the ssame on curr accelerator, use the data
    /// on cpu to perform the copy
    void try_switch_to_cpu() {
        if (is_cpu_dev(curr))
            return;
        auto cpu_queue = get_cpu_queue();
        if (devs.find(cpu_queue->getDev()) != std::end(devs))
            if (devs[cpu_queue->getDev()].state == shared)
                curr = cpu_queue;
    }

    void sync(std::shared_ptr<KalmarQueue> pQueue, bool modify, bool block = true) {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        if (devs.find(pQueue->getDev()) == std::end(devs)) {
            dev_info dev = {pQueue->getDev()->create(count, this), invalid};
            devs[pQueue->getDev()] = dev;
            if (is_cpu_dev(pQueue))
                data = dev.data;
        }
        if (!curr) {
            curr = pQueue;
            devs[pQueue->getDev()].state = modify ? modified : shared;
            return;
        }
        if (curr->getDev() == pQueue->getDev()) {
            // curr->wait();
            curr = pQueue;
            if (modify) {
                disc();
                devs[curr->getDev()].state = modified;
            }
        } else {
            try_switch_to_cpu();
            dev_info& dst = devs[pQueue->getDev()];
            dev_info& src = devs[curr->getDev()];
            if (dst.state == invalid && src.state != invalid)
                copy_helper(curr, src, pQueue, dst, count, block);
            if (modify) {
                curr = pQueue;
                disc();
                dst.state = modified;
            } else {
                dst.state = shared;
                if (src.state == modified)
                    src.state = shared;
            }
        }
    }

    void* map(size_t cnt, size_t offset, bool modify) {
        if (cnt == 0)
            cnt = count;
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
    void synchronize(bool modify) { sync(master, modify); }
    void get_cpu_access(bool modify) { sync(get_cpu_queue(), modify); }

    void write(const void* src, int cnt, int offset, bool blocking) {
        curr->write(devs[curr->getDev()].data, src, cnt, offset, blocking, false);
        dev_info& dev = devs[curr->getDev()];
        if (dev.state != modified) {
            disc();
            dev.state = modified;
        }
    }

    void read(void* dst, int cnt, int offset) {
        curr->read(devs[curr->getDev()].data, dst, cnt, offset);
    }

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
        if (src.state == invalid) {
            src.state = shared;
            if (is_cpu_dev(curr))
                memset((char*)src.data + src_offset, 0, cnt);
            else {
                void *ptr = aligned_alloc(0x1000, cnt);
                memset(ptr, 0, cnt);
                curr->write(src.data, ptr, cnt, src_offset, true, false);
                ::operator delete(ptr);
            }
        }
        copy_helper(curr, src, other->curr, dst, cnt, true, src_offset, dst_offset);
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
        if (HostPtr)
            synchronize(false);
        auto cpu_acc = get_cpu_queue()->getDev();
        if (devs.find(cpu_acc) != std::end(devs)) {
            if (!HostPtr)
                cpu_acc->release(devs[cpu_acc].data);
            devs.erase(cpu_acc);
        }
        KalmarDevice* pDev;
        dev_info info;
        for (const auto it : devs) {
            std::tie(pDev, info) = it;
            pDev->release(info.data);
        }
    }
};

} // namespace Concurrency

#endif // __CLAMP_AMP_RUNTIME
