#ifndef __CLAMP_AMP_RUNTIME
#define __CLAMP_AMP_RUNTIME

#include <set>
#include <future>

namespace Concurrency {

struct rw_info;

struct obj_info
{
    void* device;
    size_t count;
    int ref;
};

class AMPAllocator;

class AMPManager : public std::enable_shared_from_this<AMPManager>
{
private:
    virtual void* create(size_t count, void *data, bool hasSrc) { return data; }
    virtual void release(void *data) {}
    std::shared_ptr<AMPAllocator> newAloc();


    std::map<void *, obj_info> mem_info;
    const std::wstring path;
protected:
    std::wstring des;
    size_t mem;
    bool is_double_;
    bool is_limited_double_;
    bool cpu_shared_memory;
    bool emulated;
    AMPManager(const std::wstring& path) : path(path) {}
public:
    AMPManager() : path(L"cpu"), des(L"dummy"), mem(0), is_double_(true),
    is_limited_double_(true), cpu_shared_memory(true), emulated(true) {}


    std::wstring get_path() { return path; }
    std::wstring get_des() { return des; }
    size_t get_mem() { return mem; }
    bool is_double() { return is_double_; }
    bool is_lim_double() { return is_limited_double_; }
    bool is_uni() { return cpu_shared_memory; }
    bool is_emu() { return emulated; }


    virtual void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }
    virtual bool check(size_t *local, size_t dim_ext) { return true; }
    virtual std::shared_ptr<AMPAllocator> createAloc() { return newAloc(); }
    virtual ~AMPManager() {}

    void regist(size_t count, void* data, bool hasSrc) {
        auto it = mem_info.find(data);
        if (it == std::end(mem_info)) {
            void* device = create(count, data, hasSrc);
            mem_info[data] = {device, count, 1};
        } else
            ++it->second.ref;
    }

    void unregist(void *data) {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info)) {
            obj_info& obj = it->second;
            if (!--obj.ref) {
                release(obj.device);
                mem_info.erase(it);
            }
        }
    }

    obj_info device_data(void* data) {
        auto it = mem_info.find(data);
        if (it != std::end(mem_info))
            return it->second;
        return obj_info();
    }
};

class AMPAllocator
{
protected:
  std::shared_ptr<AMPManager> Man;
public:
  AMPAllocator(std::shared_ptr<AMPManager> Man) : Man(Man) {}
  virtual ~AMPAllocator() {}

  void regist(size_t count, void* data, bool hasSrc) {
      Man->regist(count, data, hasSrc);
  }

  void unregist(void* data) {
      Man->unregist(data);
  }

  void* CreateKernel(const char* fun, void* size, void* source) {
      return Man->CreateKernel(fun, size, source);
  }

  std::shared_ptr<AMPManager> getMan() { return Man; }

  virtual void flush() {}
  virtual void wait() {}

  // overide function
  virtual void amp_write(void *data) {}
  virtual void amp_read(void *data) {}
  virtual void* amp_map(void *data, bool Write) { return data; }
  virtual void amp_unmap(void *data, void* addr) {}
  virtual void amp_copy(void *dst, void *src, size_t n) { memmove(dst, src, n); }
  virtual void PushArg(void* kernel, int idx, rw_info& data) {}
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {}
};

std::shared_ptr<AMPAllocator> AMPManager::newAloc() {
    return std::shared_ptr<AMPAllocator>(new AMPAllocator(shared_from_this()));
}

class accelerator;

class AMPContext
{
private:
    std::wstring def;
    friend accelerator;
protected:
    std::vector<std::shared_ptr<AMPManager>> Devices;
    std::map<std::shared_ptr<AMPManager>,
        std::shared_ptr<AMPAllocator>> default_map;
    AMPContext() : def(L"default"), Devices(0) {
        auto Man = std::shared_ptr<AMPManager>(new AMPManager());
        default_map[Man] = Man->createAloc();
        Devices.push_back(Man);
    }
public:
    virtual ~AMPContext() {}
    size_t getNumDevices() { return Devices.size(); }
    std::shared_ptr<AMPManager>* getDevices() { return Devices.data(); }
    bool set_default(const std::wstring& path) {
        if (def == L"default") {
            def = path;
            return true;
        } else
            return false;
    }
    std::shared_ptr<AMPAllocator> getView(const std::shared_ptr<AMPManager>& pMan) {
        return default_map[pMan];
    }
    std::shared_ptr<AMPManager> getDevice(std::wstring path = L"") {
        if (path == L"")
            path = def;
        if (path == L"default") {
            if (def == L"default")
                return Devices[1];
            else
                path = def;
        }
        for (const auto dev : Devices)
            if (dev->get_path() == path)
                return dev;
        return Devices[0];
    }
};

AMPContext *getContext();

namespace CLAMP {
// used in parallel_for_each.h
#ifdef __AMP_CPU__
extern bool is_cpu();
extern bool in_cpu_kernel();
extern void enter_kernel();
extern void leave_kernel();
#endif

extern void *CreateKernel(std::string, AMPAllocator*);
extern std::shared_future<void>* LaunchKernelAsync(void *, size_t, size_t *, size_t *);
extern void MatchKernelNames(std::string &);

extern void PushArg(void *, int, size_t, const void *);
extern void PushArgPtr(void *, int, size_t, const void *);

} // namespace CLAMP

class Serialize {
public:
  typedef void *kernel;
  Serialize(kernel k)
      : aloc_(), k_(k), current_idx_(0) {}
  Serialize(std::shared_ptr<AMPAllocator> aloc, kernel k)
      : aloc_(aloc), k_(k), current_idx_(0) {}
  void Append(size_t sz, const void *s) {
    CLAMP::PushArg(k_, current_idx_++, sz, s);
  }
  std::shared_ptr<AMPAllocator> get_aloc() { return aloc_; }
  void AppendPtr(size_t sz, const void *s) {
    CLAMP::PushArgPtr(k_, current_idx_++, sz, s);
  }
  void* getKernel() { return k_; }
  int getAndIncCurrentIndex() {
    int ret = current_idx_;
    current_idx_++;
    return ret;
  }
private:
  std::shared_ptr<AMPAllocator> aloc_;
  kernel k_;
  int current_idx_;
};


struct rw_info
{
    void *data;
    size_t count;
    std::shared_ptr<AMPAllocator> curr;
    const std::shared_ptr<AMPAllocator> master;
    std::set<std::shared_ptr<AMPAllocator>> Alocs;
    unsigned int discard : 1;
    unsigned int dirty : 1;
    unsigned int hasSrc : 1;

    rw_info(std::shared_ptr<AMPAllocator> Aloc, size_t count, void* p = nullptr)
        : data(p), count(count), curr(Aloc), master(Aloc), Alocs({Aloc}), discard(false),
        dirty(false), hasSrc(p != nullptr) {
        if (!hasSrc)
            data = aligned_alloc(0x1000, count);
#ifdef __AMP_CPU__
        if (!CLAMP::in_cpu_kernel())
#endif
            curr->regist(count, data, hasSrc);
    }

    void append(Serialize& s, bool isArray) {
        auto aloc = s.get_aloc();
        if (dirty) {
            if (curr != aloc) {
                if (Alocs.find(aloc) == std::end(Alocs)) {
                    aloc->regist(count, data, hasSrc);
                    Alocs.insert(aloc);
                }
                if (!discard || isArray) {
                    if (curr->getMan() != aloc->getMan()) {
                        void* dst = aloc->amp_map(data, true);
                        void* src = curr->amp_map(data, false);
                        memmove(dst, src, count);
                        aloc->amp_unmap(data, dst);
                        curr->amp_unmap(data, src);
                    } else {
                        // force previous execution finish
                        // replace with more efficient implementation in the future
                        curr->wait();
                    }
                    curr = aloc;
                }
            }
        } else {
            if (curr != aloc) {
                if (Alocs.find(aloc) == std::end(Alocs)) {
                    aloc->regist(count, data, hasSrc);
                    Alocs.insert(aloc);
                }
                curr = aloc;
            }
            if (!discard || isArray)
                curr->amp_write(data);
        }
        dirty = true;
        discard = false;
        curr->PushArg(s.getKernel(), s.getAndIncCurrentIndex(), *this);
    }

    void disc() {
        discard = true;
        dirty = false;
    }

    void stash() { dirty = false; }

    void copy(void* dst, size_t count) {
        if (dirty)
            curr->amp_copy(dst, data, count);
        else
            memmove(dst, data, count);
    }

    void synchronize() {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        if (dirty && !discard) {
            curr->amp_read(data);
            dirty = false;
        }
    }

    ~rw_info() {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel()) {
            if (!hasSrc)
                ::operator delete(data);
            return;
        }
#endif
        if (hasSrc)
            synchronize();
        for (const auto& aloc : Alocs)
            aloc->unregist(data);
        if (!hasSrc)
            ::operator delete(data);
    }
};

} // namespace Concurrency

#endif // __CLAMP_AMP_RUNTIME
