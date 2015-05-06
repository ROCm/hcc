#ifndef __CLAMP_AMP_RUNTIME
#define __CLAMP_AMP_RUNTIME

#include <set>

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
    virtual void* create(size_t count, void *data, bool hasSrc) = 0;
    virtual void release(void *data) = 0;


    std::map<void *, obj_info> mem_info;
    const std::wstring path;
    const std::wstring des;
protected:
    size_t mem;
    bool dou;
    bool lim_dou;
    bool uni;
    bool emu;
    AMPManager(const std::wstring& path) : path(path) {}
public:

    std::wstring get_path() { return path; }
    std::wstring get_des() { return des; }
    size_t get_mem() { return mem; }
    bool is_double() { return dou; }
    bool is_lim_double() { return lim_dou; }
    bool is_uni() { return uni; }
    bool is_emu() { return emu; }


    virtual void* CreateKernel(const char* fun, void* size, void* source) = 0;
    virtual bool check(size_t *local, size_t dim_ext) = 0;
    virtual std::shared_ptr<AMPAllocator> createAloc() = 0;
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
    AMPAllocator(std::shared_ptr<AMPManager> Man) : Man(Man) {}
public:
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
  virtual void amp_write(void *data) = 0;
  virtual void amp_read(void *data) = 0;
  virtual void* amp_map(void *data, bool Write) = 0;
  virtual void amp_unmap(void *data, void* addr) = 0;
  virtual void amp_copy(void *dst, void *src, size_t n) = 0;
  virtual void PushArg(void* kernel, int idx, rw_info& data) = 0;
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) = 0;
};

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
    AMPContext() : def(L"default"), Devices(0) {}
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
                return Devices[0];
            else
                path = def;
        }
        if (path == L"gpu")
            path += L"0";
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
    std::shared_ptr<AMPAllocator> latest;
    const std::shared_ptr<AMPAllocator> master;
    std::set<std::shared_ptr<AMPAllocator>> Alocs;
    unsigned int discard : 1;
    unsigned int dirty : 1;
    unsigned int hasSrc : 1;

    rw_info(std::shared_ptr<AMPAllocator> Aloc, size_t count, void* p = nullptr)
        : data(p), count(count), latest(Aloc), master(Aloc), Alocs({Aloc}), discard(false),
        dirty(false), hasSrc(p != nullptr) {
        if (!hasSrc)
            data = aligned_alloc(0x1000, count);
#ifdef __AMP_CPU__
        if (!CLAMP::in_cpu_kernel())
#endif
            latest->regist(count, data, hasSrc);
    }

    void append(Serialize& s, bool isArray) {
        auto aloc = s.get_aloc();
        if (dirty) {
            if (latest != aloc) {
                if (Alocs.find(aloc) == std::end(Alocs)) {
                    aloc->regist(count, data, hasSrc);
                    Alocs.insert(aloc);
                }
                if (!discard || isArray) {
                    void* dst = aloc->amp_map(data, true);
                    void* src = latest->amp_map(data, false);
                    memmove(dst, src, count);
                    aloc->amp_unmap(data, dst);
                    latest->amp_unmap(data, src);
                    latest = aloc;
                }
            }
        } else {
            if (latest != aloc) {
                if (Alocs.find(aloc) == std::end(Alocs)) {
                    aloc->regist(count, data, hasSrc);
                    Alocs.insert(aloc);
                }
                latest = aloc;
            }
            if (!discard || isArray)
                latest->amp_write(data);
        }
        dirty = true;
        discard = false;
        latest->PushArg(s.getKernel(), s.getAndIncCurrentIndex(), *this);
    }

    void disc() {
        discard = true;
        dirty = false;
    }

    void stash() { dirty = false; }

    void copy(void* dst, size_t count) {
        if (dirty)
            latest->amp_copy(dst, data, count);
        else
            memmove(dst, data, count);
    }

    void synchronize() {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        if (dirty && !discard) {
            latest->amp_read(data);
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

#endif // __CLAMP_AMP_ALLOCATOR
