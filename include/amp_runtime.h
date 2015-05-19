#ifndef __CLAMP_AMP_RUNTIME
#define __CLAMP_AMP_RUNTIME

#include <map>

namespace Concurrency {

class AMPAllocator;

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



class AMPManager : public std::enable_shared_from_this<AMPManager>
{
public:
    virtual std::wstring get_path() = 0;
    virtual std::wstring get_description() = 0;
    virtual size_t get_mem() = 0;
    virtual bool is_double() = 0;
    virtual bool is_lim_double() = 0;
    virtual bool is_unified() = 0;
    virtual bool is_emulated() = 0;
    access_type cpu_type;


    virtual std::shared_ptr<AMPAllocator> createAloc() = 0;
    virtual void* create(size_t count) = 0;
    virtual void create(void* ptr) = 0;
    virtual void release(void* ptr) = 0;
    virtual void* CreateKernel(const char* fun, void* size, void* source) = 0;
    virtual ~AMPManager() {}
};

class AMPAllocator
{
public:
  virtual ~AMPAllocator() {}
  virtual void flush() {}
  virtual void wait() {}
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {}
  virtual void* LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) { return nullptr; }

  virtual void read(void* device, void* dst, size_t count) = 0;
  virtual void write(void* device, void* src, size_t count) = 0;
  virtual void copy(void* dst, void* src, size_t count) = 0;
  virtual void* map(void* device, size_t count, size_t offset, bool modify) = 0;
  virtual void unmap(void* device, void* addr) = 0;
  virtual void Push(void *kernel, int idx, void*& data, void* device) = 0;

  std::shared_ptr<AMPManager> getMan() { return Man; }
  queuing_mode mode;
protected:
  AMPAllocator(std::shared_ptr<AMPManager> Man) : mode(queuing_mode_automatic), Man(Man) {}
private:
  std::shared_ptr<AMPManager> Man;
};

class CPUManager final : public AMPManager
{
    std::shared_ptr<AMPAllocator> newAloc();
    std::map<void*, int> addr;
public:
    CPUManager() : addr() {}


    std::wstring get_path() override { return L"cpu"; }
    std::wstring get_description() override { return L"CPU Device"; }
    size_t get_mem() override { return 0; }
    bool is_double() override { return true; }
    bool is_lim_double() override { return true; }
    bool is_unified() override { return true; }
    bool is_emulated() override { return true; }

    std::shared_ptr<AMPAllocator> createAloc() { return newAloc(); }
    void* create(size_t count) override { return aligned_alloc(0x1000, count); }
    void create(void* ptr) override { addr[ptr]++; }
    void release(void* ptr) override {
        auto it = addr.find(ptr);
        if (it != std::end(addr)) {
            if (!--it->second)
                addr.erase(it);
        } else
            ::operator delete(ptr);
    }
    void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }
};

class CPUAllocator final : public AMPAllocator
{
public:
    CPUAllocator(std::shared_ptr<AMPManager> Man) : AMPAllocator(Man) {}
    void* map(void* device, size_t count, size_t offset, bool modify) override { return (char*)device + offset; }
    void unmap(void* device, void* addr) override {}
    void read(void* device, void* dst, size_t count) {}
    void write(void* device, void* src, size_t count) {}
    void copy(void* device, void* src, size_t count) override {}
    void Push(void *kernel, int idx, void*& data, void* device) override {}
};

std::shared_ptr<AMPAllocator> CPUManager::newAloc() {
    return std::shared_ptr<AMPAllocator>(new CPUAllocator(shared_from_this()));
}

class AMPContext
{
protected:
    std::shared_ptr<AMPManager> def;
    std::vector<std::shared_ptr<AMPManager>> Devices;
    std::map<std::shared_ptr<AMPManager>,
        std::shared_ptr<AMPAllocator>> default_map;
    AMPContext() : def(), Devices() {
        auto Man = std::shared_ptr<AMPManager>(new CPUManager);
        default_map[Man] = Man->createAloc();
        Devices.push_back(Man);
    }
public:
    virtual ~AMPContext() {}

    std::vector<std::shared_ptr<AMPManager>> getDevices() { return Devices; }

    bool set_default(const std::wstring& path) {
        for (const auto dev : Devices)
            if (dev->get_path() == path) {
                def = dev;
                return true;
            }
        return false;
    }

    std::shared_ptr<AMPAllocator> auto_select() { return default_map[Devices[1]]; }
    std::shared_ptr<AMPAllocator> getView(const std::shared_ptr<AMPManager>& pMan) {
        return default_map[pMan];
    }
    std::shared_ptr<AMPManager> getDevice(std::wstring path = L"") {
        if (path == L"default")
            return def;
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
extern void MatchKernelNames(std::string &);

extern void PushArg(void *, int, size_t, const void *);
extern void PushArgPtr(void *, int, size_t, const void *);

} // namespace CLAMP

class Serialize {
public:
    typedef void *kernel;
    Serialize(bool flag)
        : aloc_(), k_(nullptr), current_idx_(0), collector(), collect(true) {}
    Serialize(kernel k)
        : aloc_(), k_(k), current_idx_(0), collector(), collect(false) {}
    Serialize(std::shared_ptr<AMPAllocator> aloc, kernel k)
        : aloc_(aloc), k_(k), current_idx_(0), collector(), collect(false) {}
    void Append(size_t sz, const void *s) {
        if (!collect)
            CLAMP::PushArg(k_, current_idx_++, sz, s);
    }
    std::shared_ptr<AMPAllocator> get_aloc() { return aloc_; }
    void AppendPtr(size_t sz, const void *s) {
        if (!collect)
            CLAMP::PushArgPtr(k_, current_idx_++, sz, s);
    }
    void* getKernel() { return k_; }
    int getAndIncCurrentIndex() {
        int ret = current_idx_;
        current_idx_++;
        return ret;
    }

    // select best
    void push(std::shared_ptr<AMPAllocator> aloc) { collector.push_back(aloc); }
    bool is_collec() const { return collect; }
    std::shared_ptr<AMPAllocator> best() {
        std::sort(std::begin(collector), std::end(collector));
        std::vector<std::shared_ptr<AMPAllocator>> candidate;
        int max = 0;
        for (int i = 0; i < collector.size(); ++i) {
            auto head = collector[i];
            int count = 1;
            while (head == collector[++i])
                ++count;
            if (count > max) {
                max = count;
                candidate.clear();
                candidate.push_back(head);
            }
        }
        if (candidate.size())
            return candidate[0];
        else
            return nullptr;
    }
private:
    std::shared_ptr<AMPAllocator> aloc_;
    kernel k_;
    int current_idx_;
    std::vector<std::shared_ptr<AMPAllocator>> collector;
    bool collect;
};

static const std::shared_ptr<AMPAllocator> get_cpu_view() {
    return getContext()->getView(getContext()->getDevice(L"cpu"));
}

struct rw_info
{
    void *data;
    const size_t count;
    std::shared_ptr<AMPAllocator> curr;
    std::shared_ptr<AMPAllocator> master;
    std::shared_ptr<AMPAllocator> stage;
    std::map<std::shared_ptr<AMPManager>, void*> Alocs;
    access_type mode;
    unsigned int discard : 1;
    unsigned int onDevice : 1;

    // consruct array_view
    rw_info(const size_t count, void* ptr)
        : data(ptr), count(count), curr(nullptr), master(nullptr), stage(nullptr),
        Alocs(), mode(access_type_read_write), discard(false), onDevice(false) {
            if (ptr) {
                curr = master = get_cpu_view();
                curr->getMan()->create(ptr);
                Alocs[curr->getMan()] = ptr;
            }
        }

    // construct array
    rw_info(const std::shared_ptr<AMPAllocator> Aloc, const std::shared_ptr<AMPAllocator> Stage,
            const size_t count, access_type mode) : data(nullptr), count(count),
    curr(Aloc), master(Aloc), stage(nullptr), Alocs(), mode(mode), discard(false), onDevice(false) {
        Alocs[curr->getMan()] = curr->getMan()->create(count);
        if (curr->getMan()->get_path() == L"cpu") {
            data = Alocs[curr->getMan()];
            if (Stage != curr) {
                stage = Stage;
                Alocs[stage->getMan()] = stage->getMan()->create(count);
            } else
                stage = curr;
        } else {
            stage = curr;
            if (curr->getMan()->is_unified() && mode != access_type_none)
                data = Alocs[curr->getMan()];
            else
                onDevice = true;
        }

        if (mode == access_type_auto)
            mode = curr->getMan()->cpu_type;
    }

    void append(Serialize& s, bool isArray) {
        auto aloc = s.get_aloc();
        if (!curr) {
            Alocs[aloc->getMan()] = aloc->getMan()->create(count);
            curr = aloc;
        }
        if (aloc->getMan() != curr->getMan()) {
            if (Alocs.find(aloc->getMan()) == std::end(Alocs))
                Alocs[aloc->getMan()] = aloc->getMan()->create(count);
            if (!discard || isArray) {
                void* dst = Alocs[aloc->getMan()];
                void* src = Alocs[curr->getMan()];
                // assert(aloc->getMan()->get_path() != L"cpu");
                if (curr->getMan()->get_path() == L"cpu")
                    aloc->write(dst, src, count);
                else {
                    curr->wait();
                    aloc->copy(dst, src, count);
                }
            }
            curr = aloc;
        } else {
            if (curr != aloc) {
                curr->wait();
                curr = aloc;
            }
        }
        onDevice = true;
        discard = false;
        curr->Push(s.getKernel(), s.getAndIncCurrentIndex(), data, Alocs[curr->getMan()]);
    }

    void disc() {
        onDevice = false;
        discard = true;
    }

    void* map(size_t cnt, size_t offset, bool modify) {
        if (!curr)
            get_cpu_access(modify);
        if (cnt == 0)
            cnt = count;
        return curr->map(Alocs[curr->getMan()], cnt, offset, modify);
    }
    void unmap(void* addr) { curr->unmap(Alocs[curr->getMan()], addr); }

    void synchronize(bool modify) {
        if (!discard && data && onDevice) {
            onDevice = false;
            void* src = Alocs[curr->getMan()];
            if (src != data)
                curr->read(src, data, count);
        }
        if (modify) {
            if (master)
                curr = master;
            else
                curr = get_cpu_view();
        }
    }

    void get_cpu_access(bool modify) {
        if (!discard) {
            auto cpu_view = get_cpu_view();
            if (Alocs.find(cpu_view->getMan()) == std::end(Alocs)) {
                data = cpu_view->getMan()->create(count);
                Alocs[cpu_view->getMan()] = data;
            }
            if (onDevice)
                curr->read(Alocs[curr->getMan()], data, count);
        }
        if (modify || !curr)
            curr = get_cpu_view();
        onDevice = false;
    }

    void copy(rw_info* other) {
        if (!curr) {
            if (!other->curr) {
                get_cpu_access(false);
                other->get_cpu_access(true);
            } else {
                curr = other->curr;
                Alocs[curr->getMan()] = curr->getMan()->create(count);
            }
        } else {
            if (!other->curr) {
                other->curr = curr;
                other->Alocs[curr->getMan()] = curr->getMan()->create(count);
            }
        }
        void* dst = other->Alocs[other->curr->getMan()];
        void* src = Alocs[curr->getMan()];
        if (other->curr->getMan()->get_path() == L"cpu") {
            if (curr->getMan()->get_path() == L"cpu")
                memmove(dst, src, count);
            else
                curr->read(src, dst, count);
        } else {
            other->onDevice = true;
            if (curr->getMan()->get_path() == L"cpu")
                other->curr->write(dst, src, count);
            else  {
                curr->wait();
                other->curr->copy(dst, src, count);
            }
        }
    }

    ~rw_info() {
        synchronize(false);
        std::shared_ptr<AMPManager> pMan;
        void* addr;
        for (const auto it : Alocs) {
            std::tie(pMan, addr) = it;
            pMan->release(addr);
        }
    }
};

} // namespace Concurrency

#endif // __CLAMP_AMP_RUNTIME
