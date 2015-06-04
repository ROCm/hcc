#ifndef __CLAMP_AMP_RUNTIME
#define __CLAMP_AMP_RUNTIME

#include <map>

namespace Concurrency {

class AMPView;

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

class AMPDevice : public std::enable_shared_from_this<AMPDevice>
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


    virtual std::shared_ptr<AMPView> createAloc() = 0;
    virtual void* create(size_t count) = 0;
    virtual void release(void* ptr) = 0;
    virtual void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }
    virtual ~AMPDevice() {}
};

class AMPView
{
public:
  virtual ~AMPView() {}
  virtual void flush() {}
  virtual void wait() {}
  virtual void LaunchKernel(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) {}
  virtual void* LaunchKernelAsync(void *kernel, size_t dim_ext, size_t *ext, size_t *local_size) { return nullptr; }

  virtual void read(void* device, void* dst, size_t count, size_t offset) {
      memmove(dst, (char*)device + offset, count);
  }
  virtual void write(void* device, const void* src, size_t count, size_t offset, bool blocking) {
      memmove((char*)device + offset, src, count);
  }
  virtual void copy(void* src, void* dst, size_t count, size_t src_offset, size_t dst_offset, bool blocking) {
      memmove((char*)dst + dst_offset, (char*)src + src_offset, count);
  }
  virtual void* map(void* device, size_t count, size_t offset, bool modify) {
      return (char*)device + offset;
  }
  virtual void unmap(void* device, void* addr) {}
  virtual void Push(void *kernel, int idx, void*& data, void* device, bool isConst) = 0;

  std::shared_ptr<AMPDevice> getMan() { return Man; }
  AMPDevice* getManPtr() { return Man.get(); }
  queuing_mode mode;
protected:
  AMPView(std::shared_ptr<AMPDevice> Man) : mode(queuing_mode_automatic), Man(Man) {}
private:
  std::shared_ptr<AMPDevice> Man;
};

class CPUManager final : public AMPDevice
{
    std::shared_ptr<AMPView> newAloc();
public:
    std::wstring get_path() override { return L"cpu"; }
    std::wstring get_description() override { return L"CPU Device"; }
    size_t get_mem() override { return 0; }
    bool is_double() override { return true; }
    bool is_lim_double() override { return true; }
    bool is_unified() override { return true; }
    bool is_emulated() override { return true; }


    std::shared_ptr<AMPView> createAloc() { return newAloc(); }
    void* create(size_t count) override { return aligned_alloc(0x1000, count); }
    void release(void* ptr) override { ::operator delete(ptr); }
    void* CreateKernel(const char* fun, void* size, void* source) { return nullptr; }
};

class CPUAllocator final : public AMPView
{
public:
    CPUAllocator(std::shared_ptr<AMPDevice> Man) : AMPView(Man) {}
    void Push(void *kernel, int idx, void*& data, void* device, bool isConst) override {}
};

inline std::shared_ptr<AMPView> CPUManager::newAloc() {
    return std::shared_ptr<AMPView>(new CPUAllocator(shared_from_this()));
}

class AMPContext
{
protected:
    std::shared_ptr<AMPDevice> def;
    std::vector<std::shared_ptr<AMPDevice>> Devices;
    std::map<std::shared_ptr<AMPDevice>,
        std::shared_ptr<AMPView>> default_map;
    AMPContext() : def(), Devices() {
        auto Man = std::shared_ptr<AMPDevice>(new CPUManager);
        default_map[Man] = Man->createAloc();
        Devices.push_back(Man);
    }
public:
    virtual ~AMPContext() {}

    std::vector<std::shared_ptr<AMPDevice>> getDevices() { return Devices; }

    bool set_default(const std::wstring& path) {
        auto result = std::find_if(std::begin(Devices), std::end(Devices),
                                   [&] (const std::shared_ptr<AMPDevice>& Man)
                                   { return Man->get_path() == path; });
        if (result == std::end(Devices))
            return false;
        else {
            def = *result;
            return true;
        }
    }

    std::shared_ptr<AMPView> auto_select() { return default_map[def]; }
    std::shared_ptr<AMPView> getView(const std::shared_ptr<AMPDevice>& pMan) {
        return default_map[pMan];
    }
    std::shared_ptr<AMPDevice> getDevice(std::wstring path = L"") {
        if (path == L"default" || path == L"")
            return def;
        auto result = std::find_if(std::begin(Devices), std::end(Devices),
                                   [&] (const std::shared_ptr<AMPDevice>& man)
                                   { return man->get_path() == path; });
        if (result != std::end(Devices))
            return *result;
        else
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

extern void *CreateKernel(std::string, AMPView*);
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
    Serialize(std::shared_ptr<AMPView> aloc, kernel k)
        : aloc_(aloc), k_(k), current_idx_(0), collector(), collect(false) {}
    void Append(size_t sz, const void *s) {
        if (!collect)
            CLAMP::PushArg(k_, current_idx_++, sz, s);
    }
    std::shared_ptr<AMPView> get_aloc() { return aloc_; }
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
    void push(std::shared_ptr<AMPView> aloc) { collector.push_back(aloc); }
    bool is_collec() const { return collect; }
    std::shared_ptr<AMPView> best() {
        std::sort(std::begin(collector), std::end(collector));
        std::vector<std::shared_ptr<AMPView>> candidate;
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
    std::shared_ptr<AMPView> aloc_;
    kernel k_;
    int current_idx_;
    std::vector<std::shared_ptr<AMPView>> collector;
    bool collect;
};

static const std::shared_ptr<AMPView> get_cpu_view() {
    static auto cpu_view = getContext()->getView(getContext()->getDevice(L"cpu"));
    return cpu_view;
}

enum states
{
    modified,
    shared,
    invalid
};

struct dev_info
{
    void* data;
    states state;
};

struct rw_info
{
    void *data;
    const size_t count;
    std::shared_ptr<AMPView> curr;
    std::shared_ptr<AMPView> master;
    std::shared_ptr<AMPView> stage;
    std::map<AMPDevice*, dev_info> Alocs;
    access_type mode;
    unsigned int HostPtr : 1;


    // consruct array_view
    rw_info(const size_t count, void* ptr)
        : data(ptr), count(count), curr(nullptr), master(nullptr), stage(nullptr),
        Alocs(), mode(access_type_none), HostPtr(ptr != nullptr) {
#ifdef __AMP_CPU__
            if (CLAMP::in_cpu_kernel() && ptr == nullptr) {
                data = aligned_alloc(0x1000, count);
                return;
            }
#endif
            if (ptr) {
                mode = access_type_read_write;
                curr = master = get_cpu_view();
                Alocs[curr->getManPtr()] = {ptr, modified};
            }
        }

    // construct array
    rw_info(const std::shared_ptr<AMPView> Aloc, const std::shared_ptr<AMPView> Stage,
            const size_t count, access_type mode_) : data(nullptr), count(count),
    curr(Aloc), master(Aloc), stage(nullptr), Alocs(), mode(mode_), HostPtr(false) {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel() && data == nullptr) {
            data = aligned_alloc(0x1000, count);
            return;
        }
#endif
        if (mode == access_type_auto)
            mode = curr->getManPtr()->cpu_type;
        Alocs[curr->getManPtr()] = {curr->getManPtr()->create(count), modified};
        if (curr->getManPtr()->get_path() == L"cpu") {
            data = Alocs[curr->getManPtr()].data;
            if (Stage != curr) {
                stage = Stage;
                Alocs[stage->getManPtr()] = {stage->getManPtr()->create(count), invalid};
            } else
                stage = curr;
        } else {
            stage = curr;
            if (curr->getManPtr()->is_unified() && mode != access_type_none)
                data = Alocs[curr->getManPtr()].data;
        }
    }

    void construct(std::shared_ptr<AMPView> aloc) {
        curr = aloc;
        Alocs[aloc->getManPtr()] = {aloc->getManPtr()->create(count), invalid};
        if (aloc->getManPtr()->get_path() == L"cpu")
            data = Alocs[aloc->getManPtr()].data;
    }

    void disc() {
        for (auto& it : Alocs)
            it.second.state = invalid;
    }

    void sync(std::shared_ptr<AMPView> aloc, bool modify) {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        if (curr->getManPtr() == aloc->getManPtr())
            return;
        dev_info& dst = Alocs[aloc->getManPtr()];
        dev_info& src = Alocs[curr->getManPtr()];
        if (dst.state == invalid) {
            auto cpu_view = get_cpu_view();
            if (Alocs.find(cpu_view->getManPtr()) != std::end(Alocs))
                if (Alocs[cpu_view->getManPtr()].state == shared) {
                    curr = cpu_view;
                    src = Alocs[cpu_view->getManPtr()];
                }
            if (src.state != invalid) {
                if (aloc->getManPtr()->get_path() == L"cpu")
                    curr->read(src.data, dst.data, count, 0);
                else if (curr->getManPtr()->get_path() == L"cpu")
                    aloc->write(dst.data, src.data, count, 0, false);
                else
                    curr->copy(src.data, dst.data, count, 0, 0, false);
            }
        }
        curr = aloc;
        if (modify) {
            disc();
            dst.state = modified;
        } else {
            dst.state = shared;
            if (src.state == modified)
                src.state = shared;
        }
    }

    void append(Serialize& s, bool isArray, bool isConst) {
        auto aloc = s.get_aloc();
        if (!curr) {
            construct(aloc);
            dev_info& obj = Alocs[curr->getManPtr()];
            if (isConst)
                obj.state = shared;
            else
                obj.state = modified;
        }
        if (aloc->getManPtr() != curr->getManPtr()) {
            if (Alocs.find(aloc->getManPtr()) == std::end(Alocs))
                Alocs[aloc->getManPtr()] = {aloc->getManPtr()->create(count), invalid};
            dev_info& dst = Alocs[aloc->getManPtr()];
            dev_info& src = Alocs[curr->getManPtr()];
            if (dst.state == invalid && (src.state != invalid || isArray)) {
                auto cpu_view = get_cpu_view();
                if (src.state == shared && cpu_view != curr)
                    if (Alocs.find(cpu_view->getManPtr()) != std::end(Alocs))
                        if (Alocs[cpu_view->getManPtr()].state == shared) {
                            curr = cpu_view;
                            src = Alocs[cpu_view->getManPtr()];
                        }
                if (curr->getManPtr()->get_path() == L"cpu")
                    aloc->write(dst.data, src.data, count, 0, false);
                else {
                    // curr->wait();
                    aloc->copy(src.data, dst.data, count, 0, 0, false);
                }
            }
            if (isConst) {
                if (src.state == modified)
                    src.state = shared;
                dst.state = shared;
            } else {
                curr = aloc;
                if (src.state != invalid)
                    disc();
                dst.state = modified;
            }
        } else {
            if (curr != aloc) {
                // curr->wait();
                curr = aloc;
            }
        }
        aloc->Push(s.getKernel(), s.getAndIncCurrentIndex(), data, Alocs[aloc->getManPtr()].data, isConst);
    }


    void* map(size_t cnt, size_t offset, bool modify) {
        if (cnt == 0)
            cnt = count;
        if (!curr) {
            auto def_view = getContext()->auto_select();
            curr = def_view;
            data = def_view->getManPtr()->create(count);
            if (modify)
                Alocs[def_view->getManPtr()] = {data, modified};
            else
                Alocs[def_view->getManPtr()] = {data, shared};
            return curr->map(data, cnt, offset, modify);;
        }
        auto cpu_view = get_cpu_view();
        if (Alocs.find(cpu_view->getManPtr()) != std::end(Alocs))
            if (Alocs[cpu_view->getManPtr()].state == shared)
                curr = cpu_view;
        dev_info& info = Alocs[curr->getManPtr()];
        if (info.state == shared && modify) {
            disc();
            info.state = modified;
        }
        return curr->map(info.data, cnt, offset, modify);
    }
    void unmap(void* addr) { curr->unmap(Alocs[curr->getManPtr()].data, addr); }

    void synchronize(bool modify) { sync(master, modify); }
    void sync_to(std::shared_ptr<AMPView> aloc) {
        auto Man = aloc->getManPtr();
        if (Alocs.find(Man) == std::end(Alocs))
            Alocs[Man] = {Man->create(count), invalid};
        if (curr)
            sync(aloc, false);
        else
            curr = master = aloc;
    }

    void get_cpu_access(bool modify) {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel())
            return;
#endif
        auto cpu_view = get_cpu_view();
        if (Alocs.find(cpu_view->getManPtr()) == std::end(Alocs)) {
            data = cpu_view->getManPtr()->create(count);
            Alocs[cpu_view->getManPtr()] = {data, invalid};
        }
        if (!curr) {
            curr = cpu_view;
            return;
        }
        if (curr == cpu_view)
            return;
        sync(cpu_view, modify);
    }

    void write(const void* src, int cnt, int offset, bool blocking) {
        curr->write(Alocs[curr->getManPtr()].data, src, cnt, offset, blocking);
        dev_info& dev = Alocs[curr->getManPtr()];
        if (dev.state != modified) {
            disc();
            dev.state = modified;
        }
    }

    void read(void* dst, int cnt, int offset) {
        curr->read(Alocs[curr->getManPtr()].data, dst, cnt, offset);
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
        dev_info& dst = other->Alocs[other->curr->getManPtr()];
        dev_info& src = Alocs[curr->getManPtr()];
        if (src.state == invalid) {
            src.state = shared;
            if (curr->getManPtr()->get_path() == L"cpu")
                memset((char*)src.data + src_offset, 0, cnt);
            else {
                void *ptr = aligned_alloc(0x1000, cnt);
                memset(ptr, 0, cnt);
                curr->write(src.data, ptr, cnt, src_offset, true);
                ::operator delete(ptr);
            }
        }
        if (other->curr->getManPtr()->get_path() == L"cpu") {
            if (curr->getManPtr()->get_path() == L"cpu")
                memmove((char*)dst.data + dst_offset, (char*)src.data + src_offset, cnt);
            else
                curr->read(src.data, (char*)dst.data + dst_offset, cnt, src_offset);
        } else {
            if (curr->getManPtr()->get_path() == L"cpu")
                other->curr->write(dst.data, (char*)src.data + src_offset, cnt, dst_offset, false);
            else {
                // curr->wait();
                other->curr->copy(src.data, dst.data, cnt, src_offset, dst_offset, true);
            }
        }
        other->disc();
        dst.state = modified;
    }

    ~rw_info() {
#ifdef __AMP_CPU__
        if (CLAMP::in_cpu_kernel()) {
            if (data && !HostPtr)
                ::operator delete(data);
            return;
        }
#endif
        if (HostPtr)
            synchronize(false);
        auto cpu_acc = get_cpu_view()->getManPtr();
        if (Alocs.find(cpu_acc) != std::end(Alocs)) {
            if (!HostPtr)
                cpu_acc->release(Alocs[cpu_acc].data);
            Alocs.erase(cpu_acc);
        }
        AMPDevice* pMan;
        dev_info info;
        for (const auto it : Alocs) {
            std::tie(pMan, info) = it;
            pMan->release(info.data);
        }
    }
};

} // namespace Concurrency

#endif // __CLAMP_AMP_RUNTIME
