//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __CLAMP_AMP_MANAGE
#define __CLAMP_AMP_MANAGE

#include <amp_runtime.h>

namespace Concurrency {

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

#ifndef E_FAIL
#define E_FAIL 0x80004005
#endif

static const char *__errorMsg_UnsupportedAccelerator = "concurrency::parallel_for_each is not supported on the selected accelerator \"CPU accelerator\".";


// Dummy interface that looks somewhat like std::shared_ptr<T>
template <typename T>
class _data {
public:
    _data() = delete;
    _data(int count) {}
    _data(const _data& d) restrict(cpu, amp)
        : p_(d.p_) {}
    template <typename U>
        _data(const _data<U>& d) restrict(cpu, amp)
        : p_(reinterpret_cast<T *>(d.get())) {}
    __attribute__((annotate("user_deserialize")))
        explicit _data(__global T* t) restrict(cpu, amp) { p_ = t; }
    __global T* get(void) const restrict(cpu, amp) { return p_; }
    std::shared_ptr<AMPAllocator> get_av() const { return nullptr; }
    void reset() const {}

    T* map_ptr(bool modify = false, size_t count = 0, size_t offset = 0) const { return nullptr; }
    void unmap_ptr(void* addr) const {}
    void synchronize(bool modify = false) const {}
    void copy(_data<T> other) const {}
    void refresh() const {}
    access_type get_access() const { return access_type_auto; }
    std::shared_ptr<AMPAllocator> get_stage() const { return nullptr; }

private:
    __global T* p_;
};

template <typename T>
class _data_host {
    mutable std::shared_ptr<rw_info> mm;
    bool isArray;
    template <typename U> friend class _data_host;

public:
    _data_host(size_t count, void* src = nullptr)
        : mm(std::make_shared<rw_info>(count*sizeof(T), src)), isArray(false) {}

    _data_host(std::shared_ptr<AMPAllocator> av, std::shared_ptr<AMPAllocator> stage, int count, access_type mode)
        : mm(std::make_shared<rw_info>(av, stage, count*sizeof(T), mode)), isArray(true) {}

    _data_host(const _data_host& other) : mm(other.mm), isArray(false) {}

    template <typename U>
        _data_host(const _data_host<U>& other) : mm(other.mm), isArray(false) {}

    T *get() const { return static_cast<T*>(mm->data); }
    void synchronize(bool modify = false) const { mm->synchronize(modify); }
    void discard() const { mm->disc(); }
    void refresh() const {}
    size_t size() const { return mm->count; }
    void reset() const { mm.reset(); }
    void get_cpu_access(bool modify = false) const { mm->get_cpu_access(modify); }
    std::shared_ptr<AMPAllocator> get_av() const { return mm->master; }
    std::shared_ptr<AMPAllocator> get_stage() const { return mm->stage; }
    access_type get_access() const { return mm->mode; }
    void copy(_data_host<T> other) const { mm->copy(other.mm.get()); }
    T* map_ptr(bool modify = false, size_t count = 0, size_t offset = 0) const {
        return (T*)mm->map(count * sizeof(T), offset * sizeof(T), modify);
    }
    void unmap_ptr(void* addr) const { return mm->unmap(addr); }

    __attribute__((annotate("serialize")))
        void __cxxamp_serialize(Serialize& s) const {
            if (s.is_collec()) {
                if (isArray && mm->stage->getMan()->get_path() != L"cpu")
                    s.push(mm->stage);
                return;
            }
            if (isArray) {
                auto curr = s.get_aloc()->getMan()->get_path();
                auto path = mm->master->getMan()->get_path();
                bool except = false;
                if (path == L"cpu") {
                    if (mm->stage != mm->master) {
                        auto asoc = mm->stage->getMan()->get_path();
                        if (asoc != curr)
                            except = true;
                    } else
                        except = true;
                }
                if (except)
                    throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
            }
            mm->append(s, isArray);
        }
    __attribute__((annotate("user_deserialize")))
        explicit _data_host(__global T* t) {}
};

} // namespace Concurrency

#endif // __CLAMP_AMP_MANAGE
