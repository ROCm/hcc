//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <amp_allocator.h>

namespace Concurrency {

struct mm_info
{
    size_t count;
    void *host;
    void *device;
    void *dirty;
    bool discard;
    bool sync;
    mm_info(int count)
        : count(count), host(::operator new(count)), device(host),
        dirty(host), discard(false), sync(true)
    { getAllocator()->init(device, count); }
    mm_info(int count, void *src)
        : count(count), host(src), device(::operator new(count)),
        dirty(host), discard(false), sync(true)
    { getAllocator()->init(device, count); }
    void synchronize() {
        if (dirty != host && sync) {
            memmove(host, device, count);
            dirty = host;
        }
    }
    void refresh() {
        if (device != host)
            memmove(device, host, count);
    }
    void* get() { return dirty; }
    void disc() {
        if (dirty != host)
            dirty = host;
        discard = true;
    }
    void serialize(Serialize& s) {
        if (dirty == host && device != host) {
            if (!discard)
                refresh();
            dirty = device;
        }
        sync = s.get_sync();
        discard = false;
        getAllocator()->append(s.getKernel(), s.getAndIncCurrentIndex(), device);
    }
    ~mm_info() {
        getAllocator()->free(device);
        if (host != device) {
            if (!discard)
                synchronize();
            ::operator delete(device);
        }
    }
};

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
private:
    __global T* p_;
};

template <typename T>
class _data_host {
    std::shared_ptr<mm_info> mm;
    template <typename U> friend struct _data_host;
public:
    _data_host(int count)
        : mm(std::make_shared<mm_info>(count * sizeof(T))) {}
    _data_host(int count, T* src)
        : mm(std::make_shared<mm_info>(count * sizeof(T), src)) {}
    _data_host(const _data_host& other)
        : mm(other.mm) {}
    template <typename U>
        _data_host(const _data_host<U>& other) : mm(other.mm) {}

    T *get() const { return (T *)mm->get(); }
    void synchronize() const { mm->synchronize(); }
    void discard() const { mm->disc(); }
    void refresh() const { mm->refresh(); }

    __attribute__((annotate("serialize")))
        void __cxxamp_serialize(Serialize& s) const {
            mm->serialize(s);
        }
    __attribute__((annotate("user_deserialize")))
        explicit _data_host(__global T* t)
#ifdef __AMP_CPU__
        {}
#endif
};

} // namespace Concurrency
