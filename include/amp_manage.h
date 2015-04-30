//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifdef __AMP_CPU__
#include <amp_cpu_manage.h>
#else

#pragma once

#include <amp_allocator.h>

namespace Concurrency {

// Dummy interface that looks somewhat like std::shared_ptr<T>
template <typename T>
class _data {
public:
    _data() = delete;
    _data(int count, bool) {}
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


static inline void amp_no_delete(void *p)
{
    getAllocator()->sync(p);
    getAllocator()->free(p);
}

static inline void amp_delete(void *p)
{
    amp_no_delete(p);
    operator delete(p);
}

template <typename T>
class _data_host {
    std::shared_ptr<void> mm;
    size_t count;
    bool isArray;
    template <typename U> friend class _data_host;
public:
    _data_host(int count, bool isArr = false)
        : mm(aligned_alloc(0x1000, count * sizeof(T)), amp_delete), count(count),
        isArray(isArr) { getAllocator()->init(mm.get(), count * sizeof(T)); }
    _data_host(int count, T* src, bool isArr = false)
        : mm(src, amp_no_delete), count(count), isArray(isArr)
    { getAllocator()->init(mm.get(), count * sizeof(T)); }
    _data_host(const _data_host& other)
        : mm(other.mm), count(other.count), isArray(false) {}
    template <typename U>
        _data_host(const _data_host<U>& other)
        : mm(other.mm), count(other.count), isArray(false) {}

    T *get() const { return (T *)mm.get(); }
    void synchronize() const { getAllocator()->sync(mm.get()); }
    void discard() const { getAllocator()->discard(mm.get()); }
    void refresh() const {}
    void copy(void *dst) const { getAllocator()->copy(dst, mm.get(), count * sizeof(T)); }
    size_t size() const { return count; }
    void stash() const { getAllocator()->stash(mm.get()); }

    __attribute__((annotate("serialize")))
        void __cxxamp_serialize(Serialize& s) const {
            getAllocator()->append(s.getKernel(), s.getAndIncCurrentIndex(), mm.get(), isArray);
        }
    __attribute__((annotate("user_deserialize")))
        explicit _data_host(__global T* t);
};

inline void *getDevicePointer(void *ptr) { return getAllocator()->device_data(ptr); }
inline void *getOCLQueue(void *ptr) { return getAllocator()->getQueue(); }

} // namespace Concurrency
#endif
