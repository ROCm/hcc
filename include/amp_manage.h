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
    std::shared_ptr<AMPAllocator> get_av() const { return nullptr; }
    void reset() const {}
private:
    __global T* p_;
};

template <typename T>
class _data_host {
    mutable std::shared_ptr<rw_info> mm;
    size_t count;
    bool isArray;
    template <typename U> friend class _data_host;

public:
    _data_host(std::shared_ptr<AMPAllocator> av, int count, bool isArr = false)
        : mm(std::make_shared<rw_info>(av, count*sizeof(T))), count(count), isArray(isArr) {}

    _data_host(std::shared_ptr<AMPAllocator> av, int count, T* src, bool isArr = false)
        : mm(std::make_shared<rw_info>(av, count*sizeof(T), src)), count(count), isArray(isArr) {}

    _data_host(const _data_host& other)
        : mm(other.mm), count(other.count), isArray(false) {}

    template <typename U>
        _data_host(const _data_host<U>& other)
        : mm(other.mm), count(other.count), isArray(false) {}

    _data_host(_data_host&& other) : mm(other.mm) { other.reset(); }
    _data_host& operator=(const _data_host& other) {
        mm = other.mm;
        count = other.count;
        isArray = false;
        return *this;
    }

    T *get() const { return static_cast<T*>(mm->data); }
    void synchronize() const { mm->synchronize(); }
    void discard() const { mm->disc(); }
    void refresh() const {}
    void copy(void *dst) const { mm->copy(dst, count * sizeof(T)); }
    size_t size() const { return count; }
    void stash() const { mm->stash(); }
    void reset() const { mm.reset(); }
    std::shared_ptr<AMPAllocator> get_av() const { return mm->master; }

    __attribute__((annotate("serialize")))
        void __cxxamp_serialize(Serialize& s) const {
            mm->append(s, isArray);
        }
    __attribute__((annotate("user_deserialize")))
        explicit _data_host(__global T* t) {}
};

} // namespace Concurrency

#endif // __CLAMP_AMP_MANAGE
