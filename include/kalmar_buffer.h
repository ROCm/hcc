//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "kalmar_runtime.h"
#include "kalmar_serialize.h"

/** \cond HIDDEN_SYMBOLS */
namespace Kalmar {

// Dummy interface that looks somewhat like std::shared_ptr<T>
template <typename T>
class _data {
public:
    _data() = delete;
    _data(int count) : p_(nullptr) {}
    _data(const _data& d) restrict(cpu, amp)
        : p_(d.p_) {}
    _data(int count, void* d) restrict(cpu, amp)
        : p_(static_cast<T*>(d)) {}
    template <typename U>
        _data(const _data<U>& d) restrict(cpu, amp)
        : p_(reinterpret_cast<T *>(d.get())) {}
    __attribute__((annotate("user_deserialize")))
        explicit _data(T* t) restrict(cpu, amp) { p_ = t; }
    T* get(void) const restrict(cpu, amp) { return p_; }
    T* get_device_pointer() const restrict(cpu, amp) { return p_; }
    std::shared_ptr<KalmarQueue> get_av() const { return nullptr; }
    void reset() const {}

    T* map_ptr(bool modify, size_t count, size_t offset) const { return nullptr; }
    void unmap_ptr(const void* addr, bool modify, size_t count, size_t offset) const {}
    void synchronize(bool modify = false) const {}
    void get_cpu_access(bool modify = false) const {}
    void copy(_data<T> other, int, int, int) const {}
    void write(const T*, int , int offset = 0, bool blocking = false) const {}
    void read(T*, int , int offset = 0) const {}
    void refresh() const {}
    void set_const() const {}
    access_type get_access() const { return access_type_auto; }
    std::shared_ptr<KalmarQueue> get_stage() const { return nullptr; }

private:
    T* p_;
};

template <typename T>
class _data_host {
    mutable std::shared_ptr<rw_info> mm;
    bool isArray;
    template <typename U> friend class _data_host;
public:
    _data_host(size_t count, const void* src = nullptr)
        : mm(std::make_shared<rw_info>(count*sizeof(T), const_cast<void*>(src))),
        isArray(false) {}

    _data_host(std::shared_ptr<KalmarQueue> av, std::shared_ptr<KalmarQueue> stage, int count,
               access_type mode)
        : mm(std::make_shared<rw_info>(av, stage, count*sizeof(T), mode)), isArray(true) {}

    _data_host(std::shared_ptr<KalmarQueue> av, std::shared_ptr<KalmarQueue> stage, int count,
               void* device_pointer, access_type mode)
        : mm(std::make_shared<rw_info>(av, stage, count*sizeof(T), device_pointer, mode)), isArray(true) {}

    _data_host(const _data_host& other) : mm(other.mm), isArray(false) {}

    template <typename U>
        _data_host(const _data_host<U>& other) : mm(other.mm), isArray(false) {}

    T *get() const { return static_cast<T*>(mm->data); }
    T* get_device_pointer() const { return static_cast<T*>(mm->get_device_pointer()); }
    void synchronize(bool modify = false) const { mm->synchronize(modify); }
    void discard() const { mm->disc(); }
    void refresh() const {}
    size_t size() const { return mm->count; }
    void reset() const { mm.reset(); }
    void get_cpu_access(bool modify = false) const { mm->get_cpu_access(modify); }
    std::shared_ptr<KalmarQueue> get_av() const { return mm->master; }
    std::shared_ptr<KalmarQueue> get_stage() const { return mm->stage; }
    access_type get_access() const { return mm->mode; }
    void copy(_data_host<T> other, int src_offset, int dst_offset, int size) const {
        mm->copy(other.mm.get(), src_offset * sizeof(T), dst_offset * sizeof(T), size * sizeof(T));
    }
    void write(const T* src, int size, int offset = 0, bool blocking = false) const {
        mm->write(src, size * sizeof(T), offset * sizeof(T), blocking);
    }
    void read(T* dst, int size, int offset = 0) const {
        mm->read(dst, size * sizeof(T), offset * sizeof(T));
    }
    T* map_ptr(bool modify, size_t count, size_t offset) const {
        return (T*)mm->map(count * sizeof(T), offset * sizeof(T), modify);
    }
    void unmap_ptr(const void* addr, bool modify, size_t count, size_t offset) const { return mm->unmap(const_cast<void*>(addr), count * sizeof(T), offset * sizeof(T), modify); }
    void sync_to(std::shared_ptr<KalmarQueue> pQueue) const { mm->sync(pQueue, false); }

    __attribute__((annotate("serialize")))
        void __cxxamp_serialize(Serialize& s) const {
            s.visit_buffer(mm.get(), !std::is_const<T>::value, isArray);
        }
    __attribute__((annotate("user_deserialize")))
        explicit _data_host(typename std::remove_const<T>::type* t) {}
};

} // namespace Kalmar
/** \endcond */
