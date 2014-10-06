//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __CL_MANAGE__
#define __CL_MANAGE__

#pragma once
#include <string.h>
#include <CL/opencl.h>


struct mm_info;
struct AMPAllocator
{
    AMPAllocator() {
        cl_uint          num_platforms;
        cl_int           err;
        cl_platform_id   platform_id[10];
        int i;
        err = clGetPlatformIDs(10, platform_id, &num_platforms);
        for (i = 0; i < num_platforms; i++) {
            err = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err == CL_SUCCESS)
                break;
        }
        if (err != CL_SUCCESS) {
            for (i = 0; i < num_platforms; i++) {
                err = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
                if (err == CL_SUCCESS)
                    break;
            }
        }
        assert(err == CL_SUCCESS);
        context = clCreateContext(0, 1, &device, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        queue = clCreateCommandQueue(context, device, 0, &err);
        assert(err == CL_SUCCESS);
    }
    cl_mem setup(void *data, int count) {
        cl_int err;
        cl_mem dm = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, count, data, &err);
        assert(err == CL_SUCCESS);
        mem_info[data] = dm;
        return dm;
    }
    void unregister(void *data) {
        clReleaseMemObject(mem_info[data]);
        mem_info.erase(data);
    }
    ~AMPAllocator() {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }
    std::map<void *, cl_mem> mem_info;
    cl_context       context;
    cl_device_id     device;
    cl_kernel        kernel;
    cl_command_queue queue;
    cl_program       program;
};

AMPAllocator& getAllocator();

struct mm_info
{
    size_t count;
    void *src_ptr;
    void *data_ptr;
    bool isArray;
    bool discard;
    bool dirty;
    mm_info(int count)
        : count(count), src_ptr(nullptr), data_ptr(::operator new(count)),
        isArray(false), discard(false), dirty(false) {}
    mm_info(int count, void *src)
        : count(count), src_ptr(src), data_ptr(::operator new(count)),
        isArray(false), discard(false), dirty(false) {
            memmove(data_ptr, src_ptr, count);
        }
    void synchronize() {
        memmove(src_ptr, data_ptr, count);
    }
    void refresh() {
        if (src_ptr != nullptr)
            memmove(data_ptr, src_ptr, count);
    }
    void* get() {
        if (dirty) {
            dirty = false;
            synchronize();
        }
        return data_ptr;
    }
    void disc() {
        dirty = false;
        discard = !isArray;
    }
    void isArr() { isArray = true;}
    void serialize(Serialize& s) {
        dirty = src_ptr != nullptr;
        cl_mem dm = getAllocator().setup(data_ptr, count);
        s.Append(sizeof(cl_mem), &dm);
    }
    ~mm_info() {
        getAllocator().unregister(data_ptr);
        if (src_ptr != nullptr) {
            if (!discard)
                synchronize();
            ::operator delete(data_ptr);
        }
    }
};


// Dummy interface that looks somewhat like std::shared_ptr<T>
template <typename T>
class _data {
    typedef typename std::remove_const<T>::type nc_T;
    friend _data<const T>;
    friend _data<nc_T>;
public:
    _data() = delete;
    _data(int count) {}
    _data(const _data& d) restrict(cpu, amp):p_(d.p_) {}
    template <class = typename std::enable_if<std::is_const<T>::value>::type>
        _data(const _data<nc_T>& d) restrict(cpu, amp):p_(d.p_) {}
    template <class = typename std::enable_if<!std::is_const<T>::value>::type>
        _data(const _data<const T>& d) restrict(cpu, amp):p_(const_cast<T*>(d.p_)) {}
    template <typename T2>
        _data(const _data<T2>& d) restrict(cpu, amp):p_(reinterpret_cast<T *>(d.get())) {}
    __attribute__((annotate("user_deserialize")))
        explicit _data(__global T* t) restrict(cpu, amp) { p_ = t; }
    __global T* get(void) const restrict(cpu, amp) { return p_; }
    __global T* get_mutable(void) const restrict(cpu, amp) { return p_; }
    __global T* get_data() const { return get(); }
    void reset(__global T *t = NULL) restrict(cpu, amp) { p_ = t; }
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
    _data_host(const _data_host& other) : mm(other.mm) {}
    template <typename U>
        _data_host(const _data_host<U>& other) : mm(other.mm) {}

    T *get() const { return (T *)mm->get(); }
    void synchronize() const { mm->synchronize(); }
    void discard() const { mm->disc(); }
    void isArray() { mm->isArr(); }
    void refresh() const { mm->refresh(); }

    __attribute__((annotate("serialize")))
        void __cxxamp_serialize(Serialize& s) const {
            mm->serialize(s);
        }
    __attribute__((annotate("user_deserialize")))
        explicit _data_host(__global T* t);
};
#endif // __CL_MANAGE__
