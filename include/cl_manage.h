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
struct rw_info
{
    void *data;
    int count;
    cl_mem dm;
};
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
    void setup(Serialize& s, void *data, int count) {
        auto iter = mem_info.find(data);
        cl_mem dm = nullptr;
        if (iter == std::end(mem_info)) {
            cl_int err;
            dm = clCreateBuffer(context, CL_MEM_READ_WRITE, count, NULL, &err);
            assert(err == CL_SUCCESS);
            mem_info[data] = dm;
            rw_que.push_back({data, count, dm});
        } else
            dm = iter->second;
        s.Append(sizeof(cl_mem), &dm);
    }
    void unregister(void *data) {
        auto iter = mem_info.find(data);
        if (iter != std::end(mem_info)) {
            clReleaseMemObject(iter->second);
            mem_info.erase(iter);
        }
    }
    void write() {
        cl_int err;
        for (auto& it : rw_que) {
            err = clEnqueueWriteBuffer(queue, it.dm, CL_TRUE, 0,
                                       it.count, it.data, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        }
    }
    void read() {
        cl_int err;
        for (auto& it : rw_que) {
            err = clEnqueueWriteBuffer(queue, it.dm, CL_TRUE, 0,
                                       it.count, it.data, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        }
        rw_que.clear();
    }
    ~AMPAllocator() {
        for (auto& iter : mem_info)
            clReleaseMemObject(iter.second);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }
    std::map<void *, cl_mem> mem_info;
    std::vector<rw_info> rw_que;
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
    void *host;
    void *device;
    void *dirty;
    bool discard;
    mm_info(int count)
        : count(count), host(::operator new(count)), device(host),
        dirty(host), discard(false) {}
    mm_info(int count, void *src)
        : count(count), host(src), device(::operator new(count)),
        dirty(host), discard(false) { refresh(); }
    void synchronize() {
        if (dirty != host) {
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
        discard = false;
        if (dirty == host && device != host) {
            refresh();
            dirty = device;
        }
        getAllocator().setup(s, device, count);
    }
    ~mm_info() {
        getAllocator().unregister(device);
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
        explicit _data_host(__global T* t);
};
#endif // __CL_MANAGE__
