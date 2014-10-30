//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <string.h>
#include <CL/opencl.h>


namespace CLAMP {
extern void CompileKernels(cl_program& program, cl_context& context, cl_device_id& device);
}

#if defined(CXXAMP_NV)
struct rw_info
{
    int count;
    bool used;
};
#endif
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
    void compile() {
        CLAMP::CompileKernels(program, context, device);
    }
    void init(void *data, int count) {
        if (count > 0) {
            cl_int err;
#if defined(CXXAMP_NV)
            cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE, count, NULL, &err);
            rwq[data] = {count, false};
#else
            cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, count, data, &err);
#endif
            assert(err == CL_SUCCESS);
            mem_info[data] = dm;
        }
    }
    void append(Serialize& s, void *data) {
        s.Append(sizeof(cl_mem), &mem_info[data]);
#if defined(CXXAMP_NV)
        rwq[data].used = true;
#endif
    }
#if defined(CXXAMP_NV)
    void write() {
        cl_int err;
        for (auto& it : rwq) {
            rw_info& rw = it.second;
            if (rw.used) {
                err = clEnqueueWriteBuffer(queue, mem_info[it.first], CL_TRUE, 0,
                                           rw.count, it.first, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
            }
        }
    }
    void read() {
        cl_int err;
        for (auto& it : rwq) {
            rw_info& rw = it.second;
            if (rw.used) {
                err = clEnqueueReadBuffer(queue, mem_info[it.first], CL_TRUE, 0,
                                          rw.count, it.first, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
                rw.used = false;
            }
        }
    }
#endif
    void free(void *data) {
        auto iter = mem_info.find(data);
        clReleaseMemObject(iter->second);
        mem_info.erase(iter);
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
#if defined(CXXAMP_NV)
    std::map<void *, rw_info> rwq;
#endif
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
        dirty(host), discard(false) { getAllocator().init(device, count); }
    mm_info(int count, void *src)
        : count(count), host(src), device(::operator new(count)),
        dirty(host), discard(false) { getAllocator().init(device, count); }
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
        if (dirty == host && device != host) {
            if (!discard)
                refresh();
            dirty = device;
        }
        discard = false;
        getAllocator().append(s, device);
    }
    ~mm_info() {
        getAllocator().free(device);
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
