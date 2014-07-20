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

struct mm_info
{
    cl_mem dm;
    size_t count;
    bool toDel;
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
    void AMPMalloc(void **cpu_ptr, size_t count) {
        cl_int err;
        *cpu_ptr = ::operator new(count);
        cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE, count, NULL, &err);
        assert(err == CL_SUCCESS);
        al_info[cpu_ptr] = {dm, count, true};
    }
    void AMPMalloc(void **cpu_ptr, size_t count, void **data_ptr) {
        cl_int err;
        cpu_ptr = data_ptr;
        cl_mem dm = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   count, *cpu_ptr, &err);
        assert(err == CL_SUCCESS);
        al_info[cpu_ptr] = {dm, count, false};
    }
    void write() {
        cl_int err;
        for (auto& iter : al_info) {
            err = clEnqueueWriteBuffer(queue, iter.second.dm, CL_TRUE, 0, iter.second.count, *(iter.first), 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        }
    }
    template <typename T>
    void write(T *p) {
        cl_int err;
        void **ptr = (void**)(const_cast<T**>(&p));
        mm_info mm = al_info[ptr];
        err = clEnqueueWriteBuffer(queue, mm.dm, CL_TRUE, 0, mm.count, *ptr, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    void read() {
        for (auto& iter : al_info)
            clEnqueueReadBuffer(queue, iter.second.dm, CL_TRUE, 0, iter.second.count, *(iter.first), 0, NULL, NULL);
    }
    template <typename T>
    void read(T *p) {
        void **ptr = (void**)(const_cast<T**>(&p));
        mm_info mm = al_info[ptr];
        clEnqueueReadBuffer(queue, mm.dm, CL_TRUE, 0, mm.count, *ptr, 0, NULL, NULL);
    }
    template <typename T>
    cl_mem getmem(T *p) {
        void **ptr = (void**)(const_cast<T**>(&p));
        return al_info[ptr].dm;
    }
    void AMPFree() {
        for (auto& iter : al_info) {
            mm_info mm = iter.second;
            if (mm.toDel)
                ::operator delete(*(iter.first));
            clReleaseMemObject(mm.dm);
        }
        al_info.clear();
    }
    void AMPFree(void **cpu_ptr) {
        mm_info mm = al_info[cpu_ptr];
        if (mm.toDel)
            ::operator delete(*cpu_ptr);
        clReleaseMemObject(mm.dm);
        al_info.erase(cpu_ptr);
    }
    ~AMPAllocator() {
        read();
        AMPFree();
        clReleaseContext(context);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
    }
    std::map<void **, mm_info>  al_info;
    cl_context       context;
    cl_device_id     device;
    cl_kernel        kernel;
    cl_command_queue queue;
    cl_program       program;
};

AMPAllocator& getAllocator();

template <class T>
struct CLAllocator
{
    T* allocate(unsigned n) {
        T *p = nullptr;
        getAllocator().AMPMalloc((void**)(const_cast<T**>(&p)), n * sizeof(T));
        return p;
    }
    T* allocate(unsigned n, T *ptr) {
        T *p = nullptr;
        getAllocator().AMPMalloc((void**)(const_cast<T**>(&p)), n * sizeof(T), (void**)(const_cast<T**>(&ptr)));
        return p;
    }
};

template <class T>
struct CLDeleter {
    void operator()(T* ptr) {
        getAllocator().AMPFree((void**)(const_cast<T**>(&ptr)));
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
class _data_host: public std::shared_ptr<T> {
 public:
  _data_host(const _data_host &other):std::shared_ptr<T>(other) {}
  template <class = typename std::enable_if<!std::is_const<T>::value>::type>
  _data_host(const _data_host<const T> &other):std::shared_ptr<T>(other) {}
  _data_host(std::nullptr_t x = nullptr):std::shared_ptr<T>(nullptr) {}

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const {
      cl_mem mm = getAllocator().getmem(std::shared_ptr<T>::get());
      s.Append(sizeof(cl_mem), &mm);
  }
  __attribute__((annotate("user_deserialize")))
  explicit _data_host(__global T* t);
};
#endif // __CL_MANAGE__
