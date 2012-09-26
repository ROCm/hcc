#pragma once
template<class T>
class GMACAllocator
{
 public:
  typedef T value_type;
  T* allocate(unsigned n) {
    T *p;
    cl_int ret = clMalloc(accelerator().get_default_view().clamp_get_command_queue(),
        (void**)&p, n * sizeof(T));
    assert(ret == CL_SUCCESS);
    return p;
  }
};

template<class T>
class GMACDeleter {
 public:
  void operator()(T* ptr) {
    cl_int ret = clFree(accelerator().get_default_view().clamp_get_command_queue(),
        reinterpret_cast<void*>(ptr));
    assert(ret == CL_SUCCESS);
  }
};

