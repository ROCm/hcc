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

// Dummy interface that looks somewhat like std::shared_ptr<T>
template <typename T>
class _data {
 public:
  _data() = delete;
  _data(const _data& d) restrict(cpu, amp):p_(d.p_) {}
  __attribute__((annotate("deserialize")))
  explicit _data(__global T* t) restrict(cpu, amp) { p_ = t; }
  __global T* get(void) const restrict(cpu, amp) { return p_; }
  void reset(__global T *t = NULL) restrict(cpu, amp) { p_ = t; }
 private:
  __global T* p_;
};

template <typename T>
class _data_host: public std::shared_ptr<T> {
 public:
  _data_host(const _data_host &other):std::shared_ptr<T>(other) {}

  _data_host(std::nullptr_t x = nullptr):std::shared_ptr<T>(nullptr) {}

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const {
    cl_context context = s.getContext();
    cl_mem t = clGetBuffer(context, (const void *)std::shared_ptr<T>::get());
    s.Append(sizeof(cl_mem), &t);
  }
};
