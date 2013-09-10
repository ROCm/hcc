#pragma once
template<class T>
class GMACAllocator
{
 public:
  typedef T value_type;
  T* allocate(unsigned n) {
    T *p = NULL;
    ecl_error ret = eclMalloc(
	(void**)(const_cast<T**>(&p)), n * sizeof(T));
    assert(ret == eclSuccess);
    return p;
  }
};

template<class T>
class GMACDeleter {
 public:
  void operator()(T* ptr) {
    ecl_error ret = eclFree(
        const_cast<void*>(reinterpret_cast<const void*>(ptr)));
    assert(ret == eclSuccess);
  }
};

// Dummy interface that looks somewhat like std::shared_ptr<T>
template <typename T>
class _data {
 public:
  _data() = delete;
  _data(const _data& d) restrict(cpu, amp):p_(d.p_) {}
  __attribute__((annotate("user_deserialize")))
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
    s.AppendPtr((const void *)std::shared_ptr<T>::get());
  }
  __attribute__((annotate("user_deserialize")))
  explicit _data_host(__global T* t) restrict(amp);
};
