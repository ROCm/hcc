// CAVEATS: There could be walkarounds for quick evaluation purposes. Here we
// list such features that are used in the code with description.
//
// ACCELERATOR
//  According to specification, each array should have its binding accelerator
//  instance. For now, we haven't implemented such binding nor actual
//  implementation of accelerator. For a quick and dirty walkaround for
//  OpenCL based prototype, we allow key OpenCL objects visible globally so
//  that we don't have to be bothered with such implementation effort.

#pragma once

#include <cassert>
#include <exception>
#include <string>
#include <vector>
#include <chrono>
#include <future>
#include <string.h> //memcpy
#ifndef CXXAMP_ENABLE_HSA_OKRA
#include <gmac/opencl.h>
#endif
#include <memory>
#include <algorithm>
#include <set>
#include <type_traits>
// CLAMP
#include <serialize.h>
// End CLAMP

/* COMPATIBILITY LAYER */
#define STD__FUTURE_STATUS__FUTURE_STATUS std::future_status

#ifndef WIN32
#define __declspec(ignored) /* */
#endif

namespace Concurrency {
typedef int HRESULT;
class runtime_exception : public std::exception
{
public:
  runtime_exception(const char * message, HRESULT hresult) throw() : _M_msg(message), err_code(hresult) {}
  explicit runtime_exception(HRESULT hresult) throw() : err_code(hresult) {}
  runtime_exception(const runtime_exception& other) throw() : _M_msg(other.what()), err_code(other.err_code) {}
  runtime_exception& operator=(const runtime_exception& other) throw() {
    _M_msg = *(other.what());
    err_code = other.err_code;
    return *this;
  }
  virtual ~runtime_exception() throw() {}
  virtual const char* what() const throw() {return _M_msg.c_str();}
  HRESULT get_error_code() const {return err_code;}

private:
  std::string _M_msg;
  HRESULT err_code;
};

#ifndef E_FAIL
#define E_FAIL 0x80004005
#endif

static const char *__errorMsg_UnsupportedAccelerator = "concurrency::parallel_for_each is not supported on the selected accelerator \"CPU accelerator\".";

class invalid_compute_domain : public runtime_exception
{
public:
  explicit invalid_compute_domain (const char * message) throw()
  : runtime_exception(message, E_FAIL) {}
  invalid_compute_domain() throw()
  : runtime_exception(E_FAIL) {}
};

/*
  This is not part of C++AMP standard, but borrowed from Parallel Patterns
  Library.
*/
  template <typename _Type> class task;
  template <> class task<void>;

enum queuing_mode {
  queuing_mode_immediate,
  queuing_mode_automatic
};

enum access_type
{
  access_type_none,
  access_type_read,
  access_type_write,
  access_type_read_write = access_type_read | access_type_write,
  access_type_auto
};

class completion_future;
class accelerator;
template <typename T, int N> class array_view;
template <typename T, int N> class array;

class accelerator_view {
public:
  accelerator_view() = delete;
  accelerator_view(const accelerator_view& other) { *this = other; }
  accelerator_view& operator=(const accelerator_view& other) {
    is_debug = other.is_debug;
    is_auto_selection = other.is_auto_selection;
    version = other.version;
    queuing_mode = other.queuing_mode;
    _accelerator = other._accelerator;
    return *this;
  }

  accelerator& get_accelerator() const { return *_accelerator; }
  enum queuing_mode get_queuing_mode() const { return queuing_mode; }
  bool get_is_debug() const { return is_debug; }
  bool get_version() const { return version; }
  bool get_is_auto_selection() const { return is_auto_selection; }
  void flush() {}
  void wait() {}
  completion_future create_marker();
  bool operator==(const accelerator_view& other) const;
  bool operator!=(const accelerator_view& other) const { return !(*this == other); }
  ~accelerator_view() {}
 private:
  bool is_debug;
  bool is_auto_selection;
  unsigned int version;
  enum queuing_mode queuing_mode;
  //CLAMP-specific
  friend class accelerator;
  template <typename T, int>
  friend class array;
  explicit accelerator_view(accelerator* accel) :
    is_debug(false), is_auto_selection(false), version(0), queuing_mode(queuing_mode_automatic), _accelerator(accel) {}
  //End CLAMP-specific
  accelerator* _accelerator;
};

class accelerator {
public:
  static const wchar_t default_accelerator[];   // = L"default"
  static const wchar_t gpu_accelerator[];       // = L"gpu"
  static const wchar_t cpu_accelerator[];       // = L"cpu"

  accelerator();
  explicit accelerator(const std::wstring& path);
  accelerator(const accelerator& other);
  static std::vector<accelerator> get_all() {
    std::vector<accelerator> acc;
#ifndef CXXAMP_ENABLE_HSA_OKRA
    AcceleratorInfo accInfo;
    for (unsigned i = 0; i < eclGetNumberOfAccelerators(); i++) {
      assert(eclGetAcceleratorInfo(i, &accInfo) == eclSuccess);
      if (accInfo.acceleratorType == GMAC_ACCELERATOR_TYPE_GPU)
        acc.push_back(*_gpu_accelerator);

      if (accInfo.acceleratorType == GMAC_ACCELERATOR_TYPE_CPU)
        acc.push_back(*_cpu_accelerator);
    }
#else
    acc.push_back(*_cpu_accelerator);  // in HSA path, always add CPU accelerator
    acc.push_back(*_gpu_accelerator);  // in HSA path, always add GPU accelerator
#endif
    return acc;
  }
  static bool set_default(const std::wstring& path) {
    if (_default_accelerator != nullptr) {
      return false;
    }
    if (path == std::wstring(cpu_accelerator)) {
      _default_accelerator = _cpu_accelerator;
      return true;
    } else if (path == std::wstring(gpu_accelerator)) {
      _default_accelerator = _gpu_accelerator;
      return true;
    }
    return false;
  }
  accelerator& operator=(const accelerator& other);

  const std::wstring &get_device_path() const { return device_path; }

  const std::wstring &get_description() const { return description; }
  bool get_supports_cpu_shared_memory() const {return supports_cpu_shared_memory; }
  bool get_is_debug() const { return is_debug; }
  bool get_version() const { return version; }
  accelerator_view& get_default_view() const;
  bool get_has_display() const { return has_display; }
  accelerator_view create_view();
  accelerator_view create_view(queuing_mode qmode);
  bool get_is_emulated() const { return is_emulated; }
  bool get_supports_double_precision() const { return supports_double_precision; }
  bool get_supports_limited_double_precision() const { return supports_limited_double_precision; }
  size_t get_dedicated_memory() const { return dedicated_memory; }
  bool set_default_cpu_access_type(access_type type);
  access_type get_default_cpu_access_type() const;
  bool operator==(const accelerator& other) const;
  bool operator!=(const accelerator& other) const;
 private:
  std::wstring device_path;
  unsigned int version; // hiword=major, loword=minor
  std::wstring description;
  bool is_debug;
  bool is_emulated;
  bool has_display;
  bool supports_double_precision;
  bool supports_limited_double_precision;
  bool supports_cpu_shared_memory;
  size_t dedicated_memory;
  access_type default_access_type;
  std::shared_ptr<accelerator_view> default_view;
#ifndef CXXAMP_ENABLE_HSA_OKRA
  typedef GmacAcceleratorInfo AcceleratorInfo;
  AcceleratorInfo accInfo;
#endif

  // static class members
  static std::shared_ptr<accelerator> _default_accelerator; // initialized as nullptr
  static std::shared_ptr<accelerator> _gpu_accelerator;
  static std::shared_ptr<accelerator> _cpu_accelerator;
};

//CLAMP
extern "C" __attribute__((pure)) int get_global_id(int n) restrict(amp);
extern "C" __attribute__((pure)) int get_local_id(int n) restrict(amp);
extern "C" __attribute__((pure)) int get_group_id(int n) restrict(amp);
#ifdef __APPLE__
#define tile_static static __attribute__((section("clamp,opencl_local")))
#else
#define tile_static static __attribute__((section("clamp_opencl_local")))
#endif
extern "C" void barrier(int n) restrict(amp);
//End CLAMP
class completion_future {
public:

    completion_future() {};

    completion_future(const completion_future& _Other)
        : __amp_future(_Other.__amp_future) {}

    completion_future(completion_future&& _Other)
        : __amp_future(std::move(_Other.__amp_future)) {}

    ~completion_future() {}

    completion_future& operator=(const completion_future& _Other) {
        if (this != &_Other)
           __amp_future = _Other.__amp_future;
        return (*this);
    }

    completion_future& operator=(completion_future&& _Other) {
        if (this != &_Other)
            __amp_future = std::move(_Other.__amp_future);
        return (*this);
    }

    void get() const {
        __amp_future.get();
    }

    bool valid() const {
        return __amp_future.valid();
    }
    void wait() const {
        if(this->valid())
          __amp_future.wait();
    }

    template <class _Rep, class _Period>
    std::future_status wait_for(const std::chrono::duration<_Rep, _Period>& _Rel_time) const {
        return __amp_future.wait_for(_Rel_time);
    }

    template <class _Clock, class _Duration>
    std::future_status wait_until(const std::chrono::time_point<_Clock, _Duration>& _Abs_time) const {
        return __amp_future.wait_until(_Abs_time);
    }

    operator std::shared_future<void>() const {
        return __amp_future;
    }

    template<typename functor>
    void then(const functor & func) const {
      this->wait();
      if(this->valid())
        func();
    }

private:
    std::shared_future<void> __amp_future;

    completion_future(const std::shared_future<void> &__future)
        : __amp_future(__future) {}

    template <typename InputType, typename OutputType>
        friend completion_future __amp_copy_async_impl(InputType& src, OutputType& dst);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, array<T, N>& dest);
    template <typename InputIter, typename T, int N>
        friend completion_future copy_async(InputIter srcBegin, const array_view<T, N>& dest);
    template <typename OutputIter, typename T, int N>
        friend completion_future copy_async(const array<T, N>& src, OutputIter destBegin);
    template <typename OutputIter, typename T, int N>
        friend completion_future copy_async(const array_view<T, N>& src, OutputIter destBegin);
    template <typename T, int N> friend class array_view;
};

template <int N> class extent;

template <int...> struct __indices {};

template <int _Sp, class _IntTuple, int _Ep>
    struct __make_indices_imp;

template <int _Sp, int ..._Indices, int _Ep>
    struct __make_indices_imp<_Sp, __indices<_Indices...>, _Ep>
    {
        typedef typename __make_indices_imp<_Sp+1, __indices<_Indices..., _Sp>, _Ep>::type type;
    };

template <int _Ep, int ..._Indices>
    struct __make_indices_imp<_Ep, __indices<_Indices...>, _Ep>
    {
        typedef __indices<_Indices...> type;
    };

template <int _Ep, int _Sp = 0>
    struct __make_indices
    {
        static_assert(_Sp <= _Ep, "__make_indices input error");
        typedef typename __make_indices_imp<_Sp, __indices<>, _Ep>::type type;
    };

template <int _Ip>
    class __index_leaf {
        int __idx;
        int dummy;
    public:
        explicit __index_leaf(int __t) restrict(amp,cpu) : __idx(__t) {}

        __index_leaf& operator=(const int __t) restrict(amp,cpu) {
            __idx = __t;
            return *this;
        }
        __index_leaf& operator+=(const int __t) restrict(amp,cpu) {
            __idx += __t;
            return *this;
        }
        __index_leaf& operator-=(const int __t) restrict(amp,cpu) {
            __idx -= __t;
            return *this;
        }
        __index_leaf& operator*=(const int __t) restrict(amp,cpu) {
            __idx *= __t;
            return *this;
        }
        __index_leaf& operator/=(const int __t) restrict(amp,cpu) {
            __idx /= __t;
            return *this;
        }
        __index_leaf& operator%=(const int __t) restrict(amp,cpu) {
            __idx %= __t;
            return *this;
        }
              int& get()       restrict(amp,cpu) { return __idx; }
        const int& get() const restrict(amp,cpu) { return __idx; }
    };


template <class _Indx> struct index_impl;
template <int ...N>
    struct index_impl<__indices<N...> >
    : public __index_leaf<N>...
    {
        index_impl() restrict(amp,cpu) : __index_leaf<N>(0)... {}

        template<class ..._Up>
            explicit index_impl(_Up... __u) restrict(amp,cpu)
            : __index_leaf<N>(__u)... {}

        index_impl(const index_impl& other) restrict(amp,cpu)
            : index_impl(static_cast<const __index_leaf<N>&>(other).get()...) {}

        index_impl(int components[]) restrict(amp,cpu)
            : __index_leaf<N>(components[N])... {}
        index_impl(const int components[]) restrict(amp,cpu)
            : __index_leaf<N>(components[N])... {}

        template<class ..._Tp>
            inline void __swallow(_Tp...) restrict(amp,cpu) {}

        int operator[] (unsigned int c) const restrict(amp,cpu) {
            return static_cast<const __index_leaf<0>&>(*((__index_leaf<0> *)this + c)).get();
        }
        int& operator[] (unsigned int c) restrict(amp,cpu) {
            return static_cast<__index_leaf<0>&>(*((__index_leaf<0> *)this + c)).get();
        }
        index_impl& operator=(const index_impl& __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator=(static_cast<const __index_leaf<N>&>(__t).get())...);
            return *this;
        }
        index_impl& operator+=(const index_impl& __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator+=(static_cast<const __index_leaf<N>&>(__t).get())...);
            return *this;
        }
        index_impl& operator-=(const index_impl& __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator-=(static_cast<const __index_leaf<N>&>(__t).get())...);
            return *this;
        }
        index_impl& operator*=(const index_impl& __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator*=(static_cast<const __index_leaf<N>&>(__t).get())...);
            return *this;
        }
        index_impl& operator/=(const index_impl& __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator/=(static_cast<const __index_leaf<N>&>(__t).get())...);
            return *this;
        }
        index_impl& operator%=(const index_impl& __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator%=(static_cast<const __index_leaf<N>&>(__t).get())...);
            return *this;
        }
        index_impl& operator+=(const int __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator+=(__t)...);
            return *this;
        }
        index_impl& operator-=(const int __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator-=(__t)...);
            return *this;
        }
        index_impl& operator*=(const int __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator*=(__t)...);
            return *this;
        }
        index_impl& operator/=(const int __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator/=(__t)...);
            return *this;
        }
        index_impl& operator%=(const int __t) restrict(amp,cpu) {
            __swallow(__index_leaf<N>::operator%=(__t)...);
            return *this;
        }
    };

template<int N> class index;
template<int N> class extent;

template <int N, typename _Tp>
struct index_helper
{
    static inline void set(_Tp& now) restrict(amp,cpu) {
        now[N - 1] = get_global_id(_Tp::rank - N);
        index_helper<N - 1, _Tp>::set(now);
    }
    static inline bool equal(const _Tp& _lhs, const _Tp& _rhs) restrict(amp,cpu) {
        return (_lhs[N - 1] == _rhs[N - 1]) &&
            (index_helper<N - 1, _Tp>::equal(_lhs, _rhs));
    }
    static inline int count_size(const _Tp& now) restrict(amp,cpu) {
        return now[N - 1] * index_helper<N - 1, _Tp>::count_size(now);
    }
};
template<typename _Tp>
struct index_helper<1, _Tp>
{
    static inline void set(_Tp& now) restrict(amp,cpu) {
        now[0] = get_global_id(_Tp::rank - 1);
    }
    static inline bool equal(const _Tp& _lhs, const _Tp& _rhs) restrict(amp,cpu) {
        return (_lhs[0] == _rhs[0]);
    }
    static inline int count_size(const _Tp& now) restrict(amp,cpu) {
        return now[0];
    }
};

template <int N, typename _Tp1, typename _Tp2>
struct amp_helper
{
    static bool inline contains(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[N - 1] >= 0 && idx[N - 1] < ext[N - 1] &&
            amp_helper<N - 1, _Tp1, _Tp2>::contains(idx, ext);
    }

    static bool inline contains(const _Tp1& idx, const _Tp2& ext,const _Tp2& ext2) restrict(amp,cpu) {
        return idx[N - 1] >= 0 && ext[N - 1] > 0 && (idx[N - 1] + ext[N - 1]) <= ext2[N - 1] &&
            amp_helper<N - 1, _Tp1, _Tp2>::contains(idx, ext,ext2);
    }

    static int inline flatten(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[N - 1] + ext[N - 1] * amp_helper<N - 1, _Tp1, _Tp2>::flatten(idx, ext);
    }
    static void inline minus(const _Tp1& idx, _Tp2& ext) restrict(amp,cpu) {
        ext.base_ -= idx.base_;
    }
};
template <typename _Tp1, typename _Tp2>
struct amp_helper<1, _Tp1, _Tp2>
{
    static bool inline contains(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[0] >= 0 && idx[0] < ext[0];
    }

    static bool inline contains(const _Tp1& idx, const _Tp2& ext,const _Tp2& ext2) restrict(amp,cpu) {
        return idx[0] >= 0 && ext[0] > 0 && (idx[0] + ext[0]) <= ext2[0] ;
    }

    static int inline flatten(const _Tp1& idx, const _Tp2& ext) restrict(amp,cpu) {
        return idx[0];
    }
    static void inline minus(const _Tp1& idx, _Tp2& ext) restrict(amp,cpu) {
        ext.base_ -= idx.base_;
    }
};

template <int N>
class index {
public:
    static const int rank = N;
    typedef int value_type;

    index() restrict(amp,cpu) : base_() {
        static_assert( N>0, "rank should bigger than 0 ");
    };
    index(const index& other) restrict(amp,cpu)
        : base_(other.base_) {}
    template <typename ..._Tp>
        explicit index(_Tp ... __t) restrict(amp,cpu)
        : base_(__t...) {
            static_assert(sizeof...(_Tp) <= 3, "Explicit constructor with rank greater than 3 is not allowed");
            static_assert(sizeof...(_Tp) == N, "rank should be consistency");
        }
    explicit index(int components[]) restrict(amp,cpu)
        : base_(components) {}
    explicit index(const int components[]) restrict(amp,cpu)
        : base_(components) {}

    index& operator=(const index& __t) restrict(amp,cpu) {
        base_.operator=(__t.base_);
        return *this;
    }

    int operator[] (unsigned int c) const restrict(amp,cpu) {
        return base_[c];
    }
    int& operator[] (unsigned int c) restrict(amp,cpu) {
        return base_[c];
    }

    bool operator== (const index& other) const restrict(amp,cpu) {
        return index_helper<N, index<N> >::equal(*this, other);
    }
    bool operator!= (const index& other) const restrict(amp,cpu) {
        return !(*this == other);
    }

    index& operator+=(const index& __r) restrict(amp,cpu) {
        base_.operator+=(__r.base_);
        return *this;
    }
    index& operator-=(const index& __r) restrict(amp,cpu) {
        base_.operator-=(__r.base_);
        return *this;
    }
    index& operator*=(const index& __r) restrict(amp,cpu) {
        base_.operator*=(__r.base_);
        return *this;
    }
    index& operator/=(const index& __r) restrict(amp,cpu) {
        base_.operator/=(__r.base_);
        return *this;
    }
    index& operator%=(const index& __r) restrict(amp,cpu) {
        base_.operator%=(__r.base_);
        return *this;
    }
    index& operator+=(int __r) restrict(amp,cpu) {
        base_.operator+=(__r);
        return *this;
    }
    index& operator-=(int __r) restrict(amp,cpu) {
        base_.operator-=(__r);
        return *this;
    }
    index& operator*=(int __r) restrict(amp,cpu) {
        base_.operator*=(__r);
        return *this;
    }
    index& operator/=(int __r) restrict(amp,cpu) {
        base_.operator/=(__r);
        return *this;
    }
    index& operator%=(int __r) restrict(amp,cpu) {
        base_.operator%=(__r);
        return *this;
    }

    index& operator++() restrict(amp,cpu) {
        base_.operator+=(1);
        return *this;
    }
    index operator++(int) restrict(amp,cpu) {
        index ret = *this;
        base_.operator+=(1);
        return ret;
    }
    index& operator--() restrict(amp,cpu) {
        base_.operator-=(1);
        return *this;
    }
    index operator--(int) restrict(amp,cpu) {
        index ret = *this;
        base_.operator-=(1);
        return ret;
    }

    template<int T>
    friend class extent;
private:
    typedef index_impl<typename __make_indices<N>::type> base;
    base base_;
    template <int K, typename Q> friend struct index_helper;
    template <int K, typename Q1, typename Q2> friend struct amp_helper;

    template<int K, class Y>
        friend void parallel_for_each(extent<K>, const Y&);
    __attribute__((annotate("__cxxamp_opencl_index")))
        void __cxxamp_opencl_index() restrict(amp,cpu)
#ifdef __GPU__
        {
            index_helper<N, index<N>>::set(*this);
        }
#else
    ;
#endif
};


#ifndef CLK_LOCAL_MEM_FENCE
#define CLK_LOCAL_MEM_FENCE (1)
#endif

#ifndef CLK_GLOBAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE (2)
#endif

// C++AMP LPM 4.5
class tile_barrier {
 public:
  tile_barrier(const tile_barrier& other) restrict(amp,cpu) {}
  void wait() const restrict(amp) {
#ifdef __GPU__
    wait_with_all_memory_fence();
#endif
  }
  void wait_with_all_memory_fence() const restrict(amp) {
#ifdef __GPU__
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#endif
  }
  void wait_with_global_memory_fence() const restrict(amp) {
#ifdef __GPU__
    barrier(CLK_GLOBAL_MEM_FENCE);
#endif
  }
  void wait_with_tile_static_memory_fence() const restrict(amp) {
#ifdef __GPU__
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
  }
 private:
  tile_barrier() restrict(amp) {}
  template<int D0, int D1, int D2>
  friend class tiled_index;
};

template <typename T, int N> class array;
template <typename T, int N> class array_view;

// forward decls
template <int D0, int D1=0, int D2=0> class tiled_extent;

template <int N>
class extent {
public:
    static const int rank = N;
    typedef int value_type;

    extent() restrict(amp,cpu) : base_() {
      static_assert(N > 0, "Dimensionality must be positive");
    };
    extent(const extent& other) restrict(amp,cpu)
        : base_(other.base_) {}
    template <typename ..._Tp>
        explicit extent(_Tp ... __t) restrict(amp,cpu)
        : base_(__t...) {
      static_assert(sizeof...(__t) <= 3, "Can only supply at most 3 individual coordinates in the constructor");
      static_assert(sizeof...(__t) == N, "rank should be consistency");
    }
    explicit extent(int components[]) restrict(amp,cpu)
        : base_(components) {}
    explicit extent(const int components[]) restrict(amp,cpu)
        : base_(components) {}
    template <int D0, int D1, int D2>
        explicit extent(const tiled_extent<D0, D1, D2>& other) restrict(amp,cpu)
            : base_(other.base_) {}

    extent& operator=(const extent& other) restrict(amp,cpu) {
        base_.operator=(other.base_);
        return *this;
    }

    int operator[] (unsigned int c) const restrict(amp,cpu) {
        return base_[c];
    }
    int& operator[] (unsigned int c) restrict(amp,cpu) {
        return base_[c];
    }

    bool operator==(const extent& other) const restrict(amp,cpu) {
        return index_helper<N, extent<N> >::equal(*this, other);
    }
    bool operator!=(const extent& other) const restrict(amp,cpu) {
        return !(*this == other);
    }

    unsigned int size() const restrict(amp,cpu) {
        return index_helper<N, extent<N>>::count_size(*this);
    }
    bool contains(const index<N>& idx) const restrict(amp,cpu) {
        return amp_helper<N, index<N>, extent<N>>::contains(idx, *this);
    }
    template <int D0>
        typename std::enable_if<N == 1, tiled_extent<D0> >::type tile() const {
            static_assert(D0 > 0, "Tile size must be positive");
            return tiled_extent<D0>(*this);
        }
    template <int D0, int D1>
        typename std::enable_if<N == 2, tiled_extent<D0, D1> >::type tile() const {
            static_assert(D0 > 0, "Tile size must be positive");
            static_assert(D1 > 0, "Tile size must be positive");
            return tiled_extent<D0, D1>(*this);
        }
    template <int D0, int D1, int D2>
        typename std::enable_if<N == 3, tiled_extent<D0, D1, D2> >::type tile() const {
            static_assert(D0 > 0, "Tile size must be positive");
            static_assert(D1 > 0, "Tile size must be positive");
            static_assert(D2 > 0, "Tile size must be positive");
            return tiled_extent<D0, D1, D2>(*this);
        }

    extent operator+(const index<N>& idx) restrict(amp,cpu) {
        extent __r = *this;
        __r += idx;
        return __r;
    }
    extent operator-(const index<N>& idx) restrict(amp,cpu) {
        extent __r = *this;
        __r -= idx;
        return __r;
    }
    extent& operator+=(const index<N>& idx) restrict(amp,cpu) {
        base_.operator+=(idx.base_);
        return *this;
    }
    extent& operator-=(const index<N>& idx) restrict(amp,cpu) {
        base_.operator-=(idx.base_);
        return *this;
    }
    extent& operator+=(const extent& __r) restrict(amp,cpu) {
        base_.operator+=(__r.base_);
        return *this;
    }
    extent& operator-=(const extent& __r) restrict(amp,cpu) {
        base_.operator-=(__r.base_);
        return *this;
    }
    extent& operator*=(const extent& __r) restrict(amp,cpu) {
        base_.operator*=(__r.base_);
        return *this;
    }
    extent& operator/=(const extent& __r) restrict(amp,cpu) {
        base_.operator/=(__r.base_);
        return *this;
    }
    extent& operator%=(const extent& __r) restrict(amp,cpu) {
        base_.operator%=(__r.base_);
        return *this;
    }
    extent& operator+=(int __r) restrict(amp,cpu) {
        base_.operator+=(__r);
        return *this;
    }
    extent& operator-=(int __r) restrict(amp,cpu) {
        base_.operator-=(__r);
        return *this;
    }
    extent& operator*=(int __r) restrict(amp,cpu) {
        base_.operator*=(__r);
        return *this;
    }
    extent& operator/=(int __r) restrict(amp,cpu) {
        base_.operator/=(__r);
        return *this;
    }
    extent& operator%=(int __r) restrict(amp,cpu) {
        base_.operator%=(__r);
        return *this;
    }
    extent& operator++() restrict(amp,cpu) {
        base_.operator+=(1);
        return *this;
    }
    extent operator++(int) restrict(amp,cpu) {
        extent ret = *this;
        base_.operator+=(1);
        return ret;
    }
    extent& operator--() restrict(amp,cpu) {
        base_.operator-=(1);
        return *this;
    }
    extent operator--(int) restrict(amp,cpu) {
        extent ret = *this;
        base_.operator-=(1);
        return ret;
    }
private:
    typedef index_impl<typename __make_indices<N>::type> base;
    base base_;
    template <int K, typename Q> friend struct index_helper;
    template <int K, typename Q1, typename Q2> friend struct amp_helper;
};


// C++AMP LPM 4.4.1

template <int D0, int D1=0, int D2=0>
class tiled_index {
 public:
  static const int rank = 3;
  const index<3> global;
  const index<3> local;
  const index<3> tile;
  const index<3> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<3>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0, D1, D2>& o) restrict(amp, cpu):
    global(o.global), local(o.local), tile(o.tile), tile_origin(o.tile_origin), barrier(o.barrier) {}
  operator const index<3>() const restrict(amp,cpu) {
    return global;
  }
  const Concurrency::extent<3> tile_extent;
  Concurrency::extent<3> get_tile_extent() const restrict(amp, cpu) {
    return tile_extent;
  }
  static const int tile_dim0 = D0;
  static const int tile_dim1 = D1;
  static const int tile_dim2 = D2;
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  __attribute__((always_inline)) tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<3>(get_global_id(2), get_global_id(1), get_global_id(0))),
    local(index<3>(get_local_id(2), get_local_id(1), get_local_id(0))),
    tile(index<3>(get_group_id(2), get_group_id(1), get_group_id(0))),
    tile_origin(index<3>(get_global_id(2)-get_local_id(2),
                         get_global_id(1)-get_local_id(1),
                         get_global_id(0)-get_local_id(0))),
    tile_extent(D0, D1, D2)
#endif // __GPU__
  {}
  template<int D0_, int D1_, int D2_, typename K>
  friend void parallel_for_each(tiled_extent<D0_, D1_, D2_>, const K&);
};
template <int N> class extent;
template <int D0>
class tiled_index<D0, 0, 0> {
 public:
  const index<1> global;
  const index<1> local;
  const index<1> tile;
  const index<1> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<1>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0>& o) restrict(amp, cpu):
    global(o.global), local(o.local), tile(o.tile), tile_origin(o.tile_origin), barrier(o.barrier) {}
  operator const index<1>() const restrict(amp,cpu) {
    return global;
  }
  const Concurrency::extent<1> tile_extent;
  Concurrency::extent<1> get_tile_extent() const restrict(amp, cpu) {
    return tile_extent;
  }
  static const int tile_dim0 = D0;
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  __attribute__((always_inline)) tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<1>(get_global_id(0))),
    local(index<1>(get_local_id(0))),
    tile(index<1>(get_group_id(0))),
    tile_origin(index<1>(get_global_id(0)-get_local_id(0))),
    tile_extent(D0)
#endif // __GPU__
  {}
  template<int D, typename K>
  friend void parallel_for_each(tiled_extent<D>, const K&);
};

template <int D0, int D1>
class tiled_index<D0, D1, 0> {
 public:
  const index<2> global;
  const index<2> local;
  const index<2> tile;
  const index<2> tile_origin;
  const tile_barrier barrier;
  tiled_index(const index<2>& g) restrict(amp, cpu):global(g){}
  tiled_index(const tiled_index<D0, D1>& o) restrict(amp, cpu):
    global(o.global), local(o.local), tile(o.tile), tile_origin(o.tile_origin), barrier(o.barrier) {}
  operator const index<2>() const restrict(amp,cpu) {
    return global;
  }
  const Concurrency::extent<2> tile_extent;
  Concurrency::extent<2> get_tile_extent() const restrict(amp, cpu) {
    return tile_extent;
  }
  static const int tile_dim0 = D0;
  static const int tile_dim1 = D1;
 private:
  //CLAMP
  __attribute__((annotate("__cxxamp_opencl_index")))
  __attribute__((always_inline)) tiled_index() restrict(amp)
#ifdef __GPU__
  : global(index<2>(get_global_id(1), get_global_id(0))),
    local(index<2>(get_local_id(1), get_local_id(0))),
    tile(index<2>(get_group_id(1), get_group_id(0))),
    tile_origin(index<2>(get_global_id(1)-get_local_id(1),
                         get_global_id(0)-get_local_id(0))),
    tile_extent(D0, D1)
#endif // __GPU__
  {}
  template<int D0_, int D1_, typename K>
  friend void parallel_for_each(tiled_extent<D0_, D1_>, const K&);
};



template <int D0, int D1/*=0*/, int D2/*=0*/>
class tiled_extent : public extent<3>
{
public:
  static const int rank = 3;
  tiled_extent() restrict(amp,cpu) {
    static_assert(D0 > 0, "Tile size must be positive");
    static_assert(D1 > 0, "Tile size must be positive");
    static_assert(D2 > 0, "Tile size must be positive");
  }
  tiled_extent(const tiled_extent& other) restrict(amp,cpu): extent(other[0], other[1], other[2]) {}
  tiled_extent(const extent<3>& ext) restrict(amp,cpu): extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu) {
    tiled_extent padded(*this);
    padded[0] = (padded[0] <= D0) ? D0 : (((padded[0] + D0 - 1) / D0) * D0);
    padded[1] = (padded[1] <= D1) ? D1 : (((padded[1] + D1 - 1) / D1) * D1);
    padded[2] = (padded[2] <= D2) ? D2 : (((padded[2] + D2 - 1) / D2) * D2);
    return padded;
  }
  tiled_extent truncate() const restrict(amp,cpu) {
    tiled_extent trunc(*this);
    trunc[0] = (trunc[0]/D0) * D0;
    trunc[1] = (trunc[1]/D1) * D1;
    trunc[2] = (trunc[2]/D2) * D2;
    return trunc;
  }

  // __declspec(property(get)) extent<3> tile_extent;
  extent<3> get_tile_extent() const;
  static const int tile_dim0 = D0;
  static const int tile_dim1 = D1;
  static const int tile_dim2 = D2;
  friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
  friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
};

template <int D0, int D1>
class tiled_extent<D0,D1,0> : public extent<2>
{
public:
  static const int rank = 2;
  tiled_extent() restrict(amp,cpu) {
    static_assert(D0 > 0, "Tile size must be positive");
    static_assert(D1 > 0, "Tile size must be positive");
  }
  tiled_extent(const tiled_extent& other) restrict(amp,cpu):extent(other[0], other[1]) {}
  tiled_extent(const extent<2>& ext) restrict(amp,cpu):extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu) {
    tiled_extent padded(*this);
    padded[0] = (padded[0] <= D0) ? D0 : (((padded[0] + D0 - 1) / D0) * D0);
    padded[1] = (padded[1] <= D1) ? D1 : (((padded[1] + D1 - 1) / D1) * D1);
    return padded;
  }
  tiled_extent truncate() const restrict(amp,cpu) {
    tiled_extent trunc(*this);
    trunc[0] = (trunc[0]/D0) * D0;
    trunc[1] = (trunc[1]/D1) * D1;
    return trunc;
  }
  // __declspec(property(get)) extent<2> tile_extent;
  extent<2> get_tile_extent() const;
  static const int tile_dim0 = D0;
  static const int tile_dim1 = D1;
  friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
  friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
};


template <int D0>
class tiled_extent<D0,0,0> : public extent<1>
{
public:
  static const int rank = 1;
  tiled_extent() restrict(amp,cpu) {
    static_assert(D0 > 0, "Tile size must be positive");
  }
  tiled_extent(const tiled_extent& other) restrict(amp,cpu):
    extent(other[0]) {}
  tiled_extent(const extent<1>& ext) restrict(amp,cpu):extent(ext) {}
  tiled_extent& operator=(const tiled_extent& other) restrict(amp,cpu);
  tiled_extent pad() const restrict(amp,cpu) {
    tiled_extent padded(*this);
    padded[0] = (padded[0] <= D0) ? D0 : (((padded[0] + D0 - 1) / D0) * D0);
    return padded;
  }
  tiled_extent truncate() const restrict(amp,cpu) {
    tiled_extent trunc(*this);
    trunc[0] = (trunc[0]/D0) * D0;
    return trunc;
  }
  // __declspec(property(get)) extent<1> tile_extent;
  extent<1> get_tile_extent() const;
  static const int tile_dim0 = D0;
  friend bool operator==(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
  friend bool operator!=(const tiled_extent& lhs, const tiled_extent& rhs) restrict(amp,cpu);
};


#define __global
#ifdef CXXAMP_ENABLE_HSA_OKRA
//include okra-specific files here
} //namespace Concurrency
#include "okra_manage.h"
namespace Concurrency {
#else
#include "gmac_manage.h"
#endif
template <typename T, int N>
struct projection_helper
{
    typedef array_view<T, N - 1> result_type;
    typedef array_view<const T, N - 1> const_result_type;
    static result_type project(array<T, N>& now, int stride) restrict(amp,cpu) {
#ifndef __GPU__
        if( stride < 0)
          throw runtime_exception("errorMsg_throw", 0);
#endif
        int comp[N - 1], i;
        for (i = N - 1; i > 0; --i)
            comp[i - 1] = now.extent[i];
        Concurrency::extent<N - 1> ext(comp);
        int offset = ext.size() * stride;
#ifndef __GPU__
        if( offset >= now.extent.size())
          throw runtime_exception("errorMsg_throw", 0);
#endif
        array_view<T, N - 1> av(ext, ext, index<N - 1>(), now.m_device, now.data(), offset);
        return av;
    }
    static const_result_type project(const array<const T, N>& now, int stride) restrict(amp,cpu) {
        int comp[N - 1], i;
        for (i = N - 1; i > 0; --i)
            comp[i - 1] = now.extent[i];
        Concurrency::extent<N - 1> ext(comp);
        int offset = ext.size() * stride;
        return const_result_type(ext, ext, index<N - 1>(), now.m_device, now.data(), offset);
    }
    static result_type project(const array_view<T, N>& now, int stride) restrict(amp,cpu) {
        int ext[N - 1], i, idx[N - 1], ext_o[N - 1];
        for (i = N - 1; i > 0; --i) {
            ext_o[i - 1] = now.extent[i];
            ext[i - 1] = now.extent_base[i];
            idx[i - 1] = now.index_base[i];
        }
        stride += now.index_base[0];
        Concurrency::extent<N - 1> ext_now(ext_o);
        Concurrency::extent<N - 1> ext_base(ext);
        Concurrency::index<N - 1> idx_base(idx);
        array_view<T, N - 1> av(ext_now, ext_base, idx_base, now.cache,
                                now.p_, now.offset + ext_base.size() * stride);
        return av;
    }
};

template <typename T>
struct projection_helper<T, 1>
{
    typedef __global T& result_type;
    typedef __global const T& const_result_type;
    static result_type project(array<T, 1>& now, int i) restrict(amp,cpu) {
        __global T *ptr = reinterpret_cast<__global T *>(now.m_device.get() + i);
        return *ptr;
    }
    static const_result_type& project(const array<T, 1>& now, int i) restrict(amp,cpu) {
        __global const T *ptr = reinterpret_cast<__global const T *>(now.m_device.get() + i);
        return *ptr;
    }
    static result_type& project(const array_view<T, 1>& now, int i) restrict(amp,cpu) {
        __global T *ptr = reinterpret_cast<__global T *>(now.cache.get() + i + now.offset + now.index_base[0]);
        return *ptr;
    }
};

template <typename T, int N = 1>
class array_helper {
public:
  __attribute__((annotate("user_deserialize")))
  array_helper() restrict(cpu, amp) {}

  void setArray(array<T, N>* arr) restrict(cpu) { m_arr = arr; }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const {
    array<T, N>* p_arr = (array<T, N>*)m_arr;
    if (p_arr && p_arr->pav && p_arr->pav->get_accelerator() == accelerator(accelerator::cpu_accelerator)) {
      throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
    }    
  }
private:
  void* m_arr;
};

// ------------------------------------------------------------------------

template <typename T, int N = 1>
class array {
public:
#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host<T> gmac_buffer_t;
#endif
  typedef array_helper<T, N> array_helper_t;

  static const int rank = N;
  typedef T value_type;
  array() = delete;

  explicit array(const Concurrency::extent<N>& ext);
  explicit array(int e0);
  explicit array(int e0, int e1);
  explicit array(int e0, int e1, int e2);


  array(const Concurrency::extent<N>& ext, accelerator_view av,
        access_type cpu_access_type = access_type_auto);
  array(int e0, accelerator_view av,
        access_type cpu_access_type = access_type_auto);
  array(int e0, int e1, accelerator_view av,
        access_type cpu_access_type = access_type_auto);
  array(int e0, int e1, int e2, accelerator_view av,
        access_type cpu_access_type = access_type_auto);


  array(const Concurrency::extent<N>& extent, accelerator_view av, accelerator_view associated_av);
  array(int e0, accelerator_view av, accelerator_view associated_av);
  array(int e0, int e1, accelerator_view av, accelerator_view associated_av); //staging
  array(int e0, int e1, int e2, accelerator_view av, accelerator_view associated_av); //staging


  template <typename InputIter>
      array(const Concurrency::extent<N>& ext, InputIter srcBegin);
  template <typename InputIter>
      array(const Concurrency::extent<N>& ext, InputIter srcBegin, InputIter srcEnd);
  template <typename InputIter>
      array(int e0, InputIter srcBegin);
  template <typename InputIter>
      array(int e0, InputIter srcBegin, InputIter srcEnd);
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin);
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, InputIter srcEnd);
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin);
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd);


  template <typename InputIter>
      array(const Concurrency::extent<N>& ext, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(const Concurrency::extent<N>& ext, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(int e0, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(int e0, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, accelerator_view av,
            access_type cpu_access_type = access_type_auto);
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, access_type cpu_access_type = access_type_auto);


  template <typename InputIter>
      array(const Concurrency::extent<N>& ext, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(const Concurrency::extent<N>& ext, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(int e0, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(int e0, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(int e0, int e1, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin,
            accelerator_view av, accelerator_view associated_av);
  template <typename InputIter>
      array(int e0, int e1, int e2, InputIter srcBegin, InputIter srcEnd,
            accelerator_view av, accelerator_view associated_av);


  explicit array(const array_view<const T, N>& src) : array(src.extent) {
      memmove(const_cast<void*>(reinterpret_cast<const void*>(m_device.get())),
      reinterpret_cast<const void*>(src.cache.get()), extent.size() * sizeof(T));
  }


  array(const array_view<const T, N>& src, accelerator_view av,
        access_type cpu_access_type = access_type_auto);
  array(const array_view<const T, N>& src, accelerator_view av,
        accelerator_view associated_av);


  array(const array& other);
  array(array&& other);

  array& operator=(const array& other) {
    if(this != &other) {
      extent = other.extent;
      this->cpu_access_type = other.cpu_access_type;
#ifndef __GPU__
      this->initialize();
#endif
      copy(other, *this);
    }
    return *this;
  }
  array& operator=(array&& other) {
    if(this != &other) {
      extent = other.extent;
      this->cpu_access_type = other.cpu_access_type;
      other.m_device = nullptr;
      copy(other, *this);
    }
    return *this;
  }
  array& operator=(const array_view<T,N>& src) {
    extent = src.get_extent();
#ifndef __GPU__
    this->initialize();
#endif
    src.copy_to(*this);

    return *this;
  }

  void copy_to(array& dest) const {
#ifndef __GPU__
      for(int i = 0 ; i < N ; i++)
      {
        if(dest.extent[i] < this->extent[i] )
          throw runtime_exception("errorMsg_throw", 0);
      }
#endif
      copy(*this, dest);
  }

  void copy_to(const array_view<T,N>& dest) const {
      copy(*this, dest);
  }

  Concurrency::extent<N> get_extent() const restrict(amp,cpu) {
      return extent;
  }


  accelerator_view get_accelerator_view() const {return *pav;}
  accelerator_view get_associated_accelerator_view() const {return *paav;}
  access_type get_cpu_access_type() const {return cpu_access_type;}

  __global T& operator[](const index<N>& idx) restrict(amp,cpu) {
#ifndef __GPU__
      if(pav && (pav->get_accelerator() == accelerator(accelerator::gpu_accelerator))) {
          throw runtime_exception("The array is not accessible on CPU.", 0);
      }
#endif
      __global T *ptr = reinterpret_cast<__global T*>(m_device.get());
      return ptr[amp_helper<N, index<N>, Concurrency::extent<N> >::flatten(idx, extent)];
  }
  __global const T& operator[](const index<N>& idx) const restrict(amp,cpu) {
#ifndef __GPU__
      if(pav && (pav->get_accelerator() == accelerator(accelerator::gpu_accelerator))) {
          throw runtime_exception("The array is not accessible on CPU.", 0);
      }
#endif
      __global T *ptr = reinterpret_cast<__global T*>(m_device.get());
      return ptr[amp_helper<N, index<N>, Concurrency::extent<N> >::flatten(idx, extent)];
  }

  typename projection_helper<T, N>::result_type
      operator[] (int i) restrict(amp,cpu) {
          return projection_helper<T, N>::project(*this, i);
      }
  typename projection_helper<T, N>::const_result_type
      operator[] (int i) const restrict(amp,cpu) {
          return projection_helper<const T, N>::project(*this, i);
      }

  __global T& operator()(const index<N>& idx) restrict(amp,cpu) {
    return (*this)[idx];
  }
  __global const T& operator()(const index<N>& idx) const restrict(amp,cpu) {
    return (*this)[idx];
  }
  typename projection_helper<T, N>::result_type
      operator()(int i0) restrict(amp,cpu) {
          return (*this)[i0];
  }
  __global const T& operator()(int i0) const restrict(amp,cpu) {
      return (*this)[i0];
  }
  __global T& operator()(int i0, int i1) restrict(amp,cpu) {
      return (*this)[index<2>(i0, i1)];
  }
  __global const T& operator()(int i0, int i1) const restrict(amp,cpu) {
      return (*this)[index<2>(i0, i1)];
  }
  __global T& operator()(int i0, int i1, int i2) restrict(amp,cpu) {
      return (*this)[index<3>(i0, i1, i2)];
  }
  __global const T& operator()(int i0, int i1, int i2) const restrict(amp,cpu) {
      return (*this)[index<3>(i0, i1, i2)];
  }

  array_view<T, N> section(const Concurrency::index<N>& idx, const Concurrency::extent<N>& ext) restrict(amp,cpu) {
#ifndef __GPU__
      if(  !amp_helper<N, index<N>, Concurrency::extent<N>>::contains(idx,  ext ,this->extent) )
        throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<T, N> av(*this);
      return av.section(idx, ext);
  }
  array_view<const T, N> section(const Concurrency::index<N>& idx, const Concurrency::extent<N>& ext) const restrict(amp,cpu) {
      array_view<const T, N> av(*this);
      return av.section(idx, ext);
  }
  array_view<T, N> section(const index<N>& idx) restrict(amp,cpu) {
#ifndef __GPU__
      if(  !amp_helper<N, index<N>, Concurrency::extent<N>>::contains(idx, this->extent ) )
        throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<T, N> av(*this);
      return av.section(idx);
  }
  array_view<const T, N> section(const index<N>& idx) const restrict(amp,cpu) {
      array_view<const T, N> av(*this);
      return av.section(idx);
  }
  array_view<T,N> section(const extent<N>& ext) restrict(amp,cpu) {
      array_view<T, N> av(*this);
      return av.section(ext);
  }
  array_view<const T,N> section(const extent<N>& ext) const restrict(amp,cpu) {
      array_view<const T, N> av(*this);
      return av.section(ext);
  }

  array_view<T, 1> section(int i0, int e0) restrict(amp,cpu) {
      static_assert(N == 1, "Rank must be 1");
      return section(Concurrency::index<1>(i0), Concurrency::extent<1>(e0));
  }
  array_view<const T, 1> section(int i0, int e0) const restrict(amp,cpu) {
      static_assert(N == 1, "Rank must be 1");
      return section(Concurrency::index<1>(i0), Concurrency::extent<1>(e0));
  }
  array_view<T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return section(Concurrency::index<2>(i0, i1), Concurrency::extent<2>(e0, e1));
  }
  array_view<T, 2> section(int i0, int i1, int e0, int e1) restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return section(Concurrency::index<2>(i0, i1), Concurrency::extent<2>(e0, e1));
  }
  array_view<T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return section(Concurrency::index<3>(i0, i1, i2), Concurrency::extent<3>(e0, e1, e2));
  }
  array_view<const T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) const restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return section(Concurrency::index<3>(i0, i1, i2), Concurrency::extent<3>(e0, e1, e2));
  }

  template <typename ElementType>
    array_view<ElementType, 1> reinterpret_as() restrict(amp,cpu) {
#ifndef __GPU__
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
        int size = extent.size() * sizeof(T) / sizeof(ElementType);
#ifndef __GPU__
        if( (extent.size() * sizeof(T)) % sizeof(ElementType))
          throw runtime_exception("errorMsg_throw", 0);
#endif
        array_view<ElementType, 1> av(Concurrency::extent<1>(size), reinterpret_cast<ElementType*>(m_device.get()));
        return av;
    }
  template <typename ElementType>
    array_view<const ElementType, 1> reinterpret_as() const restrict(amp,cpu) {
#ifndef __GPU__
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
        int size = extent.size() * sizeof(T) / sizeof(ElementType);
        array_view<const ElementType, 1> av(Concurrency::extent<1>(size), reinterpret_cast<const ElementType*>(m_device.get()));
        return av;
    }
  template <int K> array_view<T, K>
      view_as(const Concurrency::extent<K>& viewExtent) restrict(amp,cpu) {
          array_view<T, 1> av(Concurrency::extent<1>(viewExtent.size()), data());
          return av.view_as(viewExtent);
      }
  template <int K> array_view<const T, K>
      view_as(const Concurrency::extent<K>& viewExtent) const restrict(amp,cpu) {
          const array_view<T, 1> av(Concurrency::extent<1>(viewExtent.size()), data());
          return av.view_as(viewExtent);
      }

  operator std::vector<T>() const {
      T *begin = reinterpret_cast<T*>(m_device.get()),
        *end = reinterpret_cast<T*>(m_device.get() + extent.size());
      return std::vector<T>(begin, end);
  }

  T* data() const restrict(amp,cpu) {
#ifndef __GPU__
    // TODO: If array's buffer is inaccessible on CPU, host pointer to that buffer must be NULL
    if(cpu_access_type == access_type_none) {
      //return reinterpret_cast<T*>(NULL);
    }      
#endif
    return reinterpret_cast<T*>(m_device.get());
  }
  ~array() { // For GMAC
    m_device.reset();
    if(pav) delete pav;
    if(paav) delete paav;
  }


  const gmac_buffer_t& internal() const restrict(amp,cpu) { return m_device; }
  Concurrency::extent<N> extent;
private:
  template <int K, typename Q> friend struct index_helper;
  template <int K, typename Q1, typename Q2> friend struct amp_helper;
  template <typename K, int Q> friend struct projection_helper;
  template <typename K, int Q> friend class array_helper;
  gmac_buffer_t m_device;
  access_type cpu_access_type;
  array_helper_t m_array_helper;
  __attribute__((cpu)) accelerator_view *pav, *paav;

#ifndef __GPU__
  void initialize() {
      m_device.reset(GMACAllocator<T>().allocate(extent.size()), GMACDeleter<T>());
      m_array_helper.setArray(this);
  }
  template <typename InputIter>
      void initialize(InputIter srcBegin, InputIter srcEnd) {
          initialize();
          std::copy(srcBegin, srcEnd, m_device.get());
      }
#endif
};

template <typename T, int N = 1>
class array_view
{
  typedef typename std::remove_const<T>::type nc_T;
public:
#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host_view<T> gmac_buffer_t;
#endif

  static const int rank = N;
  typedef T value_type;
  array_view() = delete;

  ~array_view() restrict(amp,cpu) {
#ifndef __GPU__
      if (p_ && cache.is_last()) {
          synchronize();
          cache.reset();
      }
#endif
  }

  array_view(array<T, N>& src) restrict(amp,cpu)
      : extent(src.extent), p_(NULL), cache(src.internal()), offset(0),
        index_base(), extent_base(src.extent) {}

  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(const Concurrency::extent<N>& extent, Container& src)
      : array_view(extent, src.data()) {}
  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(int e0, Container& src)
      : array_view(Concurrency::extent<1>(e0), src)
  { static_assert(N == 1, "Rank must be 1"); }
  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(int e0, int e1, Container& src)
      : array_view(Concurrency::extent<2>(e0, e1), src)
  { static_assert(N == 2, "Rank must be 2"); }
  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(int e0, int e1, int e2, Container& src)
      : array_view(Concurrency::extent<3>(e0, e1, e2), src)
  { static_assert(N == 3, "Rank must be 3"); }


  array_view(const Concurrency::extent<N>& extent, value_type* src) restrict(amp,cpu);
  array_view(int e0, value_type *src) restrict(amp,cpu)
      : array_view(Concurrency::extent<1>(e0), src)
  { static_assert(N == 1, "Rank must be 1"); }
  array_view(int e0, int e1, value_type *src) restrict(amp,cpu)
      : array_view(Concurrency::extent<2>(e0, e1), src)
  { static_assert(N == 2, "Rank must be 2"); }
  array_view(int e0, int e1, int e2, value_type *src) restrict(amp,cpu)
      : array_view(Concurrency::extent<3>(e0, e1, e2), src)
  { static_assert(N == 3, "Rank must be 3"); }


  explicit array_view(const Concurrency::extent<N>& extent);
  explicit array_view(int e0)
      : array_view(Concurrency::extent<1>(e0))
  { static_assert(N == 1, "Rank must be 1"); }
  explicit array_view(int e0, int e1)
      : array_view(Concurrency::extent<2>(e0, e1))
  { static_assert(N == 2, "Rank must be 2"); }
  explicit array_view(int e0, int e1, int e2)
      : array_view(Concurrency::extent<3>(e0, e1, e2))
  { static_assert(N == 3, "Rank must be 3"); }

  template <class = typename std::enable_if<std::is_const<T>::value>::type>
    array_view(const array_view<nc_T, N>& other) restrict(amp,cpu) : extent(other.extent),
      p_(other.p_), cache(other.cache), offset(other.offset), index_base(other.index_base),
      extent_base(other.extent_base) {}
  template <class = typename std::enable_if<!std::is_const<T>::value>::type>
    array_view(const array_view<const T, N>& other) restrict(amp,cpu) : extent(other.extent),
      p_(const_cast<T*>(other.p_)), cache(other.cache), offset(other.offset), index_base(other.index_base),
      extent_base(other.extent_base) {
      }

  array_view(const array_view<const T, N>& other) restrict(amp,cpu) : extent(other.extent),
    p_(const_cast<T*>(other.p_)), cache(other.cache), offset(other.offset), index_base(other.index_base),
    extent_base(other.extent_base) {
    }
   array_view(const array_view& other) restrict(amp,cpu) : extent(other.extent),
    p_(other.p_), cache(other.cache), offset(other.offset), index_base(other.index_base),
    extent_base(other.extent_base) {}
  array_view& operator=(const array_view& other) restrict(amp,cpu) {
      if (this != &other) {
          extent = other.extent;
          p_ = other.p_;
          cache = other.cache;
          index_base = other.index_base;
          extent_base = other.extent_base;
          offset = other.offset;
      }
      return *this;
  }
  array_view& operator=(const array_view<const T,N>& other) restrict(amp,cpu) {
    extent = other.extent;
    p_ = const_cast<T*>(other.p_);
    cache = other.cache;
    index_base = other.index_base;
    extent_base = other.extent_base;
    offset = other.offset;
    return *this;
  }

  void copy_to(array<T,N>& dest) const {
#ifndef __GPU__
      for(int i= 0 ;i< N;i++)
      {
        if(dest.extent[i] < this->extent[i])
          throw runtime_exception("errorMsg_throw", 0);
      }
#endif
      copy(*this, dest);
  }
  void copy_to(const array_view& dest) const {
      copy(*this, dest);
  }

  extent<N> get_extent() const restrict(amp,cpu) {
      return extent;
  }

  __global T& operator[](const index<N>& idx) const restrict(amp,cpu) {
      __global T *ptr = reinterpret_cast<__global T*>(cache.get() + offset);
      return ptr[amp_helper<N, index<N>, Concurrency::extent<N>>::flatten(idx + index_base, extent_base)];
  }
  template <int D0, int D1=0, int D2=0>
  __global T& operator[](const tiled_index<D0, D1, D2>& idx) const restrict(amp,cpu) {
      __global T *ptr = reinterpret_cast<__global T*>(cache.get() + offset);
      return ptr[amp_helper<N, index<N>, Concurrency::extent<N>>::flatten(idx.global + index_base, extent_base)];
  }

  typename projection_helper<T, N>::result_type
      operator[] (int i) const restrict(amp,cpu) {
          return projection_helper<T, N>::project(*this, i);
      }
  __global T& operator()(const index<N>& idx) const restrict(amp,cpu) {
    return (*this)[idx];
  }
  typename projection_helper<T, N>::result_type
      operator()(int i0) const restrict(amp,cpu) {
          return (*this)[index<1>(i0)];
  }
  __global T& operator()(int i0, int i1) const restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return (*this)[index<2>(i0, i1)];
  }
  __global T& operator()(int i0, int i1, int i2) const restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return (*this)[index<3>(i0, i1, i2)];
  }

  template <typename ElementType>
      array_view<ElementType, 1> reinterpret_as() restrict(amp,cpu) {
#ifndef __GPU__
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
          int size = extent.size() * sizeof(T) / sizeof(ElementType);
#ifndef __GPU__
          if( (extent.size() * sizeof(T)) % sizeof(ElementType))
            throw runtime_exception("errorMsg_throw", 0);
#endif
          array_view<ElementType, 1> av(Concurrency::extent<1>(size), reinterpret_cast<ElementType*>(cache.get_mutable() + offset + index_base[0]));
          return av;
      }
  template <typename ElementType>
      array_view<const ElementType, 1> reinterpret_as() const restrict(amp,cpu) {
#ifndef __GPU__
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
          int size = extent.size() * sizeof(T) / sizeof(ElementType);
          array_view<const ElementType, 1> av(Concurrency::extent<1>(size), reinterpret_cast<const ElementType*>(cache.get() + offset + index_base[0]));
          return av;
      }
  array_view<T, N> section(const Concurrency::index<N>& idx,
                           const Concurrency::extent<N>& ext) const restrict(amp,cpu) {
#ifndef __GPU__
      if(  !amp_helper<N, index<N>, Concurrency::extent<N>>::contains(idx, ext,this->extent ) )
        throw runtime_exception("errorMsg_throw", 0);
#endif
      array_view<T, N> av(ext, extent_base, idx + index_base, cache, p_, offset);
      return av;
  }
  array_view<T, N> section(const Concurrency::index<N>& idx) const restrict(amp,cpu) {
      Concurrency::extent<N> ext(extent);
      amp_helper<N, Concurrency::index<N>, Concurrency::extent<N>>::minus(idx, ext);
      return section(idx, ext);
  }
  array_view<T, N> section(const Concurrency::extent<N>& ext) const restrict(amp,cpu) {
      Concurrency::index<N> idx;
      return section(idx, ext);
  }
  array_view<T, 1> section(int i0, int e0) const restrict(amp,cpu) {
      static_assert(N == 1, "Rank must be 1");
      return section(Concurrency::index<1>(i0), Concurrency::extent<1>(e0));
  }
  array_view<T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
      static_assert(N == 2, "Rank must be 2");
      return section(Concurrency::index<2>(i0, i1), Concurrency::extent<2>(e0, e1));
  }
  array_view<T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) const restrict(amp,cpu) {
      static_assert(N == 3, "Rank must be 3");
      return section(Concurrency::index<3>(i0, i1, i2), Concurrency::extent<3>(e0, e1, e2));
  }

  template <int K>
  array_view<T, K> view_as(Concurrency::extent<K> viewExtent) const restrict(amp,cpu) {
    static_assert(N == 1, "view_as is only permissible on array views of rank 1");
#ifndef __GPU__
    if( viewExtent.size() > extent.size())
      throw runtime_exception("errorMsg_throw", 0);
#endif
    array_view<T, K> av(viewExtent, cache, p_, index_base[0]);
    return av;
  }

  void synchronize() const;
  completion_future synchronize_async() const;
  void refresh() const;
  void discard_data() const {
#ifndef __GPU__
    cache.refresh();
#endif
  }
  T* data() const restrict(amp,cpu) {
    static_assert(N == 1, "data() is only permissible on array views of rank 1");
    return reinterpret_cast<T*>(cache.get() + offset + index_base[0]);
  }

private:
  template <int K, typename Q> friend struct index_helper;
  template <int K, typename Q1, typename Q2> friend struct amp_helper;
  template <typename K, int Q> friend struct projection_helper;
  template <typename Q, int K> friend class array;
  template <typename Q, int K> friend class array_view;

  // used by view_as
  array_view(const Concurrency::extent<N>& ext, const gmac_buffer_t& cache,
             T *p, int offset) restrict(amp,cpu)
      : extent(ext), cache(cache), offset(offset), p_(p), extent_base(ext) {}
  // used by section and projection
  array_view(const Concurrency::extent<N>& ext_now,
             const Concurrency::extent<N>& ext_b,
             const Concurrency::index<N>& idx_b,
             const gmac_buffer_t& cache, T *p, int off) restrict(amp,cpu)
      : extent(ext_now), index_base(idx_b), extent_base(ext_b),
      p_(p), cache(cache), offset(off) {}

  __attribute__((cpu)) T *p_;
  gmac_buffer_t cache;
  Concurrency::extent<N> extent;
  Concurrency::extent<N> extent_base;
  Concurrency::index<N> index_base;
  int offset;
};

template <typename T, int N>
class array_view<const T, N>
{
public:
  typedef typename std::remove_const<T>::type nc_T;
  static const int rank = N;
  typedef const T value_type;

#ifdef __GPU__
  typedef _data<T> gmac_buffer_t;
#else
  typedef _data_host_view<T> gmac_buffer_t;
#endif

  array_view() = delete;

  ~array_view() restrict(amp,cpu) {
#ifndef __GPU__
  if (p_ && cache.is_last()) {
    synchronize();
    cache.reset();
  }
#endif
  }

  array_view(const array<T,N>& src) restrict(amp,cpu)
      : extent(src.extent), p_(NULL), cache(src.internal()), offset(0),
        index_base(), extent_base(src.extent) {}
  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
    array_view(const extent<N>& extent, const Container& src)
        : array_view(extent, src.data()) {}
    template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(int e0, Container& src)
      : array_view(Concurrency::extent<1>(e0), src)
  { static_assert(N == 1, "Rank must be 1"); }
  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(int e0, int e1, Container& src)
      : array_view(Concurrency::extent<2>(e0, e1), src)
  { static_assert(N == 2, "Rank must be 2"); }
  template <typename Container, class = typename std::enable_if<!std::is_array<Container>::value>::type>
      array_view(int e0, int e1, int e2, Container& src)
      : array_view(Concurrency::extent<3>(e0, e1, e2), src)
  { static_assert(N == 3, "Rank must be 3"); }

  array_view(const extent<N>& extent, const value_type* src) restrict(amp,cpu);
  array_view(int e0, value_type *src) restrict(amp,cpu)
      : array_view(Concurrency::extent<1>(e0), src)
  { static_assert(N == 1, "Rank must be 1"); }
  array_view(int e0, int e1, value_type *src) restrict(amp,cpu)
      : array_view(Concurrency::extent<2>(e0, e1), src)
  { static_assert(N == 2, "Rank must be 2"); }
  array_view(int e0, int e1, int e2, value_type *src) restrict(amp,cpu)
      : array_view(Concurrency::extent<3>(e0, e1, e2), src)
  { static_assert(N == 3, "Rank must be 3"); }

  array_view(const array_view<T, N>& other) restrict(amp,cpu) : extent(other.extent),
      p_(other.p_), cache(other.cache), offset(other.offset), index_base(other.index_base),
      extent_base(other.extent_base) {}

  array_view(const array_view& other) restrict(amp,cpu) : extent(other.extent),
    p_(other.p_), cache(other.cache), offset(other.offset), index_base(other.index_base),
    extent_base(other.extent_base) {}

  array_view& operator=(const array_view<T,N>& other) restrict(amp,cpu) {
    extent = other.extent;
    p_ = other.p_;
    cache = other.cache;
    index_base = other.index_base;
    extent_base = other.extent_base;
    offset = other.offset;
    return *this;
  }

  array_view& operator=(const array_view& other) restrict(amp,cpu) {
    if (this != &other) {
      extent = other.extent;
      p_ = other.p_;
      cache = other.cache;
      index_base = other.index_base;
      extent_base = other.extent_base;
      offset = other.offset;
    }
    return *this;
  }

  void copy_to(array<T,N>& dest) const {
    copy(*this, dest);
  }

  void copy_to(const array_view<T,N>& dest) const {
    copy(*this, dest);
  }

  extent<N> get_extent() const restrict(amp,cpu) {
    return extent;
  }
  accelerator_view get_source_accelerator_view() const;

  __global const T& operator[](const index<N>& idx) const restrict(amp,cpu) {
    __global T *ptr = reinterpret_cast<__global T*>(cache.get() + offset);
    return ptr[amp_helper<N, index<N>, Concurrency::extent<N>>::flatten(idx + index_base, extent_base)];
  }

  typename projection_helper<const T, N>::result_type
      operator[] (int i) const restrict(amp,cpu) {
    return projection_helper<const T, N>::project(*this, i);
  }

  const T& get_ref(const index<N>& idx) const restrict(amp,cpu);

  __global const T& operator()(const index<N>& idx) const restrict(amp,cpu) {
    return (*this)[idx];
  }
  __global const T& operator()(int i0) const restrict(amp,cpu) {
    static_assert(N == 1, "Rank must be 1");
    return (*this)[index<1>(i0)];
  }
  __global const T& operator()(int i0, int i1) const restrict(amp,cpu) {
    static_assert(N == 2, "Rank must be 2");
    return (*this)[index<2>(i0, i1)];
  }
  __global const T& operator()(int i0, int i1, int i2) const restrict(amp,cpu) {
    static_assert(N == 3, "Rank must be 3");
    return (*this)[index<3>(i0, i1, i2)];
  }
/*
  typename projection_helper<const T, N>::result_type
      operator()(int i) const restrict(amp,cpu) {
    return (*this)[idx];
  }
*/
  template <typename ElementType>
    array_view<ElementType, 1> reinterpret_as() restrict(amp,cpu) {
#ifndef __GPU__
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
      int size = extent.size() * sizeof(T) / sizeof(ElementType);
      array_view<ElementType, 1> av(Concurrency::extent<1>(size), reinterpret_cast<ElementType*>(cache.get_mutable() + offset + index_base[0]));
      return av;
    }
  template <typename ElementType>
    array_view<const ElementType, 1> reinterpret_as() const restrict(amp,cpu) {
#ifndef __GPU__
          static_assert( ! (std::is_pointer<ElementType>::value ),"can't use pointer in the kernel");
          static_assert( ! (std::is_same<ElementType,short>::value ),"can't use short in the kernel");
#endif
      int size = extent.size() * sizeof(T) / sizeof(ElementType);
      array_view<const ElementType, 1> av(Concurrency::extent<1>(size), reinterpret_cast<const ElementType*>(cache.get() + offset + index_base[0]));
      return av;
    }
  array_view<const T, N> section(const Concurrency::index<N>& idx,
                     const Concurrency::extent<N>& ext) const restrict(amp,cpu) {
    array_view<const T, N> av(ext, extent_base, idx + index_base, cache, p_, offset);
    return av;
  }
  array_view<const T, N> section(const Concurrency::index<N>& idx) const restrict(amp,cpu) {
    Concurrency::extent<N> ext(extent);
    amp_helper<N, Concurrency::index<N>, Concurrency::extent<N>>::minus(idx, ext);
    return section(idx, ext);
  }

  array_view<const T, N> section(const Concurrency::extent<N>& ext) const restrict(amp,cpu) {
    Concurrency::index<N> idx;
    return section(idx, ext);
  }
  array_view<const T, 1> section(int i0, int e0) const restrict(amp,cpu) {
    static_assert(N == 1, "Rank must be 1");
    return section(Concurrency::index<1>(i0), Concurrency::extent<1>(e0));
  }
  array_view<const T, 2> section(int i0, int i1, int e0, int e1) const restrict(amp,cpu) {
    static_assert(N == 2, "Rank must be 2");
    return section(Concurrency::index<2>(i0, i1), Concurrency::extent<2>(e0, e1));
  }
  array_view<const T, 3> section(int i0, int i1, int i2, int e0, int e1, int e2) const restrict(amp,cpu) {
    static_assert(N == 3, "Rank must be 3");
    return section(Concurrency::index<3>(i0, i1, i2), Concurrency::extent<3>(e0, e1, e2));
  }

  template <int K>
    array_view<const T, K> view_as(Concurrency::extent<K> viewExtent) const restrict(amp,cpu) {
      static_assert(N == 1, "view_as is only permissible on array views of rank 1");
      array_view<const T, K> av(viewExtent, cache, p_, offset);
      return av;
    }

  void synchronize() const;
  completion_future synchronize_async() const;

  void synchronize_to(const accelerator_view& av) const;
  completion_future synchronize_to_async(const accelerator_view& av) const;

  void refresh() const;

  const T* data() const restrict(amp,cpu) {
    static_assert(N == 1, "data() is only permissible on array views of rank 1");
    return reinterpret_cast<T*>(cache.get() + offset + index_base[0]);
  }
private:
  template <int K, typename Q> friend struct index_helper;
  template <int K, typename Q1, typename Q2> friend struct amp_helper;
  template <typename K, int Q> friend struct projection_helper;
  template <typename Q, int K> friend class array;
  template <typename Q, int K> friend class array_view;
/*
  // used by view_as
  array_view(const Concurrency::extent<N>& ext, const gmac_buffer_t& cache,
             T *p, int offset) restrict(amp,cpu)
      : extent(ext), cache(cache), offset(offset), p_(p), extent_base(ext) {}
*/
  // used by section and projection
  array_view(const Concurrency::extent<N>& ext_now,
             const Concurrency::extent<N>& ext_b,
             const Concurrency::index<N>& idx_b,
             const gmac_buffer_t& cache, value_type *p, int off) restrict(amp,cpu)
      : extent(ext_now), index_base(idx_b), extent_base(ext_b),
      p_(p), cache(cache), offset(off) {}

  __attribute__((cpu)) value_type *p_;
  gmac_buffer_t cache;
  Concurrency::extent<N> extent;
  Concurrency::extent<N> extent_base;
  Concurrency::index<N> index_base;
  int offset;
};

#undef __global

template <int N, typename Kernel>
void parallel_for_each(extent<N> compute_domain, const Kernel& f);

template <int D0, int D1, int D2, typename Kernel>
void parallel_for_each(tiled_extent<D0,D1,D2> compute_domain, const Kernel& f);

template <int D0, int D1, typename Kernel>
void parallel_for_each(tiled_extent<D0,D1> compute_domain, const Kernel& f);

template <int D0, typename Kernel>
void parallel_for_each(tiled_extent<D0> compute_domain, const Kernel& f);

template <int N, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, extent<N> compute_domain, const Kernel& f){
    if (accl_view.get_accelerator() == accelerator(accelerator::cpu_accelerator)) {
      throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
    }
    parallel_for_each(compute_domain, f);
}

template <int D0, int D1, int D2, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, tiled_extent<D0,D1,D2> compute_domain, const Kernel& f) {
    if (accl_view.get_accelerator() == accelerator(accelerator::cpu_accelerator)) {
      throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
    }
    parallel_for_each(compute_domain, f);
}

template <int D0, int D1, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, tiled_extent<D0,D1> compute_domain, const Kernel& f) {
    if (accl_view.get_accelerator() == accelerator(accelerator::cpu_accelerator)) {
      throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
    }
    parallel_for_each(compute_domain, f);
}

template <int D0, typename Kernel>
void parallel_for_each(const accelerator_view& accl_view, tiled_extent<D0> compute_domain, const Kernel& f) {
    if (accl_view.get_accelerator() == accelerator(accelerator::cpu_accelerator)) {
      throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
    }
    parallel_for_each(compute_domain, f);
}

} // namespace Concurrency
namespace concurrency = Concurrency;
// Specialization and inlined implementation of C++AMP classes/templates
#include "amp_impl.h"
#include "parallel_for_each.h"

namespace Concurrency {

template <typename T>
void copy(const array_view<const T, 1>& src, const array_view<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        dest[i] = src[i];
}
template <typename T, int N>
void copy(const array_view<const T, N>& src, const array_view<T, N>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        Concurrency::copy(src[i], dest[i]);
}

template <typename T>
void copy(const array_view<T, 1>& src, const array_view<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        dest[i] = src[i];
}
template <typename T, int N>
void copy(const array_view<T, N>& src, const array_view<T, N>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        Concurrency::copy(src[i], dest[i]);
}

template <typename T>
void copy(const array<T, 1>& src, const array_view<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        dest[i] = src[i];
}
template <typename T, int N>
void copy(const array<T, N>& src, const array_view<T, N>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        Concurrency::copy(src[i], dest[i]);
}

template <typename T>
void copy(const array<T, 1>& src, array<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        dest[i] = src[i];
}
template <typename T, int N>
void copy(const array<T, N>& src, array<T, N>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        Concurrency::copy(src[i], dest[i]);
}

template <typename T>
void copy(const array_view<const T, 1>& src, array<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        dest[i] = src[i];
}
template <typename T, int N>
void copy(const array_view<const T, N>& src, array<T, N>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        Concurrency::copy(src[i], dest[i]);
}

template <typename T>
void copy(const array_view<T, 1>& src, array<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        dest[i] = src[i];
}
template <typename T, int N>
void copy(const array_view<T, N>& src, array<T, N>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i)
        Concurrency::copy(src[i], dest[i]);
}

// TODO: __global should not be allowed in CPU Path
template <typename InputIter, typename T>
void copy(InputIter srcBegin, InputIter srcEnd, const array_view<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        reinterpret_cast<T&>(dest[i]) = *srcBegin;
        ++srcBegin;
    }
}
template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest) {
    int adv = dest.get_extent().size() / dest.get_extent()[0];
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        Concurrency::copy(srcBegin, srcEnd, dest[i]);
        std::advance(srcBegin, adv);
    }
}

// TODO: Boundary Check
template <typename InputIter, typename T>
void copy(InputIter srcBegin, InputIter srcEnd, array<T, 1>& dest) {
#ifndef __GPU__
    if( ( std::distance(srcBegin,srcEnd) <=0 )||( std::distance(srcBegin,srcEnd) < dest.get_extent()[0] ))
      throw runtime_exception("errorMsg_throw ,copy between different types", 0);
#endif
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        dest[i] = *srcBegin;
        ++srcBegin;
    }
}
template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest) {
    int adv = dest.get_extent().size() / dest.get_extent()[0];
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        Concurrency::copy(srcBegin, srcEnd, dest[i]);
        std::advance(srcBegin, adv);
    }
}

template <typename InputIter, typename T>
void copy(InputIter srcBegin, const array_view<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        reinterpret_cast<T&>(dest[i]) = *srcBegin;
        ++srcBegin;
    }
}
template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, const array_view<T, N>& dest) {
    int adv = dest.get_extent().size() / dest.get_extent()[0];
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        Concurrency::copy(srcBegin, dest[i]);
        std::advance(srcBegin, adv);
    }
}

template <typename InputIter, typename T>
void copy(InputIter srcBegin, array<T, 1>& dest) {
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        dest[i] = *srcBegin;
        ++srcBegin;
    }
}
template <typename InputIter, typename T, int N>
void copy(InputIter srcBegin, array<T, N>& dest) {
    int adv = dest.get_extent().size() / dest.get_extent()[0];
    for (int i = 0; i < dest.get_extent()[0]; ++i) {
        Concurrency::copy(srcBegin, dest[i]);
        std::advance(srcBegin, adv);
    }
}

template <typename OutputIter, typename T>
void copy(const array_view<T, 1> &src, OutputIter destBegin) {
    for (int i = 0; i < src.get_extent()[0]; ++i) {
        *destBegin = (src[i]);
        destBegin++;
    }
}
template <typename OutputIter, typename T, int N>
void copy(const array_view<T, N> &src, OutputIter destBegin) {
    int adv = src.get_extent().size() / src.get_extent()[0];
    for (int i = 0; i < src.get_extent()[0]; ++i) {
        copy(src[i], destBegin);
        std::advance(destBegin, adv);
    }
}

template <typename OutputIter, typename T>
void copy(const array<T, 1> &src, OutputIter destBegin) {
    for (int i = 0; i < src.get_extent()[0]; ++i) {
        *destBegin = src[i];
        destBegin++;
    }
}
template <typename OutputIter, typename T, int N>
void copy(const array<T, N> &src, OutputIter destBegin) {
    int adv = src.get_extent().size() / src.get_extent()[0];
    for (int i = 0; i < src.get_extent()[0]; ++i) {
        copy(src[i], destBegin);
        std::advance(destBegin, adv);
    }
}


template <typename InputType, typename OutputType>
completion_future __amp_copy_async_impl(InputType& src, OutputType& dst) {
    std::future<void> fut = std::async([&]() mutable { copy(src, dst); });
    fut.wait();
    return completion_future(fut.share());
}


template <typename T, int N>
completion_future copy_async(const array<T, N>& src, array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array<T, N>& src, const array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array<T, N>& src, const array_view<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}


template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, const array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<const T, N>& src, const array_view<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}


template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, const array<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}

template <typename T, int N>
completion_future copy_async(const array_view<T, N>& src, const array_view<T, N>& dest) {
    return __amp_copy_async_impl(src, dest);
}


template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, InputIter srcEnd, array<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, srcEnd, dest); });
    fut.wait();
    return completion_future(fut.share());
}

template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, InputIter srcEnd, const array_view<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, srcEnd, dest); });
    fut.wait();
    return completion_future(fut.share());
}


template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, array<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, dest); });
    fut.wait();
    return completion_future(fut.share());
}
template <typename InputIter, typename T, int N>
completion_future copy_async(InputIter srcBegin, const array_view<T, N>& dest) {
    std::future<void> fut = std::async([&]() mutable { copy(srcBegin, dest); });
    fut.wait();
    return completion_future(fut.share());
}


template <typename OutputIter, typename T, int N>
completion_future copy_async(const array<T, N>& src, OutputIter destBegin) {
    std::future<void> fut = std::async([&]() mutable { copy(src, destBegin); });
    fut.wait();
    return completion_future(fut.share());
}
template <typename OutputIter, typename T, int N>
completion_future copy_async(const array_view<T, N>& src, OutputIter destBegin) {
    std::future<void> fut = std::async([&]() mutable { copy(src, destBegin); });
    fut.wait();
    return completion_future(fut.share());
}

#ifdef __GPU__
extern "C" unsigned atomic_add_local(volatile __attribute__((address_space(3))) unsigned *p, unsigned val) restrict(amp,cpu);
static inline unsigned atomic_fetch_add(unsigned *x, unsigned y) restrict(amp,cpu) { 
  return atomic_add_local(reinterpret_cast<volatile __attribute__((address_space(3))) unsigned *>(x), y);
}
#else
extern unsigned atomic_fetch_add(unsigned *x, unsigned y) restrict(amp,cpu);
#endif

#ifdef __GPU__
extern "C" int atomic_add_global(volatile __attribute__((address_space(1))) int *p, int val) restrict(amp, cpu);
static inline int atomic_fetch_add(int *x, int y) restrict(amp,cpu) {
  return atomic_add_global(reinterpret_cast<volatile __attribute__((address_space(1))) int *>(x), y);
}
#else
extern int atomic_fetch_add(int *x, int y) restrict(amp, cpu);
#endif

#ifdef __GPU__
extern "C" unsigned atomic_max_local(volatile __attribute__((address_space(3))) unsigned *p, unsigned val) restrict(amp,cpu);
extern "C" int atomic_max_global(volatile __attribute__((address_space(1))) int *p, int val) restrict(amp, cpu);
static inline unsigned atomic_fetch_max(unsigned *x, unsigned y) restrict(amp,cpu) {
  return atomic_max_local(reinterpret_cast<volatile __attribute__((address_space(3))) unsigned *>(x), y);
}
static inline int atomic_fetch_max(int *x, int y) restrict(amp,cpu) {
  return atomic_max_global(reinterpret_cast<volatile __attribute__((address_space(1))) int *>(x), y);
}

extern "C" unsigned atomic_inc_local(volatile __attribute__((address_space(3))) unsigned *p) restrict(amp,cpu);
extern "C" int atomic_inc_global(volatile __attribute__((address_space(1))) int *p) restrict(amp, cpu);
static inline unsigned atomic_fetch_inc(unsigned *x) restrict(amp,cpu) {
  return atomic_inc_local(reinterpret_cast<volatile __attribute__((address_space(3))) unsigned *>(x));
}
static inline int atomic_fetch_inc(int *x) restrict(amp,cpu) {
  return atomic_inc_global(reinterpret_cast<volatile __attribute__((address_space(1))) int *>(x));
}
#else

extern int atomic_fetch_inc(int * _Dest) restrict(amp, cpu);
extern unsigned atomic_fetch_inc(unsigned * _Dest) restrict(amp, cpu);

extern int atomic_fetch_max(int * dest, int val) restrict(amp, cpu);
extern unsigned int atomic_fetch_max(unsigned int * dest, unsigned int val) restrict(amp, cpu);

#endif

}//namespace Concurrency
