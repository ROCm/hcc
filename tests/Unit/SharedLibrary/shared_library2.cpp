
// RUN: %hc -fPIC -Wl,-Bsymbolic -shared -DSHARED_LIBRARY_1 %s -o %T/libtest2_foo.so
// RUN: %hc -fPIC -Wl,-Bsymbolic -shared -DSHARED_LIBRARY_2 %s -o %T/libtest2_bar.so
// RUN: %clang -std=c++11 -ldl -lpthread %s -o %t.out -ldl && LD_LIBRARY_PATH=%T %t.out

// kernels built as multiple shared libraries
// loaded dynamically via dlopen()

#if SHARED_LIBRARY_1

#include <hc.hpp>

extern "C" int foo(int grid_size) {
  using namespace hc;

  extent<1> ex(grid_size);
  array_view<int, 1> av(grid_size);

  parallel_for_each(ex, [=](index<1>& idx) [[hc]] {
    av(idx) = 1;
  }).wait();

  int ret = 0;
  for (int i = 0; i < grid_size; ++i) {
    ret += av[i];
  }

  return ret;
}

#else

#if SHARED_LIBRARY_2

#include <hc.hpp>

extern "C" int bar(int grid_size) {
  using namespace hc;

  extent<1> ex(grid_size);
  array_view<int, 1> av(grid_size);

  parallel_for_each(ex, [=](index<1>& idx) [[hc]] {
    av(idx) = 2;
  }).wait();

  int ret = 0;
  for (int i = 0; i < grid_size; ++i) {
    ret += av[i];
  }

  return ret;
}

#else

#include <dlfcn.h>

int main() {
  bool ret = true;

  int (*foo_handle)(int) = nullptr;
  int (*bar_handle)(int) = nullptr;

  void* libfoo_handle = dlopen("libtest2_foo.so", RTLD_LAZY);
  ret &= (libfoo_handle != nullptr);
  void* libbar_handle = dlopen("libtest2_bar.so", RTLD_LAZY);
  ret &= (libbar_handle != nullptr);

  if (libfoo_handle) {
    foo_handle = (int(*)(int)) dlsym(libfoo_handle, "foo");
    ret &= (foo_handle != nullptr);
  }

  if (libbar_handle) {
    bar_handle = (int(*)(int)) dlsym(libbar_handle, "bar");
    ret &= (bar_handle != nullptr);
  }
 
  if (foo_handle && bar_handle) {
    for (int i = 0; i < 16; ++i) {
      ret &= (foo_handle(i) == i);
      ret &= (bar_handle(i * 2) == (i * 4));
    }
  }

  if (libfoo_handle) {
    dlclose(libfoo_handle);
  }

  if (libbar_handle) {
    dlclose(libbar_handle);
  }

  return !(ret == true);
}

#endif // if SHARED_LIBRARY_2

#endif // if SHARED_LIBRARY_1
