
// RUN: %hc -fPIC -Wl,-Bsymbolic -shared -DSHARED_LIBRARY_1 %s -o %T/libtest3_foo.so
// RUN: %hc -fPIC -Wl,-Bsymbolic -shared -DSHARED_LIBRARY_2 %s -o %T/libtest3_bar.so
// RUN: %clang -std=c++11 %s -L%T -ltest3_foo -ltest3_bar -o %t.out && LD_LIBRARY_PATH=%T %t.out

// kernels built as multiple shared libraries
// linked dynamically with the main program

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

extern "C" int foo(int);
extern "C" int bar(int);

int main() {
  bool ret = true;

  for (int i = 0; i < 16; ++i) {
    ret &= (foo(i) == i);
    ret &= (bar(i * 2) == (i * 4));
  }

  return !(ret == true);
}

#endif // if SHARED_LIBRARY_2

#endif // if SHARED_LIBRARY_1
