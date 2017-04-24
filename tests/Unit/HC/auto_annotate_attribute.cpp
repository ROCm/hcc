// XFAIL: *
// RUN: %hc -Xclang -fauto-compile-for-accelerator %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <vector>

// foo is a global function which doesn't have [[hc]] attribute
// if compiled with -Xclang -fauto-compile-for-accelerator, [[hc]] would be
// annotated automatically
int foo() {
  return 1;
}

template<int GRID_SIZE>
bool test1() {
  using namespace hc;
  bool ret = true;
  array<int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table[idx] = foo();
  }).wait();

  std::vector<int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    if (result[i] != 1) {
      std::cerr << "Verify failed at index: " << i << " , expected: " << 1 << " , actual: " << result[i] << "\n";
      ret = false;
      break;
    }
  }
  return ret;
}

// bar is a static function which doesn't have [[hc]] attribute
// if compiled with -Xclang -fauto-compile-for-accelerator, [[hc]] would be
// annotated automatically
static int bar() {
  return 1;
}

template<int GRID_SIZE>
bool test2() {
  using namespace hc;
  bool ret = true;
  array<int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table[idx] = bar();
  }).wait();

  std::vector<int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    if (result[i] != 1) {
      std::cerr << "Verify failed at index: " << i << " , expected: " << 1 << " , actual: " << result[i] << "\n";
      ret = false;
      break;
    }
  }
  return ret;
}

// baz is a class with a member function test() which doesn't have [[hc]] attribute
// if compiled with -Xclang -fauto-compile-for-accelerator, [[hc]] would be
// annotated automatically
class baz {
public:
  int test() {
    return 1;
  }

  static int test2() {
    return 1;
  }
};

template<int GRID_SIZE>
bool test3() {
  using namespace hc;
  bool ret = true;
  array<int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  baz obj;
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table[idx] = obj.test();
  }).wait();

  std::vector<int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    if (result[i] != 1) {
      std::cerr << "Verify failed at index: " << i << " , expected: " << 1 << " , actual: " << result[i] << "\n";
      ret = false;
      break;
    }
  }
  return ret;
}

template<int GRID_SIZE>
bool test4() {
  using namespace hc;
  bool ret = true;
  array<int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  parallel_for_each(ex, [&](index<1>& idx) [[hc]] {
    table[idx] = baz::test2();
  }).wait();

  std::vector<int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    if (result[i] != 1) {
      std::cerr << "Verify failed at index: " << i << " , expected: " << 1 << " , actual: " << result[i] << "\n";
      ret = false;
      break;
    }
  }
  return ret;
}

int main() {
  bool ret = true;

  // test with global function
  ret &= test1<64>();

  // test with static function
  ret &= test2<64>();

  // test with member function
  ret &= test3<64>();

  // test with static member function
  ret &= test4<64>();

  return !(ret == true);
}

