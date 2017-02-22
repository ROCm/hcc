
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>

#define TEST_SIZE (100)

template<typename _Tp, size_t SIZE>
bool test(void) {
  int x = 0;
  bool ret = true;
  int N = TEST_SIZE;

  std::vector<_Tp> v1(SIZE);
  std::vector<_Tp> v2(SIZE);
  std::generate(v1.begin(), v1.end(), [&x]{ x++; return x; });
  std::copy(v1.begin(), v1.end(), v2.begin());

  std::srand(std::time(0));

  // run N times
  while (N--) {
    bool a = std::lexicographical_compare(v1.begin(), v1.end(),
                                          v2.begin(), v2.end());

    bool b = std::lexicographical_compare(v1.begin(), v1.end(),
                                          v2.begin(), v2.end(),
                                          std::greater<_Tp>());

    using namespace std::experimental::parallel;
    // parallel STL lexicographical_compare
    bool c = lexicographical_compare(par, v1.begin(), v1.end(),
                                     v2.begin(), v2.end());

    bool d = lexicographical_compare(par, v1.begin(), v1.end(),
                                     v2.begin(), v2.end(),
                                     std::greater<_Tp>());

    ret &= (a == c);
    ret &= (b == d);

    std::random_shuffle(v1.begin(), v1.end());
    std::random_shuffle(v2.begin(), v2.end());
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<char, TEST_SIZE>();
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  return !(ret == true);
}

