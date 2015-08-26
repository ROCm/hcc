#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <iomanip>


#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

// f(first, last) => f(first, first+SIZE)
template <typename _Tp, size_t SIZE>
bool run(std::function<void(_Tp *,_Tp *)> f) {

  // first
  _Tp input1[SIZE] { 0 };
  _Tp input2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));

  bool ret = true;

  f(input1, input2);
  ret &= std::equal(std::begin(input1), std::end(input1), std::begin(input2));

#if _DEBUG
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << input1[i * COL + j];
    }
    std::cout << "\n";
  }

  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << input2[i * COL + j];
    }
    std::cout << "\n";
  }
#endif

  return ret;
}


// f(first, last, d_first) => f(first, first+SIZE, d_first)
template <typename _Tp, size_t SIZE>
bool run(std::function<void(_Tp *, _Tp *,
                            _Tp *, _Tp *)> f) {

  // first
  _Tp input1[SIZE] { 0 };
  _Tp input2[SIZE] { 0 };

  // d_first
  _Tp output1[SIZE] { 0 };
  _Tp output2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));

  bool ret = true;

  f(input1, output1,
    input2, output2);
  ret &= std::equal(std::begin(output1), std::end(output1), std::begin(output2));

#if _DEBUG
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << output1[i * COL + j];
    }
    std::cout << "\n";
  }

  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << output2[i * COL + j];
    }
    std::cout << "\n";
  }
#endif

  return ret;
}


// f(first1, last1, first2, d_first) => f(first1, first1+SIZE, first2, d_first)
template <typename _Tp, size_t SIZE>
bool run(std::function<void(_Tp *, _Tp *, _Tp *,
                            _Tp *, _Tp *, _Tp *)> f) {

  // first1
  _Tp input1[SIZE] { 0 };
  _Tp input2[SIZE] { 0 };
  // first2
  _Tp input3[SIZE] { 0 };
  _Tp input4[SIZE] { 0 };
  // d_first
  _Tp output1[SIZE] { 0 };
  _Tp output2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));
  std::copy(std::begin(input1), std::end(input1), std::begin(input3));
  std::copy(std::begin(input1), std::end(input1), std::begin(input4));

  bool ret = true;

  f(input1, input3, output1,
    input2, input4, output2);
  ret &= std::equal(std::begin(output1), std::end(output1), std::begin(output2));

#if _DEBUG
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << output1[i * COL + j];
    }
    std::cout << "\n";
  }

  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << output2[i * COL + j];
    }
    std::cout << "\n";
  }
#endif

  return ret;
}
