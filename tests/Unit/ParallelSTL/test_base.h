#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <iomanip>


#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

// for floating numbers
#define EQ(a,b) ((((a) - (b)) > 0 ? ((a) - (b)) : ((b) - (a))) < 0.001)

// check outputs
// PSTL(std::begin(first), std::end(first))
template <typename T, size_t SIZE>
bool run(std::function<void(T (&)[SIZE],
                            T (&)[SIZE])> f,
                       bool checkOutput=true) {

  // first
  T input1[SIZE] { 0 };
  T input2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 0);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));

  bool ret = true;

  f(input1, input2);
  if (checkOutput) {
    ret &= std::equal(std::begin(input1), std::end(input1), std::begin(input2));
  }

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


// PSTL(std::begin(first), std::end(first), std::begin(d_first))
template <typename T, size_t SIZE>
bool run(std::function<void(T (&)[SIZE], T (&)[SIZE],
                                         T (&)[SIZE])> f,
                       bool checkOutput=true) {

  // first
  T input1[SIZE] { 0 };

  // d_first
  T output1[SIZE] { 0 };
  T output2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);

  bool ret = true;

  f(input1, output1,
            output2);
  if (checkOutput) {
    ret &= std::equal(std::begin(output1), std::end(output1), std::begin(output2));
  }

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


// PSTL(std::begin(first1), std::end(first1), std::begin(first2), std::begin(d_first))
template <typename T, size_t SIZE>
bool run(std::function<void(T (&)[SIZE], T (&)[SIZE], T (&)[SIZE],
                                                      T (&)[SIZE])> f,
                       bool checkOutput=true) {

  // first1
  T input1[SIZE] { 0 };

  // first2
  T input2[SIZE] { 0 };

  // d_first
  T output1[SIZE] { 0 };
  T output2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));

  bool ret = true;

  f(input1, input2, output1,
                    output2);
  if (checkOutput) {
    ret &= std::equal(std::begin(output1), std::end(output1), std::begin(output2));
  }

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
