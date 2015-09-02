#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <iomanip>
#include <array>


#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

template<typename T>
void dump(T &output1, T &output2) {
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
}


// for floating numbers
#define EQ(a,b) ((((a) - (b)) > 0 ? ((a) - (b)) : ((b) - (a))) < 0.001)

template<bool Condition, class T=void>
using EnableIf = typename std::enable_if<Condition, T>::type;

// ArrayRef:    C style array reference by default
//
// T (&)[SIZE]
//   or
// std::array<T, SIZE> &
//
template<typename T, size_t SIZE, typename U>
using ArrayRef = EnableIf<std::is_array<U>::value ||
                          std::is_base_of<std::array<T, SIZE>, U>::value,
                          U> &;

// These functions will generate input data and accept different kinds of
// lambda to run between std::STL and our PSTL implementation
// callers are able to change the input data if needed
//
// PSTL(std::begin(first), std::end(first))
template <typename T, size_t SIZE, typename U=T[SIZE]>
bool run_and_compare(std::function<void(ArrayRef<T, SIZE, U>,
                                        ArrayRef<T, SIZE, U>)> f,
                     bool checkOutput=true) {

  // first
  U input1 { 0 };
  U input2 { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 0);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));

  bool ret = true;

  f(input1, input2);

  if (checkOutput) {
    ret &= std::equal(std::begin(input1), std::end(input1), std::begin(input2));
  }

  dump(input1, input2);

  return ret;
}


// PSTL(std::begin(first), std::end(first), std::begin(d_first))
template <typename T, size_t SIZE, typename U=T[SIZE]>
bool run_and_compare(std::function<void(ArrayRef<T, SIZE, U>,
                                        ArrayRef<T, SIZE, U>,
                                        ArrayRef<T, SIZE, U>)> f,
                     bool checkOutput=true) {

  // first
  U  input1 { 0 };

  // d_first
  U output1 { 0 };
  U output2 { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);

  bool ret = true;

  f(input1, output1, output2);

  if (checkOutput) {
    ret &= std::equal(std::begin(output1), std::end(output1), std::begin(output2));
  }

  dump(output1, output2);

  return ret;
}


// PSTL(std::begin(first1), std::end(first1), std::begin(first2), std::begin(d_first))
template <typename T, size_t SIZE, typename U=T[SIZE]>
bool run_and_compare(std::function<void(ArrayRef<T, SIZE, U>,
                                        ArrayRef<T, SIZE, U>,
                                        ArrayRef<T, SIZE, U>,
                                        ArrayRef<T, SIZE, U>)> f,
                     bool checkOutput=true) {

  // first1
  U  input1 { 0 };

  // first2
  U  input2 { 0 };

  // d_first
  U output1 { 0 };
  U output2 { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 1);
  std::copy(std::begin(input1), std::end(input1), std::begin(input2));

  bool ret = true;

  f(input1, input2, output1, output2);

  if (checkOutput) {
    ret &= std::equal(std::begin(output1), std::end(output1), std::begin(output2));
  }

  dump(output1, output2);

  return ret;
}
