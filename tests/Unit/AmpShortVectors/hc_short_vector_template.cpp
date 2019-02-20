// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_short_vector.hpp>
#include <type_traits>

using namespace hc;
using namespace hc::short_vector;

int main(void) {

  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned char, 1>::type, uchar_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned char, 2>::type, uchar_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned char, 3>::type, uchar_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned char, 4>::type, uchar_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<char, 1>::type, char_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<char, 2>::type, char_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<char, 3>::type, char_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<char, 4>::type, char_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned short, 1>::type, ushort_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned short, 2>::type, ushort_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned short, 3>::type, ushort_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned short, 4>::type, ushort_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<short, 1>::type, short_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<short, 2>::type, short_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<short, 3>::type, short_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<short, 4>::type, short_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned int, 1>::type, uint_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned int, 2>::type, uint_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned int, 3>::type, uint_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<unsigned int, 4>::type, uint_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<int, 1>::type, int_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<int, 2>::type, int_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<int, 3>::type, int_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<int, 4>::type, int_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<float, 1>::type, float_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<float, 2>::type, float_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<float, 3>::type, float_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<float, 4>::type, float_4>::value,
                "Mismatched vector types!");

  static_assert(std::is_same<typename hc::short_vector::short_vector<double, 1>::type, double_1>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<double, 2>::type, double_2>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<double, 3>::type, double_3>::value,
                "Mismatched vector types!");
  static_assert(std::is_same<typename hc::short_vector::short_vector<double, 4>::type, double_4>::value,
                "Mismatched vector types!");

  return 0;
}
