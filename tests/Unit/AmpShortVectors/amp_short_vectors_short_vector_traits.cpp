// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_short_vectors.h>

using namespace concurrency;
using namespace concurrency::graphics;

// type trait and helper function
template<typename T, typename U> struct is_same
{
  static const bool result = false;
};

template<typename T> struct is_same<T, T>
{
  static const bool result = true;
};

template<typename T, typename U> 
bool eqTypes() { return is_same<T, U>::result; }

int main(void) {

  {
    bool ret1 = eqTypes<short_vector_traits<unsigned int>::value_type, 
                         unsigned int>();
    bool ret2 = short_vector_traits<unsigned int>::size == 1;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<uint_2>::value_type, 
                         unsigned int>();
    bool ret2 = short_vector_traits<uint_2>::size == 2;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<uint_3>::value_type, 
                         unsigned int>();
    bool ret2 = short_vector_traits<uint_3>::size == 3;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<uint_4>::value_type, 
                         unsigned int>();
    bool ret2 = short_vector_traits<uint_4>::size == 4;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<int>::value_type, int>();
    bool ret2 = short_vector_traits<int>::size == 1;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<int_2>::value_type, int>();
    bool ret2 = short_vector_traits<int_2>::size == 2;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<int_3>::value_type, int>();
    bool ret2 = short_vector_traits<int_3>::size == 3;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<int_4>::value_type, int>();
    bool ret2 = short_vector_traits<int_4>::size == 4;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<float>::value_type, float>();
    bool ret2 = short_vector_traits<float>::size == 1;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<float_2>::value_type, float>();
    bool ret2 = short_vector_traits<float_2>::size == 2;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<float_3>::value_type, float>();
    bool ret2 = short_vector_traits<float_3>::size == 3;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<float_4>::value_type, float>();
    bool ret2 = short_vector_traits<float_4>::size == 4;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<unorm>::value_type, unorm>();
    bool ret2 = short_vector_traits<unorm>::size == 1;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<unorm_2>::value_type, unorm>();
    bool ret2 = short_vector_traits<unorm_2>::size == 2;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<unorm_3>::value_type, unorm>();
    bool ret2 = short_vector_traits<unorm_3>::size == 3;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<unorm_4>::value_type, unorm>();
    bool ret2 = short_vector_traits<unorm_4>::size == 4;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<norm>::value_type, norm>();
    bool ret2 = short_vector_traits<norm>::size == 1;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<norm_2>::value_type, norm>();
    bool ret2 = short_vector_traits<norm_2>::size == 2;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<norm_3>::value_type, norm>();
    bool ret2 = short_vector_traits<norm_3>::size == 3;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<norm_4>::value_type, norm>();
    bool ret2 = short_vector_traits<norm_4>::size == 4;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<double>::value_type, double>();
    bool ret2 = short_vector_traits<double>::size == 1;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<double_2>::value_type, double>();
    bool ret2 = short_vector_traits<double_2>::size == 2;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<double_3>::value_type, double>();
    bool ret2 = short_vector_traits<double_3>::size == 3;
    assert(ret1 && ret2);
  }

  {
    bool ret1 = eqTypes<short_vector_traits<double_4>::value_type, double>();
    bool ret2 = short_vector_traits<double_4>::size == 4;
    assert(ret1 && ret2);
  }

  return 0;
}
