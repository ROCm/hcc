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
    bool ret = eqTypes<short_vector<unsigned int, 1>::type, unsigned int>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unsigned int, 2>::type, uint_2>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unsigned int, 3>::type, uint_3>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unsigned int, 4>::type, uint_4>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<int, 1>::type, int>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<int, 2>::type, int_2>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<int, 3>::type, int_3>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<int, 4>::type, int_4>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<float, 1>::type, float>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<float, 2>::type, float_2>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<float, 3>::type, float_3>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<float, 4>::type, float_4>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unorm, 1>::type, unorm>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unorm, 2>::type, unorm_2>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unorm, 3>::type, unorm_3>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<unorm, 4>::type, unorm_4>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<norm, 1>::type, norm>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<norm, 2>::type, norm_2>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<norm, 3>::type, norm_3>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<norm, 4>::type, norm_4>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<double, 1>::type, double>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<double, 2>::type, double_2>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<double, 3>::type, double_3>();
    assert(ret);
  }

  {
    bool ret = eqTypes<short_vector<double, 4>::type, double_4>();
    assert(ret);
  }

  return 0;
}
