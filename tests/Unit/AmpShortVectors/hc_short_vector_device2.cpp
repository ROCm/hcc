// RUN: %hc %s -o %t.out && %t.out

#include <algorithm>
#include <hc.hpp>
#include <hc_short_vector.hpp>

using namespace hc;
using namespace hc::short_vector;

template <typename T>
class add {
public:
  T operator()(T& a, T& b) [[hc]] { return a+b; }
};

template <typename T>
class sub {
public:
  T operator()(T& a, T& b) [[hc]] { return a-b; }
};

template <typename T>
class mul {
public:
  T operator()(T& a, T& b) [[hc]] { return a*b; }
};

template <typename T>
class divide {
public:
  T operator()(T& a, T& b) [[hc]] { return a/b; }
};

template<typename VECTOR_TYPE, template<typename> class OPERATOR, int N>
int test() {

  // initialize the input data
  std::vector<typename VECTOR_TYPE::value_type> in1(N * VECTOR_TYPE::size);
  std::vector<typename VECTOR_TYPE::value_type> in2(N * VECTOR_TYPE::size);
  int value = 0;
  for(auto&& d : in1)
    d = static_cast<typename VECTOR_TYPE::value_type>(value++);
  value = 10;
  for(auto&& d : in2)
    d = static_cast<typename VECTOR_TYPE::value_type>(value++);

  // generate output using the short vector type operator
  hc::array_view<VECTOR_TYPE, 1> av_vector_in1(N, reinterpret_cast<VECTOR_TYPE*>(in1.data()));
  hc::array_view<VECTOR_TYPE, 1> av_vector_in2(N, reinterpret_cast<VECTOR_TYPE*>(in2.data()));
  hc::array_view<VECTOR_TYPE, 1> av_vector_out(N);
  hc::parallel_for_each(av_vector_in1.get_extent().tile(64), [=](hc::tiled_index<1> i) [[hc]] {
    auto idx = i.global[0];
    av_vector_out[idx] = OPERATOR<VECTOR_TYPE>()(av_vector_in1[idx], av_vector_in2[idx]);
  });

  // generate expected output using the scalar type operator
  hc::array_view<typename VECTOR_TYPE::value_type, 1> av_scalar_in1(in1.size(), in1.data());
  hc::array_view<typename VECTOR_TYPE::value_type, 1> av_scalar_in2(in1.size(), in2.data());
  hc::array_view<typename VECTOR_TYPE::value_type, 1> av_scalar_out(in1.size());
  hc::parallel_for_each(av_scalar_in1.get_extent().tile(64), [=](hc::tiled_index<1> i) [[hc]] {
    auto idx = i.global[0];
    av_scalar_out[idx] = OPERATOR<typename VECTOR_TYPE::value_type>()(av_scalar_in1[idx], av_scalar_in2[idx]);
  });
 
  // compare the results
  typename VECTOR_TYPE::value_type* vector_out = reinterpret_cast<typename VECTOR_TYPE::value_type*>(av_vector_out.data());
  typename VECTOR_TYPE::value_type* scalar_out = reinterpret_cast<typename VECTOR_TYPE::value_type*>(av_scalar_out.data());
  return std::equal(scalar_out, scalar_out + in1.size(), vector_out)?0:1;
}

template <typename VECTOR_TYPE, int N>
int run_tests() {
  int errors = 0;
  errors += test<VECTOR_TYPE, add, N>();
  errors += test<VECTOR_TYPE, sub, N>();
  errors += test<VECTOR_TYPE, mul, N>();
  errors += test<VECTOR_TYPE, divide, N>();
  return errors;
}


int main() {

  int errors = 0;

  #if 0
  errors += run_tests<short1,1024>();
  errors += run_tests<short2,1024>();
  errors += run_tests<short4,1024>();
  errors += run_tests<short8,1024>();
  #endif

  errors += run_tests<int1,1024>();
  errors += run_tests<int2,1024>();
  errors += run_tests<int4,1024>();
  errors += run_tests<int8,1024>();

  errors += run_tests<uint1,1024>();
  errors += run_tests<uint2,1024>();
  errors += run_tests<uint4,1024>();
  errors += run_tests<uint8,1024>();

  errors += run_tests<long1,1024>();
  errors += run_tests<long2,1024>();
  errors += run_tests<long4,1024>();
  errors += run_tests<long8,1024>();

  errors += run_tests<ulong1,1024>();
  errors += run_tests<ulong2,1024>();
  errors += run_tests<ulong4,1024>();
  errors += run_tests<ulong8,1024>();

  errors += run_tests<half1,1024>();
  errors += run_tests<half2,1024>();
  errors += run_tests<half4,1024>();
  errors += run_tests<half8,1024>();

  errors += run_tests<float1,1024>();
  errors += run_tests<float2,1024>();
  errors += run_tests<float4,1024>();
  errors += run_tests<float8,1024>();

  errors += run_tests<double1,1024>();
  errors += run_tests<double2,1024>();
  errors += run_tests<double4,1024>();
  errors += run_tests<double8,1024>();

  return errors;
}
