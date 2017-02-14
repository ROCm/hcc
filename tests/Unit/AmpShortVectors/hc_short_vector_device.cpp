// RUN: %hc %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_short_vector.hpp>

#define GRID_SIZE (64)

using namespace hc;
using namespace hc::short_vector;

template<typename T>
bool test_norm() {
    extent<1> ex(GRID_SIZE);
    array_view<int, 1> av(GRID_SIZE);
    parallel_for_each(ex, [=](index<1>& idx) restrict(amp) {
        T val;
        av[idx] = (int)val;
    }).wait();

    av.synchronize();
    return av[0] == 0;
}

template<typename T>
bool test() {
    extent<1> ex(GRID_SIZE);
    array_view<int, 1> av(GRID_SIZE);
    parallel_for_each(ex, [=](index<1>& idx) restrict(amp) {
        T val;
        av[idx] = (int)(val.get_x());
    }).wait();

    av.synchronize();
    return av[0] == 0;
}

int main(void) {
    bool ret = true;

    // scalar 1 types
    ret &= test<char_1>();
    ret &= test<uchar_1>();
    ret &= test<int_1>();
    ret &= test<uint_1>();
    ret &= test<float_1>();
    ret &= test<double_1>();
    ret &= test<short_1>();
    ret &= test<ushort_1>();
    ret &= test<long_1>();
    ret &= test<ulong_1>();
    ret &= test<longlong_1>();
    ret &= test<ulonglong_1>();
    ret &= test<half_1>();

    // scalar 2 types
    ret &= test<char_2>();
    ret &= test<uchar_2>();
    ret &= test<int_2>();
    ret &= test<uint_2>();
    ret &= test<float_2>();
    ret &= test<double_2>();
    ret &= test<short_2>();
    ret &= test<ushort_2>();
    ret &= test<long_2>();
    ret &= test<ulong_2>();
    ret &= test<longlong_2>();
    ret &= test<ulonglong_2>();
    ret &= test<half_2>();

    // scalar 3 types
    ret &= test<char_3>();
    ret &= test<uchar_3>();
    ret &= test<int_3>();
    ret &= test<uint_3>();
    ret &= test<float_3>();
    ret &= test<double_3>();
    ret &= test<short_3>();
    ret &= test<ushort_3>();
    ret &= test<long_3>();
    ret &= test<ulong_3>();
    ret &= test<longlong_3>();
    ret &= test<half_3>();

    // scalar 4 types
    ret &= test<char_4>();
    ret &= test<uchar_4>();
    ret &= test<int_4>();
    ret &= test<uint_4>();
    ret &= test<float_4>();
    ret &= test<double_4>();
    ret &= test<short_4>();
    ret &= test<ushort_4>();
    ret &= test<long_4>();
    ret &= test<ulong_4>();
    ret &= test<longlong_4>();
    ret &= test<half_4>();

    // norm and unorm
    ret &= test_norm<norm>();
    ret &= test_norm<unorm>();

    // norm and unorm in 2D/3D/4D cases
    ret &= test<norm_2>();
    ret &= test<unorm_2>();
    ret &= test<norm_3>();
    ret &= test<unorm_3>();
    ret &= test<norm_4>();
    ret &= test<unorm_4>();

    return !(ret == true);
}
