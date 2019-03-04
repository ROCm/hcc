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
    parallel_for_each(ex, [=](hc::index<1>& idx) [[hc]] {
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
    parallel_for_each(ex, [=](hc::index<1>& idx) [[hc]] {
        T val;
        av[idx] = (int)(val.get_x());
    }).wait();

    av.synchronize();
    return av[0] == 0;
}

int main(void) {
    bool ret = true;

    ret &= test<hc::short_vector::short_vector<double,1>::type>();
    ret &= test<hc::short_vector::short_vector<int,2>::type>();
    ret &= test<hc::short_vector::short_vector<unsigned int,3>::type>();
    ret &= test<hc::short_vector::short_vector<float,4>::type>();

    return !(ret == true);
}
