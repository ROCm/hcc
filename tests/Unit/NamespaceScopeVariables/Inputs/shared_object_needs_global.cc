#include "test_parameters.hpp"

#include <hc.hpp>

#include <algorithm>

bool test_scalar()
{
    using namespace hc;

    array_view<int> read_scalar{1};

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        read_scalar[0] = global_scalar;
    });

    if (read_scalar[0] != global_scalar) return false;

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        ++global_scalar;
    });

    if (global_scalar != read_scalar[0] + 1) return false;

    return true;
}

bool test_array()
{
    using namespace hc;
    using namespace std;

    array_view<int> read_array(array_size);

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        for (auto i = 0u; i != array_size; ++i) read_array[i] = global_array[i];
    });


    if (!equal(global_array, global_array + array_size, read_array.data())) {
        return false;
    }

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        for (auto&& x : global_array) ++x;
    });

    if (!equal(
        global_array,
        global_array + array_size,
        read_array.data(),
        [](int x, int y) { return x == y + 1; })) {
        return false;
    }

    return true;
}