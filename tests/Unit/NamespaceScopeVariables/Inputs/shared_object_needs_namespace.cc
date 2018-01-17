#include "test_parameters.hpp"

#include <hc.hpp>

#include <algorithm>

bool test_scalar()
{
    using namespace ns;
    using namespace hc;

    array_view<int> read_scalar{1};

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        read_scalar[0] = namespace_scalar;
    });

    if (read_scalar[0] != namespace_scalar) return false;

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        ++namespace_scalar;
    });

    if (namespace_scalar != read_scalar[0] + 1) return false;

    return true;
}

bool test_array()
{
    using namespace hc;
    using namespace ns;
    using namespace std;

    array_view<int> read_array(array_size);

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        for (auto i = 0u; i != array_size; ++i) {
            read_array[i] = namespace_array[i];
        }
    });


    if (!equal(
        namespace_array, namespace_array + array_size, read_array.data())) {
        return false;
    }

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        for (auto&& x : namespace_array) ++x;
    });

    if (!equal(
        namespace_array,
        namespace_array + array_size,
        read_array.data(),
        [](int x, int y) { return x == y + 1; })) {
        return false;
    }

    return true;
}