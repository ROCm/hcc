// RUN: %hc -fPIC -shared %S/Inputs/shared_object_with_namespace.cc -o %T/libnamespace_from.so
// RUN: %hc %s -L%T -lnamespace_from -o %t.out && LD_LIBRARY_PATH=%T %t.out

#include "Inputs/test_parameters.hpp"

#include <hc.hpp>

#include <algorithm>
#include <cstdlib>

int main()
{
    using namespace hc;
    using namespace ns;
    using namespace std;

    namespace_scalar = rand();
    generate_n(namespace_array, array_size, rand);

    array_view<int> read_scalar{1};
    array_view<int> read_array(array_size);

    parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
        read_scalar[0] = namespace_scalar;
        for (auto i = 0u; i != array_size; ++i) {
            read_array[i] = namespace_array[i];
        }
    });

    if (read_scalar[0] != namespace_scalar) return EXIT_FAILURE;
    if (!equal(
        namespace_array, namespace_array + array_size, read_array.data())) {
        return EXIT_FAILURE;
    }

    #if false // 10/09/2017 - GPU writes to globals are not correctly observed.
        parallel_for_each(hc::extent<1>{1}, [=](index<1>) [[hc]] {
            ++namespace_scalar;
            for (auto&& x : namespace_array) ++x;
        }).wait();

        if (namespace_scalar != read_scalar[0] + 1) return EXIT_FAILURE;
        if (!equal(
            namespace_array,
            namespace_array + array_size,
            read_array.data(),
            [](int x, int y) { return x == y + 1; })) {
            return EXIT_FAILURE;
        }
    #endif

    return EXIT_SUCCESS;
}