// TODO: 10/09/2017 we do not support shared libraries referencing namespace
//       scope variables in main in HC, until further notice.
// XFAIL: *
// RUN: %hc -fPIC -shared %S/Inputs/shared_object_needs_namespace.cc -o %T/libnamespace_to.so
// RUN: %hc %s -L%T -lnamespace_to -o %t.out && LD_LIBRARY_PATH=%T %t.out

#include "Inputs/test_parameters.hpp"

#include <algorithm>
#include <cstdlib>

extern bool test_scalar();
extern bool test_array();

namespace ns
{
    int namespace_scalar;
    int namespace_array[array_size];
}

int main()
{
    using namespace ns;
    using namespace std;

    namespace_scalar = rand();
    generate_n(namespace_array, array_size, rand);

    return (test_scalar() && test_array()) ? EXIT_SUCCESS : EXIT_FAILURE;
}

