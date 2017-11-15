// TODO: 10/09/2017 we do not support shared libraries referencing globals in
//       main in HC, until further notice.
// XFAIL: *
// RUN: %hc -fPIC -shared %S/Inputs/shared_object_needs_global.cc -o %T/libglobal_to.so
// RUN: %hc %s -L%T -lglobal_to -o %t.out && LD_LIBRARY_PATH=%T %t.out

#include "Inputs/test_parameters.hpp"

#include <algorithm>
#include <cstdlib>

extern bool test_scalar();
extern bool test_array();

int global_scalar;
int global_array[array_size];

int main()
{
    using namespace std;

    global_scalar = rand();
    generate_n(global_array, array_size, rand);

    return (test_scalar() && test_array()) ? EXIT_SUCCESS : EXIT_FAILURE;
}

