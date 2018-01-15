// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>test for nested texture / writeonly_texture_view in array. test with const</summary>


#include <amptest.h>
#include <amptest_main.h>
#include <amp_graphics.h>
using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;
using namespace Concurrency::graphics;
runall_result test_main()
{
    const array<const texture<int, 1>, 1> a(16);
    const array<writeonly_texture_view<int, 2>, 1> b(16);

    struct Type1
    {
       texture<float, 1> a;
    };
    const array<Type1, 1> c(16);

    struct Type2
    {
       const texture<float, 3> &a;
    };
    array<Type2, 1> d(16);

    struct Type3
    {
       const writeonly_texture_view<float, 2> a;
    };
    array<Type3, 1> e(16);

    struct Type4
    {
       const writeonly_texture_view<float, 1> a;
       writeonly_texture_view<double, 2> b;
    };
    const array<Type4, 1> f(16);
    return runall_fail;
}

//#Expects: Error: error C2973
//#Expects: Error: test.cpp\(19\)
//#Expects: Error: error C2973
//#Expects: Error: test.cpp\(20\)
//#Expects: Error: error C2973
//#Expects: Error: test.cpp\(26\)
//#Expects: Error: error C2973
//#Expects: Error: test.cpp\(32\)
//#Expects: Error: error C2973
//#Expects: Error: test.cpp\(38\)
//#Expects: Error: error C2973
//#Expects: Error: test.cpp\(45\)
