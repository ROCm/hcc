// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Check that applying the operator on index objects of incompatible ranks results in a compilation error</summary>
//#Expects: Error: test.cpp\(40\) : error C2679
//#Expects: Error: test.cpp\(40\) : error C2088
//#Expects: Error: test.cpp\(41\) : error C2679
//#Expects: Error: test.cpp\(41\) : error C2088
//#Expects: Error: test.cpp\(52\) : error C2679
//#Expects: Error: test.cpp\(52\) : error C2088
//#Expects: Error: test.cpp\(53\) : error C2679
//#Expects: Error: test.cpp\(53\) : error C2088
//#Expects: Error: test.cpp\(64\) : error C2679
//#Expects: Error: test.cpp\(64\) : error C2088
//#Expects: Error: test.cpp\(65\) : error C2679
//#Expects: Error: test.cpp\(65\) : error C2088
//#Expects: Error: test.cpp\(76\) : error C2679
//#Expects: Error: test.cpp\(76\) : error C2088
//#Expects: Error: test.cpp\(77\) : error C2679
//#Expects: Error: test.cpp\(77\) : error C2088

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;


bool test1() restrict(cpu,amp)
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<1> i1;
    index<2> i2;

    i1 += i2;
    i1 -= i2;

    return false;
}

bool test2() restrict(cpu,amp)
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<3> i1;
    index<4> i2;

    i1 += i2;
    i1 -= i2;

    return false;
}

bool test3() restrict(cpu,amp)
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<4> i1;
    index<5> i2;

    i1 += i2;
    i1 -= i2;

    return false;
}

bool test4() restrict(cpu,amp)
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<10> i1;
    index<11> i2;

    i1 += i2;
    i1 -= i2;

    return false;
}

runall_result test_main()
{
    //Always fail if this runs to completion. Test is expected to fail compilation
    return runall_fail;
}

