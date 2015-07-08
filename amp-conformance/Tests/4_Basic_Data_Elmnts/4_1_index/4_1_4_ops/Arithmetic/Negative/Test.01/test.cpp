// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that applying the operator on 2 index of incompatible ranks rerults in a compilation error</summary>
//#Expects: Error: test.cpp\(33\)
//#Expects: Error: test.cpp\(34\)
//#Expects: Error: test.cpp\(46\)
//#Expects: Error: test.cpp\(47\)
//#Expects: Error: test.cpp\(59\)
//#Expects: Error: test.cpp\(60\)
//#Expects: Error: test.cpp\(72\)
//#Expects: Error: test.cpp\(73\)


#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

bool test1() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<1> i1;
    index<2> i2;
    index<1> ir;

    ir = i1 + i2;
    ir = i1 - i2;

    return false;
}

bool test2() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<3> i1;
    index<4> i2;
    index<3> ir;

    ir = i1 + i2;
    ir = i1 - i2;

    return false;
}

bool test3() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<4> i1;
    index<5> i2;
    index<4> ir;

    ir = i1 + i2;
    ir = i1 - i2;

    return false;
}

bool test4() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    index<10> i1;
    index<11> i2;
    index<10> ir;

    ir = i1 + i2;
    ir = i1 - i2;

    return false;
}

bool test() __GPU
{
    return (test1() && test2() && test3() && test4());
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    if (!test())
    {
        result[idx] = 1;
    }
}

const int size = 10;

int test_device()
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return 2;
    }
    accelerator_view av = device.get_default_view();

    extent<1> e(size);
    array<int, 1> result(e, av);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) __GPU {
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] == 1)
        {
            return 1;
        }
    }

    return 0;
}

int main()
{
    test();
    test_device();

    // always fail. this test is expected to fail compilation
    return 1;
}

