// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Check that comparing extents of incompatible ranks results in a compilation error </summary>
//#Expects: Error: test.cpp\(29\)
//#Expects: Error: test.cpp\(47\)

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

bool test_equal() __GPU
{
    int flag = 0;

    int data1[] = {-10, -1, 0, 1, 10};
    int data2[] = {-10, -1, 0, 1};

    extent<5> e1(data1);
    extent<4> e2(data2);

    if (e1 == e2)
    {
        flag = 1;
    }

    return false;
}

bool test_not_equal() __GPU
{
    int flag = 0;

    int data1[] = {-10, -1, 0, 1, 10};
    int data2[] = {-100, -11, 1, 11, 100, 7};

    extent<5> e1(data1);
    extent<6> e2(data2);

    if (e1 != e2)
    {
        flag = 1;
    }

    return false;
}

bool test() __GPU
{
    return (test_equal() && test_not_equal());
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    if (!test())
    {
        result[idx] = 1;
    }
}

const int size = 10;

bool test_device()
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

    parallel_for_each(e, [&](index<1> idx) __GPU{
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] == 1)
        {
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    test();
    test_device();

    //Always Fail if we run to completion. this test should fail to compile
    return 1;
}

