// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that applying the operator on extents of incompatible ranks rerults in a compilation error</summary>
//#Expects: Error: test.cpp\(42\)
//#Expects: Error: test.cpp\(43\)
//#Expects: Error: test.cpp\(44\)
//#Expects: Error: test.cpp\(45\)
//#Expects: Error: test.cpp\(57\)
//#Expects: Error: test.cpp\(58\)
//#Expects: Error: test.cpp\(59\)
//#Expects: Error: test.cpp\(60\)
//#Expects: Error: test.cpp\(72\)
//#Expects: Error: test.cpp\(73\)
//#Expects: Error: test.cpp\(74\)
//#Expects: Error: test.cpp\(75\)
//#Expects: Error: test.cpp\(87\)
//#Expects: Error: test.cpp\(88\)
//#Expects: Error: test.cpp\(89\)
//#Expects: Error: test.cpp\(90\)


#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;


int test1() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    extent<1> e1;
    extent<2> e2;
    extent<1> er;

    er = e1 + e2;
    er = e1 - e2;
    er = e1 * e2;
    er = e1 / e2;

    return 1;
}

int test2() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    extent<3> e1;
    extent<4> e2;
    extent<3> er;

    er = e1 + e2;
    er = e1 - e2;
    er = e1 * e2;
    er = e1 / e2;

    return 1;
}

int test3() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    extent<4> e1;
    extent<5> e2;
    extent<4> er;

    er = e1 + e2;
    er = e1 - e2;
    er = e1 * e2;
    er = e1 / e2;

    return 1;
}

int test4() __GPU
{
    int data[] = {200, 100, 2000, 0, -100, -10, -1, 0,  1,  10, 100};
    extent<10> e1;
    extent<11> e2;
    extent<10> er;

    er = e1 + e2;
    er = e1 - e2;
    er = e1 * e2;
    er = e1 / e2;

    return 1;
}

int test() __GPU
{
    int result = test1();

    result = (result == 0) ? test2() : result;

    result = (result == 0) ? test3() : result;

    result = (result == 0) ? test4() : result;

    return result;
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    result[idx] = test();
}

const int size = 10;

int test_device()
{
    accelerator acc;
    if (!get_device(Device::ALL_DEVICES, acc))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }

    accelerator_view av = acc.get_default_view();
    extent<1> e(size);
    array<int, 1> result(e, av);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) __GPU {
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] != 0)
        {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    int result = test();

    printf("Test %s on host\n", ((result == 0) ? "passed" : "failed"));
    if(result != 0) return result;

    result = test_device();
    printf("Test %s on device\n", ((result == 0) ? "passed" : "failed"));
    return result;
}

