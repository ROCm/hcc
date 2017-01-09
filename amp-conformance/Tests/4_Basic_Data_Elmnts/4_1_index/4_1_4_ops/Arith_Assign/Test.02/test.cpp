// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check binary assignment operators between one index object and a scalar type</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

// scalartype (RHS): 1
int test1() restrict(cpu,amp)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1o(data1);
    index<7> i1;
    int dataa[] = {-99, -9, 0, 1, 2, 11, 101};
    int datas[] = {-101, -11, -2, -1, 0, 9, 99};
    int datam[] = {-100, -10, -1, 0, 1, 10, 100};
    int datad[] = {-100, -10, -1, 0, 1, 10, 100};
    int datar[] = { 0, 0, 0, 0, 0, 0, 0};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);
    index<7> id(datad);
    index<7> ir(datar);

    i1 = i1o;
    i1 += 1;
    if (!(i1 == ia))
    {
        return 11;   // test1 scenario1 failed
    }

    i1 = i1o;
    i1 -= 1;
    if (!(i1 == is))
    {
        return 12;  // test1 scenario2 failed
    }

    i1 = i1o;
    i1 *= 1;
    if (!(i1 == im))
    {
        return 13;  // test1 scenario3 failed
    }

    i1 = i1o;
    i1 /= 1;
    if (!(i1 == id))
    {
        return 14;  // test1 scenario4 failed
    }

    i1 = i1o;
    i1 %= 1;
    if (!(i1 == ir))
    {
        return 15;  // test1 scenario5 failed
    }

    return 0;
}

// scalartype (RHS): -1
int test2() restrict(cpu,amp)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1o(data1);
    index<7> i1;
    int dataa[] = {-101, -11, -2, -1, 0, 9, 99};
    int datas[] = {-99, -9, 0, 1, 2, 11, 101};
    int datam[] = {100, 10, 1, 0, -1, -10, -100};
    int datad[] = {100, 10, 1, 0, -1, -10, -100};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);
    index<7> id(datad);
    index<7> ir;

    i1 = i1o;
    i1 += -1;
    if (!(i1 == ia))
    {
        return 21;  // test2 scenario1 failed
    }

    i1 = i1o;
    i1 -= -1;
    if (!(i1 == is))
    {
        return 22;  // test2 scenario2 failed
    }

    i1 = i1o;
    i1 *= -1;
    if (!(i1 == im))
    {
        return 23;  // test2 scenario3 failed
    }

    i1 = i1o;
    i1 /= -1;
    if (!(i1 == id))
    {
        return 24;  // test2 scenario4 failed
    }

    i1 = i1o;
    i1 %= -1;
    if (!(i1 == ir))
    {
        return 25;  // test2 scenario5 failed
    }

    return 0;
}

// scalartype (RHS): 0
int test3() restrict(cpu,amp)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1o(data1);
    index<7> i1;
    int dataa[] = {-100, -10, -1, 0, 1, 10, 100};
    int datas[] = {-100, -10, -1, 0, 1, 10, 100};
    int datam[] = {0, 0, 0, 0, 0, 0, 0};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);

    i1 = i1o;
    i1 += 0;
    if (!(i1 == ia))
    {
        return 31;  // test3 scenario1 failed
    }

    i1 = i1o;
    i1 -= 0;
    if (!(i1 == is))
    {
        return 32;  // test3 scenario2 failed
    }

    i1 = i1o;
    i1 *= 0;
    if (!(i1 == im))
    {
        return 33;  // test3 scenario3 failed
    }

    return 0;
}

// scalartype (RHS): -9
int test4() restrict(cpu,amp)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1o(data1);
    index<7> i1;
    int dataa[] = {-109, -19, -10, -9, -8, 1, 91};
    int datas[] = {-91, -1, 8, 9, 10, 19, 109};
    int datam[] = {900, 90, 9, 0, -9, -90, -900};
    int datad[] = {11, 1, 0, 0, 0, -1, -11};
    int datar[] = {-1, -1, -1, 0, 1, 1, 1};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);
    index<7> id(datad);
    index<7> ir(datar);

    i1 = i1o;
    i1 += -9;
    if (!(i1 == ia))
    {
        return 41;  // test4 scenario1 failed
    }

    i1 = i1o;
    i1 -= -9;
    if (!(i1 == is))
    {
        return 42;  // test4 scenario2 failed
    }

    i1 = i1o;
    i1 *= -9;
    if (!(i1 == im))
    {
        return 43;  // test4 scenario3 failed
    }

    i1 = i1o;
    i1 /= -9;
    if (!(i1 == id))
    {
        return 44;  // test4 scenario4 failed
    }

    i1 = i1o;
    i1 %= -9;
    if (!(i1 == ir))
    {
        return 45;  // test4 scenario5 failed
    }

    return 0;
}

int test() restrict(cpu,amp)
{
    int result = test1();

    result = (result == 0) ? test2() : result;

    result = (result == 0) ? test3() : result;

    result = (result == 0) ? test4() : result;

    return result;
}

void kernel(array<int, 1>& result) restrict(cpu,amp)
{
    result(0) = test1();
    result(1) = test2();
    result(2) = test3();
    result(3) = test4();
}

const int size = 4;

int test_device()
{
    accelerator device = require_device(device_flags::NOT_SPECIFIED);
    accelerator_view av = device.get_default_view();

    extent<1> e(size);
    array<int, 1> result(e, av);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) restrict(cpu,amp){
        kernel(result);
    });
    presult = result;

    for(int i = 0; i < size; i++)
    {
        if(presult[i] != 0)
        {
            return presult[i];
        }
    }

    return 0;
}

runall_result test_main()
{
    // Test on host
    int host_result = test();
    if(host_result == 0)
    {
        printf("Test passed on host\n");
    }
    else
    {
        printf("Test failed on host. Failing test: %d\n", host_result);
        return runall_fail;
    }

    // Test on device
    int device_result = test_device();
    if(device_result == 0)
    {
        printf("Test passed on device\n");
    }
    else
    {
        printf("Test failed on device. Failed test: %d\n", device_result);
	return runall_fail;
    }

    return runall_pass;
}
