// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check binary assignment operators between one extent and a scalar type</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

using std::vector;

// scalartype (RHS): 1
int test1() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1o(data1);
    extent<7> e1;
    int dataa[] = {-99, -9, 0, 1, 2, 11, 101};
    int datas[] = {-101, -11, -2, -1, 0, 9, 99};
    int datam[] = {-100, -10, -1, 0, 1, 10, 100};
    int datad[] = {-100, -10, -1, 0, 1, 10, 100};
	int datar[] = {0, 0, 0, 0, 0, 0, 0};
    extent<7> ea(dataa);
    extent<7> es(datas);
    extent<7> em(datam);
    extent<7> ed(datad);
	extent<7> er(datar);

    e1 = e1o;
    e1 += 1;
    if (!(e1 == ea))
    {
        return 11;
    }

    e1 = e1o;
    e1 -= 1;
    if (!(e1 == es))
    {
        return 12;
    }

    e1 = e1o;
    e1 *= 1;
    if (!(e1 == em))
    {
        return 13;
    }

    e1 = e1o;
    e1 /= 1;
    if (!(e1 == ed))
    {
        return 14;
    }

	e1 = e1o;
    e1 %= 1;
    if (!(e1 == er))
    {
        return 15;
    }

    return 0;
}

// scalartype (RHS): -1
int test2() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1o(data1);
    extent<7> e1;
    int dataa[] = {-101, -11, -2, -1, 0, 9, 99};
    int datas[] = {-99, -9, 0, 1, 2, 11, 101};
    int datam[] = {100, 10, 1, 0, -1, -10, -100};
    int datad[] = {100, 10, 1, 0, -1, -10, -100};
    extent<7> ea(dataa);
    extent<7> es(datas);
    extent<7> em(datam);
    extent<7> ed(datad);
	extent<7> er;

    e1 = e1o;
    e1 += -1;
    if (!(e1 == ea))
    {
        return 21;
    }

    e1 = e1o;
    e1 -= -1;
    if (!(e1 == es))
    {
        return 22;
    }

    e1 = e1o;
    e1 *= -1;
    if (!(e1 == em))
    {
        return 23;
    }

    e1 = e1o;
    e1 /= -1;
    if (!(e1 == ed))
    {
        return 24;
    }

	e1 = e1o;
    e1 %= -1;
    if (!(e1 == er))
    {
        return 25;
    }

    return 0;
}

// scalartype (RHS): 0
int test3() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1o(data1);
    extent<7> e1;
    int dataa[] = {-100, -10, -1, 0, 1, 10, 100};
    int datas[] = {-100, -10, -1, 0, 1, 10, 100};
    int datam[] = {0, 0, 0, 0, 0, 0, 0};
    extent<7> ea(dataa);
    extent<7> es(datas);
    extent<7> em(datam);

    e1 = e1o;
    e1 += 0;
    if (!(e1 == ea))
    {
        return 31;
    }

    e1 = e1o;
    e1 -= 0;
    if (!(e1 == es))
    {
        return 32;
    }

    e1 = e1o;
    e1 *= 0;
    if (!(e1 == em))
    {
        return 33;
    }

    return 0;
}

// scalartype (RHS): -9
int test4() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1o(data1);
    extent<7> e1;
    int dataa[] = {-109, -19, -10, -9, -8, 1, 91};
    int datas[] = {-91, -1, 8, 9, 10, 19, 109};
    int datam[] = {900, 90, 9, 0, -9, -90, -900};
    int datad[] = {11, 1, 0, 0, 0, -1, -11};
	int datar[] = {-1, -1, -1, 0, 1, 1, 1};
    extent<7> ea(dataa);
    extent<7> es(datas);
    extent<7> em(datam);
    extent<7> ed(datad);
	extent<7> er(datar);

    e1 = e1o;
    e1 += -9;
    if (!(e1 == ea))
    {
        return 41;
    }

    e1 = e1o;
    e1 -= -9;
    if (!(e1 == es))
    {
        return 42;
    }

    e1 = e1o;
    e1 *= -9;
    if (!(e1 == em))
    {
        return 43;
    }

    e1 = e1o;
    e1 /= -9;
    if (!(e1 == ed))
    {
        return 44;
    }

	e1 = e1o;
    e1 %= -9;
    if (!(e1 == er))
    {
        return 45;
    }

    return 0;
}

int test() restrict(amp,cpu)
{
    int result = test1();
    if(result != 0)
    {
        return result;
    }

    result = test2();
    if(result != 0)
    {
        return result;
    }

    result = test3();
    if(result != 0)
    {
        return result;
    }

    return test4();
}

void kernel(index<1>& idx, array<int, 1>& result) restrict(amp,cpu)
{
    result[idx] = test();
}

const int size = 10;

int test_device()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    extent<1> e(size);
    array<int, 1> result(e, av);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) restrict(amp,cpu) {
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] != 0)
        {
            int ret = presult[i];
            return ret;
        }
    }

    return 0;
}

runall_result test_main()
{
    runall_result result;

	result &= REPORT_RESULT(test());
	result &= REPORT_RESULT(test_device());

    return result;
}

