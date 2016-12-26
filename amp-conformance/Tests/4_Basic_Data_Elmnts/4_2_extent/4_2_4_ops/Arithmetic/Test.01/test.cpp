// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check binary arithmetic operators between one extent and a scalar type</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

// scalartype (RHS): 1
int test1() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1(data1);
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

    if (!((e1 + 1) == ea))
    {
        return 11;
    }

    if (!((e1 - 1) == es))
    {
        return 12;
    }

    if (!((e1 * 1) == em))
    {
        return 13;
    }

    if (!((e1 / 1) == ed))
    {
        return 14;
    }

	if (!((e1 % 1) == er))
    {
        return 15;
    }

    return 0;
}

// scalartype (RHS): -1
int test2() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1(data1);
    int dataa[] = {-101, -11, -2, -1, 0, 9, 99};
    int datas[] = {-99, -9, 0, 1, 2, 11, 101};
    int datam[] = {100, 10, 1, 0, -1, -10, -100};
    int datad[] = {100, 10, 1, 0, -1, -10, -100};
	int datar[] = {0, 0, 0, 0, 0, 0, 0};
    extent<7> ea(dataa);
    extent<7> es(datas);
    extent<7> em(datam);
    extent<7> ed(datad);
	extent<7> er(datar);

    if (!((e1 - 1) == ea))
    {
        return 21;
    }

    if (!((e1 + 1) == es))
    {
        return 22;
    }

    if (!((e1 * -1) == em))
    {
        return 23;
    }

    if (!((e1 / -1) == ed))
    {
        return 24;
    }

	if (!((e1 % -1) == er))
    {
        return 25;
    }

    return 0;
}

// scalartype (RHS): 0
int test3() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<7> e1(data1);
    int dataa[] = {-100, -10, -1, 0, 1, 10, 100};
    int datas[] = {-100, -10, -1, 0, 1, 10, 100};
    int datam[] = {0, 0, 0, 0, 0, 0, 0};
    extent<7> ea(dataa);
    extent<7> es(datas);
    extent<7> em(datam);

    if (!((e1 + 0) == ea))
    {
        return 31;
    }

    if (!((e1 - 0) == es))
    {
        return 32;
    }

    if (!((e1 * 0) == em))
    {
        return 33;
    }

    return 0;
}

// scalartype (RHS): -9
int test4() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    extent<6> e1(data1);
    int dataa[] = {-109, -19, -10, -9, -8, 1, 91};
    int datas[] = {-91, -1, 8, 9, 10, 19, 109};
    int datam[] = {900, 90, 9, 0, -9, -90, -900};
    int datad[] = {11, 1, 0, 0, 0, -1, -11};
	int datar[] = {-1, -1, -1, 0, 1, 1, 1};
    extent<6> ea(dataa);
    extent<6> es(datas);
    extent<6> em(datam);
    extent<6> ed(datad);
	extent<6> er(datar);

    if (!((e1 - 9) == ea))
    {
        return 41;
    }

    if (!((e1 + 9) == es))
    {
        return 42;
    }

    if (!((e1 * -9) == em))
    {
        return 43;
    }

    if (!((e1 / -9) == ed))
    {
        return 44;
    }

	if (!((e1 % -9) == er))
    {
        return 45;
    }

    return 0;
}

// scalartype (LHS): 1
int test5() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    extent<6> e1(data1);
    int dataa[] = {-99, -9, 0, 2, 11, 101};
    int datas[] = {101, 11, 2, 0, -9, -99};
    int datam[] = {-100, -10, -1, 1, 10, 100};
    int datad[] = {0, 0, -1, 1, 0, 0};
	int datar[] = {1, 1, 0, 0, 1, 1};
    extent<6> ea(dataa);
    extent<6> es(datas);
    extent<6> em(datam);
    extent<6> ed(datad);
	extent<6> er(datar);

    if (!((1 + e1) == ea))
    {
        return 51;
    }

    if (!((1 - e1) == es))
    {
        return 52;
    }

    if (!((1 * e1) == em))
    {
        return 53;
    }

    if (!((1 / e1) == ed))
    {
        return 54;
    }

	if (!((1 % e1) == er))
    {
        return 55;
    }

    return 0;
}

// scalartype (LHS): -1
int test6() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    extent<6> e1(data1);
    int dataa[] = {-101, -11, -2, 0, 9, 99};
    int datas[] = {99, 9, 0, -2, -11, -101};
    int datam[] = {100, 10, 1, -1, -10, -100};
    int datad[] = {0, 0, 1, -1, 0, 0};
	int datar[] = {-1, -1, 0, 0, -1, -1};
    extent<6> ea(dataa);
    extent<6> es(datas);
    extent<6> em(datam);
    extent<6> ed(datad);
	extent<6> er(datar);

    if (!(((-1) + e1) == ea))
    {
        return 61;
    }

    if (!(((-1) - e1) == es))
    {
        return 62;
    }

    if (!(((-1) * e1) == em))
    {
        return 63;
    }

    if (!(((-1) / e1) == ed))
    {
        return 64;
    }

	if (!(((-1) % e1) == er))
    {
        return 65;
    }

    return 0;
}

// scalartype (LHS): 0
int test7() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    extent<6> e1(data1);
    int dataa[] = {-100, -10, -1, 1, 10, 100};
    int datas[] = {100, 10, 1, -1, -10, -100};
    int datam[] = {0, 0, 0, 0, 0, 0};
    extent<6> ea(dataa);
    extent<6> es(datas);
    extent<6> em(datam);
	extent<6> ed;
	extent<6> er;

    if (!((0 + e1) == ea))
    {
        return 71;
    }

    if (!((0 - e1) == es))
    {
        return 72;
    }

    if (!((0 * e1) == em))
    {
        return 73;
    }

	if (!((0 / e1) == ed))
    {
        return 74;
    }

	if (!((0 % e1) == er))
    {
        return 75;
    }
    return 0;
}

// scalartype (LHS): -9
int test8() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    extent<6> e1(data1);
    int dataa[] = {-109, -19, -10, -8, 1, 91};
    int datas[] = {91, 1, -8, -10, -19, -109};
    int datam[] = {900, 90, 9, -9, -90, -900};
    int datad[] = {0, 0, 9, -9, 0, 0};
	int datar[] = {-9, -9, 0, 0, -9, -9};
    extent<6> ea(dataa);
    extent<6> es(datas);
    extent<6> em(datam);
    extent<6> ed(datad);
	extent<6> er(datar);

    if (!(((-9) + e1) == ea))
    {
        return 81;
    }

    if (!(((-9) -  e1) == es))
    {
        return 82;
    }

    if (!(((-9) * e1) == em))
    {
        return 83;
    }

    if (!(((-9) / e1) == ed))
    {
        return 84;
    }

	if (!(((-9) % e1) == er))
    {
        return 85;
    }

    return 0;
}

int test() restrict(amp,cpu)
{
    int result = test1();

    result = (result == 0) ? test2() : result;

    result = (result == 0) ? test3() : result;

    result = (result == 0) ? test4() : result;

    result = (result == 0) ? test4() : result;

    result = (result == 0) ? test5() : result;

    result = (result == 0) ? test6() : result;

    result = (result == 0) ? test7() : result;

    result = (result == 0) ? test8() : result;

    return result;
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

    parallel_for_each(e, [&](index<1> idx) restrict(amp,cpu) {
            kernel(idx, result);
    });

    vector<int> presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] != 0)
        {
            printf("Test failed. Return code: %d\n", presult[i]);
            return 1;
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

