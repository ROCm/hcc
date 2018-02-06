// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check binary arithmetic operators +,-,+=,-= between extent and index</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

template <int R>
void fill(int arr[], int val) restrict(amp,cpu)
{
    for (int i = 0; i < R; ++i)
	    arr[i] = val;
}

//extent + index
// extent - index
// extent += index
// extent -= index
template <int R>
int test1() restrict(amp,cpu)
{
    int vIndex[R];
	fill<R>(vIndex, 6);
	index<R> idx(vIndex);

	int vExtent[R];
	fill<R>(vExtent, 25);
	extent<R> exOrig(vExtent);
	extent<R> exAddend(exOrig);
	extent<R> exMinuend(exOrig);


	int vAddResult[R];
	fill<R>(vAddResult, 25 + 6);
	extent<R> addResult(vAddResult);

	int vSubResult[R];
	fill<R>(vSubResult, 25 - 6);
	extent<R> subResult(vSubResult);

	exAddend += idx;
	if (exAddend != addResult)
	{
		return 11*R;
	}

	exMinuend -= idx;
	if (exMinuend != subResult)
	{
		return 12*R;
	}

	exAddend = exOrig;
	if ((exAddend + idx) != addResult)
	{
		return 13*R;
	}

	exMinuend = exOrig;
	if ((exMinuend - idx) != subResult)
	{
		return 14*R;
	}

	return 0;
}

// extent + extent
// extent - extent
// extent += extent
// extent -= extent
template <int R>
int test2() restrict(amp,cpu)
{
    int vOperand[R];
	fill<R>(vOperand, 6);
	extent<R> operand(vOperand);

	int vExtent[R];
	fill<R>(vExtent, 25);
	extent<R> exOrig(vExtent);
	extent<R> exAddend(vExtent);
	extent<R> exMinuend(vExtent);

	int vAddResult[R];
	fill<R>(vAddResult, 25 + 6);
	extent<R> addResult(vAddResult);

	int vSubResult[R];
	fill<R>(vSubResult, 25 - 6);
	extent<R> subResult(vSubResult);

	exAddend += operand;
	if (exAddend != addResult)
	{
		return 21*R;
	}

	exMinuend -= operand;
	if (exMinuend != subResult)
	{
		return 22*R;
	}

	exAddend = exOrig;
	if ((exAddend + operand) != addResult)
	{
		return 23*R;
	}

	exMinuend = exOrig;
	if ((exMinuend - operand) != subResult)
	{
		return 24*R;
	}

	return 0;
}


int test() restrict(amp,cpu)
{
    int result = test1<1>();
	if (result == 0)
	    result = test1<2>();
	if (result == 0)
	    result = test1<3>();
	if (result == 0)
	    result = test1<6>();

	if (result == 0)
        result = test2<1>();
	if (result == 0)
	    result = test2<2>();
	if (result == 0)
	    result = test2<3>();
	if (result == 0)
	    result = test2<6>();

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

    parallel_for_each(e, [&] (index<1> idx) restrict(amp,cpu) {
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

