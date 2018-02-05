// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>checking operators explicitly.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() restrict(amp,cpu)
{
    extent<3> ea(4, 4, 4);
    extent<3> er, et;

    er = extent<3>(6, 6, 6);
    et = operator+(ea, 2);

    if (!(et == er))
    {
        return 11;
    }

    er = extent<3>(2, 2, 2);
    et = operator-(ea, 2);

    if (!(et == er))
    {
        return 12;
    }

    er = extent<3>(8, 8, 8);
    et = operator*(ea, 2);

    if (!(et == er))
    {
        return 13;
    }

    er = extent<3>(2, 2, 2);
    et = operator/(ea, 2);

    if (!(et == er))
    {
        return 14;
    }

	et = ea + extent<3>(1,2,3); // et <=> (5,6,7)
	er = extent<3>(1, 0, 1);
    et = operator%(et, 2);

    if (!(et == er))
    {
        return 15;
    }

    return 0;
}

void kernel(index<1>& idx, array<int, 1>& result) restrict(amp,cpu)
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
