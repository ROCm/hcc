// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that accessing each dimension returns the correct index component</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int test() __GPU
{
    index<1> i1(100);

    if (i1[0] != 100)
    {
        return 11;
    }

    index<3> i3(100, 200, 300);

    if ((i3[0] != 100) ||(i3[1] != 200) ||(i3[2] != 300))
    {
        return 12;
    }

    int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    index<10> i10(data);

    for (int i = 0; i < 10; i++)
    {
        if (i10[i] != i + 1)
        {
            return 13;
        }
    }

    return 0;
}

runall_result test_main()
{
	runall_result result;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    result &= EVALUATE_TEST_ON_CPU_AND_GPU(av, test());
	return result;
}
