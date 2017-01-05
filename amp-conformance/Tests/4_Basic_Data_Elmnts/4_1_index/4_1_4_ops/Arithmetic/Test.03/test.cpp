// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check for addition and muplication operators, a @ b is equal to b @ a.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int test() __GPU
{
    int data1[] = {-100, -10, -1, 0,  1,  10, 100};
    int data2[] = {  99,  10,  1, 1, -1, -10, -9};
    index<7> i1(data1);
    index<7> i2(data2);
    index<7> ir1;
    index<7> ir2;

    ir1 = i1 + i2;
    ir2 = i2 + i1;

    if (!(ir1 == ir2))
    {
        return 11;  //test1 scenario1 failed
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
