// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test chaining of assignment operators.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int test() __GPU
{
    index<3> ir, it, i1, i2, i3, i4, i5, i6, i7, i8, i9;

    i1 = index<3>(1, 1, 1);
    i2 = index<3>(2, 2, 2);
    i3 = index<3>(3, 3, 3);
    i4 = index<3>(4, 4, 4);
    i5 = index<3>(5, 5, 5);
    i6 = index<3>(6, 6, 6);
    i7 = index<3>(7, 7, 7);
    i8 = index<3>(8, 8, 8);
    i9 = index<3>(9, 9, 9);

    ir = index<3>(23, 23, 23);

    it = i1 += i2 -= i3 -= i4 += i5 +=i6 +=i7 -= i8 -= i9 ;

    if (!(it == ir))
    {
        return 11;      // test1 scenario1 failed
    }

    i1 = index<3>(1, 1, 1);
    i2 = index<3>(2, 2, 2);
    i3 = index<3>(53, 53, 53);
    i4 = index<3>(4, 4, 4);
    i5 = index<3>(5, 5, 5);
    i6 = index<3>(6, 6, 6);
    i7 = index<3>(17, 17, 17);
    i8 = index<3>(18, 18, 18);
    i9 = index<3>(9, 9, 9);

    ir = index<3>(61, 61, 61);

    it = (i1 += i2) += (i3 -= i4 += i5) += (i6 += i7 -= i8 -= i9);

    if (!(it == ir))
    {
        return 12;      // test1 scenario2 failed
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

