// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check binary assginment operators between two index objects.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int test() __GPU
{
    int data1[] = {-100, -10, -1, 0,  1,  10, 100};
    int data2[] = {  99,  10,  1, 1, -1, -10, -9};
    index<7> i1o(data1);
    index<7> i2o(data2);
    index<7> i1;
    index<7> i2;
    int dataa[] = {-1, 0, 0, 1, 0, 0, 91};
    int datas[] = {-199, -20, -2, -1, 2, 20, 109};
    int datam[] = {-9900, -100, -1, 0, -1, -100, -900};
    int datad[] = {-1, -1, -1, 0, -1, -1, -11};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);
    index<7> id(datad);

    i1 = i1o;
    i2 = i2o;
    i1 += i2;
    if (!(i1 == ia))
    {
        return 11;  // test1 scenario1 failed
    }

    i1 = i1o;
    i2 = i2o;
    i1 -= i2;
    if (!(i1 == is))
    {
        return 12;  // test1 scenario2 failed
    }

    return 0;   // all pass
}

runall_result test_main()
{
	runall_result result;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    result &= EVALUATE_TEST_ON_CPU_AND_GPU(av, test());
	return result;
}

