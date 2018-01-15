// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check for Increment Operator</summary>

#include "amptest.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

//Post IncrementOperator
bool test_post_increment() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_inc[] = {-99,  -9,  0, 1, 2, 11, 101};
    index<7> io(data);
    index<7> i_inc(data_inc);
    index<7> i1,ir;

	i1 = io;
	ir = i1++;

    return ((ir == io) && (i1 == i_inc));
}

//Pre IncrementOperator
bool test_pre_increment() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_inc[] = {  -99,  -9,  0, 1, 2, 11, 101};
    index<7> io(data);
    index<7> i_inc(data_inc);
    index<7> i1,ir;

	i1 = io;
	ir = ++i1;

    return ((ir == i1) && (i1 == i_inc));
}

runall_result test_main()
{
    runall_result result;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    result &= EVALUATE_TEST_ON_CPU_AND_GPU(av, test_post_increment());
    result &= EVALUATE_TEST_ON_CPU_AND_GPU(av, test_pre_increment());

    return runall_pass;
}

