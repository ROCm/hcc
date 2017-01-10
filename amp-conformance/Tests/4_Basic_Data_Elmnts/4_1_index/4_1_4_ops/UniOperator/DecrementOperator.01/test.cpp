// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check for Decrement Operator</summary>

#include "amptest.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

//Post DecrementOperator
bool test_post_decrement() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_dec[] = {-101, -11, -2, -1, 0, 9, 99};
    index<7> io(data);
    index<7> i_dec(data_dec);
    index<7> i1,ir;

	i1 = io;
	ir = i1--;

    return ((ir == io) && (i1 == i_dec));
}

//Pre DecrementOperator
bool test_pre_decrement() restrict(cpu,amp)
{
    int data[] = {-100, -10, -1, 0,  1,  10, 100};
    int data_dec[] = {-101, -11, -2, -1, 0, 9, 99};
    index<7> io(data);
    index<7> i_dec(data_dec);
    index<7> i1,ir;

	i1 = io;
	ir = --i1;

    return ((ir == i1) && (i1 == i_dec));
}

runall_result test_main()
{
    runall_result result;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    result &= EVALUATE_TEST_ON_CPU_AND_GPU(av, test_post_decrement());
    result &= EVALUATE_TEST_ON_CPU_AND_GPU(av, test_pre_decrement());

    return runall_pass;
}

