// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests creation of array on the default accelerator_view</summary>

#include <amptest_main.h>
#include <amptest.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <wchar.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

const size_t DATA_SIZE = 1024;

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    vector<int> inData(DATA_SIZE);
    vector<int> outData(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i)
    {
        inData[i] = rand();
    }

    Concurrency::extent<1> domain(DATA_SIZE);
    array<int, 1> f(domain, inData.begin(), av);

    outData = inData;

	return Verify(outData, inData);
}
