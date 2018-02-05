// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test a user-defined call operator on the GPU</summary>

#include "amptest.h"
#include "amptest_main.h"

using namespace concurrency;
using namespace concurrency::Test;


class testclass {

public:
    int r;

    testclass() __GPU
        : r(0)
    {}

    testclass& operator ()(int a, int b) __GPU  {
         r = a+b;
         return (*this);
    };

};


runall_result kernel() __GPU
{
    testclass t;

    // this will be 7 if the call operator is invoked
    return t(4, 3).r == 7;
}

runall_result test_main()
{
    return GPU_INVOKE(require_device(device_flags::NOT_SPECIFIED).get_default_view(), runall_result, kernel);
}
