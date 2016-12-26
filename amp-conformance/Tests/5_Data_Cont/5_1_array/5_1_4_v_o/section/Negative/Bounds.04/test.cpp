// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test a negative section offset </summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    std::vector<int> v(10);
    array<int, 1> av(extent<1>(10), v.begin());
    try
    {
        array_view<int, 1> section = av.section(-1, 5); // this should throw
        return runall_fail;
    }
    catch (runtime_exception &re)
    {
        Log(LogType::Info, true) << re.what() << std::endl;
        return runall_pass;
    }
}

