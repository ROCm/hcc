// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// struct cannot have av pointer.
//#Expects: Error: error C3581

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace concurrency;

struct S
{
    S(array_view<int>& a) restrict(cpu,amp) : m(&a) {}
    ~S() restrict(cpu,amp) {}

    array_view<int> *m;
};

// Diagnostics may be deferred until the use of the class.
void func_cpu_amp(array_view<int>& a) restrict(cpu,amp)
{
    S s(a);
}

runall_result test_main()
{
    return runall_fail;
}
