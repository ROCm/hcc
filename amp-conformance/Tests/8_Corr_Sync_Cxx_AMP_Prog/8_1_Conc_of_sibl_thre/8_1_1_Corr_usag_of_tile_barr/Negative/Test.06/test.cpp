// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Use barrier in non amp-restricted functions. Compile error is prompted.</summary>

#include <iostream>
#include <amptest.h>
#include <amptest_main.h>

using namespace std;
using namespace Concurrency;
using namespace Concurrency::Test;

static
inline
void foo(tiled_index<1> idx)
{
    idx.barrier.wait();
}

runall_result test_main()
{
    typename std::result_of<decltype(&foo), tiled_index<1>>::type* foo = nullptr;
    // Should not get here.
    return !foo ? runall_fail : runall_cascade_fail;
}

//#Expects: Error: \(19\) : .+ C2512
//#Expects: Error: \(20\) : .+ C3930

