// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Default argument expression</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int foo() restrict(amp) { return 1; }
int foo() restrict(cpu) { return 2; }

void boo1(int x = foo()) restrict(amp) {} // binds to foo 'cpu'

void hoo1() restrict(amp)
{
    boo1(); // expects an error, because foo was bound to foo 'cpu'
}

runall_result test_main()
{
    parallel_for_each(extent<1>{1}, [=](index<1>) restrict(amp) { hoo1(); });

    // Should not get here.
    return runall_pass;
}
//#Expects: Error: test.cpp\(16\) : error C3931

