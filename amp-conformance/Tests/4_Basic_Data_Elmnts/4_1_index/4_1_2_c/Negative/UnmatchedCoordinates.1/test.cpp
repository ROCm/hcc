// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests that checks the constructing an index of rank 2 with 1 co-ordinate fails</summary>
//#Expects: Error: amp.h\(\d+\) : error C2338
//#Expects: Error: test.cpp\(16\)

#include <amptest_main.h>

#include <amp.h>

using namespace Concurrency;
using namespace Concurrency::Test;

void compile_only() {
    index<2> idx(1);
}

runall_result test_main()
{
    compile_only();

    return runall_pass;
}
