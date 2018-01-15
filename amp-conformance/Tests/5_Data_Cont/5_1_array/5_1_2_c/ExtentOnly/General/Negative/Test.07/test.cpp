// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create an array of a user defined union with not supported type</summary>
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581

#include "./../../../../constructor.h"
#include <amptest_main.h>

typedef union
{
    int* n;
    float* u;
} usr_union;

template<typename _type, int _rank>
void kernel_userdefined(index<_rank>& idx, array<_type, _rank>& f) __GPU
{
    f[idx].n = -5;
    f[idx].u = -5.0f;
}

runall_result test_main()
{
    int extdata[] = {2, 2};
    test_array_userdefined<usr_union, 1>(extdata);

	// We shouldn't compile
    return runall_fail;
}

