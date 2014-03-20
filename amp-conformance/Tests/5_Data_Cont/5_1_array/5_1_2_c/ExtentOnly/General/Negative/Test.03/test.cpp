// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create an array of unsupported integral types char, unsigned char, short, unsigned short, long long, unsigned long long, long double</summary>
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
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C2338
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
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C2664
//#Expects: Error: error C2664
//#Expects: Error: error C2664


#include "./../../../../constructor.h"
#include <amptest_main.h>

runall_result test_main()
{
    int extdata[] = {1};

    array<char, 1> achar(extdata);
    array<unsigned char, 1> auchar(extdata);
    array<bool, 1> abool(extdata);
    array<short, 1> ashort(extdata);
    array<unsigned short, 1> aushort(extdata);
    array<long long, 1> allong(extdata);
    array<unsigned long long, 1> aullong(extdata);
    array<long double, 1> aldouble(extdata);

	// We shouldn't compile
    return runall_fail;
}

