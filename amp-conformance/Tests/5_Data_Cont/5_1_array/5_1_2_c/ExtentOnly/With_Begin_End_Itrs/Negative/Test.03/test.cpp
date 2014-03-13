// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Compile error when using map based iterator</summary>
//#Expects: Error: error C2440

#include <map>
#include "./../../../../constructor.h"
#include <amptest_main.h>

runall_result test_main()
{
    const int _rank = 1;
    const int _D0 = 10;

    std::map<int, double> data;
    typedef std::pair <int, double> data_pair;
    for (int i = 0; i < _D0; i++)
        data.insert( data_pair(i, (double)rand()));

    array<double, _rank> src1(_D0, data.begin(), data.end());
    array<double, _rank> src2(_D0, data.begin());

	// We shouldn't compile
    return runall_fail;
}

