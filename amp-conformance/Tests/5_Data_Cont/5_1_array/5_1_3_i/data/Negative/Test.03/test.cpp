// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify NON-availablity of array's x properties for any rank</summary>
//#Expects: Error: error C2039
//#Expects: Error: error C2039
//#Expects: Error: error C2039
//#Expects: Error: error C2039

#include "./../../../index.h"

template<int _rank>
bool test_feature_d()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;

    extent<_rank> e1(edata);
    array<int, _rank> src(e1);

    if (src.x)
        return false;

    return false;
}

int main(int argc, char **argv)
{
    test_feature_d<1>();
    test_feature_d<2>();
    test_feature_d<3>();
    test_feature_d<4>();

    return runall_fail;
}
