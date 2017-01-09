// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify NON-availability of array's x, y and z properties in 1D</summary>
//#Expects: Error: error C2039
//#Expects: Error: error C2039
//#Expects: Error: error C2039

#include "./../../../index.h"

template<typename _type>
bool test_feature()
{
    {
        const int _rank = 2;

        int edata[_rank];
        for (int i = 0; i < _rank; i++)
            edata[i] = i;

        extent<_rank> e1(edata);
        array<_type, _rank> src(e1);

		int x = src.x;
		int y = src.y;
        int z = src.z;
    }

    return false;
}

int main(int argc, char **argv)
{
    test_feature<int>();

    printf("Failed!");

    return runall_fail;
}

