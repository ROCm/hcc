// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Copy between array of diff dimension but same size</summary>
//#Expects: Error: error C2664
//#Expects: Error: error C2664

#include "./../../../member.h"

template<typename _type>
bool test_feature()
{
    {
        int dim_x = 10;
        array<int, 1> src(dim_x*dim_x);
        array<int, 2> dst(dim_x, dim_x);

        src.copy_to(dst);
    }

    {
        int dim_x = 10;
        array<int, 1> src(dim_x*dim_x*dim_x);
        array<int, 3> dst(dim_x, dim_x, dim_x);

        src.copy_to(dst);
    }

    return false;
}

int main(int argc, char **argv)
{
    test_feature<int>();

    printf("Failed!");

    return runall_fail;
}

