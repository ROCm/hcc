// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify availablity of array.get_extent() properties for 1D, 2D and 3D</summary>

#include "./../../../member.h"

bool test_feature_1d(int *edata)
{
    const int rank = 1;
    extent<rank> e1(edata);
    array<int, rank> src(e1);

    if (src.get_extent()[0] != edata[0])
        return false;

    printf ("Pass 1D\n");
    return true;
}

bool test_feature_2d(int *edata)
{
    const int rank = 2;
    extent<rank> e1(edata);
    array<int, rank> src(e1);

    if (src.get_extent()[0] != edata[0])
        return false;

    if (src.get_extent()[1] != edata[1])
        return false;

    printf ("Pass 2D\n");
    return true;
}

bool test_feature_3d(int *edata)
{
    const int rank = 3;
    extent<rank> e1(edata);
    array<int, rank> src(e1);

    if (src.get_extent()[0] != edata[0])
        return false;

    if (src.get_extent()[1] != edata[1])
        return false;

    if (src.get_extent()[2] != edata[2])
        return false;

    printf ("Pass 3D\n");
    return true;
}

#define MAX 5
int main()
{
    int edata[MAX];
    for (int i = 0; i < MAX; i++)
        edata[i] = i+1;

    int passed =
        test_feature_1d(edata) && test_feature_2d(edata) && test_feature_3d(edata)
            ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

