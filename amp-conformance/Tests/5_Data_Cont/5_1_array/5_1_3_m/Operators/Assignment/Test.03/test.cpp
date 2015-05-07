// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>test array's assignment operator from an array_view.</summary>

#include <amptest.h>
#include <vector>

using  namespace concurrency;

static const int ev = 32;

template<typename _type, int _rank>
bool test_feature()
{
    // setup the extent and size based on _rank
    extent<_rank> ex;
    int size = 1;
    for (int i = 0; i <_rank; i++)
    {
        size *= ev;
        ex[i] = ev;
    }

    // fill in data and zero out the result vector
    std::vector<_type> vsrc(size);
    std::vector<_type> vdst(size);
    for (int i = 0; i < size; i++)
    {
       vsrc[i] = (_type)rand();
       vdst[i] = (_type)0;
    }

    array_view<_type, _rank> av(ex, vsrc);
    array<_type, _rank> a(ex);

    a = av;  // array's assignment operator from an array_view
    vdst = a;

    // verify results
    for (int i = 0; i < size; i++)
    {
        if (vdst[i] != vsrc[i])
        {
            return false;
        }
    }

    return true;
}

int main()
{

    int passed =
        test_feature<int, 1>() && test_feature<int, 2>() && test_feature<int, 4>() &&
        test_feature<float, 1>() && test_feature<float, 2>() && test_feature<float, 4>()
        ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return runall_pass;
}
