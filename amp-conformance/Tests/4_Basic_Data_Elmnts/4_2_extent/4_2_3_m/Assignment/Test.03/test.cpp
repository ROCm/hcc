// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that the assignment works the same as copy constructor</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

template<int N>
bool compare_extent(const extent<N> & e1, const extent<N> & e2) __GPU
{
    for (int i = 0; i < N; i++)
    {
        if (e1[i] != e2[i])
        {
            return false;
        }
    }

    return true;
}

int test() __GPU
{
    extent<1> e1(100);
    extent<1> e1n;
    extent<1> e1n2(e1);

    e1n = e1;

    if (!(compare_extent(e1n, e1n2)))
    {
        return 11;
    }

    extent<3> e3(100, 200, 300);
    extent<3> e3n;
    extent<3> e3n2(e3);

    e3n = e3;

    if (!(compare_extent(e3n, e3n2)))
    {
        return 12;
    }

    int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    extent<10> e10(data);
    extent<10> e10n;
    extent<10> e10n2(e10);

    e10n = e10;

    if (!(compare_extent(e10n, e10n2)))
    {
        return 13;
    }

    return 0;
}

int main()
{
    int result = test();

    printf("Test %s\n", ((result == 0) ? "passed" : "failed"));
    return result;
}

