// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Assign an initialized extent of the same rank to this extent and ensure that dimensions are copied correctly. Repeat by assigning to an uninitialized const.</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

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

    e1n = e1;

    if (!(compare_extent(e1, e1n)))
    {
        return 11;
    }

    extent<3> e3(100, 200, 300);

    extent<3> e3n;

    e3n = e3;

    if (!(compare_extent(e3, e3n)))
    {
        return 12;
    }

    int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    extent<10> e10(data);

    extent<10> e10n;

    e10n = e10;

    if (!(compare_extent(e10, e10n)))
    {
        return 13;
    }

    return 0;
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    result[idx] = test();
}

const int size = 10;

int test_device()
{
    accelerator acc;
    if (!get_device(Device::ALL_DEVICES, acc))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view av = acc.get_default_view();

    extent<1> e(size);
    array<int, 1> result(e, av);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) __GPU {
        kernel(idx , result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] != 0)
        {
            printf("Test failed. Return code: %d\n", presult[i]);
            return 1;
        }
    }

    return 0;
}

int main()
{
    int result = test();

    printf("Test %s on host\n", ((result == 0) ? "passed" : "failed"));
    if(result != 0) return result;

    result = test_device();
    printf("Test %s on device\n", ((result == 0) ? "passed" : "failed"));
    return result;
}

