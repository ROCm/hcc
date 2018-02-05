// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Arrays passed by value</summary>
//#Expects: error C3597

#include <iostream>
#include <functional>
#include <amptest.h>

using namespace std;

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

void init(vector<int> &a, int size)
{
    for(int i=0; i<size; ++i)
    {
        a[i] = rand();
    }
}

int main()
{
    srand(668);
    accelerator device;

    if (!get_device(Device::ALL_DEVICES, device))
    {
        cout << "Unable to get required device to run this test" << endl;
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    const int size = 2048;
    vector<int> a(size);
    vector<int> b(size);
    vector<int> c(size);

    init(a, size);
    init(b, size);
    c.assign(c.size(), 0);

    Concurrency::extent<1> e(size);
    Concurrency::array<int, 1> aa(e, a.begin(), rv);
    Concurrency::array<int, 1> ab(e, b.begin(), rv);
    Concurrency::array<int, 1> ac(e, c.begin(), rv);

    // error ac and ab is passed by reference
    parallel_for_each(aa.get_extent(), [=](index<1> idx) __GPU {
        ac[idx] = aa[idx] + ab[idx];
    });
    c = ac;

    bool passed = true;
    for(int i=0; i<size; ++i)
    {
        int expectedResult = a[i] + b[i];
        if (c[i] != expectedResult)
        {
            cout << "c[" << i << "] = " << c[i] << " expected:" << expectedResult << endl;
            passed = false;
            break;
        }
    }

    cout << "lambda test: " << (passed ? "pass" : "fail") << endl;

    return passed ? 0 : 1;
}

