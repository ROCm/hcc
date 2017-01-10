// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Lambda capture-by-reference for selected variables</summary>
//#Expects: error C3590

#include <iostream>
#include <amptest.h>

using namespace std;

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

void init(vector<int> &a, int size)
{
    for(int i=0; i<size; ++i)
    {
        a[i] = i;
    }
}

int main()
{
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

    Concurrency::extent<1> e(size);
    Concurrency::array<int, 1> aa(e, a.begin(), rv);
    Concurrency::array<int, 1> ab(e, b.begin(), rv);
    Concurrency::array<int, 1> ac(e, rv);

    int x = 5;
    float y = 10.0f;

    parallel_for_each(aa.get_extent(), [&](index<1> idx) __GPU { // error capture by reference is not allowed
        ac[idx] = ac[idx] * x + ab[idx] * static_cast<int>(y);
    });
    c = ac;

    bool passed = true;
    for(int i=0; i<size; ++i)
    {
        int expectedResult = a[i] * x + b[i] * static_cast<int>(y);
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

