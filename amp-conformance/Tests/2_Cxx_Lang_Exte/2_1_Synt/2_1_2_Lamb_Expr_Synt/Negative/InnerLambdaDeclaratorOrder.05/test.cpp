// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Incorrect order of mutable, throw, return-type-clause and __GPU on lambda inside vector code</summary>
//#Expects: Error: error C3409

#include <iostream>
#include <amptest.h>

using namespace std;

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    accelerator device;

    if (!get_device(Device::ALL_DEVICES, device))
    {
        cout << "Unable to get required device to run this test" << endl;
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    const int size = 5;
    vector<int> c(size);
    c.assign(c.size(), 0);

    Concurrency::extent<1> e(size);
    Concurrency::array<int, 1> ac(e, c, rv);

    // error: inner lambda has incorrect lamda-declarator clause
    parallel_for_each(ac.get_extent(), [&](index<1> idx) __GPU { []() mutable -> void __GPU {}(); ac[idx] += 1; });

    c = ac;

    bool passed = true;
    for(int i=0; i<size; ++i)
    {
        int expectedResult = 1;
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

