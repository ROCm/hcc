// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Capture by reference in user-defined type</summary>
//#Expects: Error: error C3590
//#Expects: Error: error C3581

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

    // Integral type
    int x = 5;

    // Floating-point type
    float y = 10.0f;

    // Abstract data type
    struct MyT
    {
    public:
        MyT(int x) __GPU :m_x(x) {}
        MyT(const MyT& that) __GPU { m_x = that.m_x; }

        int get() __GPU
        {
            //error capture-by-reference not allowed in __GPU restricted code
            int y = 33;
            return [&]() mutable __GPU -> int { y *= 2; return m_x * y; } ();
        }

    private:
        int m_x;
    };

    parallel_for_each(aa.get_extent(), [&](index<1> idx) __GPU {
        MyT t(x);
        ac[idx] = aa[idx] * ab[idx] * t.get();
    });
    c = ac;

    bool passed = true;
    MyT t(x);

    for(int i=0; i<size; ++i)
    {
        int expectedResult = a[i] * b[i] * t.get();
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

