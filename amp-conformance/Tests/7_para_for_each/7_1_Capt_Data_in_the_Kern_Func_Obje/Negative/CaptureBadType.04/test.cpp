// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Lambda expression unsupported user-defined type</summary>
//#Expects: Error: error C3596
//#Expects: Error: error C3581

#include <iostream>
#include <amptest.h>

using namespace std;

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

class A
{
public:
    A(float a) : m_a(a) {}
    virtual ~A() { }
    int get() restrict(cpu,amp) const
    {
        return m_a;
    }

private:
    float m_a;
};

int main()
{
    const int size = 11;
    vector<int> c(size);
    Concurrency::extent<1> e(size);
    Concurrency::array<int, 1> ac(e);

    A a(22);
    parallel_for_each(ac.get_extent(), [&, a](index<1> idx) __GPU {
        ac[idx] = a.get();
    });
    c = ac;

    bool passed = true;
    for(int i=0; i<size; ++i)
    {
        int expectedResult = 22;
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

