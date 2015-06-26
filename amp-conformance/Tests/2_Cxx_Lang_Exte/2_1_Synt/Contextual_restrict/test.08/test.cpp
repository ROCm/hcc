// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This test checks that a function qualifier can be used as a class function member name</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

class Foo
{
    int data;
public:

    explicit Foo(int d) __GPU
    {
        data = d;
    }

    int restrict() __GPU
    {
        return data;
    }
};

int Test1(int d)
{
    Foo f(d);

    return((f.restrict() == d) ? 0 : 1);
}

int Test2(int d) __GPU
{
    Foo f(d);

    return((f.restrict() == d) ? 0 : 1);
}

// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;

    cout << "Test: declare class function member name" << endl;
    if(Test1(10) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare classs function member name. function also qualified with modifier" << endl;
    if(Test2(-2) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    return passed ? 0 : 1;
}
