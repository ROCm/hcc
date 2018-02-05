// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This test checks that a function modifier can be used as an overloaded function name</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

int restrict(int a) __GPU
{
    int x = a * a;
    return ((x == a*a) ? 0 : 1);
}

int restrict(int a, int b) __GPU
{
    int y = a * b;
    return ((y == a*b) ? 0 : 1);
}

// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;
    int x = 10;
    int y = 20;

    cout << "Test: declare function modifier as an overloaded function name" << endl;
    if(restrict(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifier as an overloaded function name. function qualified with modifier" << endl;
    if(restrict(x, y) == 0)
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

