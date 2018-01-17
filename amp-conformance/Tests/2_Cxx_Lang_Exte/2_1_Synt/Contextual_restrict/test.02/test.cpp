// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This test checks that a function qualifier can be used as a function parameter</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

static int test1(int restrict)
{
    int expected = restrict * restrict;
    restrict *= restrict;
    return ((restrict == expected) ? 0 : 1);
}

static int test2(int restrict) __GPU
{
    int expected = restrict * restrict;
    restrict *= restrict;
    return ((restrict == expected) ? 0 : 1);
}

static int test3(int restrict = 0)
{
    int expected = restrict * restrict;
    restrict *= restrict;
    return ((restrict == expected) ? 0 : 1);
}

static int test4(const int restrict)
{
    int expected = restrict * restrict;
    return ((restrict * restrict == expected) ? 0 : 1);
}



// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;
    int x = 10;

    cout << "Test: declare function modifier as function paramter" << endl;
    if(test1(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifier as function parameter in function marked with modifier" << endl;
    if(test2(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifier as function parameter with default value" << endl;
    if(test3(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifier as const function parameter" << endl;
    if(test4(x) == 0)
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

