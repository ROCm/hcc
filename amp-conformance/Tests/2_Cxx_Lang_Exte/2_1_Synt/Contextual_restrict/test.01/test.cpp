// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test checks that a function modifier can be used as a local variable</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

class Foo
{

public:
    int data;

    explicit Foo(int d) __GPU
    {
        data = d;
    }
};

int test1(int a)
{
    int restrict = a * a;

    return ((restrict == a*a) ? 0 : 1);
}

int test2(int a) __GPU
{
    int restrict = a * a;
    return ((restrict == a*a) ? 0 : 1);
}

int test3(const int start)
{
    int restrict[2];
    for(int i = 0; i < 2;i++)
    {
       restrict[i] = start + i;
    }

    return ((restrict[1] == start + 1) ? 0 : 1);
}

int test4(const int start) __GPU
{
    int restrict[2];
    for(int i = 0; i < 2;i++)
    {
       restrict[i] = start + i;
    }

    return ((restrict[1] == start + 1) ? 0 : 1);
}

int test5(const int x)
{
    Foo *restrict = new Foo(x);

    int result = (restrict->data == x) ? 0 : 1;
    delete restrict;

    return result;
}

int test6(const int x) __GPU
{
    Foo restrict(x);

    return ((restrict.data == x) ? 0: 1);
}

int test7(const int x) __GPU
{
    int result = 0;

    if(x > 1)
    {
         Foo restrict(x);

         return ((restrict.data == x) ? 0: 1);
    }
    else
    {
         int restrict = x  + 1;


         result = restrict * 10;
    }

    return (result == (x + 1)*10);
}

// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;
    int x = 10;

    cout << "Test: declare function modifier as a local variable of built-in data type" << endl;
    if(test1(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifieras a local variable of built-in data type in function with modifier" << endl;
    if(test2(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifieras a local variable of type array" << endl;
    if(test3(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifieras a local variable of type array in function with modifier" << endl;
    if(test4(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed1" << endl;
    }

    cout << "Test: declare function modifieras a local variable of user defined data type" << endl;
    if(test5(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifieras a local variable of user defined data type in function with modifier" << endl;
    if(test6(x) == 0)
    {
        cout << "Passed!" << endl;
    }
    else
    {
        passed = false;
        cout << "Failed!" << endl;
    }

    cout << "Test: declare function modifieras a local variable in nested scope" << endl;
    if(test7(x) == 0)
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
