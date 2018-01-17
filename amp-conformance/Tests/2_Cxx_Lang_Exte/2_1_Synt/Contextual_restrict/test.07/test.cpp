// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This test checks that a function modifier can be used as a class data member name</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

class Foo1
{
public:
    int restrict;

    Foo1(int g)
    {
        restrict = g;
    }
};


class Foo2
{
public:
    static int restrict;

    Foo2(int g)
    {
        restrict = g;
    }
};

int Foo2::restrict;


class Foo3
{
public:
    const int restrict;

    Foo3(int g) restrict(cpu) : restrict(g)
    {

    }
};

class Foo4
{
public:
    static const int restrict = 10;
};

// Main entry point
int main(int argc, char **argv)
{
    cout << "Test: declare modifer as a class data member name" << endl;

    bool passed = true;

    int x = 10;
    Foo1 foo1 = Foo1(x);
    passed &= (foo1.restrict == x);

    Foo2 foo2 = Foo2(x);
    passed &= (foo2.restrict == x);

    Foo3 foo3 = Foo3(x);
    passed &= (foo3.restrict == x);

    Foo4 foo4 = Foo4();
    passed &= (foo4.restrict == 10);

    cout << (passed? "Passed!" : "Failed!") << endl;

    return passed ? 0 : 1;
}
