// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>This test checks that a function qualifier can be used as a class name</summary>

#include <amptest.h>
#include <iostream>

using namespace Concurrency;
using namespace std;

class restrict
{
   int size;

public:
    restrict() __GPU
    {
        this->size = 10;
    }

    restrict(const restrict& other) __GPU
    {
        this->size = other.size;
    }

    restrict(unsigned int _size) __GPU
    {
        this->size = _size;
    }

    ~restrict() __GPU
    {

    }

    int GetSize() __GPU
    {
        return size;
    }

    int add(restrict&& r) __GPU
    {
       return r.size + 1;
    }

};


restrict func(restrict x)
{
    restrict y = x;
    return y;
}


restrict func1(restrict x) __GPU
{
    restrict y = x;
    return y;
}


// Main entry point
int main(int argc, char **argv)
{
    bool passed = true;

    restrict *x = new restrict();
    passed = (x->GetSize() == 10);

    restrict *y = new restrict(20);
    passed &= (y->GetSize() == 20);

    restrict z = func(*x);
    passed &= (z.GetSize() == 10);

    restrict a = func1(*x);
    passed &= (a.GetSize() == x->GetSize());

    passed &= (a.add(restrict()) == 11);
    cout << (passed? "Passed!" : "Failed!") << endl;

    delete x;
    delete y;

    return passed ? 0 : 1;
}

