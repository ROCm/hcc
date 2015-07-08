// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>test virtual member function with restrict(amp)</summary>
//#Expects: Error: error C3581

#include <amptest.h>

class A
{
public:
    A() __GPU {}
    virtual long get() __GPU { return m; }
private:
    long  m;
};

void foo() __GPU
{
    A a;
    a.get();
}

int main(int argc, char **argv)
{
    foo();
    return 1;
}

