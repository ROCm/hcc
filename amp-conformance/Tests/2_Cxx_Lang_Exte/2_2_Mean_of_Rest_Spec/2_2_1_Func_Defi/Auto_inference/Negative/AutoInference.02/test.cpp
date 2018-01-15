// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>B class is defined outside __GPU context, while D is defined in __GPU context. D inherits from B. Make sure that member functions inherited from B which are not __GPU member function are not __GPU member functions</summary>
//#Expects: Error: C3930

#include <amptest.h>

class B
{
public:
    float f1(int &flag)
    {
        flag = 1;
        return 0.0;
    }
};

bool test() __GPU
{
    bool passed = true;
    int flag = 0;

    class D: public B
    {
    public:
        float f2(int &flag) {return 0.0;}
    };

    D o;

    o.f1(flag);

    if (flag == 1)
    {
        return false;
    }

    return passed;
}

int main(int argc, char **argv)
{
    return test() ? 0 : 1;
}

