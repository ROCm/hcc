// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Use typeid to compare two equal member function pointers, one with restrict(cpu)</summary>

// RUN: %cxxamp %s -o %t.out && %t.out

#include <typeinfo>
struct S
{
    int foo(float a, double b)
    {
        return 1;
    }
};

int main()
{
    int (S::*p1)(float a, double b) = &S::foo;
    int (S::*p2)(float a, double b) restrict(cpu) = &S::foo;
    
    return typeid(p1) == typeid(p2) ? 0 : 1;
}
