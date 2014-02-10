// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Attempt to "add restrict" to an object declaration</summary>
//#Expects: Error: error C2146

// RUN: %clang_cc1 -std=c++amp -fsyntax-only %ampneg -verify %s

typedef int binary_math(int, int);
    
int main()
{
    binary_math *foo restrict(cpu);
    // if this compiles it fails
    return 1;
}
// expected-error@16 {{expected ';' at end of declaration}}
// expected-error@16 {{C++ requires a type specifier for all declarations}}
// expected-error@16 {{restrict requires a pointer or reference ('int' is invalid)}}
