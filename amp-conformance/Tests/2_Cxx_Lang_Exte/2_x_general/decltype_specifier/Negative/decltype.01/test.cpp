// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>negative, P0</tags>
/// <summary>test decltype with illegal expression for amp</summary>

char x = 3;
int y;
float z;
int f() restrict(cpu) { return 1;}

void foo() restrict(amp)
{
    decltype(x == 4 ? y : z);

    decltype(f());

    int p;
    decltype(typeid(p));

    decltype(throw(1));
}

//#Expects: Error: test.cpp\(16\) : error C3586:.*(\bx\b)
//#Expects: Error: test.cpp\(16\) : error C3586:.*(\by\b)
//#Expects: Error: test.cpp\(16\) : error C3586:.*(\bz\b)
//#Expects: Error: test.cpp\(18\) : error C3930:.*(\bf\b)
//#Expects: Error: test.cpp\(21\) : error C3591:.*(\btypeid\b)
//#Expects: Error: test.cpp\(23\) : error C3594:.*(exception handling)?

