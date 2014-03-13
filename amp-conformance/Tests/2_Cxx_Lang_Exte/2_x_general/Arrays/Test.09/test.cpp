// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Positive test for array initialization/destruction</summary>

struct A1
{
    int var;
    A1() restrict(cpu,amp) {}
    ~A1() restrict(cpu,amp) {}
};

void test1() restrict(amp,cpu)
{
    A1 arr[5];
}

//------------------------------------

struct A2
{
    int var;
    A2() restrict(amp) {}
    A2() restrict(cpu) {}
    ~A2() restrict(amp,cpu) {}
};

void test2() restrict(amp,cpu)
{
    A2 arr[5];
}

//------------------------------------

struct A3
{
    int var;
    A3() restrict(amp) {}
    ~A3() restrict(amp) {}
};

void test3() restrict(amp)
{
    A3 arr[5];
}

//------------------------------------

struct A4
{
    int var;
    A4() restrict(cpu) {}
    ~A4() restrict(cpu) {}
};

void test4() restrict(cpu)
{
    A4 arr[5];
}

//------------------------------------

struct A5
{
    int var;
    A5(int) restrict(amp) {}
    A5() restrict(cpu) {}
};

void test5() restrict(amp)
{
    A5 arr[1] = { A5(1) };
}

//------------------------------------

struct A6 {
    int m;
    A6() restrict(amp,cpu) {}
    ~A6() restrict(amp,cpu) {}
};

void test6() restrict(amp, cpu)
{
    A6 arr1[2][2];
    A6 arr2[2][2][2];
}

//------------------------------------

int main()
{
    // if it compiles then we are happy
    return 0;
}
