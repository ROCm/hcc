// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Select cpu  through multiple layer call</summary>

#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

int f(int) restrict(amp)
{
    return 0;
}

int f(float) restrict(amp)
{
    return 0;
}

int f(int) restrict(cpu)
{
    return 0;
}

int f(float) restrict(cpu)
{
    return 1;
}

int b1(float x) restrict(amp, cpu)
{
    return f(x);
}

int b2(float f) restrict(amp, cpu)
{
    return b1(f);
}

int b3(float f) restrict(amp, cpu)
{
    return b2(f);
}

int p(float f) restrict(cpu)
{
    return b3(f);
}

bool test()
{
    float f = 0;

    int flag = p(f);

    return ((flag == 1) ? true : false);
}

int main(int argc, char **argv)
{
    bool ret = test();

    printf("%s\n", (ret)? "passed\n" : "failed");

    return (ret) ? 0 : 1;
}

