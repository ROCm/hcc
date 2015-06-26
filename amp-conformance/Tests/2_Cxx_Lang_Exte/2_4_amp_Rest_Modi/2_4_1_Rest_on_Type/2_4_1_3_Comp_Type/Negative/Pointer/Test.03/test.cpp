// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>(NEG) Define pointers to pointer as parameter and return value</summary>
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581
// ref bug: 226039

#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;


void f1(double **p1, float ***p2, int ****p3) __GPU {}

double ***** f2() __GPU
{
    double ***** pd = NULL;
    return pd;
}

void f11(double **p1, float ***p2, int ****p3) __GPU_ONLY {}

double ***** f22() __GPU_ONLY
{
    double ***** pd = NULL;
    return pd;
}


int main(int argc, char **argv)
{
    bool passed = false;

    return passed ? 0 : 1;
}

