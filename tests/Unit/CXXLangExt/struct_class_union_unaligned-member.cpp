// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>create array with type which is not 4 byte aligned</summary>
//#Expects: Error: error C3581
//#Expects: Error: test.cpp\(30\)
//#Expects: Error: test.cpp\(33\)

// XFAIL: Linux, hsa
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

#include <amp.h>

//struct of size 12
#pragma pack(2)
struct S
{
    int a;    
    double b;
};

// An HSA version of C++AMP program
int main ()
{

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    S s;
    s.a = (int)idx[0];
    s.b = (double)idx[0];
    p_ans[idx[0]] = (int)s.a + (int)s.b;
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs(ans[i] - (2 * i));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}

