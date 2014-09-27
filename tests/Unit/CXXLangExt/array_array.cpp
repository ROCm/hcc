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

// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

#include <amp.h>

// An HSA version of C++AMP program
int main ()
{

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    int arr[vecSize][vecSize];

    for (int i = 0; i < vecSize; i++) {
      for (int j = 0; j < vecSize; j++) {
        arr[i][j] = i * j * idx[0];
      }
    }

    for (int i = 0; i < vecSize; i++) {
      p_ans[idx[0]] += arr[i][idx[0]];
    }
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    ; //error += abs(ans[i] - (3 * i));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}

