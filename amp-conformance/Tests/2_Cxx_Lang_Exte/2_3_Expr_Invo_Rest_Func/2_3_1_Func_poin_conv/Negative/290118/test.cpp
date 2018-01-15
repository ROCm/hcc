// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0, restrictionqualifier</tags>
/// <summary>Negative: initialize function reference with a function with incompatible restriction specifier</summary>

#include <amptest.h>
#include <stdio.h>

static
inline
int glorp(int x) __GPU_ONLY {
  return 668 + x;
}

int main()
{
  typedef int (&FT)(int);
  FT p = glorp;
  printf("%d\n", p(-2));
  return 1;
}

//#Expects: Error: test.cpp\(18\) : error C2440

