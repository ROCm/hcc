// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>C6721: tile_static cannot be applied with type, which is not a POD</summary>

// We now support non-POD tile_static variable. this is not a negative test anymore.

#include <amptest.h>

class NonPodClass
{
private:
    int var;

public:
    NonPodClass(int i)
    {
        var = i;
    }
};

void NonPodTypeNotSupported(int x) __GPU_ONLY
{
    tile_static NonPodClass var;
    tile_static NonPodClass arr[10]; // array of non POD should also be allowed

}

int main()
{
    return 0;
}

