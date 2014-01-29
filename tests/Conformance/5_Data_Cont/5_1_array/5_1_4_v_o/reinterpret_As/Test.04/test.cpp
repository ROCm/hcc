// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Reinterpret an Array of 3 floats as float (CPU)</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %llc -march=c -o %t/kernel_.cl < %t.ll
// RUN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// RUN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include "../../../../amp.compare.h"
#include "../../../../data.h"
using namespace Concurrency;
using namespace Concurrency::Test;
class Foo
{
public:
    float r;
    float b;
    float g;
};
    
int main()
{
    std::vector<float> v(10 * 3);
    Fill(v);
    
    array<Foo, 1> arr_rbg(10, reinterpret_cast<Foo *>(v.data()));
    array_view<Foo, 1> av_rbg(arr_rbg);
    
    array_view<float, 1> av_float = arr_rbg.reinterpret_as<float>();
    
    int expected_size = arr_rbg.get_extent().size() * sizeof(Foo) / sizeof(float);

    if (av_float.get_extent()[0] != expected_size)
    {
        return 1;
    }
    
    return Verify<float>(reinterpret_cast<float *>(av_rbg.data()), av_float.data(), expected_size) ? 0 : 1;
}


