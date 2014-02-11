// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Reinterpret an Array of float as double</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %llc -march=c -o %t/kernel_.cl < %t.ll
// RUN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include "../../../../amp.compare.h"
#include "../../../../data.h"
using namespace Concurrency;
using namespace Concurrency::Test;
int main()
{
    std::vector<float> v(10);
    Fill(v);
    
    array<float, 1> arr_float(static_cast<int>(v.size()), v.begin());
    array_view<float,1> av_float(arr_float);  // Created for verification
    array_view<double, 1> av_double = arr_float.reinterpret_as<double>();
    
    int expected_size = arr_float.get_extent().size() * sizeof(float) / sizeof(double);

    if (av_double.get_extent()[0] != expected_size)
    {
        return 1;
    }
    
    return Verify<double>(reinterpret_cast<double *>(av_float.data()), av_double.data(), expected_size) ? 0 : 1;
}

