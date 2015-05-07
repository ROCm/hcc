// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Passing lambda object as argument to lambda</summary>

#include <iostream>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

// Kernel function using lambda as parameter
template<typename T>
void kernel(T _Pred, int &c, int a) __GPU
{
    c = _Pred(a) ? a : -a;
}

// One level of indirection in order to get the type of lambda
template<typename T>
void start(T lambda, array<int, 1> &ac, array<int, 1> &aa)
{
    parallel_for_each(aa.get_extent(), [&, lambda](index<1> idx) __GPU {
        kernel<T>(lambda, ac[idx], aa[idx]);
    });
}

int main()
{
    accelerator device;

    if (!get_device(Device::ALL_DEVICES, device))
    {
        std::cout << "Unable to get required device to run this test" << std::endl;
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    const int size = 2020;
    std::vector<int> a(size);

    for(int i=0;i<size;++i)
    {
        a[i] = i;
    }

    Concurrency::extent<1> e(size);
    array<int, 1> aa(e, a.begin(), rv);
    array<int, 1> ac(e, rv);

    auto lambda = [](int x) __GPU -> int { return x % 2; };

    start(lambda, ac, aa);
    a = ac;

    bool passed = true;
    for(int i=0;i<size;++i)
    {
        if(a[i] != (lambda(i) ? i : -i))
        {
            std::cout << "a[ " << i << "] = " << a[i] << " expected: " << (lambda(i) ? i : -i) << std::endl;
            passed = false;
            break;
        }
    }

    printf("test: %s\n", passed ? "passed" : "failed");
    return !passed;
}

