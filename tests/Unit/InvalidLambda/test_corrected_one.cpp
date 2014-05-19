// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test the correctness of C++ AMP copy APIs involving array_views without a data source</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl 
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <vector>
#include <algorithm>
#include <amp.h>

using namespace Concurrency;

// Tests copy from an array to an array_view without a data source
template<typename T>
bool TestCopy1(accelerator_view av, int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    array<T, 2> srcArray(HEIGHT, WIDTH, av);
    parallel_for_each(srcArray.get_extent(), [&, WIDTH](const index<2> &idx) restrict(amp) {
        srcArray[idx] = (T)(idx[0] * WIDTH + idx[1]);
    });

    array_view<T, 2> tempArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> destArrayView = tempArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);

    // A functor to verify the correctness of the copy
    auto verificationFunc = [tempArrayView, WIDTH, HEIGHT, BORDER]() -> bool {
        bool passed = true;
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                if (tempArrayView(BORDER + i, BORDER + j) != (T)((i * WIDTH) + j)) {
                    passed = false;
                }
            }
        }

        return passed;
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcArray, destArrayView);
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcArray, destArrayView);
        passed = verificationFunc();
    }

    return passed;
}   

// Tests copy from begin and end iterators to an array_view without a data source
template<typename T>
bool TestCopy2(int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    std::vector<T> srcVec(HEIGHT * WIDTH);
    std::generate(srcVec.begin(), srcVec.end(), rand);

    array_view<T, 2> tempArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> destArrayView = tempArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);

    // A functor to verify the correctness of the copy
    auto verificationFunc = [tempArrayView, WIDTH, HEIGHT, BORDER, &srcVec]() -> bool {
        bool passed = true;
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                if (tempArrayView(BORDER + i, BORDER + j) != srcVec[(i * WIDTH) + j]) {
                    passed = false;
                }
            }
        }

        return passed;
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcVec.cbegin(), srcVec.cend(), destArrayView);
        fut.get();
        passed = verificationFunc(); 
    }
    else 
    {
        copy(srcVec.cbegin(), srcVec.cend(), destArrayView);
        passed = verificationFunc();
    }

    return passed;
}   

// Tests copy from a begin iterator to an array_view without a data source
template<typename T>
bool TestCopy2_1(int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    std::vector<T> srcVec(HEIGHT * WIDTH);
    std::generate(srcVec.begin(), srcVec.end(), rand);

    array_view<T, 2> tempArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> destArrayView = tempArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);

    // A functor to verify the correctness of the copy
    auto verificationFunc = [tempArrayView, WIDTH, HEIGHT, BORDER, &srcVec]() -> bool {
        bool passed = true;
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                if (tempArrayView(BORDER + i, BORDER + j) != srcVec[(i * WIDTH) + j]) {
                    passed = false;
                }
            }
        }

        return passed;
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcVec.cbegin(), destArrayView);
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcVec.cbegin(), destArrayView);
        passed = verificationFunc();
    }

    return passed;
}   

// Tests copy from an array_view to an array_view without a data source
template<typename T>
bool TestCopy3(int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    array_view<T, 2> srcArrayView(HEIGHT, WIDTH);
    parallel_for_each(srcArrayView.get_extent(), [=](const index<2> &idx) restrict(amp) {
        srcArrayView[idx] = (T)(idx[0] * WIDTH + idx[1]);
    });

    array_view<T, 2> tempArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> destArrayView = tempArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);

    // A functor to verify the correctness of the copy
    auto verificationFunc = [tempArrayView, WIDTH, HEIGHT, BORDER]() -> bool {
        bool passed = true;
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; ++j) {
                if (tempArrayView(BORDER + i, BORDER + j) != (T)((i * WIDTH) + j)) {
                    passed = false;
                }
            }
        }

        return passed;
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcArrayView, destArrayView);
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcArrayView, destArrayView);
        passed = verificationFunc();
    }

    return passed;
}   

// Tests copy from an array_view without data source to an array
template<typename T>
bool TestCopy4(const accelerator_view &av, int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    array_view<T, 2> tempSrcArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> srcArrayView = tempSrcArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);
    parallel_for_each(srcArrayView.get_extent(), [=](const index<2> &idx) restrict(amp) {
        srcArrayView[idx] = (T)(idx[0] * WIDTH + idx[1]);
    });

    array<T, 2> destArray(HEIGHT, WIDTH, av);

    // A functor to verify the correctness of the copy
    auto verificationFunc = [&destArray, WIDTH, HEIGHT]() -> bool {
        int passed = 1;
        array_view<int> passedView(1, &passed);
        parallel_for_each(destArray.get_extent(), [=, &destArray](const index<2> &idx) restrict(amp) {
            if (destArray[idx] != (T)(idx[0] * WIDTH + idx[1])) {
                passedView(0) = 0;
            }
        });

        passedView.synchronize();
        return (passed == 1);
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcArrayView, destArray);
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcArrayView, destArray);
        passed = verificationFunc();
    }

    return passed;
}   

// Tests copy from an array_view without data source to an array_view
template<typename T>
bool TestCopy4_1(const accelerator_view &av, int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    array_view<T, 2> tempSrcArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> srcArrayView = tempSrcArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);
    parallel_for_each(srcArrayView.get_extent(), [=](const index<2> &idx) restrict(amp) {
        srcArrayView[idx] = (T)(idx[0] * WIDTH + idx[1]);
    });

    array_view<T, 2> destArrayView(HEIGHT, WIDTH);

    // A functor to verify the correctness of the copy
    auto verificationFunc = [destArrayView, WIDTH, HEIGHT]() -> bool {
        int passed = 1;
        array_view<int> passedView(1, &passed);
        parallel_for_each(destArrayView.get_extent(), [=](const index<2> &idx) restrict(amp) {
            if (destArrayView[idx] != (T)(idx[0] * WIDTH + idx[1])) {
                passedView(0) = 0;
            }
        });

        passedView.synchronize();
        return (passed == 1);
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcArrayView, destArrayView);
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcArrayView, destArrayView);
        passed = verificationFunc();
    }

    return passed;
}   

// Tests copy from an array_view without data source to an output iterator
template<typename T>
bool TestCopy5(int numElems, bool async)
{
    const int WIDTH = 257;
    const int HEIGHT = (numElems + WIDTH - 1)/ WIDTH;
    const int BORDER = 3;
    array_view<T, 2> tempSrcArrayView(HEIGHT + (2 * BORDER), WIDTH + (2 * BORDER));
    array_view<T, 2> srcArrayView = tempSrcArrayView.section(BORDER, BORDER, HEIGHT, WIDTH);
    parallel_for_each(srcArrayView.get_extent(), [=](const index<2> &idx) restrict(amp) {
        srcArrayView[idx] = (T)(idx[0] * WIDTH + idx[1]);
    });

    T *destIter = new T[HEIGHT * WIDTH];

    // A functor to verify the correctness of the copy
    auto verificationFunc = [destIter, WIDTH, HEIGHT]() -> bool {
        bool passed = true;
        for (int i = 0; i < HEIGHT * WIDTH; ++i) {
            if (destIter[i] != (T)(i)) {
                passed = false;
            }
        }

        return passed;
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcArrayView, (destIter));
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcArrayView, (destIter));
        passed = verificationFunc();
    }

    delete [] destIter;
    return passed;
}   

// Tests copy from an array_view without data source to an output iterator
// Here the input data is uninitialized and the test just ensures that there is no
// crash for such scenarios
template<typename T>
bool TestCopy6(bool async)
{
    const int size = 1023;
    array_view<T> srcArrayView(size);

    T *destIter = new T[size];

    // A functor to verify the correctness of the copy
    auto verificationFunc = []() -> bool {
        return true;
    };

    bool passed = false;
    if (async) 
    {
        auto fut = copy_async(srcArrayView, (destIter));
        fut.get();
        passed = verificationFunc();
    }
    else 
    {
        copy(srcArrayView, (destIter));
        passed = verificationFunc();
    }

    delete [] destIter;
    return passed;
}   

int main()
{
    Concurrency::accelerator def;
    Concurrency::accelerator_view av = def.create_view();

    int result = 1;

    // Test sync and async copy from array to array_view without data source
    result &= (TestCopy1<int>(av, (1 << 16), false)); // 256 KB
    result &= (TestCopy1<int>(av, (1 << 20), false)); // 4 MB
    result &= (TestCopy1<int>(av, (1 << 16), true)); // 256 KB
    result &= (TestCopy1<int>(av, (1 << 20), true)); // 4 MB

    // Test sync and async copy from begin and end iterators to array_view without data source
    result &= (TestCopy2<int>((1 << 16), false)); // 256 KB
    result &= (TestCopy2<int>((1 << 20), false)); // 4 MB
    result &= (TestCopy2<int>((1 << 16), true)); // 256 KB
    result &= (TestCopy2<int>((1 << 20), true)); // 4 MB
    result &= (TestCopy2_1<int>((1 << 20), false)); // 4 MB
    result &= (TestCopy2_1<int>((1 << 16), true)); // 256 KB
    result &= (TestCopy2_1<int>((1 << 20), true)); // 4 MB

    // Test sync and async copy from array_view to array_view without data source
    result &= (TestCopy3<int>((1 << 16), false)); // 256 KB
    result &= (TestCopy3<int>((1 << 20), false)); // 4 MB
    result &= (TestCopy3<int>((1 << 16), true)); // 256 KB
    result &= (TestCopy3<int>((1 << 20), true)); // 4 MB

    // Test sync and async copy from array_view without data source to an array
    result &= (TestCopy4<int>(av, (1 << 16), false)); // 256 KB
    result &= (TestCopy4<int>(av, (1 << 20), false)); // 4 MB
    result &= (TestCopy4<int>(av, (1 << 16), true)); // 256 KB
    result &= (TestCopy4<int>(av, (1 << 20), true)); // 4 MB

	 // Test sync and async copy from array_view without data source to an array view
    result &= TestCopy4_1<int>(av, (1 << 16), false); // 256 KB
    result &= TestCopy4_1<int>(av, (1 << 20), false); // 4 MB
    result &= TestCopy4_1<int>(av, (1 << 16), true); // 256 KB
    result &= TestCopy4_1<int>(av, (1 << 20), true); // 4 MB

    // Test sync and async copy from array_view without data source to an output iterator
    result &= TestCopy5<int>((1 << 16), false); // 256 KB
    result &= TestCopy5<int>((1 << 20), false); // 4 MB
    result &= TestCopy5<int>((1 << 16), true); // 256 KB
    result &= TestCopy5<int>((1 << 20), true); // 4 MB

    // Test sync and async copy from uninitialized array_view without data source to an output iterator
    result &= TestCopy6<int>(false);
    result &= TestCopy6<int>(true);

    return !result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Wrong scripts to cause unknown issues in kernel.cl
//
#if 0
// R1UN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O3 -o %t.ll && mkdir -p %t
// R1UN: %llc -march=c -o %t/kernel_.cl < %t.ll
// R1UN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// R1UN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// R1UN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////

