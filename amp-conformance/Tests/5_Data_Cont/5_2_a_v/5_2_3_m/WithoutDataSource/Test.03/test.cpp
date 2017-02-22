// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test the correctness of C++ AMP copy APIs involving array_views without a data source</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>
#include <amp_short_vectors.h>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;

// Overloaded std::ostream::operator<< for graphics::int_2 type
std::ostream& operator<<(std::ostream &outStream, const graphics::int_3 &val)
{
    outStream << "(" << val.get_x() << ", " << val.get_y() << ", " << val.get_z() << ")";
    return outStream;
}

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
                    Log(LogType::Info, true) << "destArrayView(" << i << ", " << j << ") = " << tempArrayView(BORDER + i, BORDER + j) << ", Expected = " << (T)((i * WIDTH) + j) << std::endl;
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
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
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
                    Log(LogType::Info, true) << "destArrayView(" << i << ", " << j << ") = " << tempArrayView(BORDER + i, BORDER + j) << ", Expected = " << srcVec[(i * WIDTH) + j] << std::endl;
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
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
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
                    Log(LogType::Info, true) << "destArrayView(" << i << ", " << j << ") = " << tempArrayView(BORDER + i, BORDER + j) << ", Expected = " << srcVec[(i * WIDTH) + j] << std::endl;
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
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
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
                    Log(LogType::Info, true) << "destArrayView(" << i << ", " << j << ") = " << tempArrayView(BORDER + i, BORDER + j) << ", Expected = " << (T)((i * WIDTH) + j) << std::endl;
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
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
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
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
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
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
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
                Log(LogType::Info, true) << "destIter[" << i << "] = " << destIter[i] << ", Expected = " << (T)(i) << std::endl;
                passed = false;
            }
        }

        return passed;
    };

    bool passed = false;
    if (async)
    {
        auto fut = copy_async(srcArrayView, destIter);
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
    }
    else
    {
        copy(srcArrayView, destIter);
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
        auto fut = copy_async(srcArrayView, destIter);
        passed = fut.to_task().then([&]() -> bool { return verificationFunc(); }).get();
    }
    else
    {
        copy(srcArrayView, destIter);
        passed = verificationFunc();
    }

    delete [] destIter;
    return passed;
}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    runall_result result;

#ifdef Copy1
    // Test sync and async copy from array to array_view without data source
    result &= REPORT_RESULT(TestCopy1<int>(av, (1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy1<graphics::int_3>(av, (1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy1<graphics::int_3>(av, (1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy1<int>(av, (1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy1<graphics::int_3>(av, (1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy1<int>(av, (1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy1<graphics::int_3>(av, (1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy1<graphics::int_3>(av, (1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy1<int>(av, (1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy1<graphics::int_3>(av, (1 << 20), true)); // 12 MB
#endif

#ifdef Copy2
    // Test sync and async copy from begin and end iterators to array_view without data source
    result &= REPORT_RESULT(TestCopy2<int>((1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy2<graphics::int_3>((1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy2<graphics::int_3>((1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy2<int>((1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy2<graphics::int_3>((1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy2<int>((1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy2<graphics::int_3>((1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy2<graphics::int_3>((1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy2<int>((1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy2<graphics::int_3>((1 << 20), true)); // 12 MB
#endif

#ifdef Copy2_1
	// Test sync and async copy from begin iterator to array_view without data source
    result &= REPORT_RESULT(TestCopy2_1<int>((1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy2_1<graphics::int_3>((1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy2_1<graphics::int_3>((1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy2_1<int>((1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy2_1<graphics::int_3>((1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy2_1<int>((1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy2_1<graphics::int_3>((1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy2_1<graphics::int_3>((1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy2_1<int>((1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy2_1<graphics::int_3>((1 << 20), true)); // 12 MB
#endif

#ifdef Copy3
    // Test sync and async copy from array_view to array_view without data source
    result &= REPORT_RESULT(TestCopy3<int>((1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy3<graphics::int_3>((1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy3<graphics::int_3>((1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy3<int>((1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy3<graphics::int_3>((1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy3<int>((1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy3<graphics::int_3>((1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy3<graphics::int_3>((1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy3<int>((1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy3<graphics::int_3>((1 << 20), true)); // 12 MB
#endif

#ifdef Copy4
    // Test sync and async copy from array_view without data source to an array
    result &= REPORT_RESULT(TestCopy4<int>(av, (1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy4<graphics::int_3>(av, (1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy4<graphics::int_3>(av, (1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy4<int>(av, (1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy4<graphics::int_3>(av, (1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy4<int>(av, (1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy4<graphics::int_3>(av, (1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy4<graphics::int_3>(av, (1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy4<int>(av, (1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy4<graphics::int_3>(av, (1 << 20), true)); // 12 MB
#endif

#ifdef Copy4_1
	 // Test sync and async copy from array_view without data source to an array view
    result &= REPORT_RESULT(TestCopy4_1<int>(av, (1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy4_1<graphics::int_3>(av, (1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy4_1<graphics::int_3>(av, (1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy4_1<int>(av, (1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy4_1<graphics::int_3>(av, (1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy4_1<int>(av, (1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy4_1<graphics::int_3>(av, (1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy4_1<graphics::int_3>(av, (1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy4_1<int>(av, (1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy4_1<graphics::int_3>(av, (1 << 20), true)); // 12 MB
#endif

#ifdef Copy5
    // Test sync and async copy from array_view without data source to an output iterator
    result &= REPORT_RESULT(TestCopy5<int>((1 << 16), false)); // 256 KB
    result &= REPORT_RESULT(TestCopy5<graphics::int_3>((1 << 16), false)); // 768 KB
    result &= REPORT_RESULT(TestCopy5<graphics::int_3>((1 << 18), false)); // 3 MB
    result &= REPORT_RESULT(TestCopy5<int>((1 << 20), false)); // 4 MB
    result &= REPORT_RESULT(TestCopy5<graphics::int_3>((1 << 20), false)); // 12 MB
    result &= REPORT_RESULT(TestCopy5<int>((1 << 16), true)); // 256 KB
    result &= REPORT_RESULT(TestCopy5<graphics::int_3>((1 << 16), true)); // 768 KB
    result &= REPORT_RESULT(TestCopy5<graphics::int_3>((1 << 18), true)); // 3 MB
    result &= REPORT_RESULT(TestCopy5<int>((1 << 20), true)); // 4 MB
    result &= REPORT_RESULT(TestCopy5<graphics::int_3>((1 << 20), true)); // 12 MB
#endif

#ifdef Copy6
    // Test sync and async copy from uninitialized array_view without data source to an output iterator
    result &= REPORT_RESULT(TestCopy6<int>(false));
    result &= REPORT_RESULT(TestCopy6<graphics::int_3>(false));
    result &= REPORT_RESULT(TestCopy6<int>(true));
    result &= REPORT_RESULT(TestCopy6<graphics::int_3>(true));
#endif

    return result;
}

