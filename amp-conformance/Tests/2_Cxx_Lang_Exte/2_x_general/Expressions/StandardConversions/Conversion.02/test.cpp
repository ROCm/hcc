// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test check the Integral and Floating point promotions for arithmetic operations</summary>

#include <cstdio>
#include <math.h>
#include <limits>
#include <typeinfo>
#include <assert.h>
#include <iostream>
#include <amptest.h>

#define DEBUG 0

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// IsEqual is generic equality test for all types
template<typename T>
bool IsEqual(T v1, T v2)
{
    if (v1 != v2)
    {
        return false;
    }
    return true;
}

// IsEqual specialization for floats
template<>
bool IsEqual(float v1, float v2)
{
    // Max difference values picked based on input values we are dealing with
    const float maxAbsoluteDiff = 0.000005;
    const float maxRelativeDiff = 0.0001; // (0.01%)

    if (fabs(v1 - v2) < maxAbsoluteDiff)
    {
        return true;
    }

    float relativeDiff = 0.0f;
    if (fabs(v1) > fabs(v2))
    {
        relativeDiff = fabs((v1 - v2) / v1);
    }
    else
    {
        relativeDiff = fabs((v2 - v1) / v2);
    }

    if (relativeDiff < maxRelativeDiff)
    {

        return true;
    }

    if (DEBUG)
    {
        printf("Failed float comaparison. Relative Diffrence is %.10f\n", relativeDiff);
    }
    return false;
}

// IsEqual specialization for doubles
template<>
bool IsEqual(double v1, double v2)
{
    // Max difference values picked based on input values we are dealing with
    const double maxAbsoluteDiff = 0.00000000005;
    const double maxRelativeDiff = 0.0001; // (0.01%)

    if (fabs(v1 - v2) < maxAbsoluteDiff)
    {
        return true;
    }

    float relativeDiff = 0.0f;
    if (fabs(v1) > fabs(v2))
    {
        relativeDiff = fabs((v1 - v2) / v1);
    }
    else
    {
        relativeDiff = fabs((v2 - v1) / v2);
    }

    if (relativeDiff < maxRelativeDiff)
    {
        return true;
    }

    if (DEBUG)
    {
        printf("Failed double comparison. Relative Diffrence is %.10f\n", relativeDiff);
    }
    return false;
}

template<typename T, typename R>
bool VerifyConversion(T input, R gpu_result)
{
    R cpu_result = input;

    // DX11 specific: doubles on gpu are first converted to float, then to uint or int
    // Additionally if destination type is double then we have to convert it to float first
    if (typeid(R) == typeid(double) || typeid(T) == typeid(double))
    {
        float f = input;
        cpu_result = f;
    }

    if (DEBUG)
    {
        printf("(generic) verification %s to %s\n", typeid(T).name(), typeid(R).name());
        std::cout << "input:" << input << " gpu_result:" << gpu_result << " cpu_result:" << cpu_result << std::endl;
    }

    bool result = false;

    if (typeid(input) == typeid(float))
    {
        // If right side of our arithmetic calculation is float, then lets do floats comparison
        // e.g. int = unsigned int + float + unsigned int * float
        result = IsEqual(static_cast<float>(cpu_result), static_cast<float>(gpu_result));
    }
    else if (typeid(input) == typeid(double))
    {
        // If right side of our arithmetic calculation is double, then lets do doubles comparison
        // e.g. unsigned int = int + double + int * double
        result = IsEqual(static_cast<double>(cpu_result), static_cast<double>(gpu_result));
    }
    else
    {
        // In this case cpu_result and gpu_result types has to match
        result = IsEqual(cpu_result, gpu_result);
    }

    return result;
}

// Initialize the input with random data
template<typename dstType, typename srcType1, typename srcType2>
void InitializeArrays(vector<srcType1> &vInput1, vector<srcType2> &vInput2, int size)
{
    // Pick some arbitrary min/max values for random numbers
    double max = 10000;
    double min = -10000;

    //Make sure that we don't generate negative numbers if unsigned int is destination type
    if (typeid(dstType) == typeid(unsigned int))
    {
        min = 0;
    }

    // Random within range for values in the middle
    for(int i=0; i<size; ++i)
    {
        int min1 = min;
        double scale = rand() / static_cast<double>(RAND_MAX);

        // min for unsigned int has to be adjusted
        if (typeid(srcType1) == typeid(unsigned int))
        {
            min1 = 0;
        }

        vInput1[i] = static_cast<srcType1>(scale * (max - min1) + min1);

        if (DEBUG)
        {
            std::cout << "generated input[" << i << "]:" << vInput1[i] << std::endl;
        }
    }

    for(int i=0; i<size; ++i)
    {
        int min2 = min;
        double scale = rand() / static_cast<double>(RAND_MAX);

        // min for unsigned int has to be adjusted
        if (typeid(srcType2) == typeid(unsigned int))
        {
            min2 = 0;
        }

        vInput2[i] = static_cast<srcType2>(scale * (max - min2) + min2);

        if (DEBUG)
        {
            std::cout << "generated input[" << i << "]:" << vInput2[i] << std::endl;
        }
    }
}

template<typename dstType, typename srcType1, typename srcType2>
void arithmetic_conversion( dstType &c, srcType1 a, srcType2 b) __GPU
{
    c = a + b + a * b;
}

template<typename dstType, typename srcType1, typename srcType2>
bool test_arithmetic_conversion()
{
    const int size = 1024;

    vector<dstType> C(size);
    vector<srcType1> A(size);
    vector<srcType2> B(size);

    // Use reference since some devices do not support double.
    accelerator device = require_device_with_double(Device::ALL_DEVICES);
    accelerator_view rv = device.get_default_view();

    // Initialize input
    InitializeArrays<dstType, srcType1, srcType2>(A, B, size);

    Concurrency::extent<1> e(size);

    array<srcType1, 1> aA(e, A.begin(), A.end(), rv);
    array<srcType2, 1> aB(e, B.begin(), B.end(), rv);
    array<dstType, 1> aC(e, rv);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) __GPU {
        arithmetic_conversion<dstType, srcType1, srcType2>(aC[idx], aA[idx], aB[idx]);
    });

    C = aC;

    bool passed = true;

    // Verify results
    for (int i = 0; i < size; i++)
    {
        auto input = A[i] + B[i] + A[i] * B[i];

        if (!VerifyConversion(input, C[i]))
        {
            passed = false;
            break;
        }
    }

    printf("test: %s = %s + %s + %s * %s: %s\n", typeid(dstType).name(), typeid(srcType1).name(), \
        typeid(srcType2).name(), typeid(srcType1).name(), typeid(srcType2).name(), passed?"pass":"fail");

    return passed;
}

// Main entry point
int main(int argc, char **argv)
{
    srand(2010);
    std::cout.setf(std::ios::fixed | std::ios::showpoint);

    bool result = true;

    result = test_arithmetic_conversion<int, unsigned int, float>() ? result: false;
    result = test_arithmetic_conversion<unsigned int, int, float>() ? result: false;
    result = test_arithmetic_conversion<float, int, unsigned int>() ? result: false;

    return !result;

}

