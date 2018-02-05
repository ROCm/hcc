// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Control Flow test: test max nesting level for do-while statements </summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

#define MAX_NESTING_LEVEL 63

// a == nesting level + 1 == specifies the number of if statements executed here
void kernel(index<1> idx, array<int, 1> & fA, int a, int initialValue, int newValue) __GPU
{
    fA[idx] = initialValue;

    if(a > 1) if(a > 2) if(a > 3) if(a > 4) if(a > 5) if(a > 6) if(a > 7) if(a > 8) if(a > 9) if(a > 10)
        if(a > 11) if(a > 12) if(a > 13) if(a > 14) if(a > 15) if(a > 16) if(a > 17) if(a > 18) if(a > 19) if(a > 20)
            if(a > 21) if(a > 22) if(a > 23) if(a > 24) if(a > 25) if(a > 26) if(a > 27) if(a > 28) if(a > 29) if(a > 30)
                if(a > 31) if(a > 32) if(a > 33) if(a > 34) if(a > 35) if(a > 36) if(a > 37) if(a > 38) if(a > 39) if(a > 40)
                    if(a > 41) if(a > 42) if(a > 43) if(a > 44) if(a > 45) if(a > 46) if(a > 47) if(a > 48) if(a > 49) if(a > 50)
                        if(a > 51) if(a > 52) if(a > 53) if(a > 54) if(a > 55) if(a > 56) if(a > 57) if(a > 58) if(a > 59) if(a > 60)
                            if(a > 61) if(a > 62) if(a > 63) /*if(a > 64) if(a > 65) if(a > 66) if(a > 67) if(a > 68) if(a > 69) if(a > 70)
                                                             if(a > 71) if(a > 72) if(a > 73) if(a > 74) if(a > 75) if(a > 76) if(a > 77) if(a > 78) if(a > 79) if(a > 80)
                                                             if(a > 81) if(a > 82) if(a > 83) if(a > 84) if(a > 85) if(a > 86) if(a > 87) if(a > 88) if(a > 89) if(a > 90)
                                                             if(a > 91) if(a > 92) if(a > 93) if(a > 94) if(a > 95) if(a > 96) if(a > 97) if(a > 98) if(a > 99) if(a > 100)
                                                             if(a > 101) if(a > 102) if(a > 103) if(a > 104) if(a > 105) if(a > 106) if(a > 107) if(a > 108) if(a > 109) if(a > 110)
                                                             if(a > 111) if(a > 112) if(a > 113) if(a > 114) if(a > 115) if(a > 116) if(a > 117) if(a > 118) if(a > 119) if(a > 120)
                                                             if(a > 121) if(a > 122)*/
                            {
                                fA[idx] = newValue;
                            }
}

runall_result test(int level, int initialValue, int newValue, int expected)
{
    const size_t size = 255;
    int * A = new int[size];

    accelerator_view av =  require_device(Device::ALL_DEVICES).get_default_view();

    extent<1> vector(size);
    array<int, 1> fA(vector, av);

    parallel_for_each(fA.get_extent(), [&,level,initialValue, newValue](index<1> idx) __GPU {
        kernel(idx, fA, level, initialValue, newValue);
    });


    copy(fA, A);

    bool passed = true;
    for(int i = 0; i < size;i++)
    {
        if(A[i] != expected)
        {
            fprintf(stderr, "A[%d] = %d. Expected: %d\n", i, A[i], expected);
            passed = false;
            break;
        }
    }

    delete[] A;

    printf("Number of conditions: [%d] - %s\n", level, passed? "Passed!" : "Failed!");

    return passed;
}

runall_result test_main()
{
    int initialValue = 0;
    int newValue = 1;
    int expected = initialValue;

    for(int level = 1; level <= MAX_NESTING_LEVEL; level++)
    {
        runall_result result = test(level, initialValue, newValue, expected);
        if(result != runall_pass)
        {
            return result;
        }
    }

    expected = newValue;
    runall_result ret;

    ret = test(MAX_NESTING_LEVEL + 1, initialValue, newValue, expected);


    return ret;
}
