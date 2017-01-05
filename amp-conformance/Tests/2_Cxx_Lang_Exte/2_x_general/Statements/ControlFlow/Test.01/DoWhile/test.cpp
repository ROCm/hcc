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

#define MAX_NESTING_LEVEL 63   // current supported max

// x specifies the number of if statements executed here
void kernel(index<1> idx, array<int, 1> & fA, int x, int initialValue, int newValue) __GPU
{
    fA[idx] = initialValue;

    do
    {
        if(x <= 1)   break; do { if(x <= 2)   break; do { if(x <= 3)   break; do { if(x <= 4)   break; do { if(x <= 5)   break; do {
            if(x <= 6)   break; do { if(x <= 7)   break; do { if(x <= 8)   break; do { if(x <= 9)   break; do { if(x <= 10)  break; do {
                if(x <= 11)  break; do { if(x <= 12)  break; do { if(x <= 13)  break; do { if(x <= 14)  break; do { if(x <= 15)  break; do {
                    if(x <= 16)  break; do { if(x <= 17)  break; do { if(x <= 18)  break; do { if(x <= 19)  break; do { if(x <= 20)  break; do {
                        if(x <= 21)  break; do { if(x <= 22)  break; do { if(x <= 23)  break; do { if(x <= 24)  break; do { if(x <= 25)  break; do {
                            if(x <= 26)  break; do { if(x <= 27)  break; do { if(x <= 28)  break; do { if(x <= 29)  break; do { if(x <= 30)  break; do {
                                if(x <= 31)  break; do { if(x <= 32)  break; do { if(x <= 33)  break; do { if(x <= 34)  break; do { if(x <= 35)  break; do {
                                    if(x <= 36)  break; do { if(x <= 37)  break; do { if(x <= 38)  break; do { if(x <= 39)  break; do { if(x <= 40)  break; do {
                                        if(x <= 41)  break; do { if(x <= 42)  break; do { if(x <= 43)  break; do { if(x <= 44)  break; do {
                                            if(x <= 45)  break; do { if(x <= 46)  break; do { if(x <= 47)  break; do { if(x <= 48)  break; do { if(x <= 49)  break; do {
                                                if(x <= 50)  break; do { if(x <= 51)  break; do { if(x <= 52)  break; do { if(x <= 53)  break; do { if(x <= 54)  break; do {
                                                    if(x <= 55)  break; do { if(x <= 56)  break; do { if(x <= 57)  break; do { if(x <= 58)  break; do { if(x <= 59)  break; do {
                                                        if(x <= 60)  break; do { if(x <= 61)  break; do { if(x <= 62)  break; do { if(x <= 63)  break; /*do { if(x <= 64)  break; do {
                                                                                                                                                       if(x <= 65)  break; do { if(x <= 66)  break; do { if(x <= 67)  break; do { if(x <= 68)  break; do { if(x <= 69)  break; do {
                                                                                                                                                       if(x <= 70)  break; do { if(x <= 71)  break; do { if(x <= 72)  break; do { if(x <= 73)  break; do { if(x <= 74)  break; do {
                                                                                                                                                       if(x <= 75)  break; do { if(x <= 76)  break; do { if(x <= 77)  break; do { if(x <= 78)  break; do { if(x <= 79)  break; do {
                                                                                                                                                       if(x <= 80)  break; do { if(x <= 81)  break; do { if(x <= 82)  break; do { if(x <= 83)  break; do { if(x <= 84)  break; do {
                                                                                                                                                       if(x <= 85)  break; do { if(x <= 86)  break; do { if(x <= 87)  break; do { if(x <= 88)  break; do { if(x <= 89)  break; do {
                                                                                                                                                       if(x <= 90)  break; do { if(x <= 91)  break; do { if(x <= 92)  break; do { if(x <= 93)  break; do { if(x <= 94)  break; do {
                                                                                                                                                       if(x <= 95)  break; do { if(x <= 96)  break; do { if(x <= 97)  break; do { if(x <= 98)  break; do { if(x <= 99)  break; do {
                                                                                                                                                       if(x <= 100) break; do { if(x <= 101) break; do { if(x <= 102) break; do { if(x <= 103) break; do { if(x <= 104) break; do {
                                                                                                                                                       if(x <= 105) break; do { if(x <= 106) break; do { if(x <= 107) break; do { if(x <= 108) break; do { if(x <= 109) break; do {
                                                                                                                                                       if(x <= 110) break; do { */
                                                        fA[idx] = newValue;
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                        break;} while(true); break;} while(true); break;} while(true); break;} while(true); /*break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
                                                                                                                                            break;} while(true); break;} while(true);*/
}

runall_result test(int level, int initialValue, int newValue, int expected)
{
    const size_t size = 1;
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
        while(A[i] != expected)
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
