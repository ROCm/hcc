// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Control Flow test: test max nesting level of while statement</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

#define MAX_NESTING_LEVEL 63

// x == nesting level + 1 == specifies the number of if statements executed here
void kernel(index<1> idx, array<int, 1> & fA, int x, int initialValue, int newValue) __GPU
{
    fA[idx] = initialValue;

    while(x > 1) { while(x > 2) { while(x > 3) { while(x > 4) { while(x > 5) {
        while(x > 6) { while(x > 7) { while(x > 8) { while(x > 9) { while(x > 10) {
            while(x > 11) { while(x > 12) { while(x > 13) { while(x > 14) { while(x > 15) {
                while(x > 16) { while(x > 17) { while(x > 18) { while(x > 19) { while(x > 20) {
                    while(x > 21) { while(x > 22) { while(x > 23) { while(x > 24) { while(x > 25) {
                        while(x > 26) { while(x > 27) { while(x > 28) { while(x > 29) { while(x > 30) {
                            while(x > 31) { while(x > 32) { while(x > 33) { while(x > 34) { while(x > 35) {
                                while(x > 36) { while(x > 37) { while(x > 38) { while(x > 39) { while(x > 40) {
                                    while(x > 41) { while(x > 42) { while(x > 42) { while(x > 43) { while(x > 44) {
                                        while(x > 45) { while(x > 46) { while(x > 47) { while(x > 48) { while(x > 49) {
                                            while(x > 50) { while(x > 51) { while(x > 52) { while(x > 53) { while(x > 54) {
                                                while(x > 55) { while(x > 56) { while(x > 57) { while(x > 58) { while(x > 59) {
                                                    while(x > 60) { while(x > 61) { while(x > 62) { while(x > 63) { /*while(x > 64) {
                                                                                                                    while(x > 65) { while(x > 66) { while(x > 67) { while(x > 68) { while(x > 69) {
                                                                                                                    while(x > 70) { while(x > 71) { while(x > 72) { while(x > 73) { while(x > 74) {
                                                                                                                    while(x > 75) { while(x > 76) { while(x > 77) { while(x > 78) { while(x > 79) {
                                                                                                                    while(x > 80) { while(x > 81) { while(x > 82) { while(x > 83) { while(x > 84) {
                                                                                                                    while(x > 85) { while(x > 86) { while(x > 87) { while(x > 88) { while(x > 89) {
                                                                                                                    while(x > 90) { while(x > 91) { while(x > 92) { while(x > 93) { while(x > 94) {
                                                                                                                    while(x > 95) { while(x > 96) { while(x > 97) { while(x > 98) { while(x > 99) {
                                                                                                                    while(x > 100) { while(x > 101) { while(x > 102) { while(x > 103) { while(x > 104) {
                                                                                                                    while(x > 105) { while(x > 106) { while(x > 107) { while(x > 108) { while(x > 109) {
                                                                                                                    while(x > 110) { while(x > 111) {*/
                                                        fA[idx] = newValue;
                                                        break;} break;} break;} break;} break;}
                                                break;} break;} break;} break;} break;}
                                            break;} break;} break;} break;} break;}
                                        break;} break;} break;} break;} break;}
                                    break;} break;} break;} break;} break;}
                                break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                        break;} break;} break;} break;} break;}
                    break;} break;} break;} break;} break;}
                break;} break;} break;} break;} break;}
            break;} break;} break;} break;} break;}
        break;} break;} break;} break;} break;}
    break;} break;} break;} /*break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} break;} break;} break;} break;}
                            break;} */break;}
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
            break;}
    }

    expected = newValue;

    return test(MAX_NESTING_LEVEL + 1, initialValue, newValue, expected);
}
