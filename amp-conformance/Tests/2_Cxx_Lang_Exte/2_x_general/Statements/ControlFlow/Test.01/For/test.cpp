// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Control Flow test: test nesting level of 'for' loops</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

#define MAX_NESTING_LEVEL 63

// x == nesting level + 1 == specifies the number of if statements executed here
void kernel(index<1> idx, array<int,1> & fA, int x, int initialValue, int newValue) __GPU
{
    fA[idx] = initialValue;

    for(;x > 1;)  { for(;x > 2;)  { for(;x > 3;)  { for(;x > 4;)  { for(;x > 5;)  {
        for(;x > 6;)  { for(;x > 7;)  { for(;x > 8;)  { for(;x > 9;)  { for(;x > 10;) {
            for(;x > 11;) { for(;x > 12;) { for(;x > 13;) { for(;x > 14;) { for(;x > 15;) {
                for(;x > 16;) { for(;x > 17;) { for(;x > 18;) { for(;x > 19;) { for(;x > 20;) {
                    for(;x > 21;) { for(;x > 22;) { for(;x > 23;) { for(;x > 24;) { for(;x > 25;) {
                        for(;x > 26;) { for(;x > 27;) { for(;x > 28;) { for(;x > 29;) { for(;x > 30;) {
                            for(;x > 31;) { for(;x > 32;) { for(;x > 33;) { for(;x > 34;) { for(;x > 35;) {
                                for(;x > 36;) { for(;x > 37;) { for(;x > 38;) { for(;x > 39;) { for(;x > 40;) {
                                    for(;x > 41;) { for(;x > 42;) { for(;x > 42;) { for(;x > 43;) { for(;x > 44;) {
                                        for(;x > 45;) { for(;x > 46;) { for(;x > 47;) { for(;x > 48;) { for(;x > 49;) {
                                            for(;x > 50;) { for(;x > 51;) { for(;x > 52;) { for(;x > 53;) { for(;x > 54;) {
                                                for(;x > 55;) { for(;x > 56;) { for(;x > 57;) { for(;x > 58;) { for(;x > 59;) {
                                                    for(;x > 60;) { for(;x > 61;) { for(;x > 62;) { for(;x > 63;) { /*for(;x > 64;) {
                                                                                                                    for(;x > 65;) { for(;x > 66;) { for(;x > 67;) { for(;x > 68;) { for(;x > 69;) {
                                                                                                                    for(;x > 70;) { for(;x > 71;) { for(;x > 72;) { for(;x > 73;) { for(;x > 74;) {
                                                                                                                    for(;x > 75;) { for(;x > 76;) { for(;x > 77;) { for(;x > 78;) { for(;x > 79;) {
                                                                                                                    for(;x > 80;) { for(;x > 81;) { for(;x > 82;) { for(;x > 83;) { for(;x > 84;) {
                                                                                                                    for(;x > 85;) { for(;x > 86;) { for(;x > 87;) { for(;x > 88;) { for(;x > 89;) {
                                                                                                                    for(;x > 90;) { for(;x > 91;) { for(;x > 92;) { for(;x > 93;) { for(;x > 94;) {
                                                                                                                    for(;x > 95;) { for(;x > 96;) { for(;x > 97;) { for(;x > 98;) { for(;x > 99;) {
                                                                                                                    for(;x > 100;) { for(;x > 101;) { for(;x > 102;) { for(;x > 103;) { for(;x > 104;) {
                                                                                                                    for(;x > 105;) { for(;x > 106;) { for(;x > 107;) { for(;x > 108;) { for(;x > 109;) {
                                                                                                                    for(;x > 110;) { for(;x > 111;)
                                                                                                                    {*/
                                                        fA[idx] = newValue;
                                                        break;
                                                    }
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
                            break;}    */
}

runall_result test(int level, int initialValue, int newValue, int expected)
{
    const size_t size = 1;
    int * A = new int[size];

    accelerator_view av =  require_device(Device::ALL_DEVICES).get_default_view();

    extent<1> vector(size);
    array<int,1> fA(vector, av);

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
    int InitialValue = 0;
    int NewValue = 1;
    int Expected = InitialValue;

    for(int level = 1; level <= MAX_NESTING_LEVEL; level++)
    {
        runall_result result = test(level, InitialValue, NewValue, Expected);
        if(result != runall_pass)
        {
            return result;
        }
    }

    Expected = NewValue;
    runall_result ret;

    ret = test(MAX_NESTING_LEVEL + 1, InitialValue, NewValue, Expected);

    return ret;
}
