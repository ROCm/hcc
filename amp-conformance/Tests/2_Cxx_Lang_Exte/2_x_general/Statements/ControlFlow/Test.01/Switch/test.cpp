// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Control Flow test: test max nesting level of switch statements in DPC code</summary>
//#Expects: Error: error C3566

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

#define MAX_NESTING_LEVEL 62

// x == nesting level + 1 == specifies the number of if statements executed here
void kernel(index<1> idx, array<int, 1> & fA, int x, int initialValue, int newValue) __GPU
{
    fA[idx] = initialValue;

    switch(x > 1? 1:0)  { case 0: break; case 1: switch(x > 2? 1:0)  { case 0: break; case 1: 
        switch(x > 3? 1:0)  { case 0: break; case 1: switch(x > 4? 1:0)  { case 0: break; case 1: 
        switch(x > 5? 1:0)  { case 0: break; case 1: switch(x > 6? 1:0)  { case 0: break; case 1: 
        switch(x > 7? 1:0)  { case 0: break; case 1: switch(x > 8? 1:0)  { case 0: break; case 1: 
        switch(x > 9? 1:0)  { case 0: break; case 1: switch(x > 10? 1:0) { case 0: break; case 1: 
        switch(x > 11? 1:0) { case 0: break; case 1: switch(x > 12? 1:0) { case 0: break; case 1: 
        switch(x > 13? 1:0) { case 0: break; case 1: switch(x > 14? 1:0) { case 0: break; case 1: 
        switch(x > 15? 1:0) { case 0: break; case 1: switch(x > 16? 1:0) { case 0: break; case 1: 
        switch(x > 17? 1:0) { case 0: break; case 1: switch(x > 18? 1:0) { case 0: break; case 1: 
        switch(x > 19? 1:0) { case 0: break; case 1: switch(x > 20? 1:0) { case 0: break; case 1: 
        switch(x > 21? 1:0) { case 0: break; case 1: switch(x > 22? 1:0) { case 0: break; case 1: 
        switch(x > 23? 1:0) { case 0: break; case 1: switch(x > 24? 1:0) { case 0: break; case 1: 
        switch(x > 25? 1:0) { case 0: break; case 1: switch(x > 26? 1:0) { case 0: break; case 1: 
        switch(x > 27? 1:0) { case 0: break; case 1: switch(x > 28? 1:0) { case 0: break; case 1: 
        switch(x > 29? 1:0) { case 0: break; case 1: switch(x > 30? 1:0) { case 0: break; case 1: 
        switch(x > 31? 1:0) { case 0: break; case 1: switch(x > 32? 1:0) { case 0: break; case 1: 
        switch(x > 33? 1:0) { case 0: break; case 1: switch(x > 34? 1:0) { case 0: break; case 1: 
        switch(x > 35? 1:0) { case 0: break; case 1: switch(x > 36? 1:0) { case 0: break; case 1: 
        switch(x > 37? 1:0) { case 0: break; case 1: switch(x > 38? 1:0) { case 0: break; case 1: 
        switch(x > 39? 1:0) { case 0: break; case 1: switch(x > 40? 1:0) { case 0: break; case 1: 
        switch(x > 41? 1:0) { case 0: break; case 1: switch(x > 42? 1:0) { case 0: break; case 1: 
        switch(x > 42? 1:0) { case 0: break; case 1: switch(x > 43? 1:0) { case 0: break; case 1: 
        switch(x > 44? 1:0) { case 0: break; case 1: switch(x > 45? 1:0) { case 0: break; case 1: 
        switch(x > 46? 1:0) { case 0: break; case 1: switch(x > 47? 1:0) { case 0: break; case 1: 
        switch(x > 48? 1:0) { case 0: break; case 1: switch(x > 49? 1:0) { case 0: break; case 1: 
        switch(x > 50? 1:0) { case 0: break; case 1: switch(x > 51? 1:0) { case 0: break; case 1: 
        switch(x > 52? 1:0) { case 0: break; case 1: switch(x > 53? 1:0) { case 0: break; case 1: 
        switch(x > 54? 1:0) { case 0: break; case 1: switch(x > 55? 1:0) { case 0: break; case 1: 
        switch(x > 56? 1:0) { case 0: break; case 1: switch(x > 57? 1:0) { case 0: break; case 1: 
        switch(x > 58? 1:0) { case 0: break; case 1: switch(x > 59? 1:0) { case 0: break; case 1: 
        switch(x > 60? 1:0) { case 0: break; case 1: switch(x > 61? 1:0) { case 0: break; case 1: 
        switch(x > 62? 1:0) { case 0: break; case 1: /*switch(x > 63? 1:0) { case 0: break; case 1: 
                                                     switch(x > 64? 1:0) { case 0: break; case 1: switch(x > 65? 1:0) { case 0: break; case 1: 
                                                     switch(x > 66? 1:0) { case 0: break; case 1: switch(x > 67? 1:0) { case 0: break; case 1: 
                                                     switch(x > 68? 1:0) { case 0: break; case 1: switch(x > 69? 1:0) { case 0: break; case 1: 
                                                     switch(x > 70? 1:0) { case 0: break; case 1: switch(x > 71? 1:0) { case 0: break; case 1: 
                                                     switch(x > 72? 1:0) { case 0: break; case 1: switch(x > 73? 1:0) { case 0: break; case 1: 
                                                     switch(x > 74? 1:0) { case 0: break; case 1: switch(x > 75? 1:0) { case 0: break; case 1: 
                                                     switch(x > 76? 1:0) { case 0: break; case 1: switch(x > 77? 1:0) { case 0: break; case 1: 
                                                     switch(x > 78? 1:0) { case 0: break; case 1: switch(x > 79? 1:0) { case 0: break; case 1: 
                                                     switch(x > 80? 1:0) { case 0: break; case 1: switch(x > 81? 1:0) { case 0: break; case 1: 
                                                     switch(x > 82? 1:0) { case 0: break; case 1: switch(x > 83? 1:0) { case 0: break; case 1: 
                                                     switch(x > 84? 1:0) { case 0: break; case 1: switch(x > 85? 1:0) { case 0: break; case 1: 
                                                     switch(x > 86? 1:0) { case 0: break; case 1: switch(x > 87? 1:0) { case 0: break; case 1: 
                                                     switch(x > 88? 1:0) { case 0: break; case 1: switch(x > 89? 1:0) { case 0: break; case 1: 
                                                     switch(x > 90? 1:0) { case 0: break; case 1: switch(x > 91? 1:0) { case 0: break; case 1: 
                                                     switch(x > 92? 1:0) { case 0: break; case 1: switch(x > 93? 1:0) { case 0: break; case 1: 
                                                     switch(x > 94? 1:0) { case 0: break; case 1: switch(x > 95? 1:0) { case 0: break; case 1: 
                                                     switch(x > 96? 1:0) { case 0: break; case 1: switch(x > 97? 1:0) { case 0: break; case 1: 
                                                     switch(x > 98? 1:0) { case 0: break; case 1: switch(x > 99? 1:0) { case 0: break; case 1: 
                                                     switch(x > 100? 1:0) { case 0: break; case 1: switch(x > 101? 1:0) { case 0: break; case 1: 
                                                     switch(x > 102? 1:0) { case 0: break; case 1: switch(x > 103? 1:0) { case 0: break; case 1: 
                                                     switch(x > 104? 1:0) { case 0: break; case 1: switch(x > 105? 1:0) { case 0: break; case 1: 
                                                     switch(x > 106? 1:0) { case 0: break; case 1: switch(x > 107? 1:0) { case 0: break; case 1: 
                                                     switch(x > 108? 1:0) { case 0: break; case 1: switch(x > 109? 1:0) { case 0: break; case 1: 
                                                     switch(x > 110? 1:0) { case 0: break; case 1: switch(x > 111? 1:0) { case 0: break; case 1: */
                                                     fA[idx] = newValue;
    break;    
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    /*}}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}
    }}   
    }*/}
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
        if(A[i] != expected)
        {
            fprintf(stderr, "A[%Id] = %d. Expected: %d\n", i, A[i], expected);
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
    return test(MAX_NESTING_LEVEL + 1, initialValue, newValue, expected);
}
