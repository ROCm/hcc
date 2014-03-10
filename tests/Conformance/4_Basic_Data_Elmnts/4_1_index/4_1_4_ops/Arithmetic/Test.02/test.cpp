// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check binary arithmetic operators between one index and a scalar type</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>

using namespace Concurrency;
using std::vector;

// scalartype (RHS): 1
int test1() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1(data1);
    int dataa[] = {-99, -9, 0, 1, 2, 11, 101};
    int datas[] = {-101, -11, -2, -1, 0, 9, 99};
    int datam[] = {-100, -10, -1, 0, 1, 10, 100};
    int datad[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);
    index<7> id(datad);
    index<7> ir;

    if (!((i1 + 1) == ia))
    {
        return 11;  // test1 scenario1 failed
    }

    if (!((i1 - 1) == is))
    {
        return 12;  // test1 scenario2 failed
    }

    if (!((i1 * 1) == im))
    {
        return 13;  // test1 scenario3 failed
    }

    if (!((i1 / 1) == id))
    {
        return 14;  // test1 scenario4 failed
    }

    if (!((i1 % 1) == ir))
    {
        return 15;  // test1 scenario5 failed
    }
    return 0;
}

// scalartype (RHS): -1
int test2() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1(data1);
    int dataa[] = {-101, -11, -2, -1, 0, 9, 99};
    int datas[] = {-99, -9, 0, 1, 2, 11, 101};
    int datam[] = {100, 10, 1, 0, -1, -10, -100};
    int datad[] = {100, 10, 1, 0, -1, -10, -100};
    int datar[] = {0, 0, 0, 0, 0, 0, 0};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);
    index<7> id(datad);
    index<7> ir(datar);

    if (!((i1 - 1) == ia))
    {
        return 21;  // test2 scenario1 failed
    }

    if (!((i1 + 1) == is))
    {
        return 22;  // test2 scenario2 failed
    }

    if (!((i1 * -1) == im))
    {
        return 23;  // test2 scenario3 failed
    }

    if (!((i1 / -1) == id))
    {
        return 24;  // test2 scenario4 failed
    }

    if (!((i1 % -1) == ir))
    {
        return 25;  // test2 scenario5 failed
    }

    return 0;
}

// scalartype (RHS): 0
int test3() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<7> i1(data1);
    int dataa[] = {-100, -10, -1, 0, 1, 10, 100};
    int datas[] = {-100, -10, -1, 0, 1, 10, 100};
    int datam[] = {0, 0, 0, 0, 0, 0, 0};
    index<7> ia(dataa);
    index<7> is(datas);
    index<7> im(datam);

    if (!((i1 + 0) == ia))
    {
        return 31;  // test3 scenario1 failed
    }

    if (!((i1 - 0) == is))
    {
        return 32;  // test3 scenario2 failed
    }

    if (!((i1 * 0) == im))
    {
        return 33;  // test3 scenario3 failed
    }

    return 0;
}

// scalartype (RHS): -9
int test4() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0, 1, 10, 100};
    index<6> i1(data1);
    int dataa[] = {-109, -19, -10, -9, -8, 1, 91};
    int datas[] = {-91, -1, 8, 9, 10, 19, 109};
    int datam[] = {900, 90, 9, 0, -9, -90, -900};
    int datad[] = {11, 1, 0, 0, 0, -1, -11};
    int datar[] = {-1, -1, -1, 0, 1, 1, 1};
    index<6> ia(dataa);
    index<6> is(datas);
    index<6> im(datam);
    index<6> id(datad);
    index<6> ir(datar);

    if (!((i1 - 9) == ia))
    {
        return 41;  // test4 scenario1 failed
    }

    if (!((i1 + 9) == is))
    {
        return 42;  // test4 scenario2 failed
    }

    if (!((i1 * -9) == im))
    {
        return 43;  // test4 scenario3 failed
    }

    if (!((i1 / -9) == id))
    {
        return 44;  // test4 scenario4 failed
    }

    if (!((i1 % -9) == ir))
    {
        return 45;  // test4 scenario5 failed
    }

    return 0;
}

// scalartype (LHS): 1
int test5() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    index<6> i1(data1);
    int dataa[] = {-99, -9, 0, 2, 11, 101};
    int datas[] = {101, 11, 2, 0, -9, -99};
    int datam[] = {-100, -10, -1, 1, 10, 100};
    int datad[] = {0, 0, -1, 1, 0, 0};
    int datar[] = {1, 1, 0, 0, 1, 1};
    index<6> ia(dataa);
    index<6> is(datas);
    index<6> im(datam);
    index<6> id(datad);
    index<6> ir(datar);

    if (!((1 + i1) == ia))
    {
        return 51;  // test5 scenario1 failed
    }
    if (!((1 - i1) == is))
    {
        return 52;  // test5 scenario2 failed
    }
    if (!((1 * i1) == im))
    {
        return 53;  // test5 scenario3 failed
    }
    if (!((1 / i1) == id))
    {
        return 54;  // test5 scenario4 failed
    }
    if (!((1 % i1) == ir))
    {
        return 55;  // test5 scenario5 failed
    }
    return 0;
}

// scalartype (LHS): -1
int test6() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    index<6> i1(data1);
    int dataa[] = {-101, -11, -2, 0, 9, 99};
    int datas[] = {99, 9, 0, -2, -11, -101};
    int datam[] = {100, 10, 1, -1, -10, -100};
    int datad[] = {0, 0, 1, -1, 0, 0};
    int datar[] = {-1, -1, 0, 0, -1, -1};
    index<6> ia(dataa);
    index<6> is(datas);
    index<6> im(datam);
    index<6> id(datad);
    index<6> ir(datar);

    if (!(((-1) + i1) == ia))
    {
        return 61;  // test6 scenario1 failed
    }

    if (!(((-1) - i1) == is))
    {
        return 62;  //test6 scenario2 failed
    }

    if (!(((-1) * i1) == im))
    {
        return 63;  //test6 scenario3 failed
    }

    if (!(((-1) / i1) == id))
    {
        return 64;  //test6 scenario4 failed
    }

    if (!(((-1) % i1) == ir))
    {
        return 65;  //test6 scenario5 failed
    }

    return 0;
}

// scalartype (LHS): 0
int test7() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    index<6> i1(data1);
    int dataa[] = {-100, -10, -1, 1, 10, 100};
    int datas[] = {100, 10, 1, -1, -10, -100};
    int datam[] = {0, 0, 0, 0, 0, 0};
    index<6> ia(dataa);
    index<6> is(datas);
    index<6> im(datam);
    index<6> id;
    index<6> ir;

    if (!((0 + i1) == ia))
    {
        return 71;  // test7 scenario1 failed
    }

    if (!((0 - i1) == is))
    {
        return 72;  // test7 scenario2 failed
    }

    if (!((0 * i1) == im))
    {
        return 73;  // test7 scenario3 failed
    }

    if (!((0 / i1) == ir))
    {
        return 73;  // test7 scenario4 failed
    }

    if (!((0 % i1) == ir))
    {
        return 73;  // test7 scenario5 failed
    }

    return 0;
}

// scalartype (LHS): -9
int test8() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 1, 10, 100};
    index<6> i1(data1);
    int dataa[] = {-109, -19, -10, -8, 1, 91};
    int datas[] = {91, 1, -8, -10, -19, -109};
    int datam[] = {900, 90, 9, -9, -90, -900};
    int datad[] = {0, 0, 9, -9, 0, 0};
    int datar[] = {-9, -9, 0, 0, -9, -9};
    index<6> ia(dataa);
    index<6> is(datas);
    index<6> im(datam);
    index<6> id(datad);
    index<6> ir(datar);

    if (!(((-9) + i1) == ia))
    {
        return 81;  //test8 scenario1 failed
    }

    if (!(((-9) -  i1) == is))
    {
        return 82;  //test8 scenario2 failed
    }

    if (!(((-9) * i1) == im))
    {
        return 83;  //test8 scenario3 failed
    }

    if (!(((-9) / i1) == id))
    {
        return 84;  //test8 scenario4 failed
    }

    if (!(((-9) % i1) == ir))
    {
        return 85;  //test8 scenario5 failed
    }

    return 0;
}

int test() restrict(amp,cpu)
{
    int result = test1();
    if(result != 0)
    {
        return result;
    }

    result = test2();
    if(result != 0)
    {
        return result;
    }

    result = test3();
    if(result != 0)
    {
        return result;
    }
    result = test4();
    if(result != 0)
    {
        return result;
    }
    result = test5();
    if(result != 0)
    {
        return result;
    }
    result = test6();
    if(result != 0)
    {
        return result;
    }
    result = test7();
    if(result != 0)
    {
        return result;
    }
    return test8();
}

void kernel(index<1>& idx, array<int, 1>& result) restrict(amp,cpu)
{
    result[idx] = test();    
}

const int size = 10;

int test_device()
{
    extent<1> e(size);
    array<int, 1> result(e);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) restrict(amp,cpu) {
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] != 0)
        {
            int ret = presult[i];
            return ret;
        }
    }

    return 0;
}

int main() 
{
    // Test on host
    int host_result = test();
    if(host_result == 0)
    {
    }
    else
    {
        return 1;
    }

    // Test on device
    int device_result = test_device();
    if(device_result == 0)
    {
    }
    else
    {
		return 1;
    }    

    return 0;
}

