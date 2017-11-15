// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>
/// Test non-trivial and non-standard layout classes can be used as type of array/array_view
/// </summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace Concurrency::Test;

//simple inherited
class A1_base
{
protected:
    int m1;
public:
    long m2;
    A1_base() restrict(cpu,amp): m1(0), m2(1){}
    int get_m1() restrict(cpu,amp){return m1;}
    long get_m2() restrict(cpu,amp){return m2;}

};

class A1: public A1_base
{
    float m3;
public:
    A1() restrict(cpu,amp) : m3(10.0f){}
    float get_m3() restrict(cpu,amp){return m3;}
};

//multiple inherited, one base is privately inherited
class A2_base_1
{
    int m1;
public:
    A2_base_1() restrict(cpu,amp) : m1(0){}
    int get_m1() restrict(cpu,amp){return m1;}
};

class A2_base_2
{
    int m2;
public:
    A2_base_2() restrict(cpu,amp) : m2(0){}
    int get_m2() restrict(cpu,amp){return m2;}
};
class A2: public A2_base_1, private A2_base_2
{
    int m3;
public:
    A2() restrict(cpu,amp) : m3(false){}
    int get_base_m2() restrict(cpu,amp){return get_m2();}
    int get_m3() restrict(cpu,amp){return m3;}
};

// base class with static member which is incompatible
class A3_base
{
    static char m1;
    int m2;
public:
    A3_base() restrict(cpu,amp) : m2(-1){}
    int get_m2() restrict(cpu,amp){return m2;}
};

struct A3 : A3_base
{
    ~A3() restrict(amp)
    {
    }
};

// first nonstatic member has same type as base class
class A4_base
{
protected:
    int m1;
public:
    long m2;
    A4_base() restrict(cpu,amp): m1(0), m2(1){}
    A4_base(int _m1, long _m2) restrict(cpu,amp): m1(_m1), m2(_m2){}
    int get_m1() restrict(cpu,amp){return m1;}
    long get_m2() restrict(cpu,amp){return m2;}
};

class A4: public A4_base
{
    A4_base m3;
    float m4;
public:
    A4() restrict(cpu,amp) : m4(10.0f){}
    A4(int _m1, long _m2, float _m4) restrict(cpu,amp) : A4_base(_m1, _m2), m4(_m4){}
    A4_base get_m3() restrict(cpu,amp){return m3;}
    float get_m4() restrict(cpu,amp){return m4;}
};

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    runall_result result;
    array_view<runall_result, 1> result_av(1, &result);

    //A1
    array<A1, 1> arr1(1, av);
    parallel_for_each(result_av.get_extent(), [&,result_av] (index<1> idx) restrict(amp)
    {
        A1 a1;
        array_view<A1, 1> arr_view1(arr1);
        arr1[idx] = a1;
        arr_view1[idx] = a1;

        result_av[idx] &= (arr_view1[idx].get_m1() == 0);
        result_av[idx] &= (arr_view1[idx].get_m2() == 1);
        result_av[idx] &= (arr_view1[idx].get_m3() == 10.0f);
    });

    if(!result_av[0].get_is_pass())
    {
        Log(LogType::Error, true) << "a1 object was not constructed as expected" << std::endl;
        return result_av[0];
    }

    //A2
    array<A2, 1> arr2(1, av);
    parallel_for_each(result_av.get_extent(), [&,result_av] (index<1> idx) restrict(amp)
    {
        A2 a2;
        array_view<A2, 1> arr_view2(arr2);
        arr2[idx] = a2;
        arr_view2[idx] = a2;

        result_av[idx] &= (arr_view2[idx].get_m1() == false);
        result_av[idx] &= (arr_view2[idx].get_base_m2() == false);
        result_av[idx] &= (arr_view2[idx].get_m3() == false);
    });

    if(!result_av[0].get_is_pass())
    {
        Log(LogType::Error, true) << "a2 object was not constructed as expected" << std::endl;
        return runall_fail;
    }

    // A3
    array<A3, 1> arr3(1, av);
    parallel_for_each(result_av.get_extent(), [&,result_av] (index<1> idx) restrict(amp)
    {
        A3 a3;
        array_view<A3, 1> arr_view3(arr3);
        arr3[idx] = a3;
        arr_view3[idx] = a3;

        result_av[idx] &= (arr_view3[idx].get_m2() == -1);
    });

    if(!result_av[0].get_is_pass())
    {
        Log(LogType::Error, true) << "a3 object was not constructed as expected" << std::endl;
        return runall_fail;
    }

    //A4
    array<A4, 1> arr4(1, av);
    parallel_for_each(result_av.get_extent(), [&,result_av] (index<1> idx) restrict(amp)
    {
        A4 a4(10, -200, 15.0f);
        array_view<A4, 1> arr_view4(arr4);
        arr4[idx] = a4;
        arr_view4[idx] = a4;

        result_av[idx] &= (arr_view4[idx].get_m1() == 10);        // test constructor with parameters
        result_av[idx] &= (arr_view4[idx].get_m2() == -200);      // test constructor with parameters
        result_av[idx] &= (arr_view4[idx].get_m3().get_m1() == 0);  // test default constructor
        result_av[idx] &= (arr_view4[idx].get_m3().get_m2() == 1); // test default constructor
        result_av[idx] &= (arr_view4[idx].get_m4() == 15.0f);

    });

    if(!result_av[0].get_is_pass())
    {
        Log(LogType::Error, true) << "a4 object was not constructed as expected" << std::endl;
        return runall_fail;
    }

    parallel_for_each(result_av.get_extent(), [&,result_av] (index<1> idx) restrict(amp)
    {
        //local class
        class A5_base
        {
            int m2;
        public:
            A5_base() : m2(0){}
            int get_m2() {return m2;}
        };

        class A5: public A5_base
        {
            float m3;
        public:
            A5() : m3(16.0f){}
            float get_m3() {return m3;}
        };

        A5 a5[10];
        array_view<A5, 1> arr_view5(10, a5);
        arr_view5[idx] = a5[1];

        result_av[idx] &= (arr_view5[idx].get_m2() == false);
        result_av[idx] &= (arr_view5[idx].get_m3() == 16.0f);
    });

    if(!result_av[0].get_is_pass())
    {
        Log(LogType::Error, true) << "a5 object was not constructed as expected" << std::endl;
        return runall_fail;
    }

    return result;
}


