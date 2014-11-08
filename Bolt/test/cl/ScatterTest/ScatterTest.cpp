/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.
***************************************************************************/
#pragma warning(disable: 4996)
#include "common/stdafx.h"
#include "common/myocl.h"
#include "common/test_common.h"

#include "bolt/cl/functional.h"
#include "bolt/cl/iterator/constant_iterator.h"
#include "bolt/cl/iterator/counting_iterator.h"
#include "bolt/miniDump.h"
#include "bolt/cl/scatter.h"

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include<iostream>
#include <boost/range/algorithm/transform.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/config.hpp>

#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 0

#define TEMPORARY_DISABLE_STD_DV_TEST_CASES 0

BOLT_FUNCTOR( is_even,				  
struct is_even{
    bool operator () (int x)
    {
        return ( (x % 2)==0);
    }
};
);


BOLT_FUNCTOR( Int2,
struct Int2
{
    int a;
    int b;

    bool operator==(const Int2& rhs) const
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        return equal;
    }
    Int2():a(0),b(0){};
};
);

BOLT_FUNCTOR( IntFloat,
struct IntFloat
{
     int a;
     float b;

    bool operator==(const IntFloat& rhs) const
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        return equal;
    }
};
);

BOLT_FUNCTOR( IntFloatDouble,
struct IntFloatDouble
{
     int a;
     float b;
     double c;
     
    bool operator==(const IntFloatDouble& rhs) const
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        equal = ( c == rhs.c ) ? equal : false;
        return equal;
    }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< Int2 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< Int2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< IntFloat >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< IntFloat >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< IntFloatDouble >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< IntFloatDouble >::iterator, bolt::cl::deviceVectorIteratorTemplate );


class HostMemory_IntStdVector:public ::testing::TestWithParam<int>
{
protected:
    int myStdVectSize;
public:
    HostMemory_IntStdVector():myStdVectSize(GetParam()){
    }
};

typedef HostMemory_IntStdVector DeviceMemory_IntBoltdVector;

class HostMemory_UDDTestInt2:public ::testing::TestWithParam<int>
{
protected:
    int myStdVectSize;
public:
    HostMemory_UDDTestInt2():myStdVectSize(GetParam()){
    }
};
typedef HostMemory_UDDTestInt2 HostMemory_UDDTestIntFloat;

////////////////////////////////////////////
///////////scatter_if Google Test Cases ///////
////////////////////////////////////////////
TEST( HostMemory_Int, Scatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, SerialScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, MulticoreScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Int, Scatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);
    //bolt::cl::counting_iterator<int> stencil_last = stencil_first + 10;

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, SerialScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, MulticoreScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
} 

TEST( HostMemory_Int, Scatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, SerialScatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, MulticoreScatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Int, Scatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Int, SerialScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Int, MulticoreScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}


TEST_P(HostMemory_IntStdVector, SerialScatter_IfPredicate)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;
            stencil[i] = i + 5 * 1;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoresScatter_IfPredicate)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;
            stencil[i] = i + 5 * 1;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if(input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P( HostMemory_IntStdVector, SerialScatter_IfPredicate_Fancy_stencil )
{
    std::vector<int> input( myStdVectSize,0); 
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;			
        }
    std::random_shuffle( map.begin(), map.end() ); 
    bolt::cl::counting_iterator<int> stencil_first(0);

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P( HostMemory_IntStdVector, MulticoreScatter_IfPredicate_Fancy_stencil )
{
    std::vector<int> input( myStdVectSize,0); 
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;			
        }
    std::random_shuffle( map.begin(), map.end() ); 
    bolt::cl::counting_iterator<int> stencil_first(0);

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P( HostMemory_IntStdVector, SerialScatter_IfPredicate_fancyInput )
{	
    bolt::cl::counting_iterator<int> input(0); 
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            stencil[i] =  i + 2 * i;			
        }
    std::random_shuffle( map.begin(), map.end() );;

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     bolt::cl::scatter_if(ctl, input, input+myStdVectSize, map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input, input+myStdVectSize, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P( HostMemory_IntStdVector, MulticoreScatter_IfPredicate_fancyInput )
{	
    bolt::cl::counting_iterator<int> input(0); 
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            stencil[i] =  i + 2 * i;			
        }
    std::random_shuffle( map.begin(), map.end() );;

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     bolt::cl::scatter_if(ctl, input, input+myStdVectSize, map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input, input+myStdVectSize, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P( DeviceMemory_IntBoltdVector, SerialScatter_IfPredicate )
{

    std::vector<int> n_map (myStdVectSize,0);	
    std::vector<int> h_input (myStdVectSize,0);
    std::vector<int> h_stencil (myStdVectSize,0);
    for( int i=0; i < myStdVectSize ; i++ )
        {
            n_map[i] = i;
            h_input[i] =  i + 2 * i;
            h_stencil[i] = i + 5 * 1;
        }
    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> stencil ( h_stencil.begin(), h_stencil.end() );	
    bolt::cl::device_vector<int> map( n_map.begin(),n_map.end() );

    std::random_shuffle( n_map.begin(), n_map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}
TEST_P( DeviceMemory_IntBoltdVector, MulticoreScatter_IfPredicate )
{
    std::vector<int> n_map (myStdVectSize,0);	
    std::vector<int> h_input (myStdVectSize,0);
    std::vector<int> h_stencil (myStdVectSize,0);
    for( int i=0; i < myStdVectSize ; i++ )
        {
            n_map[i] = i;
            h_input[i] =  i + 2 * i;
            h_stencil[i] = i + 5 * 1;
        }
    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> stencil ( h_stencil.begin(), h_stencil.end() );	
    bolt::cl::device_vector<int> map( n_map.begin(),n_map.end() );

    std::random_shuffle( n_map.begin(), n_map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}

TEST( HostMemory_Float, Scatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, Scatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);
    //bolt::cl::counting_iterator<int> stencil_last = stencil_first + 10;

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
} 

TEST( HostMemory_Float, Scatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Float, Scatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, -1.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Float, SerialScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, -1.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Float, MulticoreScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
        exp_result.push_back(-1.5f);exp_result.push_back(8.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(6.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(4.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(2.5f);
        exp_result.push_back(-1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, -1.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}

#if(TEST_DOUBLE == 1)
TEST( HostMemory_Double, Scatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, Scatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);
    //bolt::cl::counting_iterator<int> stencil_last = stencil_first + 10;

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
} 

TEST( HostMemory_Double, Scatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<double> input(0.5);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<double> input(0.5);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_IfPredicate_fancyInput )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    bolt::cl::counting_iterator<double> input(0.5);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Double, Scatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, -1.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Double, SerialScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, -1.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Double, MulticoreScatter_IfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
        exp_result.push_back(-1.5);exp_result.push_back(8.5);
        exp_result.push_back(-1.5);exp_result.push_back(6.5);
        exp_result.push_back(-1.5);exp_result.push_back(4.5);
        exp_result.push_back(-1.5);exp_result.push_back(2.5);
        exp_result.push_back(-1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, -1.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}

/* we don not have pick_iterator corresponding to that, so commented.

TEST( DeviceMemory, Scatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    bolt::cl::device_vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    
    is_even iepred;
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory, SerialScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    bolt::cl::device_vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory, MulticoreScatter_IfPredicate_Fancy_stencil )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    bolt::cl::device_vector<int> exp_result;
    {
        exp_result.push_back(-1);exp_result.push_back(8);
        exp_result.push_back(-1);exp_result.push_back(6);
        exp_result.push_back(-1);exp_result.push_back(4);
        exp_result.push_back(-1);exp_result.push_back(2);
        exp_result.push_back(-1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    is_even iepred;
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
*/

TEST( HostMemory_Int, Scatter_If)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, SerialScatter_If)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter_If)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

TEST( HostMemory_Int, Scatter_If_Fancy_stencil)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, SerialScatter_If_Fancy_stencil)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(1);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter_If_Fancy_stencil)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(1);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

TEST( HostMemory_Int, Scatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, SerialScatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

TEST( DeviceMemory_Int, Scatter_If)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);


}
TEST( DeviceMemory_Int, SerialScatter_If)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);


}
TEST( DeviceMemory_Int, MulticoreScatter_If)
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);


}


TEST( HostMemory_Float, Scatter_If)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_If)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_If)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, Scatter_If_Fancy_stencil)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_If_Fancy_stencil)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_If_Fancy_stencil)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, Scatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float > result ( 10, 0.5f );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float > result ( 10, 0.5f );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_If_fancyInput)
{   	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float > result ( 10, 0.5f );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Float, Scatter_If)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Float, SerialScatter_If)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Float, MulticoreScatter_If)
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(0.5f);
        exp_result.push_back(7.5f);exp_result.push_back(0.5f);
        exp_result.push_back(5.5f);exp_result.push_back(0.5f);
        exp_result.push_back(3.5f);exp_result.push_back(0.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}

TEST( HostMemory_Long, Scatter_If)
{
    cl_long n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<cl_long> result ( 10, 0 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, SerialScatter_If)
{
    cl_long n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<cl_long> result ( 10, 0 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );  

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, MulticoreScatter_If)
{
    cl_long n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<cl_long> result ( 10, 0 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );  

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Long, Scatter_If_fancyInput)
{	
    bolt::cl::counting_iterator<cl_long> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<cl_long > result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, SerialScatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<cl_long> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<cl_long > result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, MulticoreScatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<cl_long> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(0);
        exp_result.push_back(7);exp_result.push_back(0);
        exp_result.push_back(5);exp_result.push_back(0);
        exp_result.push_back(3);exp_result.push_back(0);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<cl_long > result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#if(TEST_DOUBLE == 1)
TEST( HostMemory_Double, Scatter_If)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_If)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_If)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, Scatter_If_Fancy_stencil)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_If_Fancy_stencil)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_If_Fancy_stencil)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::constant_iterator<int> stencil_first(1);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil_first, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, Scatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<double> input(0.5);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double > result ( 10, 0.5 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_If_fancyInput)
{   	
    bolt::cl::counting_iterator<double> input(0.5);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double > result ( 10, 0.5 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_If_fancyInput)
{    	
    bolt::cl::counting_iterator<double> input(0.5);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double > result ( 10, 0.5 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input, input+10, map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Double, Scatter_If)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Double, SerialScatter_If)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Double, MulticoreScatter_If)
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(0.5);
        exp_result.push_back(7.5);exp_result.push_back(0.5);
        exp_result.push_back(5.5);exp_result.push_back(0.5);
        exp_result.push_back(3.5);exp_result.push_back(0.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays(exp_result, result);
}
#endif

#endif

////////////////////////////////////////////
///////////scatter Google Test Cases ///////
////////////////////////////////////////////

TEST( HostMemory_Int, Scatter )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, SerialScatter )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter( ctl,input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::scatter( ctl,input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES == 1)
TEST( HostMemory_Int, Scatter_deviceInput )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST( HostMemory_Int, SerialScatter_deviceInput )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST( HostMemory_Int, MulticoreScatter_deviceInput )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter( ctl,input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST( HostMemory_Int, Scatter_deviceMap )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}


TEST( HostMemory_Int, SerialScatter_deviceMap )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter( ctl,input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter_deviceMap )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::scatter( ctl,input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

#endif

TEST( HostMemory_Int, Scatter_fancyInput )
{    	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input, input+10, map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, SerialScatter_fancyInput )
{   	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter( ctl,input, input+10, map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter_fancyInput )
{   	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::scatter( ctl,input, input+10, map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

TEST( HostMemory_Int, Scatter_Fancy_map )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, SerialScatter_Fancy_map )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter( ctl,input.begin(), input.end(), map, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}
TEST( HostMemory_Int, MulticoreScatter_Fancy_map )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::scatter( ctl,input.begin(), input.end(), map, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);


}

TEST( DeviceMemory_Int, Scatter )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST( DeviceMemory_Int, SerialScatter )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST( DeviceMemory_Int, MulticoreScatter )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter( ctl,input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST( DeviceMemory_Int, Scatter_fancyInput )
{   	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input, input+10, map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST( DeviceMemory_Int, SerialScatter_fancyInput )
{   	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input, input+10, map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST( DeviceMemory_Int, MulticoreScatter_fancyInput )
{   	
    bolt::cl::counting_iterator<int> input(0);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(8);
        exp_result.push_back(7);exp_result.push_back(6);
        exp_result.push_back(5);exp_result.push_back(4);
        exp_result.push_back(3);exp_result.push_back(2);
        exp_result.push_back(1);exp_result.push_back(0);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter( ctl, input, input+10, map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST( DeviceMemory_Int, Scatter_Fancy_map )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST( DeviceMemory_Int, SerialScatter_Fancy_map )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST( DeviceMemory_Int, MulticoreScatter_Fancy_map )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter( ctl,input.begin(), input.end(), map, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<"   "<<exp_result[i]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST_P( HostMemory_IntStdVector, Scatter )
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;
        }
    std::random_shuffle( map.begin(), map.end() ); 


    bolt::cl::scatter(input.begin(), input.end(), map.begin(), exp_result.begin());
    bolt::cl::scatter( input.begin(), input.end(), map.begin(),result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P( HostMemory_IntStdVector, Serial_Scatter )
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), exp_result.begin());
     bolt::cl::scatter( input.begin(), input.end(), map.begin(),result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P( HostMemory_IntStdVector, MulticoreScatter )
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), exp_result.begin());
     bolt::cl::scatter( input.begin(), input.end(), map.begin(),result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P( DeviceMemory_IntBoltdVector, Scatter_fancyInput )
{ 	
    bolt::cl::counting_iterator<int> input(0);
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::scatter(input, input+myStdVectSize, map.begin(), exp_result.begin());
    bolt::cl::scatter( input, input+myStdVectSize, map.begin(),result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST_P( DeviceMemory_IntBoltdVector, Serial_Scatter_fancyInput )
{ 	
    bolt::cl::counting_iterator<int> input(0);
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::scatter(ctl, input, input+myStdVectSize, map.begin(), exp_result.begin());
     bolt::cl::scatter( input, input+myStdVectSize, map.begin(),result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST_P( DeviceMemory_IntBoltdVector, MulticoreScatter_fancyInput )
{ 	
    bolt::cl::counting_iterator<int> input(0);
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::scatter(ctl, input, input+myStdVectSize, map.begin(), exp_result.begin());
     bolt::cl::scatter( input, input+myStdVectSize, map.begin(),result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}


TEST( HostMemory_Float, Scatter )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES == 1)
TEST( HostMemory_Float, Scatter_deviceInput )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.0f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}


TEST( HostMemory_Float, SerialScatter_deviceInput )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( HostMemory_Float, MulticoreScatter_deviceInput )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
#endif

#if (TEMPORARY_DISABLE_STD_DV_TEST_CASES == 1)
TEST( HostMemory_Float, Scatter_deviceMap )///////////////////////////////////////////////////
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
} 
TEST( HostMemory_Float, SerialScatter_deviceMap )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_deviceMap )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
#endif

TEST( HostMemory_Float, Scatter_fancyInput )
{    	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input, input+10, map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_fancyInput )
{     	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input, input+10, map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_fancyInput )
{      	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input, input+10, map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, Scatter_Fancy_map )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_Fancy_map )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_Fancy_map )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Float, Scatter )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( DeviceMemory_Float, SerialScatter )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( DeviceMemory_Float, MulticoreScatter )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST( DeviceMemory_Float, Scatter_fancyInput )
{   	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::scatter( input, input+10, map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( DeviceMemory_Float, SerialScatter_fancyInput )
{    	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input, input+10, map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( DeviceMemory_Float, MulticoreScatter_fancyInput )
{   	
    bolt::cl::counting_iterator<float> input(0.5f);
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(8.5f);
        exp_result.push_back(7.5f);exp_result.push_back(6.5f);
        exp_result.push_back(5.5f);exp_result.push_back(4.5f);
        exp_result.push_back(3.5f);exp_result.push_back(2.5f);
        exp_result.push_back(1.5f);exp_result.push_back(0.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input, input+10, map.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST( DeviceMemory_Float, Scatter_Fancy_map )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );

    bolt::cl::scatter( input.begin(), input.end(), map, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( DeviceMemory_Float, SerialScatter_Fancy_map )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST( DeviceMemory_Float, MulticoreScatter_Fancy_map )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    bolt::cl::counting_iterator<int> map(0);

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
        
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

///////////////////////////////////////////////////////////////////
///////////scatter Google Test Cases compared with Boost //////////
///////////////////////////////////////////////////////////////////

TEST( HostMemory_Int, Scatter_comp_Boost )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<int> exp_result(10,0);    
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Int, SerialScatter_comp_Boost )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<int> exp_result(10,0);
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);

}
TEST( HostMemory_Int, MulticoreScatter_comp_Boost )
{
    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<int> exp_result(10,0);
    std::vector<int> result ( 10, 0 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter( ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);

}

TEST( HostMemoryRandomNo_Int, Scatter_comp_Boost )
{
    const int length = 1023;
     std::vector<int> input(length,0);
    for(int i=0 ; i< length;i++)
           input[i] = i;
     std::vector<int> exp_result(length,0);    
     std::vector<int> result ( length, 0 );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemoryRandomNo_Int, SerialScatter_comp_Boost )
{
    const int length = 1023;
     std::vector<int> input(length,0);
    for(int i=0 ; i< length;i++)
           input[i] = i;
     std::vector<int> exp_result(length,0);    
     std::vector<int> result ( length, 0 );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }	
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemoryRandomNo_Int, MulticoreScatter_comp_Boost )
{
    const int length = 1023;
     std::vector<int> input(length,0);
    for(int i=0 ; i< length;i++)
           input[i] = i;
     std::vector<int> exp_result(length,0);    
     std::vector<int> result ( length, 0 );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }	
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}


TEST( HostMemory_Float, Scatter_comp_Boost )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<float> exp_result(10,0.5f);    
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<float>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialScatter_comp_Boost )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<float> exp_result(10,0.5f);    
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<float>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreScatter_comp_Boost )
{
    float n_input[10] =  {0.5f,1.5f,2.5f,3.5f,4.5f,5.5f,6.5f,7.5f,8.5f,9.5f};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<float> exp_result(10,0.5f);    
    std::vector<float> result ( 10, 0.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<float>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemoryRandomNo_Float, Scatter_comp_Boost )
{
    const int length = 1023;
     std::vector<float> input(length,0.5f);
    for(int i=0 ; i< length;i++)
           input[i] = (float)i;
     std::vector<float> exp_result(length,0.5f);    
     std::vector<float> result ( length, 0.5f );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<float>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemoryRandomNo_Float, SerialScatter_comp_Boost )
{
    const int length = 1023;
     std::vector<float> input(length,0.5f);
    for(int i=0 ; i< length;i++)
           input[i] = (float)i;
     std::vector<float> exp_result(length,0.5f);    
     std::vector<float> result ( length, 0.5f );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<float>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemoryRandomNo_Float, MulticoreScatter_comp_Boost )
{
     const int length = 1023;
     std::vector<float> input(length,0.5f);
     for(int i=0 ; i< length;i++)
           input[i] = (float)i;
     std::vector<float> exp_result(length,0.5f);    
     std::vector<float> result ( length, 0.5f );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<float>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}

#if( TEMPORARY_DISABLE_STD_DV_TEST_CASES == 1)
#if(TEST_DOUBLE == 1)
TEST( HostMemory_Double, Scatter_com_Boost )
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<double> exp_result(10,0.5);    
    std::vector<double> result ( 10, 0.0 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialScatter_com_Boost )
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<double> exp_result(10,0.5);    
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreScatter_com_Boost )
{
    double n_input[10] =  {0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5};
    int n_map[10] =  {9,8,7,6,5,4,3,2,1,0};
    std::vector<double> exp_result(10,0.5);    
    std::vector<double> result ( 10, 0.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<int>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
#endif

TEST( HostMemoryRandomNo_Double, Scatter_com_Boost )
{
     const int length = 1024;
     std::vector<double> input(length,0.5);
     for(int i=0 ; i< length;i++)
           input[i] = (double)i;
     std::vector<double> exp_result(length,0.5);    
     std::vector<double> result ( length, 0.5 );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<double>( ) );
    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemoryRandomNo_Double, SerialScatter_com_Boost )
{
     const int length = 1024;
     std::vector<double> input(length,0.5);
     for(int i=0 ; i< length;i++)
           input[i] = (double)i;
     std::vector<double> exp_result(length,0.5);    
     std::vector<double> result ( length, 0.5 );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<double>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemoryRandomNo_Double, MulticoreScatter_com_Boost )
{
     const int length = 1024;
     std::vector<double> input(length,0.5);
     for(int i=0 ; i< length;i++)
           input[i] = (double)i;
     std::vector<double> exp_result(length,0.5);    
     std::vector<double> result ( length, 0.5 );
     std::vector<int> map(length,0);

    {	        	
            int count  = 1;
            std::vector<int>::iterator iter;
            typedef std::iterator_traits< std::vector<int>::iterator >::value_type Type;
            *map.begin() = (Type)(rand()%length);
            Type temp;
            while(count<=length-1)
            {
                temp  = (Type)(rand()%length);
                iter = std::find(map.begin(),map.begin()+count,temp);
                if(iter == map.begin()+count)
                {
                    *(map.begin()+count) = temp;
                     count++;
                }    
            }
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    boost::transform( input,boost::make_permutation_iterator( exp_result.begin(), map.begin() ),bolt::cl::identity<double>( ) );
    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    /*for(int i=0; i<length ; i++)
    { 
        std::cout<<result[ i ]<<"  "<<exp_result[ i ]<<std::endl; 
    }*/
    EXPECT_EQ(exp_result, result);
}
#endif

TEST( UDDTestInt2, SerialScatter_IfPredicate)
{
    int sz = 63;    
    std::vector<Int2> std_input ( sz );
    std::vector<int> std_map ( sz );
    std::vector<int> std_stencil (sz);

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = (int)i;
        std_input[i].a = (int)(i + 2 * i);
        std_input[i].b = (int)(i + 3 * i);
        std_stencil[i] = ((i%2)==0)?1:0;
    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<Int2> result ( sz );
    bolt::cl::device_vector<Int2> exp_result ( sz );
    bolt::cl::device_vector<Int2> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    bolt::cl::device_vector<int> stencil ( std_stencil.begin(), std_stencil.end() );
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::co	ut<<result[ i ]<<std::endl; }

    cmpArrays( exp_result, result );
}
TEST( UDDTestInt2, MulticoreScatter_IfPredicate)
{
    int sz = 63;    
    std::vector<Int2> std_input ( sz );
    std::vector<int> std_map ( sz );
    std::vector<int> std_stencil (sz);

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = (int)i;
        std_input[i].a = (int)(i + 2 * i);
        std_input[i].b = (int)(i + 3 * i);
        std_stencil[i] = ((i%2)==0)?1:0;
    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<Int2> result ( sz );
    bolt::cl::device_vector<Int2> exp_result ( sz );
    bolt::cl::device_vector<Int2> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    bolt::cl::device_vector<int> stencil ( std_stencil.begin(), std_stencil.end() );
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::co	ut<<result[ i ]<<std::endl; }

    cmpArrays( exp_result, result );
}

TEST( UDDTestIntFloat, SerialScatter_IfPredicate)
{
    int sz = 63;    
    std::vector<IntFloat> std_input ( sz );
    std::vector<int> std_map ( sz );
    std::vector<int> std_stencil (sz);

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_stencil[i] = ((i%2)==0)?1:0;
    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloat> result ( sz );
    bolt::cl::device_vector<IntFloat> exp_result ( sz );
    bolt::cl::device_vector<IntFloat> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    bolt::cl::device_vector<int> stencil ( std_stencil.begin(), std_stencil.end() );
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::co	ut<<result[ i ]<<std::endl; }

    cmpArrays( exp_result, result );
}
TEST( UDDTestIntFloat, MulticoreScatter_IfPredicate)
{
    int sz = 63;    
    std::vector<IntFloat> std_input ( sz );
    std::vector<int> std_map ( sz );
    std::vector<int> std_stencil (sz);

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_stencil[i] = ((i%2)==0)?1:0;
    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloat> result ( sz );
    bolt::cl::device_vector<IntFloat> exp_result ( sz );
    bolt::cl::device_vector<IntFloat> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    bolt::cl::device_vector<int> stencil ( std_stencil.begin(), std_stencil.end() );
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::co	ut<<result[ i ]<<std::endl; }

    cmpArrays( exp_result, result );
}

#if(TEST_DOUBLE == 1)
TEST(UDDTestIntFloatDouble, SerialScatter_IfPredicate )
{
    int sz = 63;    
    std::vector<IntFloatDouble> std_input ( sz );
    std::vector<int> std_map ( sz );
    std::vector<int> std_stencil (sz);

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_input[i].c = (double)(i + 5 * i);
        std_stencil[i] = ((i%2)==0)?1:0;
    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloatDouble> result ( sz );
    bolt::cl::device_vector<IntFloatDouble> exp_result ( sz );
    bolt::cl::device_vector<IntFloatDouble> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    bolt::cl::device_vector<int> stencil ( std_stencil.begin(), std_stencil.end() );
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::co	ut<<result[ i ]<<std::endl; }

    cmpArrays( exp_result, result );
}

TEST(UDDTestIntFloatDouble, MulticoreScatter_ifPredicate )
{
    int sz = 63;    
    std::vector<IntFloatDouble> std_input ( sz );
    std::vector<int> std_map ( sz );
    std::vector<int> std_stencil (sz);

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_input[i].c = (double)(i + 5 * i);
        std_stencil[i] = ((i%2)==0)?1:0;
    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloatDouble> result ( sz );
    bolt::cl::device_vector<IntFloatDouble> exp_result ( sz );
    bolt::cl::device_vector<IntFloatDouble> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    bolt::cl::device_vector<int> stencil ( std_stencil.begin(), std_stencil.end() );
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::co	ut<<result[ i ]<<std::endl; }

    cmpArrays( exp_result, result );
}


TEST_P(HostMemory_UDDTestInt2, SerialScatter_IfPredicate)
{
    std::vector<Int2> input( myStdVectSize);   
    std::vector<Int2> exp_result(myStdVectSize);    
    std::vector<Int2> result ( myStdVectSize);
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i].a =  i + 2 * i;
            input[i].b = i + 3 * i;
            stencil[i] = i + 5 * 1;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_UDDTestInt2, MulticoreScatter_IfPredicate)
{
    std::vector<Int2> input( myStdVectSize);   
    std::vector<Int2> exp_result(myStdVectSize);    
    std::vector<Int2> result ( myStdVectSize);
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i].a =  i + 2 * i;
            input[i].b = i + 3 * i;
            stencil[i] = i + 5 * 1;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_UDDTestIntFloat, SerialScatter_IfPredicate)
{
    std::vector<IntFloat> input( myStdVectSize);   
    std::vector<IntFloat> exp_result(myStdVectSize);    
    std::vector<IntFloat> result ( myStdVectSize);
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i].a =  i + 2 * i;
            input[i].b = (float)(i + 3 * i);
            stencil[i] = i + 5 * 1;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_UDDTestIntFloat, MulticoreScatter_IfPredicate)
{
    std::vector<IntFloat> input( myStdVectSize);   
    std::vector<IntFloat> exp_result(myStdVectSize);    
    std::vector<IntFloat> result ( myStdVectSize);
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i].a =  i + 2 * i;
            input[i].b = (float)(i + 3 * i);
            stencil[i] = i + 5 * 1;
        }
    std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin(), stencil.begin(), exp_result.begin(), iepred );
     bolt::cl::scatter_if( input.begin(), input.end(), map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
#endif

TEST(UDDTestInt2, Scatter )
{
    int sz = 63;
    std::vector<Int2> exp_result( sz );
    std::vector<Int2> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = i + 3 * i;

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<Int2> result ( sz );
    bolt::cl::device_vector<Int2> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,
                      boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                      bolt::cl::identity<Int2>( ) );
    cmpArrays( exp_result, result );
}
TEST(UDDTestInt2, SerialScatter )
{
    int sz = 63;
    std::vector<Int2> exp_result( sz );
    std::vector<Int2> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = i + 3 * i;

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<Int2> result ( sz );
    bolt::cl::device_vector<Int2> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform(std_input,boost::make_permutation_iterator( exp_result.begin(),std_map.begin()),
                                                                             bolt::cl::identity<Int2>());
    cmpArrays( exp_result, result );
}
TEST(UDDTestInt2, MulticoreScatter )
{
    int sz = 63;
    std::vector<Int2> exp_result( sz );
    std::vector<Int2> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = i + 3 * i;

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<Int2> result ( sz );
    bolt::cl::device_vector<Int2> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform(std_input,boost::make_permutation_iterator( exp_result.begin(),std_map.begin()),
                                                                             bolt::cl::identity<Int2>());
    cmpArrays( exp_result, result );
}

TEST(UDDTestIntFloat, Scatter )
{
    int sz = 63;
    std::vector<IntFloat> exp_result( sz );
    std::vector<IntFloat> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloat> result ( sz );
    bolt::cl::device_vector<IntFloat> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                                                                           bolt::cl::identity<IntFloat>( ) );
    cmpArrays( exp_result, result );
}
TEST(UDDTestIntFloat, SerialScatter )
{
    int sz = 63;
    std::vector<IntFloat> exp_result( sz );
    std::vector<IntFloat> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = float(i + 3.5f * i);

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloat> result ( sz );
    bolt::cl::device_vector<IntFloat> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

     bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                                                                           bolt::cl::identity<IntFloat>( ) );
    cmpArrays( exp_result, result );
}
TEST(UDDTestIntFloat, MulticoreScatter )
{
    int sz = 63;
    std::vector<IntFloat> exp_result( sz );
    std::vector<IntFloat> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = float(i + 3.5f * i);

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloat> result ( sz );
    bolt::cl::device_vector<IntFloat> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

     bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                                                                           bolt::cl::identity<IntFloat>( ) );
    cmpArrays( exp_result, result );
}

#if(TEST_DOUBLE == 1)
TEST(UDDTestIntFloatDouble, Scatter )
{
    int sz = 63;
    std::vector<IntFloatDouble> exp_result( sz );
    std::vector<IntFloatDouble> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_input[i].c = (double)(i + 5 * i);

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloatDouble> result ( sz );
    bolt::cl::device_vector<IntFloatDouble> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );

    bolt::cl::scatter( input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                                                                           bolt::cl::identity<IntFloatDouble>( ) );
    cmpArrays( exp_result, result );
}
TEST(UDDTestIntFloatDouble, SerialScatter )
{
    int sz = 63;
    std::vector<IntFloatDouble> exp_result( sz );
    std::vector<IntFloatDouble> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_input[i].c = (double)(i + 5 * i);

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloatDouble> result ( sz );
    bolt::cl::device_vector<IntFloatDouble> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                                                                           bolt::cl::identity<IntFloatDouble>( ) );
    cmpArrays( exp_result, result );
}
TEST(UDDTestIntFloatDouble, MulticoreScatter )
{
    int sz = 63;
    std::vector<IntFloatDouble> exp_result( sz );
    std::vector<IntFloatDouble> std_input ( sz );
    std::vector<int> std_map ( sz );

	for (int i = 0; i < sz; i++)
    {
        std_map[i] = i;
        std_input[i].a = i + 2 * i;
        std_input[i].b = (float)(i + 3 * i);
        std_input[i].c = (double)(i + 5 * i);

    }
    std::random_shuffle( std_map.begin(), std_map.end() );

    bolt::cl::device_vector<IntFloatDouble> result ( sz );
    bolt::cl::device_vector<IntFloatDouble> input ( std_input.begin(), std_input.end() );
    bolt::cl::device_vector<int> map ( std_map.begin(), std_map.end() );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter(ctl, input.begin(), input.end(), map.begin(), result.begin() );
    boost::transform( std_input,boost::make_permutation_iterator( exp_result.begin(), std_map.begin() ),
                                                                           bolt::cl::identity<IntFloatDouble>( ) );
    cmpArrays( exp_result, result );
}
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OFFSET TEST CASES
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TEST_P(HostMemory_IntStdVector, OffsetScatter_IfPredicate)
//{
//    size_t myStdVectSize = 512;
//    std::vector<int> input( myStdVectSize,0);   
//    std::vector<int> exp_result(myStdVectSize,0);    
//    std::vector<int> result ( myStdVectSize, 0 );
//    std::vector<int> map (myStdVectSize,0);	
//    std::vector<int> stencil (myStdVectSize,0);	
//    for( int i=0; i < myStdVectSize ; i++ )
//        {
//            map[i] = i;
//            input[i] = i;
//            stencil[i] = i + 5 * 1;
//        }
//    std::random_shuffle( map.begin(), map.end() ); 
//
//    is_even iepred;
//    bolt::cl::control ctl = bolt::cl::control::getDefault( );
//    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
//     bolt::cl::scatter_if(ctl, input.begin(), input.end(), map.begin()+54, stencil.begin()+53, exp_result.begin(), iepred );
//     bolt::cl::scatter_if( input.begin(), input.end(), map.begin()+54, stencil.begin()+53, result.begin(), iepred );
//    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
//    EXPECT_EQ(exp_result, result);
//}

TEST(HostMemory_IntStdVector, OffsetScatterPredicate)
{

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {3,2,1,0,4,5,8,7,6,9};


    std::vector<int> input( n_input, n_input+10 );   
    std::vector<int> exp_result(10,0);
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map +10 );

    bolt::cl::scatter( input.begin()+5, input.end(), map.begin(), exp_result.begin() );
    bolt::cl::scatter( input.begin()+5, input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}


TEST(HostMemory_IntStdVector, SerialOffsetScatterPredicate)
{

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {3,2,1,0,4,5,8,7,6,9};


    std::vector<int> input( n_input, n_input+10 );   
    std::vector<int> exp_result(10,0);
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map +10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter( ctl, input.begin()+5, input.end(), map.begin(), exp_result.begin() );
    bolt::cl::scatter( input.begin()+5, input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST(HostMemory_IntStdVector, MultiCoreOffsetScatterPredicate)
{

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {3,2,1,0,4,5,8,7,6,9};


    std::vector<int> input( n_input, n_input+10 );   
    std::vector<int> exp_result(10,0);
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map +10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::scatter( ctl, input.begin()+5, input.end(), map.begin(), exp_result.begin() );
    bolt::cl::scatter( input.begin()+5, input.end(), map.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST(HostMemory_IntStdVector, SerialOffsetScatterIfPredicate)
{

    int n_input[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_map[10] =  {3,2,1,0,4,5,8,7,6,9};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};


    std::vector<int> input( n_input, n_input+10 );   
    std::vector<int> exp_result(10,0);
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map +10 );
    std::vector<int> stencil ( n_stencil, n_stencil +10 );

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter_if( ctl, input.begin()+5, input.end(), map.begin(), stencil.begin()+5, exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin()+5, input.end(), map.begin(), stencil.begin()+5, result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}


TEST(HostMemory_IntStdVector, OffsetScatterPredicateMedium)
{
    size_t myStdVectSize = 1024;
    int s_offset = 27;
    int e_offset = 515;
    size_t distance = e_offset - s_offset;

    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < e_offset ; i++ )
    {
        map[i] = i;
        input[i] =  i + 2 * i;
    }


    bolt::cl::scatter( input.begin(), input.begin() + e_offset, map.begin(), exp_result.begin() );

    bolt::cl::scatter( input.begin(), input.begin() + e_offset, map.begin(), result.begin() );

    EXPECT_EQ(exp_result, result);
}

TEST(HostMemory_IntStdVector, SerialOffsetScatterPredicateMedium)
{
    size_t myStdVectSize = 1024;
    int s_offset = 27;
    int e_offset = 515;
    size_t distance = e_offset - s_offset;

    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < e_offset ; i++ )
    {
        map[i] = i;
        input[i] =  i + 2 * i;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::scatter( ctl, input.begin(), input.begin() + e_offset, map.begin(), exp_result.begin() );

    bolt::cl::scatter( input.begin(), input.begin() + e_offset, map.begin(), result.begin() );

    EXPECT_EQ(exp_result, result);
}

TEST(HostMemory_IntStdVector, MultiCoreOffsetScatterPredicateMedium)
{
    size_t myStdVectSize = 1024;
    int s_offset = 27;
    int e_offset = 515;
    size_t distance = e_offset - s_offset;

    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    for( int i=0; i < e_offset ; i++ )
    {
        map[i] = i;
        input[i] =  i + 2 * i;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::scatter( ctl, input.begin(), input.begin() + e_offset, map.begin(), exp_result.begin() );

    bolt::cl::scatter( input.begin(), input.begin() + e_offset, map.begin(), result.begin() );

    EXPECT_EQ(exp_result, result);
}

TEST(HostMemory_IntStdVector, SerialOffsetScatterIfPredicateMedium)
{
    size_t myStdVectSize = 512;
    int s_offset = 59;
    int e_offset = 400;
    size_t distance = e_offset - s_offset;

    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);
    for( int i=0; i < e_offset ; i++ )
        {
            map[i] = i;
            input[i] =  i + 2 * i;
            stencil[i] = i + 5 * 1;
        }

    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::scatter_if( ctl, input.begin(), input.begin()+e_offset, map.begin(), stencil.begin(), exp_result.begin(), iepred );
    bolt::cl::scatter_if( input.begin(), input.begin()+e_offset, map.begin(), stencil.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}



INSTANTIATE_TEST_CASE_P(ScatterIntLimit, HostMemory_IntStdVector, ::testing::Range(1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P(ScatterIntLimit, DeviceMemory_IntBoltdVector, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P(ScatterUDDLimit, HostMemory_UDDTestInt2, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P(ScatterUDDLimit, HostMemory_UDDTestIntFloat, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15

int main(int argc, char* argv[])
{
    //  Register our minidump generating logic
//    bolt::miniDumpSingleton::enableMiniDumps( );

    // Define MEMORYREPORT on windows platfroms to enable debug memory heap checking
#if defined( MEMORYREPORT ) && defined( _WIN32 )
    TCHAR logPath[ MAX_PATH ];
    ::GetCurrentDirectory( MAX_PATH, logPath );
    ::_tcscat_s( logPath, _T( "\\MemoryReport.txt") );

    // We leak the handle to this file, on purpose, so that the ::_CrtSetReportFile() can output it's memory 
    // statistics on app shutdown
    HANDLE hLogFile;
    hLogFile = ::CreateFile( logPath, GENERIC_WRITE, 
        FILE_SHARE_READ|FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );

    ::_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_WNDW | _CRTDBG_MODE_DEBUG );
    ::_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG );

    ::_CrtSetReportFile( _CRT_ASSERT, hLogFile );
    ::_CrtSetReportFile( _CRT_ERROR, hLogFile );
    ::_CrtSetReportFile( _CRT_WARN, hLogFile );

    int tmp = ::_CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
    tmp |= _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF;
    ::_CrtSetDbgFlag( tmp );

    // By looking at the memory leak report that is generated by this debug heap, there is a number with 
    // {} brackets that indicates the incremental allocation number of that block.  If you wish to set
    // a breakpoint on that allocation number, put it in the _CrtSetBreakAlloc() call below, and the heap
    // will issue a bp on the request, allowing you to look at the call stack
    // ::_CrtSetBreakAlloc( 1833 );

#endif /* MEMORYREPORT */

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    // Set the standard OpenCL wait behavior to help debugging

    //bolt::cl::control& myControl = bolt::cl::control::getDefault( );
    //myControl.waitMode( bolt::cl::control::NiceWait );
    //myControl.forceRunMode( bolt::cl::control::MultiCoreCpu );  // choose tbb

    int retVal = RUN_ALL_TESTS( );

    //  Reflection code to inspect how many tests failed in gTest
    ::testing::UnitTest& unitTest = *::testing::UnitTest::GetInstance( );

    unsigned int failedTests = 0;
    for( int i = 0; i < unitTest.total_test_case_count( ); ++i )
    {
        const ::testing::TestCase& testCase = *unitTest.GetTestCase( i );
        for( int j = 0; j < testCase.total_test_count( ); ++j )
        {
            const ::testing::TestInfo& testInfo = *testCase.GetTestInfo( j );
            if( testInfo.result( )->Failed( ) )
                ++failedTests;
        }
    }

    //  Print helpful message at termination if we detect errors, to help users figure out what to do next
    if( failedTests )
    {
        bolt::tout << _T( "\nFailed tests detected in test pass; please run test again with:" ) << std::endl;
        bolt::tout << _T( "\t--gtest_filter=<XXX> to select a specific failing test of interest" ) << std::endl;
        bolt::tout << _T( "\t--gtest_catch_exceptions=0 to generate minidump of failing test, or" ) << std::endl;
        bolt::tout << _T( "\t--gtest_break_on_failure to debug interactively with debugger" ) << std::endl;
        bolt::tout << _T( "\t    (only on googletest assertion failures, not SEH exceptions)" ) << std::endl;
    }

    return retVal;
}
