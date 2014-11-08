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

#include "common/stdafx.h"
#include "common/myocl.h"
#include "common/test_common.h"

#include "bolt/cl/functional.h"
#include "bolt/cl/iterator/constant_iterator.h"
#include "bolt/cl/iterator/counting_iterator.h"
#include "bolt/miniDump.h"
#include "bolt/cl/gather.h"

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include <array>
#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 0
#define TEMPORARY_DISABLE_STD_DV_TEST_CASES 0

BOLT_FUNCTOR( is_even,
struct is_even{
    bool operator () (int x)
    {
        return ( (x % 2) == 0 );
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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GatherIf tests
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST( HostMemory_int, GatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGatherIfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result(10, -1);
    //{
    //    exp_result.push_back(9);exp_result.push_back(-1);
    //    exp_result.push_back(7);exp_result.push_back(-1);
    //    exp_result.push_back(5);exp_result.push_back(-1);
    //    exp_result.push_back(3);exp_result.push_back(-1);
    //    exp_result.push_back(1);exp_result.push_back(-1);
    //}
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
     bolt::cl::gather_if(map.begin(), map.end(), stencil.begin(), input.begin(), exp_result.begin(), iepred );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, GatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result(10, -1);   
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::counting_iterator<int> stencil_first(0);    
    is_even iepred;
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), exp_result.begin(), iepred );
    bolt::cl::gather_if(map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};

    std::vector<int> exp_result(10, -1);   
    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::counting_iterator<int> stencil_first(0);  

    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil_first, input.begin(), exp_result.begin(), iepred );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, GatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    

    is_even iepred;
    bolt::cl::gather_if( map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<int> result ( 10, -1 );
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_int, GatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_int, SerialGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_int, MulticoreGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    bolt::cl::device_vector<int> result ( 10, -1 );
    bolt::cl::device_vector<int> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, Gather_IfPredicate)
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
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), exp_result.begin(), iepred );
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, SerialGather_IfPredicate)
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

     bolt::cl::gather_if(ctl, map.begin(), map.end(),  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), exp_result.begin(), iepred );
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoresGather_IfPredicate)
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
     bolt::cl::gather_if(ctl, map.begin(), map.end(),  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), exp_result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, Gather_IfPredicate_Fancy_stencil)
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
    bolt::cl::counting_iterator<int> stencil_first(0);

    is_even iepred;

     bolt::cl::gather_if(map.begin(), map.end(),  stencil_first, input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil_first, input.begin(), exp_result.begin(), iepred );
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, SerialGather_IfPredicate_Fancy_stencil)
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
    bolt::cl::counting_iterator<int> stencil_first(0);

    is_even iepred;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather_if(ctl, map.begin(), map.end(),  stencil_first, input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil_first, input.begin(), exp_result.begin(), iepred );
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoreGather_IfPredicate_Fancy_stencil)
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
    bolt::cl::counting_iterator<int> stencil_first(0);

    is_even iepred;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather_if(ctl, map.begin(), map.end(),  stencil_first, input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil_first, input.begin(), exp_result.begin(), iepred );
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, Gather_IfPredicate_Fancy_map)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            input[i] =  i + 2 * i;
            stencil[i] = i + 5 * 1;
        }
    bolt::cl::counting_iterator<int> map(0); 
    is_even iepred;

     bolt::cl::gather_if(map, map+myStdVectSize,  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if(  map, map+myStdVectSize,  stencil.begin(), input.begin(), exp_result.begin(), iepred );
    //for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, SerialGather_IfPredicate_Fancy_map)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            input[i] =  i + 2 * i;
            stencil[i] = i + 5 * 1;
        }
    bolt::cl::counting_iterator<int> map(0); 
    is_even iepred;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather_if(ctl, map, map+myStdVectSize,  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if(  map, map+myStdVectSize,  stencil.begin(), input.begin(), exp_result.begin(), iepred );
    //for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoreGather_IfPredicate_Fancy_map)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            input[i] =  i + 2 * i;
            stencil[i] = i + 5 * 1;
        }
    bolt::cl::counting_iterator<int> map(0); 
    is_even iepred;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather_if(ctl, map, map+myStdVectSize,  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if(  map, map+myStdVectSize,  stencil.begin(), input.begin(), exp_result.begin(), iepred );
    //for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(DeviceMemory_IntBoltdVector, Gather_IfPredicate)
{
    std::vector<int> h_input( myStdVectSize,0);   
    std::vector<int> h_map (myStdVectSize,0);	
    std::vector<int> h_stencil (myStdVectSize,0);

    for( int i=0; i < myStdVectSize ; i++ )
    {
        h_map[i] = i;
        h_input[i] =  i + 2 * i;
        h_stencil[i] = i + 5 * 1;
    }

    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> map ( h_map.begin(), h_map.end() );	
    bolt::cl::device_vector<int> stencil ( h_stencil.begin(), h_stencil.end() );	

   // std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;


     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), exp_result.begin(), iepred );
    cmpArrays(exp_result, result);
}

TEST_P(DeviceMemory_IntBoltdVector, SerialGather_IfPredicate)
{
    std::vector<int> h_input( myStdVectSize,0);   
    std::vector<int> h_map (myStdVectSize,0);	
    std::vector<int> h_stencil (myStdVectSize,0);

    for( int i=0; i < myStdVectSize ; i++ )
    {
        h_map[i] = i;
        h_input[i] =  i + 2 * i;
        h_stencil[i] = i + 5 * 1;
    }

    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> map ( h_map.begin(), h_map.end() );	
    bolt::cl::device_vector<int> stencil ( h_stencil.begin(), h_stencil.end() );	

   // std::random_shuffle( map.begin(), map.end() ); 

    is_even iepred;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather_if(ctl, map.begin(), map.end(),  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), exp_result.begin(), iepred );
    cmpArrays(exp_result, result);
}
TEST_P(DeviceMemory_IntBoltdVector, MulticoreGather_IfPredicate)
{
    std::vector<int> h_input( myStdVectSize,0);   
    std::vector<int> h_map (myStdVectSize,0);	
    std::vector<int> h_stencil (myStdVectSize,0);

    for( int i=0; i < myStdVectSize ; i++ )
    {
        h_map[i] = i;
        h_input[i] =  i + 2 * i;
        h_stencil[i] = i + 5 * 1;
    }

    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> map ( h_map.begin(), h_map.end() );	
    bolt::cl::device_vector<int> stencil ( h_stencil.begin(), h_stencil.end() );

    is_even iepred;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather_if(ctl, map.begin(), map.end(),  stencil.begin(), input.begin(), result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(),  stencil.begin(), input.begin(), exp_result.begin(), iepred );
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    cmpArrays(exp_result, result);
}


//TEST( HostMemory_Float, GatherIfPredicate )
//{
//    float n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
//    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
//    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
//
//    std::vector<float> exp_result;
//    {
//        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
//        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
//        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
//        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
//        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
//    }
//    std::vector<float> result ( 10, -1.5f );
//    std::vector<float> input ( n_input, n_input + 10 );
//    std::vector<float> map ( n_map, n_map + 10 );
//    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
//    is_even iepred;
//    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
//    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
//
//    EXPECT_EQ(exp_result, result);
//}
TEST( HostMemory_Float, SerialGatherIfPredicate )
{
    // VS2012 doesn't support initializer list

    //std::vector<int> input {0,1,2,3,4,5,6,7,8,9};
    //std::vector<int> map {9,8,7,6,5,4,3,2,1,0};
    //std::vector<int> stencil {0,1,0,1,0,1,0,1,0,1};
    //std::vector<int> result ( 10, 0 );

    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGatherIfPredicate )
{   
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }
    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, GatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }

    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }

    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }

    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, GatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }

    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    

    is_even iepred;
    bolt::cl::gather_if( map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialGatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }

    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }

    std::vector<float> result ( 10, -1.5f );
    std::vector<float> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Float, GatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }
    bolt::cl::device_vector<float> result ( 10, -1.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Float, SerialGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }
    bolt::cl::device_vector<float> result ( 10, -1.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );  

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Float, MulticoreGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10]   =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<float> exp_result;
    {
        exp_result.push_back(9.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(7.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(5.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(3.5f);exp_result.push_back(-1.5f);
        exp_result.push_back(1.5f);exp_result.push_back(-1.5f);
    }
    bolt::cl::device_vector<float> result ( 10, -1.5f );
    bolt::cl::device_vector<float> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );  

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}

TEST( HostMemory_Long, GatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    cl_long n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    std::vector<cl_long> result ( 10, -1 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, SerialGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    cl_long n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    std::vector<cl_long> result ( 10, -1 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, MulticoreGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    cl_long n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }
    std::vector<cl_long> result ( 10, -1 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Long, GatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    cl_long n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<cl_long> result ( 10, -1 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, SerialGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    cl_long n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<cl_long> result ( 10, -1 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    is_even iepred;
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Long, MulticoreGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    cl_long n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<cl_long> exp_result;
    {
        exp_result.push_back(9);exp_result.push_back(-1);
        exp_result.push_back(7);exp_result.push_back(-1);
        exp_result.push_back(5);exp_result.push_back(-1);
        exp_result.push_back(3);exp_result.push_back(-1);
        exp_result.push_back(1);exp_result.push_back(-1);
    }

    std::vector<cl_long> result ( 10, -1 );
    std::vector<cl_long> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    is_even iepred;
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#if(TEST_DOUBLE == 1)
TEST( HostMemory_Double, GatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGatherIfPredicate )
{ 
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }
    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, GatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }

    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }

    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGatherIfPredicate_Fancy_stencil )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::counting_iterator<int> stencil_first(0);  
    //bolt::cl::counting_iterator<int> stencil_last =  stencil_first+10; 

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }

    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil_first, input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, GatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }

    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    

    is_even iepred;
    bolt::cl::gather_if( map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }

    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGatherIfPredicate_Fancy_map )
{
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    bolt::cl::counting_iterator<int> map(0); 
    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }

    std::vector<double> result ( 10, -1.5 );
    std::vector<double> input ( n_input, n_input + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map, map+10, stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( DeviceMemory_Double, GatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }
    bolt::cl::device_vector<double> result ( 10, -1.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;
    bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Double, SerialGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }
    bolt::cl::device_vector<double> result ( 10, -1.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
TEST( DeviceMemory_Double, MulticoreGatherIfPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10]   =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(-1.5);
        exp_result.push_back(7.5);exp_result.push_back(-1.5);
        exp_result.push_back(5.5);exp_result.push_back(-1.5);
        exp_result.push_back(3.5);exp_result.push_back(-1.5);
        exp_result.push_back(1.5);exp_result.push_back(-1.5);
    }
    bolt::cl::device_vector<double> result ( 10, -1.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    bolt::cl::device_vector<int> stencil ( n_stencil, n_stencil + 10 );    

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    is_even iepred;
    bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }

    cmpArrays(exp_result, result);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gather tests
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST( HostMemory_int, Gather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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
    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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
    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, Gather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather( map, map+10, input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, Gather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> input(0); 

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> input(0); 

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> input(0); 

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    std::vector<int> result ( 10, 0 );
    std::vector<int> map ( n_map, n_map + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES==1)
TEST( HostMemory_int, Gather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, SerialGather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_int, MulticoreGather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
#endif

TEST(DeviceMemory_Int, Gather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST(DeviceMemory_Int, SerialGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST(DeviceMemory_Int, MulticoreGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

TEST(DeviceMemory_Int, Gather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather( map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Int, SerialGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Int, MulticoreGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST(DeviceMemory_Int, Gather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> input(0); 

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST(DeviceMemory_Int, SerialGather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> input(0); 

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}
TEST(DeviceMemory_Int, MulticoreGather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<int> input(0); 

    std::vector<int> exp_result;
    {
        exp_result.push_back(0);exp_result.push_back(1);
        exp_result.push_back(2);exp_result.push_back(3);
        exp_result.push_back(4);exp_result.push_back(5);
        exp_result.push_back(6);exp_result.push_back(7);
        exp_result.push_back(8);exp_result.push_back(9);
    }
    bolt::cl::device_vector<int> result ( 10, 0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );

}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES == 1)
TEST( DeviceMemory_Int, Gather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
TEST( DeviceMemory_Int, SerialGather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
TEST( DeviceMemory_Int, MulticoreGather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10] =  {9,8,7,6,5,4,3,2,1,0};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
#endif

TEST_P(HostMemory_IntStdVector, Gather)
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

     bolt::cl::gather(map.begin(), map.end(), input.begin(), result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, SerialGather)
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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoreGather)
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

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, Gather_Fancy_map)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            input[i] =  i + 2 * i;
        }
    bolt::cl::counting_iterator<int> map(0);

    bolt::cl::gather(map, map+myStdVectSize, input.begin(), result.begin());
    bolt::cl::gather( map, map+myStdVectSize,  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, SerialGather_Fancy_map)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            input[i] =  i + 2 * i;
        }
    bolt::cl::counting_iterator<int> map(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather(ctl, map, map+myStdVectSize, input.begin(), result.begin());
     bolt::cl::gather( map, map+myStdVectSize,  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoreGather_Fancy_map)
{
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            input[i] =  i + 2 * i;
        }
    bolt::cl::counting_iterator<int> map(0);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather(ctl, map, map+myStdVectSize, input.begin(), result.begin());
     bolt::cl::gather( map, map+myStdVectSize,  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, Gather_Fancy_Input)
{
    bolt::cl::counting_iterator<int> input(0);
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;            
        }
    std::random_shuffle( map.begin(), map.end() ); 


     bolt::cl::gather( map.begin(), map.end(), input, result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input, exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_IntStdVector, SerialGather_Fancy_Input)
{
    bolt::cl::counting_iterator<int> input(0);
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;            
        }
    std::random_shuffle( map.begin(), map.end() ); 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input, exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_IntStdVector, MulticoreGather_Fancy_Input)
{
    bolt::cl::counting_iterator<int> input(0);
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
    std::vector<int> stencil (myStdVectSize,0);	
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;            
        }
    std::random_shuffle( map.begin(), map.end() ); 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input, exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    EXPECT_EQ(exp_result, result);
}

TEST_P(DeviceMemory_IntBoltdVector, Gather)
{
    std::vector<int> h_input( myStdVectSize,0);   
    std::vector<int> h_map (myStdVectSize,0);	

    for( int i=0; i < myStdVectSize ; i++ )
    {
        h_map[i] = i;
        h_input[i] =  i + 2 * i;
    }

    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> map ( h_map.begin(), h_map.end() );	

     bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
     cmpArrays(exp_result, result);
}

TEST_P(DeviceMemory_IntBoltdVector, SerialGather)
{
    std::vector<int> h_input( myStdVectSize,0);   
    std::vector<int> h_map (myStdVectSize,0);	

    for( int i=0; i < myStdVectSize ; i++ )
    {
        h_map[i] = i;
        h_input[i] =  i + 2 * i;
    }

    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> map ( h_map.begin(), h_map.end() );	

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

     bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
     cmpArrays(exp_result, result);
}
TEST_P(DeviceMemory_IntBoltdVector, MulticoreGather)
{
    std::vector<int> h_input( myStdVectSize,0);   
    std::vector<int> h_map (myStdVectSize,0);	

    for( int i=0; i < myStdVectSize ; i++ )
    {
        h_map[i] = i;
        h_input[i] =  i + 2 * i;
    }

    bolt::cl::device_vector<int> input( h_input.begin(), h_input.end() );   
    bolt::cl::device_vector<int> exp_result(myStdVectSize,0);    
    bolt::cl::device_vector<int> result ( myStdVectSize, 0 );
    bolt::cl::device_vector<int> map ( h_map.begin(), h_map.end() );	
//    std::random_shuffle( map.begin(), map.end() ); 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

     bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin());
     bolt::cl::gather( map.begin(), map.end(),  input.begin(), exp_result.begin());
   // for(int i=0; i<myStdVectSize ; i++)	{std::cout<<exp_result[ i ]<<"    "<<result[i]<<std::endl;}
    cmpArrays(exp_result, result);
}


TEST( HostMemory_Float, Gather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, Gather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather( map, map+10, input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Float, Gather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<float> input(0.5f); 

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    std::vector<float> result ( 10, 0.0f );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialGather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<float> input(0.5f); 

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    std::vector<float> result ( 10, 0.0f );
    std::vector<int> map ( n_map, n_map + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<float> input(0.5f); 

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    std::vector<float> result ( 10, 0.0f );
    std::vector<int> map ( n_map, n_map + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES==1)
TEST( HostMemory_Float, Gather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, SerialGather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Float, MulticoreGather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );

    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#endif

TEST(DeviceMemory_Float, Gather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Float, SerialGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Float, MulticoreGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST(DeviceMemory_Float, Gather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather( map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Float, SerialGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Float, MulticoreGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST(DeviceMemory_Float, Gather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<float> input(0.5f); 

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.0f );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Float, SerialGather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<float> input(0.5f); 

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.0f );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Float, MulticoreGather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<float> input(0.5f); 

    std::vector<float> exp_result;
    {
        exp_result.push_back(0.5f);exp_result.push_back(1.5f);
        exp_result.push_back(2.5f);exp_result.push_back(3.5f);
        exp_result.push_back(4.5f);exp_result.push_back(5.5f);
        exp_result.push_back(6.5f);exp_result.push_back(7.5f);
        exp_result.push_back(8.5f);exp_result.push_back(9.5f);
    }
    bolt::cl::device_vector<float> result ( 10, 0.0f );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<input[ i ]<<std::endl; }
    //cmpArrays( exp_result, result );
}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES==1)
TEST( DeviceMemory_Float, Gather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
TEST( DeviceMemory_Float, SerialGather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
TEST( DeviceMemory_Float, MulticoreGather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    float n_input[10] =  {9.5f,8.5f,7.5f,6.5f,5.5f,4.5f,3.5f,2.5f,1.5f,0.5f};

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

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
#endif

#if(TEST_DOUBLE == 1)
TEST( HostMemory_Double, Gather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, Gather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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

    bolt::cl::gather( map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_Double, Gather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<double> input(0.5f); 

    std::vector<double> exp_result;
    {
        exp_result.push_back(0.5);exp_result.push_back(1.5);
        exp_result.push_back(2.5);exp_result.push_back(3.5);
        exp_result.push_back(4.5);exp_result.push_back(5.5);
        exp_result.push_back(6.5);exp_result.push_back(7.5);
        exp_result.push_back(8.5);exp_result.push_back(9.5);
    }
    std::vector<double> result ( 10, 0.0 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<double> input(0.5f); 

    std::vector<double> exp_result;
    {
        exp_result.push_back(0.5);exp_result.push_back(1.5);
        exp_result.push_back(2.5);exp_result.push_back(3.5);
        exp_result.push_back(4.5);exp_result.push_back(5.5);
        exp_result.push_back(6.5);exp_result.push_back(7.5);
        exp_result.push_back(8.5);exp_result.push_back(9.5);
    }
    std::vector<double> result ( 10, 0.0 );
    std::vector<int> map ( n_map, n_map + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGather_Fancy_input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<double> input(0.5f); 

    std::vector<double> exp_result;
    {
        exp_result.push_back(0.5);exp_result.push_back(1.5);
        exp_result.push_back(2.5);exp_result.push_back(3.5);
        exp_result.push_back(4.5);exp_result.push_back(5.5);
        exp_result.push_back(6.5);exp_result.push_back(7.5);
        exp_result.push_back(8.5);exp_result.push_back(9.5);
    }
    std::vector<double> result ( 10, 0.0 );
    std::vector<int> map ( n_map, n_map + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES==1)
TEST( HostMemory_Double, Gather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, SerialGather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
     
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST( HostMemory_Double, MulticoreGather_device_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    std::vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    std::vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
     
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
#endif

TEST(DeviceMemory_Double, Gather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Double, SerialGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
     
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Double, MulticoreGather )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
     
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST(DeviceMemory_Double, Gather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );

    bolt::cl::gather( map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Double, SerialGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Double, MulticoreGather_Fancy_map )
{
    bolt::cl::counting_iterator<int> map(0); 
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(9.5);exp_result.push_back(8.5);
        exp_result.push_back(7.5);exp_result.push_back(6.5);
        exp_result.push_back(5.5);exp_result.push_back(4.5);
        exp_result.push_back(3.5);exp_result.push_back(2.5);
        exp_result.push_back(1.5);exp_result.push_back(0.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.5 );
    bolt::cl::device_vector<double> input ( n_input, n_input + 10 );
    
     bolt::cl::control ctl = bolt::cl::control::getDefault( );
     ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map, map+10, input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

TEST(DeviceMemory_Double, Gather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<double> input(0.5); 

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(0.5);exp_result.push_back(1.5);
        exp_result.push_back(2.5);exp_result.push_back(3.5);
        exp_result.push_back(4.5);exp_result.push_back(5.5);
        exp_result.push_back(6.5);exp_result.push_back(7.5);
        exp_result.push_back(8.5);exp_result.push_back(9.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Double, SerialGather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<double> input(0.5); 

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(0.5);exp_result.push_back(1.5);
        exp_result.push_back(2.5);exp_result.push_back(3.5);
        exp_result.push_back(4.5);exp_result.push_back(5.5);
        exp_result.push_back(6.5);exp_result.push_back(7.5);
        exp_result.push_back(8.5);exp_result.push_back(9.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}
TEST(DeviceMemory_Double, MulticoreGather_Fancy_Input )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    bolt::cl::counting_iterator<double> input(0.5); 

    bolt::cl::device_vector<double> exp_result;
    {
        exp_result.push_back(0.5);exp_result.push_back(1.5);
        exp_result.push_back(2.5);exp_result.push_back(3.5);
        exp_result.push_back(4.5);exp_result.push_back(5.5);
        exp_result.push_back(6.5);exp_result.push_back(7.5);
        exp_result.push_back(8.5);exp_result.push_back(9.5);
    }
    bolt::cl::device_vector<double> result ( 10, 0.0 );
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input, result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    cmpArrays( exp_result, result );
}

#if(TEMPORARY_DISABLE_STD_DV_TEST_CASES == 1)

TEST( DeviceMemory_Double, Gather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
TEST( DeviceMemory_Double, SerialGather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
TEST( DeviceMemory_Double, MulticoreGather_StdInput_Stdresult )
{
    int n_map[10] =  {0,1,2,3,4,5,6,7,8,9};
    double n_input[10] =  {9.5,8.5,7.5,6.5,5.5,4.5,3.5,2.5,1.5,0.5};

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
    bolt::cl::device_vector<int> map ( n_map, n_map + 10 );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), result.begin() );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
     EXPECT_EQ(exp_result, result);
}
#endif
#endif

TEST_P(HostMemory_UDDTestInt2, SerialGather_IfPredicate)
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
     bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), exp_result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred  );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_UDDTestInt2, MulticoreGather_IfPredicate)
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
     bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), exp_result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred  );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_UDDTestIntFloat, SerialGather_IfPredicate)
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
     bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), exp_result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred  );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_UDDTestIntFloat, MulticoreGather_IfPredicate)
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
     bolt::cl::gather_if(ctl, map.begin(), map.end(), stencil.begin(), input.begin(), exp_result.begin(), iepred );
     bolt::cl::gather_if( map.begin(), map.end(), stencil.begin(), input.begin(), result.begin(), iepred  );
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

TEST_P(HostMemory_UDDTestInt2, SerialGather)
{
    std::vector<Int2> input( myStdVectSize);   
    std::vector<Int2> exp_result(myStdVectSize);    
    std::vector<Int2> result ( myStdVectSize);
    std::vector<int> map (myStdVectSize,0);   
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i].a =  i + 2 * i;
            input[i].b = i + 3 * i;            
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
     bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), exp_result.begin());
     bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}
TEST_P(HostMemory_UDDTestInt2, MulticoreGather)
{
    std::vector<Int2> input( myStdVectSize);   
    std::vector<Int2> exp_result(myStdVectSize);    
    std::vector<Int2> result ( myStdVectSize);
    std::vector<int> map (myStdVectSize,0);   
    for( int i=0; i < myStdVectSize ; i++ )
        {
            map[i] = i;
            input[i].a =  i + 2 * i;
            input[i].b = i + 3 * i;            
        }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
     bolt::cl::gather(ctl, map.begin(), map.end(), input.begin(), exp_result.begin());
     bolt::cl::gather( map.begin(), map.end(), input.begin(), result.begin());
    //for(int i=0; i<10 ; i++){ std::cout<<result[ i ]<<std::endl; }
    EXPECT_EQ(exp_result, result);
}

///////////////////////////////////
////////////OFFSET TESTCASES///////
///////////////////////////////////
TEST( HostMemory_int, OffsetGatherIfPredicate )
{
    int n_map[10]     =  {4,3,1,2,0,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result(10, -100);    
    std::vector<int> result(10, -100 );
    
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;

    bolt::cl::gather_if( map.begin(), map.begin()+5, stencil.begin()+2, input.begin(), result.begin(), iepred );

    bolt::cl::gather_if(map.begin(), map.begin()+5, stencil.begin()+2, input.begin(), exp_result.begin(), iepred );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, SerialOffsetGatherIfPredicate )
{
    int n_map[10]     =  {4,3,1,2,0,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result(10, -100);    
    std::vector<int> result(10, -100 );
    
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );
    std::vector<int> stencil ( n_stencil, n_stencil + 10 );    
    is_even iepred;

    bolt::cl::gather_if( map.begin(), map.begin()+5, stencil.begin()+2, input.begin(), result.begin(), iepred );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::gather_if(ctl, map.begin(), map.begin()+5, stencil.begin()+2, input.begin(), exp_result.begin(), iepred );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, OffsetGatherIfPredicateMedium )
{
    size_t myStdVectSize = 1024;
    int s_offset = 0;
    int e_offset = 515;
    size_t distance = e_offset - s_offset;

    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);
    std::vector<int> stencil (myStdVectSize,0);
	for (size_t i = 0; i < distance; i++)
    {
        map[i] =(int) (s_offset + i);
        input[i] =  (int)(i + 2 * i);
        stencil[i] =  (int)(i + 5 * i);
    }
    std::random_shuffle( map.begin(), map.end() );   
    is_even iepred;

    bolt::cl::gather_if( map.begin(), map.begin()+e_offset, stencil.begin()+56, input.begin(), result.begin(), iepred );

    bolt::cl::gather_if( map.begin(), map.begin()+e_offset, stencil.begin()+56, input.begin(), exp_result.begin(), iepred );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, SerialOffsetGatherIfPredicateMedium )
{
    size_t myStdVectSize = 1024;
    int s_offset = 0;
    int e_offset = 515;
    size_t distance = e_offset - s_offset;

    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);
    std::vector<int> stencil (myStdVectSize,0);
	for (size_t i = 0; i < distance; i++)
    {
        map[i] =(int) (s_offset + i);
        input[i] =  (int)(i + 2 * i);
        stencil[i] =  (int)(i + 5 * i);
    }
    std::random_shuffle( map.begin(), map.end() );   
    is_even iepred;

    bolt::cl::gather_if( map.begin(), map.begin()+e_offset, stencil.begin()+56, input.begin(), result.begin(), iepred );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::gather_if(ctl, map.begin(), map.begin()+e_offset, stencil.begin()+56, input.begin(), exp_result.begin(), iepred );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, OffsetGatherPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result(10, -100);    
    std::vector<int> result(10, -100 );
    
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.begin()+5, input.begin(), result.begin() );

    bolt::cl::gather( map.begin(), map.begin()+5, input.begin(), exp_result.begin() );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, SerialOffsetGatherPredicate )
{
    int n_map[10]     =  {0,1,2,3,4,5,6,7,8,9};
    int n_input[10]   =  {9,8,7,6,5,4,3,2,1,0};
    int n_stencil[10] =  {0,1,0,1,0,1,0,1,0,1};

    std::vector<int> exp_result(10, -100);    
    std::vector<int> result(10, -100 );
    
    std::vector<int> input ( n_input, n_input + 10 );
    std::vector<int> map ( n_map, n_map + 10 );

    bolt::cl::gather( map.begin(), map.begin()+5, input.begin(), result.begin() );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::gather(ctl, map.begin(), map.begin()+5, input.begin(), exp_result.begin() );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, OffsetGatherPredicateMedium )
{
    size_t myStdVectSize = 1024;

    int s_offset = 0;
    int e_offset = 567;
    size_t distance = e_offset - s_offset;
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
	for (size_t i = 0; i < distance; i++)
    {
        map[i] = (int)(s_offset + i);
        input[i] = (int)( i + 2 * i);
    }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::gather( map.begin(), map.begin() + e_offset, input.begin(), result.begin() );

    bolt::cl::gather( map.begin(), map.begin() + e_offset, input.begin(), exp_result.begin() );

    EXPECT_EQ(exp_result, result);
}

TEST( HostMemory_int, SerialOffsetGatherPredicateMedium )
{
    size_t myStdVectSize = 1024;

    int s_offset = 0;
    int e_offset = 567;
    size_t distance = e_offset - s_offset;
    std::vector<int> input( myStdVectSize,0);   
    std::vector<int> exp_result(myStdVectSize,0);    
    std::vector<int> result ( myStdVectSize, 0 );
    std::vector<int> map (myStdVectSize,0);	
	for (size_t i = 0; i < distance; i++)
    {
        map[i] = (int)(s_offset + i);
        input[i] = (int)( i + 2 * i);
    }
    std::random_shuffle( map.begin(), map.end() ); 

    bolt::cl::gather( map.begin(), map.begin() + e_offset, input.begin(), result.begin() );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::gather(ctl, map.begin(), map.begin() + e_offset, input.begin(), exp_result.begin() );

    EXPECT_EQ(exp_result, result);
}

#if TEST_LARGE_BUFFERS
TEST(sanity_gather_double_ptr, with_double) // EPR#391728
{
  int size = 100000;
  double *values;
  values=(double *) malloc(size * sizeof(double));

  for (int i = 0; i < size; i += 100)
  {
    values[i] = i % 2;
  }

  bolt::cl::device_vector<double> d_values(values, values + size);
  double *map;
  map=(double *)malloc(size * sizeof(double));

  for(int i=0; i<size/2 ;i += 2 )
  {
    if(i % 2 == 0)
    {
      map[i]= (double) i;
    }
  }
  std::random_shuffle(map, map+10);

  bolt::cl::device_vector<double> d_map(map, map + size);
  bolt::cl::device_vector<double> d_output(size,0);
  bolt::cl::device_vector<double> expected_output(size);

  bolt::cl::gather(d_map.begin(), d_map.end(), d_values.begin(), d_output.begin());

  for (int i = 0; i < size; i++)
  {
    expected_output[i]= 0 ;
  }

  for(int i=0; i<size; i++ )
  {
    EXPECT_EQ(expected_output[i], d_output[i] ) ;
  }

}

#endif

INSTANTIATE_TEST_CASE_P(GatherIntLimit, HostMemory_IntStdVector, ::testing::Range(10, 2400, 230));
INSTANTIATE_TEST_CASE_P(GatherIntLimit, DeviceMemory_IntBoltdVector, ::testing::Range(10, 2400, 230));
INSTANTIATE_TEST_CASE_P(GatherUDDLimit, HostMemory_UDDTestInt2, ::testing::Range(10, 2400, 230)); 
INSTANTIATE_TEST_CASE_P(GatherUDDLimit, HostMemory_UDDTestIntFloat, ::testing::Range(10, 2400, 230)); 


int main(int argc, char* argv[])
{
    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

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

