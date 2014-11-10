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
#include <vector>
#include <array>

#include "bolt/cl/scan.h"
#include "bolt/unicode.h"
#include "bolt/miniDump.h"
#include <bolt/cl/iterator/counting_iterator.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 1

#define SERIAL_TBB_OFFSET 0



/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
#if 1
/******************************************************************************
 *  Double x4
 *****************************************************************************/
BOLT_FUNCTOR(uddtD4,
struct uddtD4
{
    double a;
    double b;
    double c;
    double d;

    bool operator==(const uddtD4& rhs) const
    {
        bool equal = true;
        double th = 0.0000000001;
        if (rhs.a < th && rhs.a > -th)
            equal = ( (1.0*a - rhs.a) < th && (1.0*a - rhs.a) > -th) ? equal : false;
        else
            equal = ( (1.0*a - rhs.a)/rhs.a < th && (1.0*a - rhs.a)/rhs.a > -th) ? equal : false;
        if (rhs.b < th && rhs.b > -th)
            equal = ( (1.0*b - rhs.b) < th && (1.0*b - rhs.b) > -th) ? equal : false;
        else
            equal = ( (1.0*b - rhs.b)/rhs.b < th && (1.0*b - rhs.b)/rhs.b > -th) ? equal : false;
        if (rhs.c < th && rhs.c > -th)
            equal = ( (1.0*c - rhs.c) < th && (1.0*c - rhs.c) > -th) ? equal : false;
        else
            equal = ( (1.0*c - rhs.c)/rhs.c < th && (1.0*c - rhs.c)/rhs.c > -th) ? equal : false;
        if (rhs.d < th && rhs.d > -th)
            equal = ( (1.0*d - rhs.d) < th && (1.0*d - rhs.d) > -th) ? equal : false;
        else
            equal = ( (1.0*d - rhs.d)/rhs.d < th && (1.0*d - rhs.d)/rhs.d > -th) ? equal : false;
        return equal;
    }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_FUNCTOR(MultD4,
struct MultD4
{
    uddtD4 operator()(const uddtD4 &lhs, const uddtD4 &rhs) const
    {
        uddtD4 _result;
        _result.a = lhs.a*rhs.a;
        _result.b = lhs.b*rhs.b;
        _result.c = lhs.c*rhs.c;
        _result.d = lhs.d*rhs.d;
        return _result;
    };
}; 
);
uddtD4 identityMultD4 = { 1.0, 1.0, 1.0, 1.0 };
uddtD4 initialMultD4  = { 1.00001, 1.000003, 1.0000005, 1.00000007 };

/******************************************************************************
 *  Integer x2
 *****************************************************************************/
BOLT_FUNCTOR(uddtI2,
struct uddtI2
{
    int a;
    int b;

    bool operator==(const uddtI2& rhs) const
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        return equal;
    }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtI2 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtI2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_FUNCTOR(AddI2,
struct AddI2
{
    uddtI2 operator()(const uddtI2 &lhs, const uddtI2 &rhs) const
    {
        uddtI2 _result;
        _result.a = lhs.a+rhs.a;
        _result.b = lhs.b+rhs.b;
        return _result;
    };
}; 
);
uddtI2 identityAddI2 = {  0, 0 };
uddtI2 initialAddI2  = { -1, 2 };

/******************************************************************************
 *  Mixed float and int
 *****************************************************************************/
BOLT_FUNCTOR(uddtM3,
struct uddtM3
{
    unsigned int a;
    float        b;
    double       c;

    bool operator==(const uddtM3& rhs) const
    {
        bool equal = true;
        float ths = 0.00001f;
        double thd = 0.0000000001;
        equal = ( a == rhs.a ) ? equal : false;
        if (rhs.b < ths && rhs.b > -ths)
            equal = ( (1.0*b - rhs.b) < ths && (1.0*b - rhs.b) > -ths) ? equal : false;
        else
            equal = ( (1.0*b - rhs.b)/rhs.b < ths && (1.0*b - rhs.b)/rhs.b > -ths) ? equal : false;
        if (rhs.c < thd && rhs.c > -thd)
            equal = ( (1.0*c - rhs.c) < thd && (1.0*c - rhs.c) > -thd) ? equal : false;
        else
            equal = ( (1.0*c - rhs.c)/rhs.c < thd && (1.0*c - rhs.c)/rhs.c > -thd) ? equal : false;
        return equal;
    }
};
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtM3 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtM3 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

BOLT_FUNCTOR(MixM3,
struct MixM3
{
    uddtM3 operator()(const uddtM3 &lhs, const uddtM3 &rhs) const
    {
        uddtM3 _result;
        _result.a = lhs.a^rhs.a;
        _result.b = lhs.b+rhs.b;
        _result.c = lhs.c*rhs.c;
        return _result;
    };
}; 
);
uddtM3 identityMixM3 = { 0, 0.f, 1.0 };
uddtM3 initialMixM3  = { 1, 1.f, 1.000001 };
#endif

#if 0
template< typename T >
::testing::AssertionResult cmpArrays( const T ref, const T calc, size_t N )
{
    for( size_t i = 0; i < N; ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

template< typename T, size_t N >
::testing::AssertionResult cmpArrays( const T (&ref)[N], const T (&calc)[N] )
{
    for( size_t i = 0; i < N; ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

//  Primary class template for std::array types
//  The struct wrapper is necessary to partially specialize the member function
template< typename T, size_t N >
struct cmpStdArray
{
    static ::testing::AssertionResult cmpArrays( const std::array< T, N >& ref, const std::array< T, N >& calc )
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};

//  Partial template specialization for float types
//  Partial template specializations only works for objects, not functions
template< size_t N >
struct cmpStdArray< float, N >
{
    static ::testing::AssertionResult cmpArrays( const std::array< float, N >& ref, const std::array< float, N >& calc)
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_FLOAT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};

#if TEST_DOUBLE
//  Partial template specialization for float types
//  Partial template specializations only works for objects, not functions
template< size_t N >
struct cmpStdArray< double, N >
{
    static ::testing::AssertionResult cmpArrays( const std::array< double, N >& ref, const std::array< double,N >&calc)
    {
        for( size_t i = 0; i < N; ++i )
        {
            EXPECT_DOUBLE_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};
#endif

//  The following cmpArrays verify the correctness of std::vectors's
template< typename T >
::testing::AssertionResult cmpArrays( const std::vector< T >& ref, const std::vector< T >& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

::testing::AssertionResult cmpArrays( const std::vector< float >& ref, const std::vector< float >& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_FLOAT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

#if TEST_DOUBLE
::testing::AssertionResult cmpArrays( const std::vector< double >& ref, const std::vector< double >& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_DOUBLE_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}
#endif

//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename S, typename B >
::testing::AssertionResult cmpArrays( const S& ref, const B& calc )
{
    for( size_t i = 0; i < ref.size( ); ++i )
    {
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}
#endif

#include "test_common.h"


/******************************************************************************
 *  Scan with User Defined Data Types and Operators
 *****************************************************************************/
TEST(InclusiveScan, normalArrayTest)
{
    const int length=10000;
    float input[length] ;
   
    float refInput[length];
   
    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::plus<float> ai2;
    bolt::cl::inclusive_scan( input,    input + length,    input, ai2 );
    ::std::partial_sum(refInput, refInput + length, refInput, ai2);
    // compare results
    cmpArrays<float,length>(input, refInput);
} 

TEST(InclusiveScan, SerialnormalArrayTest)
{
    const int length=10000;
    float input[length] ;
   
    float refInput[length];
   
    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    // call scan
    bolt::cl::plus<float> ai2;
    bolt::cl::inclusive_scan( ctl, input,    input + length,    input, ai2 );
    ::std::partial_sum(refInput, refInput + length, refInput, ai2);
    // compare results
    cmpArrays<float,length>(input, refInput);
        
} 

TEST(InclusiveScan, MulticorenormalArrayTest)
{
    const int length=10000;
    float input[length] ;
   
    float refInput[length];
   
    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    // call scan
    bolt::cl::plus<float> ai2;
    bolt::cl::inclusive_scan( ctl, input,    input + length,    input, ai2 );
    ::std::partial_sum(refInput, refInput + length, refInput, ai2);
    // compare results
    cmpArrays<float,length>(input, refInput);
    
    
} 



TEST(InclusiveScan, DeviceVectorInclFloat)
{
    int length = 1<<10;
   
    //bolt::cl::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        refInput[i] = 1.f + rand()%3;
    }
    
    bolt::cl::device_vector< float > input( refInput.begin(), refInput.end());
     
    // call scan
    bolt::cl::plus<float> ai2;

	//Out of Place Scan
    //bolt::cl::inclusive_scan( input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);

	//Inplace Scan
	bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    //cmpArrays(refOutput, output);
	cmpArrays(refInput, input);
    
    
} 

TEST(InclusiveScan, SerialDeviceVectorInclFloat)
{
    int length = 1<<10;
    //bolt::cl::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
         refInput[i] = 1.f + rand()%3;
    }
    bolt::cl::device_vector< float > input( refInput.begin(), refInput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    // call scan
    bolt::cl::plus<float> ai2;
    /*bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);*/

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    //cmpArrays(refOutput, output);
	cmpArrays(refInput, input);
    
    
} 

TEST(InclusiveScan, MulticoreDeviceVectorInclFloat)
{
    int length = 1<<10;
    bolt::cl::device_vector< float > output( length);
    std::vector< float > refInput( length);
    std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
         refInput[i] = 1.f + rand()%3;
    }
    bolt::cl::device_vector< float > input( refInput.begin(), refInput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    // call scan
    bolt::cl::plus<float> ai2;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    //cmpArrays(refOutput, output);
	cmpArrays(refInput, input);
    
    
} 



#if (TEST_DOUBLE == 1)

TEST(InclusiveScan, DeviceVectorIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    bolt::cl::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::cl::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
    /*bolt::cl::inclusive_scan( input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);*/

	bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    //cmpArrays(refOutput, output);  
	cmpArrays(refInput, input); 
} 

TEST(InclusiveScan, SerialDeviceVectorIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    bolt::cl::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::cl::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    // call scan
    MixM3 M3;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);

    //cmpArrays(refOutput, output);  
	cmpArrays(refInput, input);  
} 

TEST(InclusiveScan, MulticoreDeviceVectorIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    bolt::cl::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::cl::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    // call scan
    MixM3 M3;
    /*bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 

#endif


TEST(ExclusiveScan, DeviceVectorExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > stdinput( length);
    //bolt::cl::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        stdinput[i] = 1.f + rand()%3;
        if(i != length-1)
           refInput[i+1] = stdinput[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;
    bolt::cl::device_vector< float > input( stdinput.begin(), stdinput.end());
    // call scan
    bolt::cl::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::cl::exclusive_scan( input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );

    // compare results
    cmpArrays(refInput, input);

} 

TEST(ExclusiveScan, SerialDeviceVectorExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > stdinput( length);
    //bolt::cl::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        stdinput[i] = 1.f + rand()%3;
        if(i != length-1)
           refInput[i+1] = stdinput[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;
    bolt::cl::device_vector< float > input( stdinput.begin(), stdinput.end());
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    bolt::cl::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );

    // compare results
    cmpArrays(refInput, input);

} 

TEST(ExclusiveScan, MulticoreDeviceVectorExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > stdinput( length);
    //bolt::cl::device_vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        stdinput[i] = 1.f + rand()%3;
        if(i != length-1)
           refInput[i+1] = stdinput[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;
     bolt::cl::device_vector< float > input( stdinput.begin(), stdinput.end());
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    // call scan
    bolt::cl::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );

    // compare results
    cmpArrays(refInput, input);

} 


#if (TEST_DOUBLE == 1)
TEST(ExclusiveScan, DeviceVectorExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    bolt::cl::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::cl::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
   /* bolt::cl::exclusive_scan( input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 

TEST(ExclusiveScan, SerialDeviceVectorExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    bolt::cl::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::cl::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    MixM3 M3;
   /* bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 

TEST(ExclusiveScan, MulticoreDeviceVectorExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    bolt::cl::device_vector< uddtM3 > input( length, initialMixM3  );
    //bolt::cl::device_vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    // call scan
    MixM3 M3;
    /*bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 
#endif


TEST(InclusiveScan, InclUdd)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  );
    //std::vector< uddtI2 > refOutput( length);
    // call scan
    AddI2 ai2;
    //bolt::cl::inclusive_scan( input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);

} 

TEST(InclusiveScan, SerialInclUdd)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  );
    //std::vector< uddtI2 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); // tested with serial also
    // call scan
    AddI2 ai2;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);

} 

TEST(InclusiveScan, MulticoreInclUdd)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  );
    //std::vector< uddtI2 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    AddI2 ai2;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 


TEST(InclusiveScan, InclFloat)
{
    //setup containers

    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::plus<float> ai2;
    //bolt::cl::inclusive_scan( input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 

TEST(InclusiveScan, SerialInclFloat)
{
    //setup containers

    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        refInput[i] = input[i];
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    bolt::cl::plus<float> ai2;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);

} 

TEST(InclusiveScan, MulticoreInclFloat)
{
    //setup containers

    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        refInput[i] = input[i];
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    bolt::cl::plus<float> ai2;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
} 


#if(TEST_DOUBLE == 1)
TEST(InclusiveScan, IncluddtM3)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
    /*bolt::cl::inclusive_scan( input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 

TEST(InclusiveScan, SerialIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    // call scan
    MixM3 M3;
   /* bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 

TEST(InclusiveScan, MulticoreIncluddtM3)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  );
    //std::vector< uddtM3 > refOutput( length);
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    // call scan
    MixM3 M3;
    /*bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  

} 
#endif


TEST(ExclusiveScan, ExclUdd)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
    //std::vector< uddtI2 > refOutput( length);
   // call scan
    AddI2 ai2;
    //bolt::cl::exclusive_scan( input.begin(),    input.end(),    output.begin(), initialAddI2, ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScan,SerialExclUdd)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
    //std::vector< uddtI2 > refOutput( length);
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    AddI2 ai2;
    //bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialAddI2, ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScan, MulticoreExclUdd)
{
    //setup containers
    int length = 1<<10;
    std::vector< uddtI2 > input( length, initialAddI2  );
    //std::vector< uddtI2 > output( length);
    std::vector< uddtI2 > refInput( length, initialAddI2  ); refInput[0] = initialAddI2;
    //std::vector< uddtI2 > refOutput( length);
 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    // call scan
    AddI2 ai2;
    //bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialAddI2, ai2 );
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), initialAddI2, ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ExclusiveScan, ExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;
    // call scan
    bolt::cl::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::cl::exclusive_scan( input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );

    // compare results
    cmpArrays(refInput, input);

} 

TEST(ExclusiveScan, SerialExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;


    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    bolt::cl::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );

    // compare results
    cmpArrays(refInput, input);

} 

TEST(ExclusiveScan, MulticoreExclFloat)
{
    //setup containers
    int length = 1<<10;
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        if(i != length-1)
           refInput[i+1] = input[i];
        //refInput[i] = 2.0f;
    }
    refInput[0] = 3.0f;


    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::plus<float> ai2;
    //::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), ai2);
    //bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), 3.0f, ai2 );

    //// compare results
    //cmpArrays(refOutput, output);

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );

    // compare results
    cmpArrays(refInput, input);

} 

#if(TEST_DOUBLE == 1)
TEST(ExclusiveScan, ExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);
    // call scan
    MixM3 M3;
   /* bolt::cl::exclusive_scan( input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  

} 

TEST(ExclusiveScan,SerialExcluddtM3)
{
    //setup containers
    int length = 1<<10;
  
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    // call scan
    MixM3 M3;
   /* bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output); */ 

	bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input); 
} 

//#if TEST_LARGE_BUFFERS
TEST(ExclusiveScan, MulticoreExcluddtM3)
{
    //setup containers
    int length = 1048576; // 2^20
  
    std::vector< uddtM3 > input( length, initialMixM3  );
    //std::vector< uddtM3 > output( length);
    std::vector< uddtM3 > refInput( length, initialMixM3  ); refInput[0] = initialMixM3;
    //std::vector< uddtM3 > refOutput( length);

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    MixM3 M3;
   /* bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refOutput.begin(), M3);
    
    cmpArrays(refOutput, output);  */

	bolt::cl::exclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), initialMixM3, M3 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), M3);
    
    cmpArrays(refInput, input);  
} 
//#endif
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(ScanUserDefined, IncAddInt2)
{
    //setup containers
    int length = (1<<16)+23;
    bolt::cl::device_vector< uddtI2 > input(  length, initialAddI2); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtI2 > output( length, identityAddI2); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtI2 > refInput( length, initialAddI2 );
    //std::vector< uddtI2 > refOutput( length );

    // call scan
    AddI2 ai2;
    //bolt::cl::inclusive_scan(  input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), ai2);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan(  input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    cmpArrays(refInput, input);

}

TEST(ScanUserDefined, SerialIncAddInt2)
{
    //setup containers
    int length = (1<<16)+23;
    bolt::cl::device_vector< uddtI2 > input(  length, initialAddI2); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtI2 > output( length, identityAddI2); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtI2 > refInput( length, initialAddI2 );
    //std::vector< uddtI2 > refOutput( length );
    

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    AddI2 ai2;
    //bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), ai2);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    cmpArrays(refInput, input);
}

TEST(ScanUserDefined, MulticoreIncAddInt2)
{
    //setup containers
    int length = (1<<16)+23;
    bolt::cl::device_vector< uddtI2 > input(  length, initialAddI2); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtI2 > output( length, identityAddI2); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtI2 > refInput( length, initialAddI2 );
    //std::vector< uddtI2 > refOutput( length );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    AddI2 ai2;
    //bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), ai2 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), ai2);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    cmpArrays(refInput, input);

}

#if(TEST_DOUBLE == 1)
TEST(ScanUserDefined, IncMultiplyDouble4)
{
    //setup containers
    int length = (1<<16)+11;
    bolt::cl::device_vector< uddtD4 > input(  length, initialMultD4); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtD4 > output( length, identityMultD4); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtD4 > refInput( length, initialMultD4 );
    //std::vector< uddtD4 > refOutput( length );

    // call scan
    MultD4 md4;
    //bolt::cl::inclusive_scan(  input.begin(),    input.end(),    output.begin(), md4 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), md4);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan(  input.begin(),    input.end(),    input.begin(), md4 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), md4);

    // compare results
    cmpArrays(refInput, input);

}

TEST(ScanUserDefined, SerialIncMultiplyDouble4)
{
    //setup containers
    int length = (1<<16)+11;
    bolt::cl::device_vector< uddtD4 > input(  length, initialMultD4); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtD4 > output( length, identityMultD4); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtD4 > refInput( length, initialMultD4 );
    //std::vector< uddtD4 > refOutput( length );
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    MultD4 md4;
    //bolt::cl::inclusive_scan(ctl, input.begin(),    input.end(),    output.begin(), md4 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), md4);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan(ctl, input.begin(),    input.end(),   input.begin(), md4 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), md4);

    // compare results
    cmpArrays(refInput, input);

}

TEST(ScanUserDefined, MulticoreIncMultiplyDouble4)
{
    //setup containers
    int length = (1<<16)+11;
    bolt::cl::device_vector< uddtD4 > input(  length, initialMultD4); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtD4 > output( length, identityMultD4); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtD4 > refInput( length, initialMultD4 );
    //std::vector< uddtD4 > refOutput( length );
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    MultD4 md4;
    //bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), md4 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), md4);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), md4 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), md4);

    // compare results
    cmpArrays(refInput, input);

}
#endif
#if(TEST_DOUBLE == 1)
TEST(ScanUserDefined, IncMixedM3)
{
    //setup containers
    int length = (1<<16)+57;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );

    // call scan
    MixM3 mm3;
    //bolt::cl::inclusive_scan(  input.begin(),    input.end(),    output.begin(), mm3 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), mm3);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan(  input.begin(),    input.end(),    input.begin(), mm3 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), mm3);

    // compare results
    cmpArrays(refInput, input);
}

TEST(ScanUserDefined, SerialIncMixedM3)
{
    //setup containers
    int length = (1<<16)+57;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    MixM3 mm3;
    //bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), mm3 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), mm3);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::inclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), mm3 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), mm3);

    // compare results
    cmpArrays(refInput, input);

}

TEST(ScanUserDefined, MulticoreIncMixedM3)
{
    //setup containers
    int length = (1<<16)+57;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    MixM3 mm3;
    //bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    output.begin(), mm3 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), mm3);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan( ctl,  input.begin(),    input.end(),    input.begin(), mm3 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), mm3);

    // compare results
    cmpArrays(refInput, input);

}
#endif
/////////////////////////////////////////////////  Exclusive  ///////////////////////////
TEST(ScanUserDefined, ExclAddInt2)
{
    //setup containers
    int length = (1<<16)+23;
    bolt::cl::device_vector< uddtI2 > input(  length, initialAddI2); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtI2 > output( length, identityAddI2); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtI2 > refInput( length, initialAddI2 ); refInput[0] = identityAddI2;
    //std::vector< uddtI2 > refOutput( length );

    // call scan
    AddI2 ai2;
    //bolt::cl::exclusive_scan(  input.begin(),    input.end(),    output.begin(), identityAddI2, ai2 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), ai2);

    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan(  input.begin(),    input.end(),    input.begin(), identityAddI2, ai2 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    cmpArrays(refInput, input);
}

TEST(ScanUserDefined, SerialExclAddInt2)
{
    //setup containers
    int length = (1<<16)+23;
    bolt::cl::device_vector< uddtI2 > input(  length, initialAddI2); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtI2 > output( length, identityAddI2); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtI2 > refInput( length, initialAddI2 ); refInput[0] = identityAddI2;
    //std::vector< uddtI2 > refOutput( length );
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    AddI2 ai2;
    //bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), identityAddI2, ai2 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), ai2);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), identityAddI2, ai2 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    cmpArrays(refInput, input);
}

TEST(ScanUserDefined, MulticoreExclAddInt2)
{
    //setup containers
    int length = (1<<16)+23;
    bolt::cl::device_vector< uddtI2 > input(  length, initialAddI2); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtI2 > output( length, identityAddI2); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtI2 > refInput( length, initialAddI2 ); refInput[0] = identityAddI2;
    //std::vector< uddtI2 > refOutput( length );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    AddI2 ai2;
    //bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), identityAddI2, ai2 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), ai2);

    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), identityAddI2, ai2 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), ai2);

    // compare results
    cmpArrays(refInput, input);
}


#if(TEST_DOUBLE == 1)
TEST(ScanUserDefined, ExclMultiplyDouble4)
{
    //setup containers
    int length = (1<<16)+11;
    bolt::cl::device_vector< uddtD4 > input(  length, initialMultD4); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtD4 > output( length, identityMultD4); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtD4 > refInput( length, initialMultD4 ); refInput[0] = identityMultD4;
    //std::vector< uddtD4 > refOutput( length );

    // call scan
    MultD4 md4;
    //bolt::cl::exclusive_scan(  input.begin(),    input.end(),    output.begin(), identityMultD4, md4 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), md4);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan(  input.begin(),    input.end(),    input.begin(), identityMultD4, md4 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), md4);

    //compare results
    cmpArrays(refInput, input);
}

TEST(ScanUserDefined, SerialExclMultiplyDouble4)
{
    //setup containers
    int length = (1<<16)+11;
    bolt::cl::device_vector< uddtD4 > input(  length, initialMultD4); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtD4 > output( length, identityMultD4); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtD4 > refInput( length, initialMultD4 ); refInput[0] = identityMultD4;
    //std::vector< uddtD4 > refOutput( length );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    MultD4 md4;
    //bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), identityMultD4, md4 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), md4);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), identityMultD4, md4 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), md4);

    // compare results
    cmpArrays(refInput, input);

}

TEST(ScanUserDefined, MulticoreExclMultiplyDouble4)
{
    //setup containers
    int length = (1<<16)+11;
    bolt::cl::device_vector< uddtD4 > input(  length, initialMultD4); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtD4 > output( length, identityMultD4); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtD4 > refInput( length, initialMultD4 ); refInput[0] = identityMultD4;
    //std::vector< uddtD4 > refOutput( length );
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    MultD4 md4;
    //bolt::cl::exclusive_scan(ctl, input.begin(),    input.end(),    output.begin(), identityMultD4, md4 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), md4);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan(ctl, input.begin(),    input.end(),    input.begin(), identityMultD4, md4 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), md4);

    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanUserDefined, ExclMixedM3)
{
    //setup containers
    int length = (1<<16)+57;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtM3 > refInput( length, initialMixM3 ); refInput[0] = identityMixM3;
    //std::vector< uddtM3 > refOutput( length );
    // call scan
    MixM3 mm3;
    //bolt::cl::exclusive_scan(  input.begin(),    input.end(),    output.begin(), identityMixM3, mm3 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), mm3);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan(  input.begin(),    input.end(),   input.begin(), identityMixM3, mm3 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), mm3);

    // compare results
    cmpArrays(refInput, input);
}

TEST(ScanUserDefined, SerialExclMixedM3)
{
    //setup containers
    int length = (1<<16)+57;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtM3 > refInput( length, initialMixM3 ); refInput[0] = identityMixM3;
    //std::vector< uddtM3 > refOutput( length );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    MixM3 mm3;
    //bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    output.begin(), identityMixM3, mm3 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), mm3);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan( ctl, input.begin(),    input.end(),    input.begin(), identityMixM3, mm3 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), mm3);

    // compare results
    cmpArrays(refInput, input);

}

TEST(ScanUserDefined, MulticoreExclMixedM3)
{
    //setup containers
    int length = (1<<16)+57;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3); //, CL_MEM_READ_WRITE, false );
    std::vector< uddtM3 > refInput( length, initialMixM3 ); refInput[0] = identityMixM3;
    //std::vector< uddtM3 > refOutput( length );
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    MixM3 mm3;
    //bolt::cl::exclusive_scan(ctl, input.begin(),    input.end(),    output.begin(), identityMixM3, mm3 );
    //::std::partial_sum(     refInput.begin(), refInput.end(), refOutput.begin(), mm3);

    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan(ctl, input.begin(),    input.end(),    input.begin(), identityMixM3, mm3 );
    ::std::partial_sum(     refInput.begin(), refInput.end(), refInput.begin(), mm3);

    // compare results
    cmpArrays(refInput, input);
}

#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process type parameterized tests

//  This class creates a C++ 'TYPE' out of a size_t value
template< size_t N >
class TypeValue
{
public:
    static const size_t value = N;
};

//  Test fixture class, used for the Type-parameterized tests
//  Namely, the tests that use std::array and TYPED_TEST_P macros
template< typename ArrayTuple >
class ScanArrayTest: public ::testing::Test
{
public:
    ScanArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        for( int i=0; i < ArraySize; i++ )
        {
            stdInput[ i ] = 1;
            boltInput[ i ] = 1;
        }
    };

    virtual void TearDown( )
    {};

    virtual ~ScanArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    // typedef typename std::tuple_element< 0, ArrayTuple >::type::value ArraySize;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;

    typename std::array< ArrayType, ArraySize > stdInput, boltInput;
    int m_Errors;
};

TYPED_TEST_CASE_P( ScanArrayTest );

TYPED_TEST_P( ScanArrayTest, InPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;    

    
    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::cl::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ) );


    typename ArrayCont::iterator istdEnd = ScanArrayTest< gtest_TypeParam_ >::stdInput.end( );
    typename ArrayCont::iterator iboltEnd = ScanArrayTest< gtest_TypeParam_ >::boltInput.end( );
    //  The returned iterator should be at the end of the result range
    EXPECT_EQ( istdEnd, stdEnd );
    EXPECT_EQ( iboltEnd, boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ScanArrayTest, InPlacePlusFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  

    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), std::plus< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::cl::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), bolt::cl::plus< ArrayType >( ) );

    typename ArrayCont::iterator istdEnd = ScanArrayTest< gtest_TypeParam_ >::stdInput.end( );
    typename ArrayCont::iterator iboltEnd = ScanArrayTest< gtest_TypeParam_ >::boltInput.end( );
    //  The returned iterator should be at the end of the result range
    EXPECT_EQ( istdEnd, stdEnd );
    EXPECT_EQ( iboltEnd, boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ScanArrayTest, InPlaceMaxFunction )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  

    //  Calling the actual functions under test
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), bolt::cl::maximum< ArrayType >( ) );
    typename ArrayCont::iterator boltEnd = bolt::cl::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), bolt::cl::maximum< ArrayType >( ) );

    typename ArrayCont::iterator istdEnd = ScanArrayTest< gtest_TypeParam_ >::stdInput.end( );
    typename ArrayCont::iterator iboltEnd = ScanArrayTest< gtest_TypeParam_ >::boltInput.end( );
    //  The returned iterator should be at the end of the result range
    EXPECT_EQ( istdEnd, stdEnd );
    EXPECT_EQ( iboltEnd, boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ScanArrayTest< gtest_TypeParam_ >::stdInput, ScanArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ScanArrayTest, OutofPlace )
{
    typedef typename ScanArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;  

    //  Declare temporary arrays to store results for out of place computation
    ArrayCont stdResult, boltResult;

    //  Calling the actual functions under test, out of place semantics
    typename ArrayCont::iterator stdEnd  = std::partial_sum( ScanArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::stdInput.end( ), stdResult.begin( ) );
    typename ArrayCont::iterator boltEnd = bolt::cl::inclusive_scan( ScanArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ScanArrayTest< gtest_TypeParam_ >::boltInput.end( ), boltResult.begin( ) );

    //  The returned iterator should be one past the end of the result array
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    typename ArrayCont::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    typename ArrayCont::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ScanArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( stdResult, boltResult );
}

REGISTER_TYPED_TEST_CASE_P( ScanArrayTest, InPlace, InPlacePlusFunction, InPlaceMaxFunction, OutofPlace );

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanIntegerVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanIntegerVector( ): stdInput( GetParam( ), 1 ), boltInput( GetParam( ), 1 )
    {}

protected:
    std::vector< int > stdInput, boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanFloatVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanFloatVector( ): stdInput( GetParam( ), 1.0f ), boltInput( GetParam( ), 1.0f )
    {}

protected:
    std::vector< float > stdInput, boltInput;
};


//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanDoubleVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanDoubleVector( ): stdInput( GetParam( ), 0.0 ), boltInput( GetParam( ), 0.0 )
    {}

protected:
    std::vector< double > stdInput, boltInput;
};


//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size              
class ScanIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanIntegerDeviceVector( ): stdInput( GetParam( ), 1 ), boltInput( static_cast<size_t>( GetParam( ) ), 1 )
    {}

protected:
    std::vector< int > stdInput;
    bolt::cl::device_vector< int > boltInput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ScanIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ScanIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );
        for( size_t i=0; i < size; i++ )
        {
            stdInput[ i ] = 1;
            boltInput[ i ] = 1;
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
    };

protected:
    //boost::shared_array< int > stdInput;
    //boost::shared_array< int > boltInput;
     int* stdInput;
     int* boltInput;
};

class scanStdVectorWithIters:public ::testing::TestWithParam<int>
{
protected:
    int myStdVectSize;
public:
    scanStdVectorWithIters():myStdVectSize(GetParam()){
    }
};

typedef scanStdVectorWithIters ScanOffsetTest;
typedef scanStdVectorWithIters ScanCLtypeTest;

class StdVectCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    StdVectCountingIterator():mySize(GetParam()){
    }
};



TEST_P (ScanCLtypeTest, InclTestLong)
{
	std::vector< cl_long > refInput(myStdVectSize);

	for(int i=0; i<myStdVectSize; i++) {
         refInput[i] = 1 + rand()%3;
    }
    bolt::cl::device_vector< cl_long> input( refInput.begin(), refInput.end());

    //bolt::cl::device_vector< cl_long > input( myStdVectSize, 1+rand()%3);
    //std::vector< cl_long > refInput( input.begin(), input.end());

   // call scan
    bolt::cl::plus<cl_long> ai2;
    bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);

    cmpArrays(input, refInput);

} 

TEST_P (ScanCLtypeTest, ExclTestLong)
{
  
    std::vector< cl_long > stdinput( myStdVectSize);
    std::vector< cl_long > refInput( myStdVectSize);

    for(int i=0; i<myStdVectSize; i++) {
        stdinput[i] = 1 + rand()%3;
        if(i != myStdVectSize-1)
           refInput[i+1] = stdinput[i];
    }
    refInput[0] = 3;

    bolt::cl::device_vector< cl_long > input( stdinput.begin(), stdinput.end());
    // call scan
    bolt::cl::plus<cl_long> ai2;

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );
    // compare results
    cmpArrays(refInput, input);

} 

TEST_P (ScanCLtypeTest, InclTestULong)
{
	std::vector< cl_ulong > refInput(myStdVectSize);

	for(int i=0; i<myStdVectSize; i++) {
         refInput[i] = 1 + rand()%3;
    }
    bolt::cl::device_vector< cl_ulong> input( refInput.begin(), refInput.end());

   // call scan
    bolt::cl::plus<cl_ulong> ai2;
    bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);

    cmpArrays(input, refInput);

} 

TEST_P (ScanCLtypeTest, ExclTestULong)
{
	std::vector< cl_ulong > stdinput( myStdVectSize);
    std::vector< cl_ulong > refInput( myStdVectSize);

    for(int i=0; i<myStdVectSize; i++) {
        stdinput[i] = 1 + rand()%3;
        if(i != myStdVectSize-1)
           refInput[i+1] = stdinput[i];
    }
    refInput[0] = 3;

    bolt::cl::device_vector< cl_ulong > input( stdinput.begin(), stdinput.end());
    // call scan
    bolt::cl::plus<cl_ulong> ai2;

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );
    // compare results
    cmpArrays(refInput, input);

} 

TEST_P (ScanCLtypeTest, InclTestShort)
{
	std::vector< cl_short > refInput(myStdVectSize);

	for(int i=0; i<myStdVectSize; i++) {
         refInput[i] = 1 + rand()%3;
    }
    bolt::cl::device_vector< cl_short> input( refInput.begin(), refInput.end());

   // call scan
    bolt::cl::plus<cl_short> ai2;
    bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);

    cmpArrays(input, refInput);

} 

TEST_P (ScanCLtypeTest, ExclTestShort)
{
	std::vector< cl_short > stdinput( myStdVectSize);
    std::vector< cl_short > refInput( myStdVectSize);

    for(int i=0; i<myStdVectSize; i++) {
        stdinput[i] = 1 + rand()%3;
        if(i != myStdVectSize-1)
           refInput[i+1] = stdinput[i];
    }
    refInput[0] = 3;

    bolt::cl::device_vector< cl_short > input( stdinput.begin(), stdinput.end());
    // call scan
    bolt::cl::plus<cl_short> ai2;

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );
    // compare results
    cmpArrays(refInput, input);
} 

TEST_P (ScanCLtypeTest, InclTestUShort)
{
	std::vector< cl_ushort > refInput(myStdVectSize);

	for(int i=0; i<myStdVectSize; i++) {
         refInput[i] = 1 + rand()%3;
    }
    bolt::cl::device_vector< cl_ushort> input( refInput.begin(), refInput.end());

   // call scan
    bolt::cl::plus<cl_ushort> ai2;


    bolt::cl::inclusive_scan( input.begin(),    input.end(),    input.begin(), ai2 );
    ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);



    cmpArrays(input, refInput);

} 

TEST_P (ScanCLtypeTest, ExclTestUShort)
{
	std::vector< cl_ushort > stdinput( myStdVectSize);
    std::vector< cl_ushort > refInput( myStdVectSize);

    for(int i=0; i<myStdVectSize; i++) {
        stdinput[i] = 1 + rand()%3;
        if(i != myStdVectSize-1)
           refInput[i+1] = stdinput[i];
    }
    refInput[0] = 3;

    bolt::cl::device_vector< cl_ushort > input( stdinput.begin(), stdinput.end());
    // call scan
    bolt::cl::plus<cl_ushort> ai2;

	::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), refInput[0], ai2 );
    // compare results
    cmpArrays(refInput, input);

} 


/*
//Scan With Fancy iterator as destination results in Compilation Error!
TEST_P( StdVectCountingIterator, withCountingIterator) 
{
    bolt::cl::counting_iterator<int> first(1);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> stdInput( mySize);
    std::vector<int> boltInput( mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i + 1;
        boltInput[i] = i + 1;
    }

    //This is logically incorrect!
    bolt::cl::counting_iterator<int> boltEnd = bolt::cl::inclusive_scan( boltInput.begin(), boltInput.end() , first);
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;

}


//Scan With Fancy iterator as input results in Compilation Error! -- NEED TO DEBUG

TEST_P( StdVectCountingIterator, withCountingIteratorInput) 
{
    bolt::cl::counting_iterator<int> first(1);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> stdInput(mySize);
    std::vector<int> boltOutput(mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i + 1;
    }


    std::vector<int>::iterator boltEnd = bolt::cl::inclusive_scan( first, / , boltOutput.begin());
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;

}


TEST_P( StdVectCountingIterator, SerialwithCountingIteratorInput) 
{
    bolt::cl::counting_iterator<int> first(1);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> stdInput( mySize);
    std::vector<int> boltOutput( mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i + 1;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    std::vector<int>::iterator boltEnd = bolt::cl::inclusive_scan( ctl, first, last , boltOutput.begin());
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;

}

TEST_P( StdVectCountingIterator, MultiCorewithCountingIteratorInput) 
{
    bolt::cl::counting_iterator<int> first(1);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> stdInput( mySize);
    std::vector<int> boltOutput( mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i + 1;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    std::vector<int>::iterator boltEnd = bolt::cl::inclusive_scan( ctl, first, last , boltOutput.begin());
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;

}
*/

TEST_P (scanStdVectorWithIters, intDefiniteValues){
    std::vector<int> stdInput( myStdVectSize);
    std::vector<int> boltInput( myStdVectSize);

    for (int i = 0; i < myStdVectSize; ++i){
        boltInput[i] = i + 1;
    }
        
    for (int i = 0; i < myStdVectSize; ++i){
        stdInput[i] = i + 1;
    }

    std::vector<int>::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                  boltInput.begin( ) );
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}
TEST_P (scanStdVectorWithIters, SerialintDefiniteValues){
    std::vector<int> stdInput( myStdVectSize);
    std::vector<int> boltInput( myStdVectSize);

    for (int i = 0; i < myStdVectSize; ++i){
        boltInput[i] = i + 1;
    }
        
    for (int i = 0; i < myStdVectSize; ++i){
        stdInput[i] = i + 1;
    }
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    std::vector<int>::iterator boltEnd = bolt::cl::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ),
                                                                                  boltInput.begin( ) );
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}
TEST_P (scanStdVectorWithIters, MulticoreintDefiniteValues){
    std::vector<int> stdInput( myStdVectSize);
    std::vector<int> boltInput( myStdVectSize);

    for (int i = 0; i < myStdVectSize; ++i){
        boltInput[i] = i + 1;
    }
        
    for (int i = 0; i < myStdVectSize; ++i){
        stdInput[i] = i + 1;
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    std::vector<int>::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                  boltInput.begin( ) );
    std::vector<int>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}

TEST_P (scanStdVectorWithIters, floatDefiniteValues){
    
    std::vector<float> boltInput( myStdVectSize);
    std::vector<float> stdInput( myStdVectSize);

    for (int i = 0; i < myStdVectSize; ++i){
        //stdInput[i] = 1.0f + ( static_cast<float>( rand( ) ) / RAND_MAX );
        stdInput[i] = 1.f + rand()%3;
        boltInput[i] = stdInput[i];
    }

    std::vector<float>::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                     boltInput.begin( ) );
    std::vector<float>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_FLOAT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}
TEST_P (scanStdVectorWithIters, SerialfloatDefiniteValues){
    
    std::vector<float> boltInput( myStdVectSize);
    std::vector<float> stdInput( myStdVectSize);

    for (int i = 0; i < myStdVectSize; ++i){
        //stdInput[i] = 1.0f + ( static_cast<float>( rand( ) ) / RAND_MAX );
        stdInput[i] = 1.f + rand()%3;
        boltInput[i] = stdInput[i];
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    std::vector<float>::iterator boltEnd = bolt::cl::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ),
                                                                                     boltInput.begin( ) );
    std::vector<float>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_FLOAT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}
TEST_P (scanStdVectorWithIters, MulticorefloatDefiniteValues){
    
    std::vector<float> boltInput( myStdVectSize);
    std::vector<float> stdInput( myStdVectSize);

    for (int i = 0; i < myStdVectSize; ++i){
        //stdInput[i] = 1.0f + ( static_cast<float>( rand( ) ) / RAND_MAX );
        stdInput[i] = 1.f + rand()%3;
        boltInput[i] = stdInput[i];
    }
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    std::vector<float>::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                     boltInput.begin( ) );
    std::vector<float>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_FLOAT_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}

#if TEST_LARGE_BUFFERS
TEST_P (ScanOffsetTest, InclOffsetTestFloat)
{
	float n = 1.f + rand()%3;
    bolt::cl::device_vector< float > input( myStdVectSize, n);
    std::vector< float > refInput( myStdVectSize, n);
   // call scan
    bolt::cl::plus<float> ai2;
    bolt::cl::inclusive_scan( input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestFloat)
{
	float n = 1.f + rand()%3;
    bolt::cl::device_vector< float > input( myStdVectSize,n);
    std::vector< float > refInput( myStdVectSize,n);
   
    refInput[myStdVectSize/4] = 3.0f;

    // call scan
    bolt::cl::plus<float> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::cl::exclusive_scan(input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , 3.0f, ai2  );

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
#endif

#if (SERIAL_TBB_OFFSET == 1)
TEST_P (ScanOffsetTest, InclOffsetTestFloatSerial)
{
	float n = 1.f + rand()%3;

    bolt::cl::device_vector< float > input( myStdVectSize, n);
    std::vector< float > refInput( myStdVectSize, n);
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
   // call scan
    bolt::cl::plus<float> ai2;
    bolt::cl::inclusive_scan( ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestFloatSerial)
{
	float n = 1.f + rand()%3;

    bolt::cl::device_vector< float > input( myStdVectSize,n);
    std::vector< float > refInput( myStdVectSize,n);
   
    //refInput[myStdVectSize/4] = 3.0f;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    // call scan
    bolt::cl::plus<float> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::cl::exclusive_scan(ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , refInput[myStdVectSize/4], ai2  );
    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 


TEST_P (ScanOffsetTest, InclOffsetTestFloatMultiCore)
{
	float n = 1.f + rand()%3;

    bolt::cl::device_vector< float > input( myStdVectSize, n);
    std::vector< float > refInput( myStdVectSize, n);
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
   // call scan
    bolt::cl::plus<float> ai2;
    bolt::cl::inclusive_scan( ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestFloatMultiCore)
{
	float n = 1.f + rand()%3;

    bolt::cl::device_vector< float > input( myStdVectSize,n);
    std::vector< float > refInput( myStdVectSize,n);
   
    refInput[myStdVectSize/4] = 3.0f;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    // call scan
    bolt::cl::plus<float> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::cl::exclusive_scan(ctl, input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , 3.f, ai2  );

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
#endif

#if (TEST_DOUBLE == 1)
#if TEST_LARGE_BUFFERS
TEST_P (ScanOffsetTest, InclOffsetTestDouble)
{
	double n = 1.0 + rand()%3;

    bolt::cl::device_vector< double > input( myStdVectSize, n);
    std::vector< double > refInput( myStdVectSize, n);
   // call scan
    bolt::cl::plus<double> ai2;
    bolt::cl::inclusive_scan( input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 
TEST_P (ScanOffsetTest, ExclOffsetTestDouble)
{
	double n = 1.0 + rand()%3;

    bolt::cl::device_vector< double > input( myStdVectSize,n);
    std::vector< double > refInput( myStdVectSize,n);
   
    refInput[myStdVectSize/4] = 3.0f;

    // call scan
    bolt::cl::plus<double> ai2;

    ::std::partial_sum(refInput.begin()+ (myStdVectSize/4) , refInput.end()- (myStdVectSize/4), refInput.begin()+ (myStdVectSize/4) , ai2);
    bolt::cl::exclusive_scan(input.begin() + (myStdVectSize/4),    input.end() - (myStdVectSize/4),    input.begin()+ (myStdVectSize/4) , 3.0f, ai2  );

    // ::std::partial_sum(refInput.begin(), refInput.end(), refInput.begin(), ai2);
    //  bolt::cl::exclusive_scan( input.begin(),    input.end(),    input.begin(), 3.0f, ai2 );


    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/4);
} 

TEST (ScanOffsetTest, InclOffsetTestUDD)
{

    int length = 1<<16;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    std::vector< uddtM3 > refInput( length, initialMixM3 );

    // call scan
    MixM3 ai2;
    bolt::cl::inclusive_scan( input.begin() + (length/4),    input.end() - (length/4),    input.begin()+ (length/4) , ai2 );
    ::std::partial_sum(refInput.begin()+ (length/4) , refInput.end()- (length/4), refInput.begin()+ (length/4) , ai2);

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",length, length/4);
} 
TEST (ScanOffsetTest, ExclOffsetTestUDD)
{
    int length = 1<<16;
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3); //,  CL_MEM_READ_WRITE, true  );
    std::vector< uddtM3 > refInput( length, initialMixM3 );

    refInput[length/4] = initialMixM3;

    // call scan
    MixM3 ai2;
    // call scan
    ::std::partial_sum(refInput.begin()+ (length/4) , refInput.end()- (length/4), refInput.begin()+ (length/4) , ai2);
    bolt::cl::exclusive_scan(input.begin() + (length/4),    input.end() - (length/4),    input.begin()+ (length/4) , initialMixM3, ai2  );

    cmpArrays(input, refInput);
    printf("\nPass for size=%d Offset=%d\n",length, length/4);
} 

#endif

TEST_P (scanStdVectorWithIters, doubleDefiniteValues){
    
    std::vector<double> boltInput( myStdVectSize);
    std::vector<double> stdInput( myStdVectSize);
    
    for (int i = 0; i < myStdVectSize; ++i){
        //stdInput[i] = 1.0f + ( static_cast<double>( rand( ) ) / RAND_MAX );
        stdInput[i] = 1.0 + rand()%3;
        boltInput[i] = stdInput[i];
    }

    std::vector<double>::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                      boltInput.begin( ) );
    std::vector<double>::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    
    EXPECT_DOUBLE_EQ((*(boltEnd-1)), (*(stdEnd-1)))<<std::endl;
}
#endif


INSTANTIATE_TEST_CASE_P(inclusiveScanIterIntLimit, ScanCLtypeTest, ::testing::Range( 1, 1024, 87 )); 
INSTANTIATE_TEST_CASE_P(inclusiveScanIterIntLimit, ScanOffsetTest, ::testing::Range(1025, 65535, 6552)); 
INSTANTIATE_TEST_CASE_P(inclusiveScanIterIntLimit, scanStdVectorWithIters, ::testing::Range(1025, 65535, 6552)); 
INSTANTIATE_TEST_CASE_P(withCountingIterator, StdVectCountingIterator, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15


TEST_P( ScanIntegerVector, InclusiveInplace )
{
    //cl_int err = CL_SUCCESS;

    //std::string strDeviceName = bolt::cl::control::getDefault( ).device( ).getInfo< CL_DEVICE_NAME >( &err );
    //bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    //std::cout << "Device under test : " << strDeviceName << std::endl;

    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< int >::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                     boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanIntegerVector, SerialInclusiveInplace )
{
    //cl_int err = CL_SUCCESS;

    //std::string strDeviceName = bolt::cl::control::getDefault( ).device( ).getInfo< CL_DEVICE_NAME >( &err );
    //bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    //std::cout << "Device under test : " << strDeviceName << std::endl;
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< int >::iterator boltEnd = bolt::cl::inclusive_scan( ctl, boltInput.begin( ), boltInput.end( ),
                                                                                          boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanIntegerVector, MulticoreInclusiveInplace )
{
    //cl_int err = CL_SUCCESS;

    //std::string strDeviceName = bolt::cl::control::getDefault( ).device( ).getInfo< CL_DEVICE_NAME >( &err );
    //bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    //std::cout << "Device under test : " << strDeviceName << std::endl;

   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< int >::iterator boltEnd = bolt::cl::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ),
                                                                                         boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}



TEST_P( ScanFloatVector, InclusiveInplace )
{
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< float >::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                       boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanFloatVector, SerialInclusiveInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< float >::iterator boltEnd = bolt::cl::inclusive_scan( ctl, boltInput.begin( ), boltInput.end( ),
                                                                                            boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanFloatVector, MulticoreInclusiveInplace )
{
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    std::vector< float >::iterator boltEnd = bolt::cl::inclusive_scan(ctl, boltInput.begin( ), boltInput.end( ),
                                                                                           boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< float >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< float >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}



#if (TEST_DOUBLE == 1)
TEST_P( ScanDoubleVector, InclusiveInplace )
{
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdInput.begin( ));
    std::vector< double >::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                        boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanDoubleVector, SerialInclusiveInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdInput.begin( ));
    std::vector< double >::iterator boltEnd = bolt::cl::inclusive_scan( ctl, boltInput.begin( ), boltInput.end( ),
                                                                                             boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanDoubleVector, MulticoreInclusiveInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< double >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdInput.begin( ));
    std::vector< double >::iterator boltEnd = bolt::cl::inclusive_scan( ctl, boltInput.begin( ), boltInput.end( ),
                                                                                             boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< double >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< double >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

#endif


TEST_P( ScanIntegerDeviceVector, InclusiveInplace )
{
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    bolt::cl::device_vector< int >::iterator boltEnd = bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                                                 boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanIntegerDeviceVector, SerialInclusiveInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    bolt::cl::device_vector< int >::iterator boltEnd = bolt::cl::inclusive_scan(ctl, boltInput.begin( ), 
                                                                  boltInput.end( ), boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ScanIntegerDeviceVector, MulticoreInclusiveInplace )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //  Calling the actual functions under test
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );
    bolt::cl::device_vector< int >::iterator boltEnd = bolt::cl::inclusive_scan(ctl, boltInput.begin( ), 
                                                                boltInput.end( ), boltInput.begin( ) );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdInput.end( ), stdEnd );
    EXPECT_EQ( boltInput.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdInput.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltInput.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

#if defined(_WIN32)
TEST_P( ScanIntegerNakedPointer, InclusiveInplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > stdEnd = std::partial_sum( wrapStdInput, wrapStdInput + endIndex, 
                                                                                            wrapStdInput );
    //int* stdEnd = std::partial_sum( stdInput, stdInput + endIndex, stdInput );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< int* > boltEnd = bolt::cl::inclusive_scan( wrapBoltInput,
                                                    wrapBoltInput + endIndex, wrapBoltInput );
    //int* boltEnd = bolt::cl::inclusive_scan( boltInput, boltInput + endIndex, boltInput );

    //  The returned iterator should be one past the 
    EXPECT_EQ( wrapStdInput + endIndex, stdEnd );
    EXPECT_EQ( wrapBoltInput + endIndex, boltEnd );

    size_t stdNumElements = std::distance( wrapStdInput, stdEnd );
    size_t boltNumElements = std::distance( wrapBoltInput, boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( ScanIntegerNakedPointer, SerialInclusiveInplace )
{
    size_t endIndex = GetParam( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > stdEnd = std::partial_sum( wrapStdInput, wrapStdInput + 
                                                                        endIndex, wrapStdInput );
    //int* stdEnd = std::partial_sum( stdInput, stdInput + endIndex, stdInput );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< int* > boltEnd = bolt::cl::inclusive_scan( ctl, wrapBoltInput,
                                                        wrapBoltInput + endIndex, wrapBoltInput );
    //int* boltEnd = bolt::cl::inclusive_scan( boltInput, boltInput + endIndex, boltInput );

    //  The returned iterator should be one past the 
    EXPECT_EQ( wrapStdInput + endIndex, stdEnd );
    EXPECT_EQ( wrapBoltInput + endIndex, boltEnd );

    size_t stdNumElements = std::distance( wrapStdInput, stdEnd );
    size_t boltNumElements = std::distance( wrapBoltInput, boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( ScanIntegerNakedPointer, MultiCoreInclusiveInplace )
{
    size_t endIndex = GetParam( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > stdEnd = std::partial_sum( wrapStdInput, wrapStdInput + 
                                                                        endIndex, wrapStdInput );
    //int* stdEnd = std::partial_sum( stdInput, stdInput + endIndex, stdInput );

    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
    stdext::checked_array_iterator< int* > boltEnd = bolt::cl::inclusive_scan( ctl, wrapBoltInput,
                                                        wrapBoltInput + endIndex, wrapBoltInput );
    //int* boltEnd = bolt::cl::inclusive_scan( boltInput, boltInput + endIndex, boltInput );

    //  The returned iterator should be one past the 
    EXPECT_EQ( wrapStdInput + endIndex, stdEnd );
    EXPECT_EQ( wrapBoltInput + endIndex, boltEnd );

    size_t stdNumElements = std::distance( wrapStdInput, stdEnd );
    size_t boltNumElements = std::distance( wrapBoltInput, boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif

TEST_P( ScanIntegerVector, ExclusiveOutOfPlace )
{
    //  Declare temporary arrays to store results for out of place computation
    std::vector< int > stdResult( GetParam( ) ), boltResult( GetParam( ) );
    int init = 3;

    //  Emulating a std exclusive scan
    if( stdInput.size( ) )
        stdInput[ 0 ] += init;
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdResult.begin( ) );
    if( stdInput.size( ) )
        stdInput[ 0 ] -= init;
    stdEnd = std::transform( stdResult.begin( ), stdResult.end( ), stdInput.begin( ), stdResult.begin( ),
                                                                                    std::minus< int >( ) );
    //  Calling Bolt exclusive scan
    std::vector< int >::iterator boltEnd = bolt::cl::exclusive_scan( boltInput.begin( ), boltInput.end( ),
                                                                            boltResult.begin( ), init );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdResult, boltResult );
}

TEST_P( ScanIntegerVector, SerialExclusiveOutOfPlace )
{
    //  Declare temporary arrays to store results for out of place computation
    std::vector< int > stdResult( GetParam( ) ), boltResult( GetParam( ) );
    int init = 3;

    //  Emulating a std exclusive scan
    if( stdInput.size( ) )
        stdInput[ 0 ] += init;
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdResult.begin( ) );
    if( stdInput.size( ) )
        stdInput[ 0 ] -= init;
    stdEnd = std::transform( stdResult.begin( ), stdResult.end( ), stdInput.begin( ), stdResult.begin( ), 
                                                                                std::minus< int >( ) );
    
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling Bolt exclusive scan
    std::vector< int >::iterator boltEnd = bolt::cl::exclusive_scan(ctl, boltInput.begin( ), boltInput.end( ), 
                                                                                    boltResult.begin( ), init );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdResult, boltResult );
}

TEST_P( ScanIntegerVector, MultiCoreExclusiveOutOfPlace )
{
    //  Declare temporary arrays to store results for out of place computation
    std::vector< int > stdResult( GetParam( ) ), boltResult( GetParam( ) );
    int init = 3;

    //  Emulating a std exclusive scan
    if( stdInput.size( ) )
        stdInput[ 0 ] += init;
    std::vector< int >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ), stdResult.begin( ) );
    if( stdInput.size( ) )
        stdInput[ 0 ] -= init;
    stdEnd = std::transform( stdResult.begin( ), stdResult.end( ), stdInput.begin( ), stdResult.begin( ), 
                                                                                  std::minus< int >( ) );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling Bolt exclusive scan
    std::vector< int >::iterator boltEnd = bolt::cl::exclusive_scan(ctl, boltInput.begin( ), boltInput.end( ),
                                                                                boltResult.begin( ), init );

    //  The returned iterator should be one past the 
    EXPECT_EQ( stdResult.end( ), stdEnd );
    EXPECT_EQ( boltResult.end( ), boltEnd );

    std::vector< int >::iterator::difference_type stdNumElements = std::distance( stdResult.begin( ), stdEnd );
    std::vector< int >::iterator::difference_type boltNumElements = std::distance( boltResult.begin( ), boltEnd );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdResult, boltResult );
}

 
//tring to call out-place scan
TEST_P( ScanFloatVector, intSameValuesSerialOutPlace )
{
    bolt::cl::device_vector< float > boltInput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float > boltOutput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float >::iterator boltEnd = bolt::cl::exclusive_scan(boltInput.begin( ),boltInput.end( ),
                                                                                                boltOutput.begin() );

    std::vector< float > stdOutput( GetParam( ), 1.0f );
    std::vector< float >::iterator stdEnd  = bolt::cl::exclusive_scan( stdInput.begin( ), stdInput.end( ),
                                                                                    stdOutput.begin( ) );
    
    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}
TEST_P( ScanFloatVector, SerialintSameValuesSerialOutPlace )
{
    bolt::cl::device_vector< float > boltInput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float > boltOutput( GetParam( ), 1.0f );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::device_vector< float >::iterator boltEnd = bolt::cl::exclusive_scan(  ctl,
                                                                                    boltInput.begin( ),
                                                                                    boltInput.end( ),
                                                                                    boltOutput.begin() );
    std::vector< float > stdOutput( GetParam( ), 1.0f );
    std::vector< float >::iterator stdEnd  = bolt::cl::exclusive_scan(ctl, stdInput.begin( ), stdInput.end( ),
                                                                                    stdOutput.begin( ) );
    
    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}
TEST_P( ScanFloatVector, MulticoreintSameValuesSerialOutPlace )
{
    bolt::cl::device_vector< float > boltInput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float > boltOutput( GetParam( ), 1.0f );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::device_vector< float >::iterator boltEnd = bolt::cl::exclusive_scan(  ctl,
                                                                                    boltInput.begin( ),
                                                                                    boltInput.end( ),
                                                                                    boltOutput.begin( ) );
    std::vector< float > stdOutput( GetParam( ), 1.0f );
    std::vector< float >::iterator stdEnd  = bolt::cl::exclusive_scan(ctl, stdInput.begin( ), stdInput.end( ),
                                                                                    stdOutput.begin( ) );
    
    //  Loop through the array and compare all the values with each other
    cmpArrays( stdOutput, boltOutput );
}

//tring to call in-place scan
TEST_P( ScanFloatVector, intSameValuesSerialInPlace )
{
    bolt::cl::device_vector< float > dvBoltInput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float >::iterator boltEnd = bolt::cl::exclusive_scan( dvBoltInput.begin( ), 
                                                                dvBoltInput.end( ), dvBoltInput.begin() );
    {
        bolt::cl::device_vector< float >::pointer inPlaceData      = dvBoltInput.data( );
    }

    //std::vector< float > stdInput( 1024, 1 );
    std::vector< float >::iterator stdEnd  = bolt::cl::exclusive_scan( stdInput.begin( ), stdInput.end( ), 
                                                                                    stdInput.begin( ) );
    
    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, dvBoltInput );
}
TEST_P( ScanFloatVector, SerialintSameValuesSerialInPlace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::device_vector< float > dvBoltInput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float >::iterator boltEnd = bolt::cl::exclusive_scan(ctl, dvBoltInput.begin( ), 
                                                                dvBoltInput.end( ), dvBoltInput.begin() );
    {
        bolt::cl::device_vector< float >::pointer inPlaceData      = dvBoltInput.data( );
    }
    
    //std::vector< float > stdInput( 1024, 1 );
    std::vector< float >::iterator stdEnd  = bolt::cl::exclusive_scan(ctl, stdInput.begin( ), stdInput.end( ), 
                                                                                    stdInput.begin( ) );
    
    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, dvBoltInput );
}
TEST_P( ScanFloatVector, MulticoreintSameValuesSerialInPlace )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    bolt::cl::device_vector< float > dvBoltInput( GetParam( ), 1.0f );
    bolt::cl::device_vector< float >::iterator boltEnd = bolt::cl::exclusive_scan(ctl, dvBoltInput.begin( ), 
                                                                dvBoltInput.end( ), dvBoltInput.begin() );
    {
        bolt::cl::device_vector< float >::pointer inPlaceData      = dvBoltInput.data( );
    }
    
    //std::vector< float > stdInput( 1024, 1 );
    std::vector< float >::iterator stdEnd  = bolt::cl::exclusive_scan(ctl, stdInput.begin( ), stdInput.end( ), 
                                                                                    stdInput.begin( ) );
    
    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, dvBoltInput );
}

//  Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( Inclusive, ScanIntegerVector, ::testing::Range( 0, 1024, 23 ) );
INSTANTIATE_TEST_CASE_P( Inclusive, ScanIntegerDeviceVector, ::testing::Range( 0, 1024, 73 ) );
INSTANTIATE_TEST_CASE_P( Inclusive, ScanIntegerNakedPointer, ::testing::Range( 0, 1024, 23) );
INSTANTIATE_TEST_CASE_P( Exclusive, ScanFloatVector, ::testing::Range( 1, 1024, 23 ) );

//  Test a huge range, suitable for floating point as they are less prone to overflow 
// (but floating point loses granularity at large values)
//INSTANTIATE_TEST_CASE_P( Inclusive, ScanFloatVector, ::testing::Range( 4096, 1048576, 4096 ) );
// above test takes a long time; >2hrs - DT


INSTANTIATE_TEST_CASE_P( Inclusive, ScanDoubleVector, ::testing::Range( 4096, 1048576, 4096 ) );


typedef ::testing::Types< 
    std::tuple< int, TypeValue< 1 > >,
    //std::tuple< int, TypeValue< bolt::cl::scanMultiCpuThreshold - 1 > >,
    //std::tuple< int, TypeValue< bolt::cl::scanGpuThreshold - 1 > >,
    std::tuple< int, TypeValue< 31 > >,
    std::tuple< int, TypeValue< 32 > >,
    std::tuple< int, TypeValue< 63 > >,
    std::tuple< int, TypeValue< 64 > >,
    std::tuple< int, TypeValue< 127 > >,
    std::tuple< int, TypeValue< 128 > >,
    std::tuple< int, TypeValue< 129 > >,
    std::tuple< int, TypeValue< 1000 > >,
    std::tuple< int, TypeValue< 1053 > >,
    std::tuple< int, TypeValue< 4096 > >,
    std::tuple< int, TypeValue< 4097 > >,
    std::tuple< int, TypeValue< 65535 > >,
    //std::tuple< int, TypeValue< 131032 > >,       // uncomment these to generate failures; stack overflow
    //std::tuple< int, TypeValue< 262154 > >,
    std::tuple< int, TypeValue< 65536 > >
> IntegerTests;

typedef ::testing::Types< 
    std::tuple< float, TypeValue< 1 > >,
    //std::tuple< float, TypeValue< bolt::cl::scanMultiCpuThreshold - 1 > >,
    //std::tuple< float, TypeValue< bolt::cl::scanGpuThreshold - 1 > >,
    std::tuple< float, TypeValue< 31 > >,
    std::tuple< float, TypeValue< 32 > >,
    std::tuple< float, TypeValue< 63 > >,
    std::tuple< float, TypeValue< 64 > >,
    std::tuple< float, TypeValue< 127 > >,
    std::tuple< float, TypeValue< 128 > >,
    std::tuple< float, TypeValue< 129 > >,
    std::tuple< float, TypeValue< 1000 > >,
    std::tuple< float, TypeValue< 1053 > >,
    std::tuple< float, TypeValue< 4096 > >,
    std::tuple< float, TypeValue< 4097 > >,
    std::tuple< float, TypeValue< 65535 > >,
    std::tuple< float, TypeValue< 65536 > >
> FloatTests;

INSTANTIATE_TYPED_TEST_CASE_P( Integer, ScanArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, ScanArrayTest, FloatTests );
//here


/* TEST(Scan, cpuQueue)
{
    MyOclContext ocl = initOcl(CL_DEVICE_TYPE_CPU, 0);
    bolt::cl::control c(ocl._queue);  // construct control structure from the queue.
    std::vector< float > boltInput( 1024, 1.0f );
    std::vector< float > boltOutput( 1024, 1.0f );
    std::vector< float > stdInput( 1024, 1.0f );
    std::vector< float > stdOutput( 1024, 1.0f );
    std::cout << "Doing BOLT scan\n";
    std::vector< float >::iterator boltEnd  = bolt::cl::inclusive_scan( c, boltInput.begin( ), boltInput.end( ), 
                                                                                        boltOutput.begin( ) );
    std::cout << "Doing STD scan\n";
    std::vector< float >::iterator stdEnd  = std::partial_sum( stdInput.begin( ), stdInput.end( ),stdOutput.begin( ));
    cmpArrays( stdInput, boltInput );
} */


/*
// std::deque's iteartor is not allowed in the bolt'routines because 
// unlike vectors, deques are not guaranteed to store all its elements in contiguous storage locations

TEST (sanity_exclusive_scan__stdDeque, intSameValuesSerialRange_EP377072)
{
int size = 10;
std::deque<int> boltInput( size, 1 );  
std::deque<int> stdInput( size, 1 );

//std::vector<int> boltInput( size, 1 );
//std::vector<int> stdInput( size, 1 );

std::cout<<"before exclusive_scan:\nstd input\tboltinput \n";
for (int i = 0; i < size; i++)
{
std::cout<<"      "<<stdInput[i]<<"       ";
std::cout<<"      "<<boltInput[i]<<"      \n";
}

//TAKE_THIS_CONTROL_PATH
bolt::cl::inclusive_scan( boltInput.begin( ), boltInput.end( ), boltInput.begin( ));
std::partial_sum( stdInput.begin( ), stdInput.end( ), stdInput.begin( ) );

std::cout<<"\n\nafter exclusive_scan:\nstd input\tboltinput \n";
for (int i = 0; i < size; i++)
{
std::cout<<"      "<<stdInput[i]<<"       ";
std::cout<<"      "<<boltInput[i]<<"      \n";
}

for (int i = 0; i < size; i++){
EXPECT_EQ(stdInput[i], boltInput[i]);
}
}
*/

int _tmain(int argc, _TCHAR* argv[])
{
    //  Register our minidump generating logic
   // bolt::miniDumpSingleton::enableMiniDumps( );

    //  Initialize googletest; this removes googletest specific flags from command line
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    bool print_clInfo = false;
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;

    try
    {
        // Declare supported options below, describe what they do
        po::options_description desc( "Scan GoogleTest command line options" );
        desc.add_options()
            ( "help,h",         "produces this help message" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ),
                                                      "Specify the platform under test" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ),
                                                       "Specify the device under test" )
            //( "gpu,g",         "Force instantiation of all OpenCL GPU device" )
            //( "cpu,c",         "Force instantiation of all OpenCL CPU device" )
            //( "all,a",         "Force instantiation of all OpenCL devices" )
            ;

        ////  All positional options (un-named) should be interpreted as kernelFiles
        //po::positional_options_description p;
        //p.add("kernelFiles", -1);

        //po::variables_map vm;
        //po::store( po::command_line_parser( argc, argv ).options( desc ).positional( p ).run( ), vm );
        //po::notify( vm );

        po::variables_map vm;
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );

        if( vm.count( "help" ) )
        {
            // This needs to be 'cout' as program-options does not support wcout yet
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "queryOpenCL" ) )
        {
            print_clInfo = true;
        }

        //  The following 3 options are not implemented yet; they are meant to be used with ::clCreateContextFromType()
        if( vm.count( "gpu" ) )
        {
            deviceType = CL_DEVICE_TYPE_GPU;
        }
        
        if( vm.count( "cpu" ) )
        {
            deviceType = CL_DEVICE_TYPE_CPU;
        }

        if( vm.count( "all" ) )
        {
            deviceType = CL_DEVICE_TYPE_ALL;
        }

    }
    catch( std::exception& e )
    {
        std::cout << _T( "Scan GoogleTest error condition reported:" ) << std::endl << e.what() << std::endl;
        return 1;
    }

    //  Query OpenCL for available platforms
    cl_int err = CL_SUCCESS;

    // Platform vector contains all available platforms on system
    std::vector< cl::Platform > platforms;
    //std::cout << "HelloCL!\nGetting Platform Information\n";
    bolt::cl::V_OPENCL( cl::Platform::get( &platforms ), "Platform::get() failed" );

    if( print_clInfo )
    {
        bolt::cl::control::printPlatforms( );
        return 0;
    }

    //  Do stuff with the platforms
    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        for(i = platforms.begin(); i != platforms.end(); ++i)
        {
            if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
    }
    bolt::cl::V_OPENCL( err, "Platform::getInfo() failed" );

    // Device info
    std::vector< cl::Device > devices;
    bolt::cl::V_OPENCL( platforms.front( ).getDevices( CL_DEVICE_TYPE_ALL, &devices ),"Platform::getDevices() failed");

    cl::Context myContext( devices.at( userDevice ) );
    cl::CommandQueue myQueue( myContext, devices.at( userDevice ) );
    bolt::cl::control::getDefault( ).setCommandQueue( myQueue );

    std::string strDeviceName = bolt::cl::control::getDefault( ).getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    std::cout << "Device under test : " << strDeviceName << std::endl;

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
