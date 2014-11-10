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
#if !defined( TEST_COMMON_H )
#define TEST_COMMON_H
#pragma once





#include "bolt/cl/device_vector.h"
#include "bolt/unicode.h"
#include <array>
#include <gtest/gtest.h>

#if !defined( BOLT_TEST_MAX_FAILURES )
    #define BOLT_TEST_MAX_FAILURES 8
#endif

#define BOLT_TEST_RESET_FAILURES \
    size_t numFailures = 0;

#define BOLT_TEST_INCREMENT_FAILURES(ref,calc) \
    if ( !(ref[ i ] == calc[ i ]) ) { \
        numFailures++; \
        /* std::cout << "i=" << i << ": " << ref[i] << " != " << calc[i] << std::endl;*/ \
        printf("i=%i: %i != %i\n", i, ref[i], calc[i]); \
    } \
    if ( numFailures > BOLT_TEST_MAX_FAILURES ) { \
        break; \
    }

size_t numFailures;
bool resetNumFailures = true;

template< typename T, size_t N >
::testing::AssertionResult cmpArrays( const T (&ref)[N], const T (&calc)[N] )
{
    BOLT_TEST_RESET_FAILURES
    for( size_t i = 0; i < N; ++i )
    {
        BOLT_TEST_INCREMENT_FAILURES(ref,calc)
        EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
    }

    return ::testing::AssertionSuccess( );
}

template< typename T1,typename T2>
::testing::AssertionResult cmpArrays( const T1 &ref, typename bolt::cl::device_vector<T2> &calc)
{

        typename bolt::cl::device_vector<T2>::pointer copySrc =  calc.data( );

        BOLT_TEST_RESET_FAILURES
        for( typename T1::size_type i = 0; i < ref.size(); ++i )
        {
            BOLT_TEST_INCREMENT_FAILURES(ref,copySrc)
            EXPECT_EQ( ref[ i ], copySrc[ i ] ) << _T( "Where i = " ) << i;
        }
      return ::testing::AssertionSuccess( );
}


template< typename T1,typename T2 >
::testing::AssertionResult cmpArrays( const T1 &ref,  T2 &calc )
{

        BOLT_TEST_RESET_FAILURES
        for( int i = 0; i < static_cast<int>(ref.size() ); ++i )
        {
            BOLT_TEST_INCREMENT_FAILURES(ref,calc)
            EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );

}

template< typename T1,typename T2>
::testing::AssertionResult 
cmpArrays( const T1 &ref, typename bolt::cl::device_vector<T2> &calc, size_t N )
{

        typename bolt::cl::device_vector<T2>::pointer copySrc =  calc.data( );

        BOLT_TEST_RESET_FAILURES
        for( int i = 0; i < static_cast<int>( N ); ++i )
        {
            BOLT_TEST_INCREMENT_FAILURES(ref,copySrc)
            EXPECT_EQ( ref[ i ], copySrc[ i ] ) << _T( "Where i = " ) << i;
        }
      return ::testing::AssertionSuccess( );
}


template< typename T1,typename T2 >
::testing::AssertionResult   cmpArrays( const T1 &ref,  T2 &calc, size_t N )
{
        BOLT_TEST_RESET_FAILURES
        for( int i = 0; i < static_cast<int>( N ); ++i )
        {
            BOLT_TEST_INCREMENT_FAILURES(ref,calc)
            EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }
      return ::testing::AssertionSuccess( );
}

template< typename T, size_t N >
struct cmpStdArray
{
    static ::testing::AssertionResult cmpArrays( const std::array< T, N >& ref, const std::array< T, N >& calc )
    {
        BOLT_TEST_RESET_FAILURES
        for( size_t i = 0; i < N; ++i )
        {
            BOLT_TEST_INCREMENT_FAILURES(ref,calc)
            EXPECT_EQ( ref[ i ], calc[ i ] ) << _T( "Where i = " ) << i;
        }

        return ::testing::AssertionSuccess( );
    }
};




#endif
