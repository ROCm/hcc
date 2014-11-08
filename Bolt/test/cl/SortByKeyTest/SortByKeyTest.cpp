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

#define TEST_DOUBLE 1
#define TEST_LARGE_BUFFERS 1
#define TEST_DEVICE_VECTOR 1
#define TEST_CPU_DEVICE 0
#define GOOGLE_TEST 1
#define BKND cl
#define STABLE_SORT_FUNC sort_by_key


#if (GOOGLE_TEST == 1)

#include "common/stdafx.h"
#include "common/myocl.h"
#include "bolt/cl/iterator/counting_iterator.h"

#include <bolt/cl/sort_by_key.h>
#include <bolt/miniDump.h>
#include <bolt/unicode.h>

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include <algorithm>
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//This is a compare routine for naked pointers.
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

template <typename T>
struct stdSortData {
    int key;
    T value;

    bool operator() (const stdSortData& lhs, const stdSortData& rhs) {
        return (lhs.key < rhs.key);
    }
    stdSortData& operator = (const stdSortData& other) {
        key = other.key;
        value = other.value;
        return (*this);
    }
    /*stdSortData& operator() () {
        key = rand();
        value = rand();
        return (*this);
    }*/
    bool operator < (const stdSortData& other) const {
        return (key < (other.key));
    }
    bool operator > (const stdSortData& other) const {
        return (key > other.key);
    }
    bool operator == (const stdSortData& other) const {
        return (key == other.key);
    }
    stdSortData()
        : key(0),value(0) { }

};
//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename A, typename B, typename C >
typename std::enable_if< !(std::is_same< typename C::value_type,float  >::value ||
                           std::is_same< typename C::value_type,double >::value),
                           ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref,const B& key, const C& value, int size)
{
    for( int i = 0; (i < size) ; ++i )
    {
            EXPECT_EQ( ref[ i ].key, key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_EQ( ref[ i ].value, value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename A, typename B, typename C >
typename std::enable_if< std::is_same< typename C::value_type,float  >::value ||
                         std::is_same< typename C::value_type,double >::value,
                         ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref,const B& key, const C& value, int size)
{
    for( int i = 0; (i < size) ; ++i )
    {
            EXPECT_EQ( ref[ i ].key, key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_EQ( ref[ i ].value, value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class StableSortbyKeyIntegerVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyKeyIntegerVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = i+3;
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<int> > stdValues, stdOffsetValues;
    std::vector< int > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class StableSortbyKeyUnsignedIntegerVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyKeyUnsignedIntegerVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = i+3;
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<unsigned int> > stdValues, stdOffsetValues;
    std::vector< unsigned int > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyFloatVector: public ::testing::TestWithParam< int >
{
public:
    StableSortbyKeyFloatVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                   stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (float)(i+3);
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = (float)(i+3);
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<float> > stdValues, stdOffsetValues;
    std::vector< float > boltValues, boltOffsetValues;
    std::vector< int > boltKeys, boltOffsetKeys;
    int VectorSize;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyDoubleVector: public ::testing::TestWithParam< int >
{
public:
    StableSortbyKeyDoubleVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                   stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (double)(i+3);
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = (double)(i+3);
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<double> > stdValues, stdOffsetValues;
    std::vector< double > boltValues, boltOffsetValues;
    std::vector< int > boltKeys, boltOffsetKeys;
    int VectorSize;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortbyKeyIntegerDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ),
                                           boltKeys( static_cast<size_t>( GetParam( ) ) ), VectorSize( GetParam( ) )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<unsigned int> > stdValues;
    bolt::cl::device_vector< unsigned int > boltValues, boltKeys;
    int VectorSize;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
/*class SortUDDDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortUDDDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ), VectorSize( GetParam( ) )
    {
        std::generate(stdValues.begin(), stdValues.end(), rand);
        //boltValues = stdValues;
        //FIXME - The above should work but the below loop is used.
        for (int i=0; i< GetParam( ); i++)
        {
            boltValues[i] = stdValues[i];
        }
    }

protected:
    std::vector< UDD > stdValues;
    bolt::cl::device_vector< UDD > boltValues;
    int VectorSize;
};*/

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortbyKeyFloatDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ),
                                         boltKeys( static_cast<size_t>( GetParam( ) ) ), VectorSize( GetParam( ) )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (float)(i+3);
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<float> > stdValues;
    bolt::cl::device_vector< int > boltKeys;
    bolt::cl::device_vector< float > boltValues;
    int VectorSize;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortbyKeyDoubleDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ),
                                          boltKeys( static_cast<size_t>( GetParam( ) ) ), VectorSize( GetParam( ) )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (double)(i+3);
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<double> > stdValues;
    bolt::cl::device_vector< int > boltKeys;
    bolt::cl::device_vector< double > boltValues;
    int VectorSize;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortbyKeyIntegerNakedPointer( ): stdValues( new stdSortData<int>[ GetParam( ) ] ),
                                                      boltValues( new int[ GetParam( ) ]),
                                                      boltKeys( new int[ GetParam( ) ] ), VectorSize( GetParam( ) )
    {}

    virtual void SetUp( )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    };

    virtual void TearDown( )
    {
        delete [] stdValues;
        delete [] boltValues;
        delete [] boltKeys;
    };

protected:
     stdSortData<int> *stdValues;
     int *boltValues, *boltKeys;
    int VectorSize;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortbyKeyFloatNakedPointer( ): stdValues( new stdSortData<float>[ GetParam( ) ] ),
                                                    boltValues(new float[ GetParam( ) ]),
                                                    boltKeys( new int[ GetParam( ) ] ), VectorSize( GetParam( ) )
    {}

    virtual void SetUp( )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (float)(i+3);
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }

    };

    virtual void TearDown( )
    {
        delete [] stdValues;
        delete [] boltValues;
        delete [] boltKeys;
    };

protected:
     stdSortData<float>* stdValues;
     float* boltValues;
     int   *boltKeys;
    int VectorSize;
};

#if (TEST_DOUBLE ==1 )
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class StableSortbyKeyDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    StableSortbyKeyDoubleNakedPointer( ):stdValues(new stdSortData<double>[GetParam( )]),
                                                   boltValues(new double[ GetParam( )]),
                                                   boltKeys( new int[ GetParam( ) ] ), VectorSize( GetParam( ) )
    {}

    virtual void SetUp( )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (double)(i+3);
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    };

    virtual void TearDown( )
    {
        delete [] stdValues;
        delete [] boltValues;
        delete [] boltKeys;
    };

protected:
     stdSortData<double>* stdValues;
     double* boltValues;
     int   *boltKeys;
    int VectorSize;
};
#endif


class StableSortByKeyCountingIterator :public ::testing::TestWithParam<int>{
protected:
     int mySize;
public:
    StableSortByKeyCountingIterator(): mySize(GetParam()){
    }
};


//StableSortByKey with Fancy Iterator would result in compilation error!

/* TEST_P(StableSortByKeyCountingIterator, RandomwithCountingIterator)
{

    std::vector< stdSortData<int> > stdValues;

    bolt::cl::counting_iterator<int> key_first(0);
    bolt::cl::counting_iterator<int> key_last = key_first + mySize;

    std::vector<int> value(mySize);

    for(int i=0; i<mySize; i++)
    {
        stdValues[i].key = i;
        stdValues[i].value = i;
        value[i] = i;
    }

    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( key_first, key_last, value.begin()); // This is logically Wrong!

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin(),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( value.begin(), value.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, key_first, value, mySize);

}

TEST_P(StableSortByKeyCountingIterator, DVwithCountingIterator)
{

    std::vector< stdSortData<int> > stdValues;

    bolt::cl::counting_iterator<int> key_first(0);
    bolt::cl::counting_iterator<int> key_last = key_first + mySize;

    bolt::cl::device_vector<int> value(mySize);

    for(int i=0; i<mySize; i++)
    {
        stdValues[i].key = i;
        stdValues[i].value = i;
        value[i] = i;
    }

    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( key_first, key_last, value.begin());

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin(),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( value.begin(), value.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, key_first, value, mySize);

} */

TEST_P( StableSortbyKeyUnsignedIntegerVector, Normal )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData< unsigned int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< unsigned int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( StableSortbyKeyIntegerVector, Normal )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( StableSortbyKeyIntegerVector, Serial )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyKeyIntegerVector, MultiCoreCPU )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

// Come Back here
TEST_P( StableSortbyKeyFloatVector, Normal )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<float> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                   stdValues.end() );
    std::vector< float >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                       boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC(boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyFloatVector, Serial )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<float> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                   stdValues.end() );
    std::vector< float >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                       boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyFloatVector, MultiCoreCPU )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<float> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                   stdValues.end() );
    std::vector< float >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                       boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

#if (TEST_DOUBLE == 1)
TEST_P( StableSortbyKeyDoubleVector, Normal )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<double> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                    stdValues.end() );
    std::vector< double >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                        boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyDoubleVector, Serial)
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<double> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                    stdValues.end() );
    std::vector< double >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                        boltValues.end());

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyDoubleVector, MultiCoreCPU)
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<double> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                    stdValues.end() );
    std::vector< double >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                        boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

#endif
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( StableSortbyKeyIntegerDeviceVector, Inplace )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< int >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                    stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyIntegerDeviceVector, SerialInplace )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< int >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                    stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyIntegerDeviceVector, MultiCoreInplace )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< int >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                    stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyFloatDeviceVector, Inplace )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< float >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                      stdValues.end() );
    std::vector< float >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                       boltValues.end() );
    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyFloatDeviceVector, SerialInplace )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< float >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                      stdValues.end() );
    std::vector< float >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                       boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyFloatDeviceVector, MultiCoreInplace )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< float >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                      stdValues.end() );
    std::vector< float >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                       boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

#if (TEST_DOUBLE == 1)
TEST_P( StableSortbyKeyDoubleDeviceVector, Inplace )
{
    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< double >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                       stdValues.end() );
    std::vector< double >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                        boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyDoubleDeviceVector, SerialInplace )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC(ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< double >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                       stdValues.end() );
    std::vector< double >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                        boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

TEST_P( StableSortbyKeyDoubleDeviceVector, MultiCoreInplace )
{
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC(ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< double >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                       stdValues.end() );
    std::vector< double >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                        boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, boltKeys, boltValues, VectorSize );
}

#endif
#endif
#if defined(_WIN32)
TEST_P( StableSortbyKeyIntegerNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<int>* > wrapstdValues( stdValues, endIndex );
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );
    stdext::checked_array_iterator< int* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //TODO - fix this testcase
    //Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyIntegerNakedPointer, SerialInplace )
{
    size_t endIndex = GetParam( );

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<int>* > wrapstdValues( stdValues, endIndex );
    std::sort( wrapstdValues, wrapstdValues + endIndex );
    stdext::checked_array_iterator< int* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //TODO - fix this testcase
    //Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyIntegerNakedPointer, MultiCoreInplace )
{
    size_t endIndex = GetParam( );
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<int>* > wrapstdValues( stdValues, endIndex );
    std::sort( wrapstdValues, wrapstdValues + endIndex );
    stdext::checked_array_iterator< int* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //TODO - fix this testcase
    //Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyFloatNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<float>* > wrapstdValues( stdValues, endIndex );
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );
    stdext::checked_array_iterator< float* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //TODO - fix this testcase
    //Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyFloatNakedPointer, SerialInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<float>* > wrapstdValues( stdValues, endIndex );
    std::sort( wrapstdValues, wrapstdValues + endIndex );
    stdext::checked_array_iterator< float* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //TODO - fix this testcase
    //Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyFloatNakedPointer, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t endIndex = GetParam( );

    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<float>* > wrapstdValues( stdValues, endIndex );
    std::sort( wrapstdValues, wrapstdValues + endIndex );
    stdext::checked_array_iterator< float* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //TODO - fix this testcase
    //Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}


#if (TEST_DOUBLE == 1)
TEST_P( StableSortbyKeyDoubleNakedPointer, Inplace )
{
    size_t endIndex = GetParam( );
    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<double>* > wrapstdValues( stdValues, endIndex );
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );


    stdext::checked_array_iterator< double* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyDoubleNakedPointer, SerialInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t endIndex = GetParam( );
    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<double>* > wrapstdValues( stdValues, endIndex );
    std::sort( wrapstdValues, wrapstdValues + endIndex );

    stdext::checked_array_iterator< double* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

TEST_P( StableSortbyKeyDoubleNakedPointer, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t endIndex = GetParam( );
    //  Calling the actual functions under test
    stdext::checked_array_iterator< stdSortData<double>* > wrapstdValues( stdValues, endIndex );
    std::sort( wrapstdValues, wrapstdValues + endIndex );

    stdext::checked_array_iterator< double* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

#endif
#endif
std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int, 5> TestValues2 = {2048,4096,8192,16384,32768};

//INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortByKeyCountingIterator,
//                        ::testing::ValuesIn( TestValues.begin(), TestValues.end() ) );

//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyUnsignedIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyIntegerVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
//#endif																					  
                                                                                      
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16	
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyFloatVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyFloatVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
//#endif
                                                                                      
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyDoubleVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyDoubleVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
//#endif
                                                                                       
#endif
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyIntegerDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                            TestValues.end()));
#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyIntegerDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                            TestValues2.end()));
#endif
                                                                                            
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyFloatDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                          TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyFloatDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                          TestValues2.end()));
//#endif	
                                                                                      
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyDoubleDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                           TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyDoubleDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                           TestValues2.end()));
//#endif
#endif
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyIntegerNakedPointer,::testing::ValuesIn(TestValues.begin(),
                                                                                            TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyIntegerNakedPointer,::testing::ValuesIn(TestValues2.begin(),
                                                                                            TestValues2.end()));
//#endif																							
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyFloatNakedPointer, ::testing::ValuesIn(TestValues.begin(),
                                                                                           TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyFloatNakedPointer, ::testing::ValuesIn(TestValues2.begin(),
                                                                                           TestValues2.end()));
//#endif
                                                                                           
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyDoubleNakedPointer,::testing::ValuesIn(TestValues.begin(),
                                                                                     TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyDoubleNakedPointer,::testing::ValuesIn(TestValues2.begin(),
                                                                                     TestValues2.end() ) );
//#endif																					 
#endif


int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
   // bolt::miniDumpSingleton::enableMiniDumps( );

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
    std::cout << "Test Completed. Press Enter to exit.\n .... ";
    //getchar();
    return retVal;
}

#else
Add your main if you want to
#endif
