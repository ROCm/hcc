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
#define BKND amp
#define STABLE_SORT_FUNC sort_by_key


#if (GOOGLE_TEST == 1)


#include "common/stdafx.h"
#include "bolt/amp/sort_by_key.h"
#include "bolt/unicode.h"
#include "bolt/amp/functional.h"

#include <gtest/gtest.h>
#include <type_traits>

#include "common/test_common.h"
#include "bolt/miniDump.h"

#include <array>
#include <algorithm>


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

    bool operator() (const stdSortData& lhs, const stdSortData& rhs) restrict(amp,cpu){
        return (lhs.key < rhs.key);
    }
    stdSortData& operator = (const stdSortData& other) restrict(amp,cpu){
        key = other.key;
        value = other.value;
        return (*this);
    }
    /*stdSortData& operator() () {
        key = rand();
        value = rand();
        return (*this);
    }*/
    bool operator < (const stdSortData& other) const restrict(amp,cpu){
        return (key < (other.key));
    }
    bool operator > (const stdSortData& other) const restrict(amp,cpu) {
        return (key > other.key);
    }
    bool operator == (const stdSortData& other) const restrict(amp,cpu){
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
    for( int i = 0; (i < size); ++i )
    {
            EXPECT_EQ( ref[ i ].key, key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_EQ( ref[ i ].value, value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename A, typename B, typename C, typename D >
typename std::enable_if< !(std::is_same< typename D::value_type,float  >::value ||
                           std::is_same< typename D::value_type,double >::value),
                           ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref, const B& refkey, const C& key, const D& value, int size)
{
    for( int i = 0; (i < size); ++i )
    {
            EXPECT_EQ( refkey[ i ], key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_EQ( ref[ i ], value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}


//  A very generic template that takes two container, and compares their values assuming a vector interface
template< typename A, typename B, typename C >
typename std::enable_if< std::is_same< typename C::value_type,float  >::value,
                         ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref,const B& key, const C& value, int size)
{
    for( int i = 0; (i < size); ++i )
    {
            EXPECT_FLOAT_EQ( (float) ref[ i ].key, (float) key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_FLOAT_EQ( (float) ref[ i ].value, (float) value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

template< typename A, typename B, typename C >
typename std::enable_if< std::is_same< typename C::value_type,double  >::value,
                         ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref,const B& key, const C& value, int size)
{
    for( int i = 0; (i < size); ++i )
    {
            EXPECT_DOUBLE_EQ((double) ref[ i ].key, (double)key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_DOUBLE_EQ( (double) ref[ i ].value, (double) value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

template< typename A, typename B, typename C, typename D >
typename std::enable_if< std::is_same< typename D::value_type,float  >::value,
                         ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref, const B& refkey, const C& key, const D& value, int size)
{
    for( int i = 0; (i < size); ++i )
    {
            EXPECT_FLOAT_EQ( (float) refkey[ i ], (float) key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_FLOAT_EQ( (float) ref[ i ], (float) value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

template< typename A, typename B, typename C, typename D >
typename std::enable_if< 
                         std::is_same< typename D::value_type,double >::value,
                         ::testing::AssertionResult
                       >::type
cmpArraysSortByKey(const A& ref, const B& refkey, const C& key, const D& value, int size)
{
    for( int i = 0; (i < size); ++i )
    {
            EXPECT_DOUBLE_EQ( (double) refkey[ i ],(double) key[ i ] ) << _T( "Where i = " ) << i;
            EXPECT_DOUBLE_EQ((double) ref[ i ], (double) value[ i ] ) << _T( "Where i = " ) << i;
    }
    return ::testing::AssertionSuccess( );
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class SortbyKeyIntegerVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyKeyIntegerVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = i+3;//stdValues[i].key; 
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = i+3;//stdOffsetValues[i].key; 
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<int> > stdValues, stdOffsetValues;
    std::vector< int > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};


class SortbyKeyUnsignedIntegerVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyKeyUnsignedIntegerVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = rand();//(unsigned int) stdValues[i].key;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = rand(); //(unsigned int) stdOffsetValues[i].key;
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
class SortbyKeyFloatVector: public ::testing::TestWithParam< int >
{
public:
    SortbyKeyFloatVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                   stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = /*(float) stdValues[i].key;*/ (float)(rand());
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = /*(float) stdOffsetValues[i].key;*/ (float)(rand());
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
class SortbyKeyDoubleVector: public ::testing::TestWithParam< int >
{
public:
    SortbyKeyDoubleVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                   stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (double) i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = (double) i+3;
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
class SortbyKeyIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortbyKeyIntegerDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ),
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
    bolt::amp::device_vector< unsigned int > boltValues, boltKeys;
    int VectorSize;
};



//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class SortbyKeyFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortbyKeyFloatDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ),
                                         boltKeys( static_cast<size_t>( GetParam( ) ) ), VectorSize( GetParam( ) )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (float)(rand());
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<float> > stdValues;
    bolt::amp::device_vector< int > boltKeys;
    bolt::amp::device_vector< float > boltValues;
    int VectorSize;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class SortbyKeyDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortbyKeyDoubleDeviceVector( ): stdValues( GetParam( ) ), boltValues( static_cast<size_t>( GetParam( ) ) ),
                                          boltKeys( static_cast<size_t>( GetParam( ) ) ), VectorSize( GetParam( ) )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (double)(rand());
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;
        }
    }

protected:
    std::vector< stdSortData<double> > stdValues;
    bolt::amp::device_vector< int > boltKeys;
    bolt::amp::device_vector< double > boltValues;
    int VectorSize;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class SortbyKeyIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortbyKeyIntegerNakedPointer( ): stdValues( new stdSortData<int>[ GetParam( ) ] ),
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
class SortbyKeyFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortbyKeyFloatNakedPointer( ): stdValues( new stdSortData<float>[ GetParam( ) ] ),
                                                    boltValues(new float[ GetParam( ) ]),
                                                    boltKeys( new int[ GetParam( ) ] ), VectorSize( GetParam( ) )
    {}

    virtual void SetUp( )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (float)(rand());
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

// UDD which contains four doubles
struct uddtD4
{
    double a;
    double b;
    double c;
    double d;

    uddtD4() restrict(cpu, amp) {}
    uddtD4(double x, double y, double z, double w) restrict(cpu, amp)
     : a(x), b(y), c(z), d(w) {}
    bool operator==(const uddtD4& rhs) const restrict(amp,cpu)
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

	/*uddtD4 operator = (const uddtD4 rhs) const restrict(amp,cpu)
    {
		uddtD4 temp;

		temp.a = rhs.a;
		temp.b = rhs.b;
        temp.c = rhs.c;
		temp.d = rhs.d;
        
        return temp;
    }*/

	bool operator < (const uddtD4& rhs) const restrict(amp,cpu)
    {
		bool ls = false;
        if (rhs.a < a && rhs.b < b && rhs.c < c && rhs.d < d)
            ls = true;    
        return ls;
    }

	bool operator > (const uddtD4& rhs) const restrict(amp,cpu)
    {
		bool gtr = false;
        if (rhs.a > a && rhs.b > b && rhs.c > c && rhs.d > d)
            gtr = true;    
        return gtr;
    }

	
};

// Functor for UDD. Adds all four double elements and returns true if lhs_sum > rhs_sum
struct AddD4
{
    bool operator()(const uddtD4 &lhs, const uddtD4 &rhs) const restrict(amp,cpu)
    {

        if( ( lhs.a + lhs.b + lhs.c + lhs.d ) > ( rhs.a + rhs.b + rhs.c + rhs.d) )
            return true;
        return false;
    };
}; 


uddtD4 identityUdd4 = uddtD4( 1.0, 1.0, 1.0, 1.0 );
uddtD4 initialUdd4  = uddtD4( 1.00001, 1.000003, 1.0000005, 1.00000007 );

template <typename T>
struct stdSortDataKey {
    T key;
    T value;

    bool operator() (const stdSortDataKey& lhs, const stdSortDataKey& rhs) restrict(amp,cpu){
        return (lhs.key < rhs.key);
    }
    stdSortDataKey& operator = (const stdSortDataKey& other) restrict(amp,cpu){
        key = other.key;
        value = other.value;
        return (*this);
    }

    bool operator < (const stdSortDataKey& other) const restrict(amp,cpu){
        return (key < (other.key));
    }
    bool operator > (const stdSortDataKey& other) const restrict(amp,cpu){
        return (key > other.key);
    }
    bool operator == (const stdSortDataKey& other) const restrict(amp,cpu){
        return (key == other.key);
    }
    stdSortDataKey()
        /*: key(0),value(0)*/ { }

};

class SortbyUDDKeyVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyUDDKeyVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key.a = 1.00001;
			stdValues[i].key.b = 1.000003;
			stdValues[i].key.c = 1.0000005;
			stdValues[i].key.d = 1.00000007;

			stdValues[i].value.a = 1.00000008;
			stdValues[i].value.b = 1.000004;
			stdValues[i].value.c = 1.0000005;
			stdValues[i].value.d = 1.00008;


            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key.a = 1.00001;
			stdOffsetValues[i].key.b = 1.000003;
			stdOffsetValues[i].key.c = 1.0000005;
			stdOffsetValues[i].key.d = 1.00000007;

			stdOffsetValues[i].value.a = 1.00000008;
			stdOffsetValues[i].value.b = 1.000004;
			stdOffsetValues[i].value.c = 1.0000005;
			stdOffsetValues[i].value.d = 1.00008;


            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortDataKey<uddtD4> > stdValues, stdOffsetValues;
    std::vector< uddtD4 > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class SortbyUDDDeviceKeyVector: public ::testing::TestWithParam< int >
{
    public:

   SortbyUDDDeviceKeyVector( ): stdValues( GetParam( ) ), stdKeys( GetParam( ) ), 
                                    stdOffsetValues( GetParam( ) ),  stdOffsetKeys( GetParam( ) ), VectorSize(GetParam( )) 
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].a = 1.00001;
			stdValues[i].b = 1.000003;
			stdValues[i].c = 1.0000005;
			stdValues[i].d = 1.00000007;

			stdKeys[i].a = 1.00000008;
			stdKeys[i].b = 1.000004;
			stdKeys[i].c = 1.0000005;
			stdKeys[i].d = 1.00008;


            //boltValues[i] = stdValues[i].value;
            //boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].a = 1.00001;
			stdOffsetValues[i].b = 1.000003;
			stdOffsetValues[i].c = 1.0000005;
			stdOffsetValues[i].d = 1.00000007;

			stdOffsetKeys[i].a = 1.00000008;
			stdOffsetKeys[i].b = 1.000004;
			stdOffsetKeys[i].c = 1.0000005;
			stdOffsetKeys[i].d = 1.00008;


            //boltOffsetValues[i] = stdOffsetValues[i].value;
            //boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    //std::vector< stdSortDataKey<uddtD4> > stdValues, stdOffsetValues;
	std::vector<uddtD4> stdValues, stdOffsetValues, stdKeys, stdOffsetKeys;
    //bolt::amp::device_vector< uddtD4 > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class SortbyFloatKeyVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyFloatKeyVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = (float) rand();
            stdValues[i].value = (float)  i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = (float) rand();
            stdOffsetValues[i].value = (float)i+3;
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortDataKey<float> > stdValues, stdOffsetValues;
    std::vector< float > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class SortbyFloatDeviceKeyVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyFloatDeviceKeyVector( ): stdValues( GetParam( ) ),Values( GetParam( ) ), OffsetValues(GetParam( ) ), 
                                    stdOffsetValues( GetParam( ) ),  stdKeys(GetParam()), stdOffsetKeys(GetParam()), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            /*stdKeys[i]= (float) rand();
            stdValues[i] = (float) i+3;
           
            stdOffsetKeys[i] = (float) rand();
            stdOffsetValues[i] = (float) i+3;*/

			Values[i].key = (float) rand();
            Values[i].value = (float)i+3;
            stdKeys[i] = Values[i].key;
            stdValues[i] = Values[i].value;

			OffsetValues[i].key = (float) rand();
            OffsetValues[i].value = (float)rand();
            stdOffsetKeys[i] = OffsetValues[i].key;
            stdOffsetValues[i] = OffsetValues[i].value;
          
        }
    }

protected:
	std::vector< stdSortDataKey<float> > Values, OffsetValues;
	std::vector< float > stdValues, stdOffsetValues, stdKeys, stdOffsetKeys;
    //std::vector< stdSortDataKey<float> > stdValues, stdOffsetValues;
    //bolt::amp::device_vector< float > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class SortbyDoubleKeyVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyDoubleKeyVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = rand();
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = rand();
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortDataKey<double> > stdValues, stdOffsetValues;
    std::vector< double > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class SortbyDoubleDeviceKeyVector: public ::testing::TestWithParam< int >
{
    public:

    SortbyDoubleDeviceKeyVector( ): stdValues( GetParam( ) ),Values( GetParam( ) ), OffsetValues(GetParam( ) ), 
                                    stdOffsetValues( GetParam( ) ),  stdKeys(GetParam()), stdOffsetKeys(GetParam()), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            /*stdKeys[i] = rand();
            stdValues[i] = (double) i+3;

            stdOffsetKeys[i] = rand();
            stdOffsetValues[i] = (double) i+3;*/


			Values[i].key = rand();
            Values[i].value = (double) i+3;
            stdKeys[i] = Values[i].key;
            stdValues[i] = Values[i].value;

			OffsetValues[i].key = rand();
            OffsetValues[i].value = (double)i+3;
            stdOffsetKeys[i] = OffsetValues[i].key;
            stdOffsetValues[i] = OffsetValues[i].value;
        }
    }

protected:
	std::vector< stdSortDataKey<double> > Values, OffsetValues;
	std::vector< double > stdValues, stdOffsetValues, stdKeys, stdOffsetKeys;
    //std::vector< stdSortDataKey<double> > stdValues, stdOffsetValues;
    //bolt::amp::device_vector< double > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};




#if (TEST_DOUBLE ==1 )
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class SortbyKeyDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    SortbyKeyDoubleNakedPointer( ):stdValues(new stdSortData<double>[GetParam( )]),
                                                   boltValues(new double[ GetParam( )]),
                                                   boltKeys( new int[ GetParam( ) ] ), VectorSize( GetParam( ) )
    {}

    virtual void SetUp( )
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = rand();
            stdValues[i].value = (double)rand();
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


class SortByKeyCountingIterator :public ::testing::TestWithParam<int>{
protected:
     int mySize;
public:
    SortByKeyCountingIterator(): mySize(GetParam()){
    }
};


//SortByKey with Fancy Iterator would result in compilation error!

/* TEST_P(SortByKeyCountingIterator, RandomwithCountingIterator)
{

    std::vector< stdSortData<int> > stdValues;

    bolt::amp::counting_iterator<int> key_first(0);
    bolt::amp::counting_iterator<int> key_last = key_first + mySize;

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

TEST_P(SortByKeyCountingIterator, DVwithCountingIterator)
{

    std::vector< stdSortData<int> > stdValues;

    bolt::amp::counting_iterator<int> key_first(0);
    bolt::amp::counting_iterator<int> key_last = key_first + mySize;

    bolt::amp::device_vector<int> value(mySize);

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

//#if 0
#if (TEST_DOUBLE == 1)
TEST( SortbyUDDKeyVectorTest, Normal )
{
	int length = (1<<12);

    std::vector< uddtD4  > stdKeys( length, initialUdd4);
	std::vector< uddtD4  > stdValues( length, identityUdd4);
	bolt::amp::device_vector< uddtD4  > boltKeys( stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< uddtD4  > boltValues( stdValues.begin(), stdValues.end());

	std::vector< uddtD4  > stdOffsetKeys( length, initialUdd4);
	std::vector< uddtD4  > stdOffsetValues( length, identityUdd4);
	bolt::amp::device_vector< uddtD4  > boltOffsetKeys( stdOffsetKeys.begin(), stdOffsetKeys.end());
	bolt::amp::device_vector< uddtD4  > boltOffsetValues( stdOffsetValues.begin(), stdOffsetValues.end());

	AddD4 ad4gt;

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ), ad4gt);
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ), ad4gt );
	
    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  length);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = length -17; //Some aribitrary offset position

    if( (( startIndex > length ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< length << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ), ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  length );
    }
}

TEST( SortbyUDDKeyVectorTest2, Normal )
{
	int length = (1<<12);

    std::vector< uddtD4  > stdKeys( length, initialUdd4);
	std::vector< uddtD4  > stdValues( length, identityUdd4);
	std::vector< uddtD4  > boltKeys( stdKeys.begin(), stdKeys.end());
	std::vector< uddtD4  > boltValues( stdValues.begin(), stdValues.end());

	std::vector< uddtD4  > stdOffsetKeys( length, initialUdd4);
	std::vector< uddtD4  > stdOffsetValues( length, identityUdd4);
	std::vector< uddtD4  > boltOffsetKeys( stdOffsetKeys.begin(), stdOffsetKeys.end());
	std::vector< uddtD4  > boltOffsetValues( stdOffsetValues.begin(), stdOffsetValues.end());

	AddD4 ad4gt;

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ), ad4gt);
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ), ad4gt );
	
    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  length);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = length -17; //Some aribitrary offset position

    if( (( startIndex > length ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< length << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ), ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  length );
    }
}

TEST_P( SortbyUDDDeviceKeyVector, Normal )
{
	AddD4 ad4gt;
	bolt::amp::device_vector< uddtD4 > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< uddtD4 > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< uddtD4 > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< uddtD4 > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( )/*, ad4gt*/);
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
	cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex/*, ad4gt */);
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ), ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( SortbyUDDDeviceKeyVector, Serial )
{
	AddD4 ad4gt;
	bolt::amp::device_vector< uddtD4 > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< uddtD4 > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< uddtD4 > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< uddtD4 > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( )/*, ad4gt*/);
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex/*, ad4gt */);
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ), ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

#if defined( ENABLE_TBB )
TEST_P( SortbyUDDDeviceKeyVector, MultiCore )
{
	AddD4 ad4gt;
	bolt::amp::device_vector< uddtD4 > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< uddtD4 > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< uddtD4 > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< uddtD4 > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( )/*, ad4gt*/);
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex/*, ad4gt */);
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ), ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}
#endif

TEST_P( SortbyDoubleKeyVector, Normal )
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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( SortbyDoubleKeyVector, Serial )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyDoubleKeyVector, MultiCoreCPU )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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
#endif
TEST_P( SortbyDoubleDeviceKeyVector, Normal )
{
	bolt::amp::device_vector< double > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< double > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< double > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< double > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( SortbyDoubleDeviceKeyVector, Serial )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    bolt::amp::device_vector< double > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< double > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< double > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< double > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }

}
#if defined( ENABLE_TBB )
TEST_P( SortbyDoubleDeviceKeyVector, MultiCoreCPU )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    bolt::amp::device_vector< double > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< double > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< double > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< double > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }

}
#endif
#endif

TEST_P( SortbyFloatKeyVector, Normal )
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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( SortbyFloatKeyVector, Serial )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}
#if defined( ENABLE_TBB )
TEST_P( SortbyFloatKeyVector, MultiCoreCPU )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex  );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}
#endif
TEST_P( SortbyFloatDeviceKeyVector, Normal )
{
	bolt::amp::device_vector< float > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< float > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< float > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< float > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values,  boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues,  boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( SortbyFloatDeviceKeyVector, Serial )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    bolt::amp::device_vector< float > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< float > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< float > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< float > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values,  boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues,  boltOffsetKeys, boltOffsetValues,  VectorSize );
    }

}
#if defined( ENABLE_TBB )
TEST_P( SortbyFloatDeviceKeyVector, MultiCoreCPU )
{
	
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    bolt::amp::device_vector< float > boltValues(stdValues.begin(), stdValues.end());
	bolt::amp::device_vector< float > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::amp::device_vector< float > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::amp::device_vector< float > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) );

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values,  boltKeys, boltValues,  VectorSize);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues,  boltOffsetKeys, boltOffsetValues,  VectorSize );
    }

}
#endif

TEST_P( SortbyKeyUnsignedIntegerVector, Normal )
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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( SortbyKeyIntegerVector, Normal )
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

TEST_P( SortbyKeyIntegerVector, Serial )
{
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyIntegerVector, MultiCoreCPU )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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
#endif
// Come Back here
TEST_P( SortbyKeyFloatVector, Normal )
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

TEST_P( SortbyKeyFloatVector, Serial )
{
 
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyFloatVector, MultiCoreCPU )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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
#endif

#if (TEST_DOUBLE == 1)
TEST_P( SortbyKeyDoubleVector, Normal )
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

TEST_P( SortbyKeyDoubleVector, Serial)
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyDoubleVector, MultiCoreCPU)
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::sort( stdValues.begin( ), stdValues.end( ));
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
        std::sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}
#endif
#endif
#if (TEST_DEVICE_VECTOR == 1)
TEST_P( SortbyKeyIntegerDeviceVector, Inplace )
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

TEST_P( SortbyKeyIntegerDeviceVector, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyIntegerDeviceVector, MultiCoreInplace )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
#endif
TEST_P( SortbyKeyFloatDeviceVector, Inplace )
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

TEST_P( SortbyKeyFloatDeviceVector, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P(SortbyKeyFloatDeviceVector, MultiCoreInplace )
{

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
#endif

#if (TEST_DOUBLE == 1)
TEST_P( SortbyKeyDoubleDeviceVector, Inplace )
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

TEST_P( SortbyKeyDoubleDeviceVector, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyDoubleDeviceVector, MultiCoreInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
//#endif
#endif
#if defined(_WIN32)
TEST_P( SortbyKeyIntegerNakedPointer, Inplace )
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

TEST_P( SortbyKeyIntegerNakedPointer, SerialInplace )
{
    size_t endIndex = GetParam( );

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyIntegerNakedPointer, MultiCoreInplace )
{
    size_t endIndex = GetParam( );
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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
#endif
TEST_P( SortbyKeyFloatNakedPointer, Inplace )
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

TEST_P( SortbyKeyFloatNakedPointer, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyFloatNakedPointer, MultiCoreInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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
#endif

#if (TEST_DOUBLE == 1)
TEST_P( SortbyKeyDoubleNakedPointer, Inplace )
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

TEST_P( SortbyKeyDoubleNakedPointer, SerialInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

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
#if defined( ENABLE_TBB )
TEST_P( SortbyKeyDoubleNakedPointer, MultiCoreInplace )
{
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

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

#endif
#endif
std::array<int, 9> TestValues = {2,4,8,16,32,64,128,256,512};
std::array<int, 6> TestValues2 = {1024, 2048,4096,8192,16384,32768};

//INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortByKeyCountingIterator,
//                        ::testing::ValuesIn( TestValues.begin(), TestValues.end() ) );

//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( SortByKeyRange, SortbyKeyIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																					  
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyIntegerVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
//#endif																					  

INSTANTIATE_TEST_CASE_P( SortByKeyRange, SortbyFloatKeyVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( SortByKeyValues,SortbyFloatKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyFloatVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyFloatDeviceKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyFloatDeviceKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																					  
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyFloatVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyFloatKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyFloatDeviceKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
//#endif																					  
                                                                                      
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyDoubleKeyVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyDoubleKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyDoubleDeviceKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyDoubleDeviceKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyDoubleVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                     TestValues.end() ) );


INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyUDDDeviceKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyUDDDeviceKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                     TestValues.end() ) );

INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyUDDKeyVector, ::testing::Range( 11, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyUDDKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                     TestValues.end() ) );

//#if(TEST_LARGE_BUFFERS == 1)																					   
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyDoubleVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyValues2,  SortbyDoubleKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyValues2,  SortbyUDDKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyUDDDeviceKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                     TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyDoubleDeviceKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
//#endif
                                                                                       
#endif

INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyUnsignedIntegerVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyUnsignedIntegerVector,::testing::ValuesIn(TestValues.begin(),
                                                                                            TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																							
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyUnsignedIntegerVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                            TestValues2.end()));


//INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyIntegerDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                            TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																							
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyIntegerDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                            TestValues2.end()));
//#endif																							
                                                                                            
INSTANTIATE_TEST_CASE_P(  SortByKeyRange,   SortbyKeyFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P(  SortByKeyValues,  SortbyKeyFloatDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                          TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																						  
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyFloatDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                          TestValues2.end()));
//#endif																						  
                                                                                          
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyDoubleDeviceVector, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyDoubleDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                           TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																						   
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyDoubleDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                           TestValues2.end()));	
//#endif
#endif
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyIntegerNakedPointer, ::testing::Range(1, 332768, 33276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyIntegerNakedPointer,::testing::ValuesIn(TestValues.begin(),
                                                                                            TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																							
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyIntegerNakedPointer,::testing::ValuesIn(TestValues2.begin(),
                                                                                            TestValues2.end()));
//#endif																							
                                                                                            
INSTANTIATE_TEST_CASE_P( SortByKeyRange, SortbyKeyFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyFloatNakedPointer, ::testing::ValuesIn(TestValues.begin(),
                                                                                           TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																						   
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyFloatNakedPointer, ::testing::ValuesIn(TestValues2.begin(),
                                                                                           TestValues2.end()));
//#endif																						   
                                                                                           
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( SortByKeyRange,  SortbyKeyDoubleNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( SortByKeyValues, SortbyKeyDoubleNakedPointer,::testing::ValuesIn(TestValues.begin(),
                                                                                     TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																					 
INSTANTIATE_TEST_CASE_P( SortByKeyValues2, SortbyKeyDoubleNakedPointer,::testing::ValuesIn(TestValues2.begin(),
                                                                                     TestValues2.end() ) );		
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
//Add your main if you want to
#endif
