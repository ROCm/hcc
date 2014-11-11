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
#define TEST_DEVICE_VECTOR 1
#define TEST_CPU_DEVICE 0
#define TEST_LARGE_BUFFERS 0
#define GOOGLE_TEST 1
#define BKND cl
#define STABLE_SORT_FUNC stable_sort_by_key


#if (GOOGLE_TEST == 1)

#include "common/stdafx.h"
#include "common/myocl.h"
#include "bolt/cl/iterator/counting_iterator.h"

#include <bolt/cl/stablesort_by_key.h>
#include <bolt/miniDump.h>
#include <bolt/unicode.h>
#include "common/test_common.h"

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include <algorithm>
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track
//This is a compare routine for naked pointers.


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
    std::vector< stdSortData<int> > stdValues;
    bolt::cl::device_vector< int > boltValues, boltKeys;
    int VectorSize;
};

// UDD which contains four doubles
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

	uddtD4 operator = (const uddtD4 rhs) const
    {
		uddtD4 temp;

		temp.a = rhs.a;
		temp.b = rhs.b;
        temp.c = rhs.c;
		temp.d = rhs.d;
        
        return temp;
    }


	bool operator < (const uddtD4& rhs) const
    {
		bool ls = false;
        if (rhs.a < a && rhs.b < b && rhs.c < c && rhs.d < d)
            ls = true;    
        return ls;
    }

	bool operator > (const uddtD4& rhs) const
    {
		bool gtr = false;
        if (rhs.a > a && rhs.b > b && rhs.c > c && rhs.d > d)
            gtr = true;    
        return gtr;
    }

	
};
);

// Functor for UDD. Adds all four double elements and returns true if lhs_sum > rhs_sum
BOLT_FUNCTOR(AddD4,
struct AddD4
{
    bool operator()(const uddtD4 &lhs, const uddtD4 &rhs) const
    {

        if( ( lhs.a + lhs.b + lhs.c + lhs.d ) > ( rhs.a + rhs.b + rhs.c + rhs.d) )
            return true;
        return false;
    };
}; 
);
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< AddD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< AddD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

uddtD4 identityUdd4 = { 1.0, 1.0, 1.0, 1.0 };
uddtD4 initialUdd4  = { 1.00001, 1.000003, 1.0000005, 1.00000007 };

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );

template <typename T>
struct stdSortDataKey {
    T key;
    T value;

    bool operator() (const stdSortDataKey& lhs, const stdSortDataKey& rhs) {
        return (lhs.key < rhs.key);
    }
    stdSortDataKey& operator = (const stdSortDataKey& other) {
        key = other.key;
        value = other.value;
        return (*this);
    }

    bool operator < (const stdSortDataKey& other) const {
        return (key < (other.key));
    }
    bool operator > (const stdSortDataKey& other) const {
        return (key > other.key);
    }
    bool operator == (const stdSortDataKey& other) const {
        return (key == other.key);
    }
    stdSortDataKey()
        /*: key(0),value(0)*/ { }

};

class StableSortbyUDDKeyVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyUDDKeyVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key.a =  rand()%10;
			stdValues[i].key.b =  rand()%15;
			stdValues[i].key.c =  rand()%15;
			stdValues[i].key.d =  rand()%20;

			stdValues[i].value.a = rand()%125;
			stdValues[i].value.b = rand()%17;
			stdValues[i].value.c = rand()%55;
			stdValues[i].value.d = rand()%30;

            boltValues[i].a = stdValues[i].value.a;
			boltValues[i].b = stdValues[i].value.b;
			boltValues[i].c = stdValues[i].value.c;
			boltValues[i].d = stdValues[i].value.d;

			boltKeys[i].a = stdValues[i].key.a;
			boltKeys[i].b = stdValues[i].key.b;
			boltKeys[i].c = stdValues[i].key.c;
			boltKeys[i].d = stdValues[i].key.d;

            stdOffsetValues[i].key.a = rand()%10; 
			stdOffsetValues[i].key.b = rand()%15; 
			stdOffsetValues[i].key.c = rand()%15; 
			stdOffsetValues[i].key.d = rand()%20; 

			stdOffsetValues[i].value.a = rand()%125;
			stdOffsetValues[i].value.b = rand()%17;
			stdOffsetValues[i].value.c = rand()%55;
			stdOffsetValues[i].value.d = rand()%30;


			boltOffsetValues[i].a = stdOffsetValues[i].value.a;
			boltOffsetValues[i].b = stdOffsetValues[i].value.b;
			boltOffsetValues[i].c = stdOffsetValues[i].value.c;
			boltOffsetValues[i].d = stdOffsetValues[i].value.d;

			boltOffsetKeys[i].a = stdOffsetValues[i].key.a;
			boltOffsetKeys[i].b = stdOffsetValues[i].key.b;
			boltOffsetKeys[i].c = stdOffsetValues[i].key.c;
			boltOffsetKeys[i].d = stdOffsetValues[i].key.d;

        }
    }

protected:
    std::vector< stdSortDataKey<uddtD4> > stdValues, stdOffsetValues;
    std::vector< uddtD4 > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class StableSortbyUDDDeviceKeyVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyUDDDeviceKeyVector( ): stdValues( GetParam( ) ), stdKeys( GetParam( ) ), 
                                    stdOffsetValues( GetParam( ) ),  stdOffsetKeys( GetParam( ) ), VectorSize(GetParam( )) 
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].a = rand()%10; //1.00001;
		    stdValues[i].b = rand()%15; //1.0003;
		    stdValues[i].c = rand()%15; //1.5;
		    stdValues[i].d = rand()%20; //1.7;
		    
		    stdKeys[i].a = rand()%16; //1.00001;
		    stdKeys[i].b = rand()%20; //1.003;
		    stdKeys[i].c = rand()%27; //1.01;
		    stdKeys[i].d = rand()%50; //1.0007;
		    
            stdOffsetValues[i].a = rand()%10; //1.00001;
		    stdOffsetValues[i].b = rand()%15; //1.03;
		    stdOffsetValues[i].c = rand()%15; //1.00005;
		    stdOffsetValues[i].d = rand()%20; //1.00007;
		    
		    stdOffsetKeys[i].a =  rand()%10; //1.00001;
		    stdOffsetKeys[i].b =  rand()%15; //1.003;
		    stdOffsetKeys[i].c =  rand()%15; //1.0005;
		    stdOffsetKeys[i].d =  rand()%20; //1.7;

        }
    }

protected:
	std::vector<uddtD4> stdValues, stdOffsetValues, stdKeys, stdOffsetKeys;
    int VectorSize;
};

class StableSortbyFloatKeyVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyFloatKeyVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ), boltOffsetValues( GetParam( ) ), boltOffsetKeys( GetParam( ) ), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
            stdValues[i].key = (float) rand();
            stdValues[i].value = (float) i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = (float) rand();
            stdOffsetValues[i].value = (float) i+3;
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;
        }
    }

protected:
    std::vector< stdSortDataKey<float> > stdValues, stdOffsetValues;
    std::vector< float > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class StableSortbyFloatDeviceKeyVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyFloatDeviceKeyVector( ): stdValues( GetParam( ) ),Values( GetParam( ) ), OffsetValues(GetParam( ) ), 
		                            Keys( GetParam( ) ), OffsetKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ),  stdKeys(GetParam()), stdOffsetKeys(GetParam()), VectorSize(GetParam( ))
    {
		std::generate(stdValues.begin(), stdValues.end(), rand);
		std::generate(stdKeys.begin(), stdKeys.end(), rand);
		std::generate(stdOffsetValues.begin(), stdOffsetValues.end(), rand);
		std::generate(stdOffsetKeys.begin(), stdOffsetKeys.end(), rand);

        for (int i=0;i<GetParam( );i++)
        {
            /*stdValues[i].key = (float) rand();
            stdValues[i].value = (float) i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;

            stdOffsetValues[i].key = (float) rand();
            stdOffsetValues[i].value = (float) i+3;
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;*/


			/*Values[i].key = (float)rand();
            Values[i].value = (float) i+3;
            stdKeys[i] = Values[i].key;
            stdValues[i] = Values[i].value;

            OffsetValues[i].key = (float)rand();
            OffsetValues[i].value = (float) i+3;
            stdOffsetKeys[i] = OffsetValues[i].key;
            stdOffsetValues[i] = OffsetValues[i].value;*/

			Values[i].key = stdKeys[i];
			Values[i].value = stdValues[i];
			OffsetValues[i].key = stdOffsetKeys[i];
			OffsetValues[i].value = stdOffsetValues[i];
        }
    }

protected:
	std::vector< stdSortDataKey<float> > Values, OffsetValues, Keys, OffsetKeys;
    std::vector< float > stdValues, stdOffsetValues, stdKeys, stdOffsetKeys;
    //bolt::cl::device_vector< float > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class StableSortbyDoubleKeyVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyDoubleKeyVector( ): stdValues( GetParam( ) ), boltValues( GetParam( ) ), boltKeys( GetParam( ) ),
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
    std::vector< stdSortDataKey<double> > stdValues, stdOffsetValues;
    std::vector< double > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};

class StableSortbyDoubleDeviceKeyVector: public ::testing::TestWithParam< int >
{
    public:

    StableSortbyDoubleDeviceKeyVector( ): stdValues( GetParam( ) ),Values( GetParam( ) ), OffsetValues(GetParam( ) ), 
		                            Keys( GetParam( ) ), OffsetKeys( GetParam( ) ),
                                    stdOffsetValues( GetParam( ) ),  stdKeys(GetParam()), stdOffsetKeys(GetParam()), VectorSize(GetParam( ))
    {
        for (int i=0;i<GetParam( );i++)
        {
           /* stdValues[i].key = rand();
            stdValues[i].value = (double) i+3;
            boltValues[i] = stdValues[i].value;
            boltKeys[i] = stdValues[i].key;*/

			Values[i].key = rand();
            Values[i].value = (double) i+3;
            stdKeys[i] = Values[i].key;
            stdValues[i] = Values[i].value;
 
           /* stdOffsetValues[i].key = rand();
            stdOffsetValues[i].value = (double) i+3;
            boltOffsetValues[i] = stdOffsetValues[i].value;
            boltOffsetKeys[i] = stdOffsetValues[i].key;*/

			OffsetValues[i].key = rand();
            OffsetValues[i].value = (double) i+3;
            stdOffsetKeys[i] = OffsetValues[i].key;
            stdOffsetValues[i] = OffsetValues[i].value;
 
        }
    }

protected:
	std::vector< stdSortDataKey<double> > Values, OffsetValues, Keys, OffsetKeys;
    std::vector< double >  stdValues, stdOffsetValues, stdKeys, stdOffsetKeys;
    //bolt::cl::device_vector< double > boltValues, boltKeys, boltOffsetValues, boltOffsetKeys;
    int VectorSize;
};


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

    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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

    std::stable_sort( stdValues.begin( ), stdValues.end( ));
    bolt::BKND::STABLE_SORT_FUNC( key_first, key_last, value.begin());

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin(),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( value.begin(), value.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( stdValues, key_first, value, mySize);

} */

#if (TEST_DOUBLE == 1)
TEST( StableSortbyUDDKeyVectorTest, Normal )
{
	int length = (1<<16);

	std::vector< uddtD4  > stdKeys( length);
	std::vector< uddtD4  > stdValues( length);


	bolt::cl::device_vector< uddtD4  > boltKeys(stdKeys.begin(), stdKeys.end());;
	bolt::cl::device_vector< uddtD4  > boltValues( stdValues.begin(), stdValues.end());
	bolt::cl::device_vector<uddtD4>::pointer boltKeysPtr1 =  boltKeys.data( );
	bolt::cl::device_vector<uddtD4>::pointer boltvaluesPtr1 =  boltValues.data( );

    for(int i = 0; i<length; i++)
	{
		stdKeys[i].a = rand()%length;
		stdKeys[i].b = rand()%length;
		stdKeys[i].c = rand()%length;
		stdKeys[i].d = rand()%length;

		stdValues[i].a = stdKeys[i].a;
		stdValues[i].b = stdKeys[i].b;
		stdValues[i].c = stdKeys[i].c;
		stdValues[i].d = stdKeys[i].d;

		boltKeysPtr1[i].a = stdKeys[i].a;
		boltKeysPtr1[i].b = stdKeys[i].b;
		boltKeysPtr1[i].c = stdKeys[i].c;
		boltKeysPtr1[i].d = stdKeys[i].d;

		boltvaluesPtr1[i].a = stdKeys[i].a;
		boltvaluesPtr1[i].b = stdKeys[i].b;
		boltvaluesPtr1[i].c = stdKeys[i].c;
		boltvaluesPtr1[i].d = stdKeys[i].d;

	}

	AddD4 ad4gt;

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ), ad4gt);
	std::stable_sort( stdKeys.begin( ), stdKeys.end( ), ad4gt);

    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ), ad4gt );
	
    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays(stdKeys, boltKeysPtr1); 
	cmpArrays(stdValues, boltvaluesPtr1); 

	std::vector< uddtD4  > stdOffsetKeys( length);
	std::vector< uddtD4  > stdOffsetValues( length);
	bolt::cl::device_vector< uddtD4  > boltOffsetKeys( stdOffsetKeys.begin(), stdOffsetKeys.end());
	bolt::cl::device_vector< uddtD4  > boltOffsetValues( stdOffsetValues.begin(), stdOffsetValues.end());

	bolt::cl::device_vector<uddtD4>::pointer boltKeysPtr =  boltOffsetKeys.data( );
	bolt::cl::device_vector<uddtD4>::pointer boltvaluesPtr =  boltOffsetValues.data( );

    for(int i = 0; i<length; i++)
	{
		stdOffsetKeys[i].a = rand()%length;
		stdOffsetKeys[i].b = rand()%length;
		stdOffsetKeys[i].c = rand()%length;
		stdOffsetKeys[i].d = rand()%length;

		stdOffsetValues[i].a = rand()%10;
		stdOffsetValues[i].b = rand()%250;
		stdOffsetValues[i].c = rand()%20;
		stdOffsetValues[i].d = rand()%350;

		boltKeysPtr[i].a = stdOffsetKeys[i].a;
		boltKeysPtr[i].b = stdOffsetKeys[i].b;
		boltKeysPtr[i].c = stdOffsetKeys[i].c;
		boltKeysPtr[i].d = stdOffsetKeys[i].d;

		boltvaluesPtr[i].a = stdOffsetValues[i].a;
		boltvaluesPtr[i].b = stdOffsetValues[i].b;
		boltvaluesPtr[i].c = stdOffsetValues[i].c;
		boltvaluesPtr[i].d = stdOffsetValues[i].d;

	}

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = length -17; //Some aribitrary offset position

    if( (( startIndex > length ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< length << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
		std::stable_sort( stdOffsetKeys.begin( ) + startIndex, stdOffsetKeys.begin( ) + endIndex, ad4gt );

        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
       
		cmpArrays(stdOffsetKeys, boltKeysPtr/*boltOffsetKeys*/); 
	    cmpArrays(stdOffsetValues, boltvaluesPtr/*boltOffsetValues*/); 

    }
}

TEST( StableSortbyUDDKeyVectorTest2, Normal )
{
	int length = (1<<16) + 55; // Its failing with offset

	std::vector< uddtD4  > stdKeys( length);
	std::vector< uddtD4  > stdValues( length);

    for(int i = 0; i<length; i++)
	{
		stdKeys[i].a = rand()%length;
		stdKeys[i].b = rand()%length;
		stdKeys[i].c = rand()%length;
		stdKeys[i].d = rand()%length;

		stdValues[i].a = stdKeys[i].a;
		stdValues[i].b = stdKeys[i].b;
		stdValues[i].c = stdKeys[i].c;
		stdValues[i].d = stdKeys[i].d;
	}

    //std::vector< uddtD4  > stdKeys( length, initialUdd4);
	//std::vector< uddtD4  > stdValues( length, initialUdd4);
	std::vector< uddtD4  > boltKeys( stdKeys.begin(), stdKeys.end());
	std::vector< uddtD4  > boltValues( stdValues.begin(), stdValues.end());

	std::vector< uddtD4  > stdOffsetKeys( length, initialUdd4);
	std::vector< uddtD4  > stdOffsetValues( length, initialUdd4);
	std::vector< uddtD4  > boltOffsetKeys( stdOffsetKeys.begin(), stdOffsetKeys.end());
	std::vector< uddtD4  > boltOffsetValues( stdOffsetValues.begin(), stdOffsetValues.end());

	AddD4 ad4gt;

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ), ad4gt);
	std::stable_sort( stdKeys.begin( ), stdKeys.end( ), ad4gt);
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ), ad4gt );
	
    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArrays(stdKeys, boltKeys); 
	cmpArrays(stdValues, boltValues); 

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = length -17; //Some aribitrary offset position

    if( (( startIndex > length ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< length << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
		std::stable_sort( stdOffsetKeys.begin( ) + startIndex, stdOffsetKeys.begin( ) + endIndex, ad4gt );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArrays(stdOffsetKeys, boltOffsetKeys); 
	    cmpArrays(stdOffsetValues, boltOffsetValues); 
    }
}
#endif

TEST_P( StableSortbyUDDDeviceKeyVector, Normal )
{
	AddD4 ad4gt;

	bolt::cl::device_vector< uddtD4 > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< uddtD4 > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< uddtD4 > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< uddtD4 > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ), ad4gt);
	std::stable_sort( stdKeys.begin( ), stdKeys.end( ), ad4gt);
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    //cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  VectorSize);
	cmpArrays(stdKeys, boltKeys); 
	cmpArrays(stdValues, boltValues);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
		std::stable_sort( stdOffsetKeys.begin( ) + startIndex, stdOffsetKeys.begin( ) + endIndex, ad4gt );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        //cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  VectorSize );
		cmpArrays(stdOffsetKeys, boltOffsetKeys); 
	    cmpArrays(stdOffsetValues,boltOffsetValues); 
    }
}

TEST_P( StableSortbyUDDDeviceKeyVector, Serial )
{
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    AddD4 ad4gt;

	bolt::cl::device_vector< uddtD4 > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< uddtD4 > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< uddtD4 > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< uddtD4 > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

	

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ), ad4gt);
	std::stable_sort( stdKeys.begin( ), stdKeys.end( ), ad4gt);
    bolt::BKND::STABLE_SORT_FUNC(ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

	
    //  Loop through the array and compare all the values with each other
    //cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  VectorSize);
	cmpArrays(stdKeys, boltKeys); 
	cmpArrays(stdValues, boltValues);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
		std::stable_sort( stdOffsetKeys.begin( ) + startIndex, stdOffsetKeys.begin( ) + endIndex, ad4gt );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        //cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  VectorSize );
		cmpArrays(stdOffsetKeys, boltOffsetKeys); 
	    cmpArrays(stdOffsetValues,boltOffsetValues); 
    }
}

TEST_P( StableSortbyUDDDeviceKeyVector, MultiCore )
{
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    AddD4 ad4gt;

	bolt::cl::device_vector< uddtD4 > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< uddtD4 > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< uddtD4 > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< uddtD4 > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ), ad4gt);
	std::stable_sort( stdKeys.begin( ), stdKeys.end( ), ad4gt);
    bolt::BKND::STABLE_SORT_FUNC(ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    //cmpArraysSortByKey( stdValues, stdKeys, boltKeys, boltValues,  VectorSize);
	cmpArrays(stdKeys, boltKeys); 
	cmpArrays(stdValues, boltValues);

    //  OFFSET Calling the actual functions under test
    int startIndex = 17; //Some aribitrary offset position
    int endIndex   = VectorSize -17; //Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex, ad4gt );
		std::stable_sort( stdOffsetKeys.begin( ) + startIndex, stdOffsetKeys.begin( ) + endIndex, ad4gt );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        //cmpArraysSortByKey( stdOffsetValues, stdOffsetKeys, boltOffsetKeys, boltOffsetValues,  VectorSize );
		cmpArrays(stdOffsetKeys, boltOffsetKeys); 
	    cmpArrays(stdOffsetValues,boltOffsetValues); 
    }
}

TEST_P( StableSortbyUDDKeyVector, Normal )
{
	AddD4 ad4gt;

    //  Calling the actual functions under test

	std::stable_sort( stdValues.begin( ), stdValues.end( ));

    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

	cmpArraysSortByKey(stdValues, boltKeys, boltValues, VectorSize);

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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( StableSortbyUDDKeyVector, Serial )
{

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    AddD4 ad4gt;

    //  Calling the actual functions under test

	std::stable_sort( stdValues.begin( ), stdValues.end( ));

    bolt::BKND::STABLE_SORT_FUNC(ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

	cmpArraysSortByKey(stdValues, boltKeys, boltValues, VectorSize);

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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }

}

TEST_P( StableSortbyUDDKeyVector, MultiCoreCPU )
{

	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    AddD4 ad4gt;

    //  Calling the actual functions under test

	std::stable_sort( stdValues.begin( ), stdValues.end( ));

    bolt::BKND::STABLE_SORT_FUNC(ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ) , ad4gt);

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

	cmpArraysSortByKey(stdValues, boltKeys, boltValues, VectorSize);

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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex, ad4gt );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }

}


TEST_P( StableSortbyDoubleKeyVector, Normal )
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

TEST_P( StableSortbyDoubleKeyVector, Serial )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyDoubleKeyVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyDoubleDeviceKeyVector, Normal )
{
	bolt::cl::device_vector< double > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< double > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< double > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< double > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());
	
    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues, VectorSize );

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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyDoubleDeviceKeyVector, Serial )
{
	bolt::cl::device_vector< double > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< double > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< double > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< double > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues, VectorSize );

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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyDoubleDeviceKeyVector, MultiCoreCPU )
{

	bolt::cl::device_vector< double > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< double > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< double > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< double > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues, VectorSize );

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
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyFloatKeyVector, Normal )
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

TEST_P( StableSortbyFloatKeyVector, Serial )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyFloatKeyVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )  + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyFloatDeviceKeyVector, Normal )
{
    
	bolt::cl::device_vector< float > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< float > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< float > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< float > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());
	

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues, VectorSize );
	

    //  OFFSET Calling the actual functions under test
    int startIndex = 17;//Some aribitrary offset position
    int endIndex   = VectorSize - 17 ;//Some aribitrary offset position

    if( (( startIndex > VectorSize ) || ( endIndex < 0 ) )  || (startIndex > endIndex) )
    {
        std::cout <<"\nSkipping NormalOffset Test for size "<< VectorSize << "\n";
    }
    else
    {
        std::stable_sort( OffsetValues.begin( ) + startIndex, OffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
		/*bolt::cl::device_vector<float>::pointer copydata =  boltOffsetValues.data( );
		bolt::cl::device_vector<float>::pointer copykeys =  boltOffsetKeys.data( );
		for(int i=17; i<VectorSize-17; i++)
	    {
		   if( (OffsetValues[ i ].key == copykeys[ i ]) && (OffsetValues[ i ].value == copydata[ i ]) )	   
				    continue;
		   else
		   {
			  printf("\n Mismatch for Offset! \n");
			  return;
		   }
	    }*/
    }
}

TEST_P( StableSortbyFloatDeviceKeyVector, Serial )
{
    bolt::cl::device_vector< float > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< float > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< float > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< float > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues, VectorSize );

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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyFloatDeviceKeyVector, MultiCoreCPU )
{
    bolt::cl::device_vector< float > boltValues(stdValues.begin(), stdValues.end());
	bolt::cl::device_vector< float > boltKeys(stdKeys.begin(), stdKeys.end());
	bolt::cl::device_vector< float > boltOffsetValues(stdOffsetValues.begin(), stdOffsetValues.end());
	bolt::cl::device_vector< float > boltOffsetKeys(stdOffsetKeys.begin(), stdOffsetKeys.end());
	
	bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( Values.begin( ), Values.end( ));
    bolt::BKND::STABLE_SORT_FUNC( ctl, boltKeys.begin( ), boltKeys.end( ), boltValues.begin( ));

    std::vector< stdSortData<int> >::iterator::difference_type stdValueElements = std::distance( stdValues.begin( ),
                                                                                                 stdValues.end() );
    std::vector< int >::iterator::difference_type boltValueElements = std::distance( boltValues.begin( ),
                                                                                     boltValues.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdValueElements, boltValueElements );

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( Values, boltKeys, boltValues, VectorSize );

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
        cmpArraysSortByKey( OffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues,  VectorSize );
    }
}

TEST_P( StableSortbyKeyIntegerVector, Serial )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }

}

TEST_P( StableSortbyKeyIntegerVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )  + startIndex);

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyFloatVector, MultiCoreCPU )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex);

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
        bolt::BKND::STABLE_SORT_FUNC( boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )+ startIndex );

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyDoubleVector, Serial)
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //  Calling the actual functions under test
    std::stable_sort( stdValues.begin( ), stdValues.end( ));
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
        std::stable_sort( stdOffsetValues.begin( ) + startIndex, stdOffsetValues.begin( ) + endIndex );
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( )  + startIndex);

        //  Loop through the array and compare all the values with each other
        cmpArraysSortByKey( stdOffsetValues, boltOffsetKeys, boltOffsetValues, VectorSize );
    }
}

TEST_P( StableSortbyKeyDoubleVector, MultiCoreCPU)
{
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
        bolt::BKND::STABLE_SORT_FUNC( ctl, boltOffsetKeys.begin( ) + startIndex, boltOffsetKeys.begin( ) + endIndex, boltOffsetValues.begin( ) + startIndex );

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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

TEST_P( StableSortbyKeyIntegerDeviceVector, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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

TEST_P( StableSortbyKeyFloatDeviceVector, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

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

TEST_P( StableSortbyKeyDoubleDeviceVector, MultiCoreInplace )
{
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

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
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );
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
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );
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
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );
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
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );
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
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );

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
    std::stable_sort( wrapstdValues, wrapstdValues + endIndex );

    stdext::checked_array_iterator< double* > wrapboltValues( boltValues, endIndex );
    stdext::checked_array_iterator< int* > wrapboltKeys( boltKeys, endIndex );
    bolt::BKND::STABLE_SORT_FUNC( ctl, wrapboltKeys, wrapboltKeys + endIndex , wrapboltValues);

    //  Loop through the array and compare all the values with each other
    cmpArraysSortByKey( wrapstdValues, wrapboltKeys, wrapboltValues, VectorSize );
}

#endif
#endif

//#if(TEST_LARGE_BUFFERS == 1)
/*Negative test to stable sort a buffer when all the input values are equal Say zero*/
TEST( DefaultGPU, Normal )
{
    /*This test case was a fail only on large buffer sizes. */
    int length = 2097152; //2^21
    bolt::cl::device_vector< float > boltKey(  length, 0.0, CL_MEM_READ_WRITE, true  );
    bolt::cl::device_vector< float > boltValue(  length, 0.0, CL_MEM_READ_WRITE, true  );
    std::vector< float > stdInput( length, 0.0 );

    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );

    //  Calling the actual functions under test
    //std::SORT_FUNC( stdInput.begin( ), stdInput.end( ));
    bolt::cl::stable_sort_by_key( ctl, boltKey.begin( ), boltKey.end( ), boltValue.begin() );


}
//#endif

std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int, 5> TestValues2 = {2048, 4096,8192,16384,32768/*, 1<<22*/};

//INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortByKeyCountingIterator,
//                        ::testing::ValuesIn( TestValues.begin(), TestValues.end() ) );

//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyIntegerVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																					  
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyIntegerVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
//#endif																					  

INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyFloatKeyVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyFloatKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyFloatVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                      TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyFloatDeviceKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyFloatDeviceKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
//#if(TEST_LARGE_BUFFERS == 1)																					  
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyFloatVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyFloatKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                      TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyFloatDeviceKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
//#endif																					  
                                                                                      
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyDoubleKeyVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyDoubleKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyDoubleDeviceKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyDoubleDeviceKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                       TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyDoubleVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                     TestValues.end() ) );


INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyUDDDeviceKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyUDDDeviceKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                     TestValues.end() ) );

INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyUDDKeyVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyUDDKeyVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                     TestValues.end() ) );

//#if(TEST_LARGE_BUFFERS == 1)																					   
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyDoubleVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2,  StableSortbyDoubleKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2,  StableSortbyUDDKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyUDDDeviceKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                     TestValues2.end() ) );
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyDoubleDeviceKeyVector, ::testing::ValuesIn( TestValues2.begin(),
                                                                                       TestValues2.end() ) );
//#endif
                                                                                       
#endif
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyIntegerDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                            TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																							
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyIntegerDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                            TestValues2.end()));
//#endif																							
                                                                                            
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyFloatDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                          TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																						  
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyFloatDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                          TestValues2.end()));
//#endif																						  
                                                                                          
#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyDoubleDeviceVector, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues, StableSortbyKeyDoubleDeviceVector,::testing::ValuesIn(TestValues.begin(),
                                                                                           TestValues.end()));
//#if(TEST_LARGE_BUFFERS == 1)																						   
INSTANTIATE_TEST_CASE_P( StableSortByKeyValues2, StableSortbyKeyDoubleDeviceVector,::testing::ValuesIn(TestValues2.begin(),
                                                                                           TestValues2.end()));	
//#endif
#endif
INSTANTIATE_TEST_CASE_P( StableSortByKeyRange, StableSortbyKeyIntegerNakedPointer, ::testing::Range(1, 32768, 3276 ) ); // 1 to 2^15
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
  //  bolt::miniDumpSingleton::enableMiniDumps( );

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
