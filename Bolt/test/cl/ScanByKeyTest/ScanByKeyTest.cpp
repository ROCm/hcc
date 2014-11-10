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

//#include "common/stdafx.h"
#include <vector>
//#include <array>
#include "bolt/cl/bolt.h"
//#include "bolt/cl/scan.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/scan_by_key.h"
#include "bolt/unicode.h"
#include "bolt/miniDump.h"

#define TEST_DOUBLE 1
#define SERIAL_TBB_OFFSET 1

#include <gtest/gtest.h>
//#include <boost/shared_array.hpp>
#include <boost/program_options.hpp>


#if 1


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track

/******************************************************************************
 *  Double x4
 *****************************************************************************/

#if (TEST_DOUBLE == 1)
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
    uddtD4 operator-() const
    {
        uddtD4 r;
        r.a = -a;
        r.b = -b;
        r.c = -c;
        r.d = -d;
        return r;
    }
    uddtD4 operator*(const uddtD4& rhs)
    {
        uddtD4 r;
        r.a = a*a;
        r.b = b*b;
        r.c = c*c;
        r.d = d*d;
        return r;
    }
};
);
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

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtD4 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtD4 >::iterator, bolt::cl::deviceVectorIteratorTemplate );



uddtD4 identityMultD4 = { 1.0, 1.0, 1.0, 1.0 };
uddtD4 initialMultD4  = { 1.00001, 1.000003, 1.0000005, 1.00000007 };

#endif

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
    uddtI2 operator-() const
    {
        uddtI2 r;
        r.a = -a;
        r.b = -b;
        return r;
    }
    uddtI2 operator*(const uddtI2& rhs)
    {
        uddtI2 r;
        r.a = a*a;
        r.b = b*b;
        return r;
    }
};
);
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

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtI2 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtI2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );



uddtI2 identityAddI2 = {  0, 0 };
uddtI2 initialAddI2  = { -1, 2 };


/******************************************************************************
 *  Mixed float and int
 *****************************************************************************/
#if (TEST_DOUBLE == 1)

BOLT_FUNCTOR(uddtM3,
struct uddtM3
{
    int a;
    int        b;
    double       c;

    bool operator==(const uddtM3& rhs) const
    {
        bool equal = true;
        double ths = 0.0001;
        double thd = 0.000000001;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        //if (rhs.b < ths && rhs.b > -ths)
        //    equal = ( (1.0*b - rhs.b) < ths && (1.0*b - rhs.b) > -ths) ? equal : false;
        //else
        //    equal = ( (1.0*b - rhs.b)/rhs.b < ths && (1.0*b - rhs.b)/rhs.b > -ths) ? equal : false;
        if (rhs.c < thd && rhs.c > -thd)
            equal = ( (1.0*c - rhs.c) < thd && (1.0*c - rhs.c) > -thd) ? equal : false;
        else
            equal = ( (1.0*c - rhs.c)/rhs.c < thd && (1.0*c - rhs.c)/rhs.c > -thd) ? equal : false;
        return equal;
    }
    uddtM3 operator-() const
    {
        uddtM3 r;
        r.a = -a;
        r.b = -b;
        r.c = -c;
        return r;
    }
    uddtM3 operator*(const uddtM3& rhs)
    {
        uddtM3 r;
        r.a = a*a;
        r.b = b*b;
        r.c = c*c;
        return r;
    }
};
);
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
BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtM3 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtM3 >::iterator, bolt::cl::deviceVectorIteratorTemplate );



uddtM3 identityMixM3 = { 0, 0, 1.0 };
uddtM3 initialMixM3  = { 2, 1, 1.000001 };
#endif

BOLT_FUNCTOR(uddtM2,
struct uddtM2
{
    int a;
    float b;

    bool operator==(const uddtM2& rhs) const
    {
        bool equal = true;
        float ths = 0.00001f; // thresh hold single(float)
        equal = ( a == rhs.a ) ? equal : false;
        if (rhs.b < ths && rhs.b > -ths)
            equal = ( (1.0*b - rhs.b) < ths && (1.0*b - rhs.b) > -ths) ? equal : false;
        else
            equal = ( (1.0*b - rhs.b)/rhs.b < ths && (1.0*b - rhs.b)/rhs.b > -ths) ? equal : false;
        return equal;
    }
    uddtM2 operator-() const
    {
        uddtM2 r;
        r.a = -a;
        r.b = -b;
        return r;
    }
    uddtM2 operator*(const uddtM2& rhs)
    {
        uddtM2 r;
        r.a = a*a;
        r.b = b*b;
        return r;
    }
    void operator++()
    {
        a += 1;
        b += 1.234567f;
    }
};
);

BOLT_FUNCTOR(MixM2,
struct MixM2
{
    uddtM2 operator()(const uddtM2 &lhs, const uddtM2 &rhs) const
    {
        uddtM2 _result;
        _result.a = lhs.a^rhs.a;
        _result.b = lhs.b+rhs.b;
        return _result;
    };
}; 
);

BOLT_CREATE_TYPENAME( bolt::cl::device_vector< uddtM2 >::iterator );
BOLT_CREATE_CLCODE( bolt::cl::device_vector< uddtM2 >::iterator, bolt::cl::deviceVectorIteratorTemplate );


uddtM2 identityMixM2 = { 0, 3.141596f };
uddtM2 initialMixM2  = { 2, 1.000001f };


BOLT_FUNCTOR(uddtM2_equal_to,
struct uddtM2_equal_to
{
    bool operator()(const uddtM2& lhs, const uddtM2& rhs) const
    {
        return lhs == rhs;
    }
};
);


#include "test_common.h"


/******************************************************************************
 *  Scan with User Defined Data Types and Operators
 *****************************************************************************/


template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryFunction>
OutputIterator
gold_scan_by_key(
    InputIterator1 firstKey,
    InputIterator1 lastKey,
    InputIterator2 values,
    OutputIterator result,
    BinaryFunction binary_op)
{
    if(std::distance(firstKey,lastKey) < 1)
         return result;
    typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;

    static_assert( std::is_convertible< vType, oType >::value,
        "InputIterator2 and OutputIterator's value types are not convertible." );

    if(std::distance(firstKey,lastKey) < 1)
         return result;
    // do zeroeth element
    *result = *values; // assign value

    // scan oneth element and beyond
    for ( InputIterator1 key = (firstKey+1); key != lastKey; key++)
    {
        // move on to next element
        values++;
        result++;

        // load keys
        kType currentKey  = *(key);
        kType previousKey = *(key-1);

        // load value
        oType currentValue = *values; // convertible
        oType previousValue = *(result-1);

        // within segment
        if (currentKey == previousKey)
        {
            //std::cout << "continuing segment" << std::endl;
            oType r = binary_op( previousValue, currentValue);
            *result = r;
        }
        else // new segment
        {
            //std::cout << "new segment" << std::endl;
            *result = currentValue;
        }
    }

    return result;
}

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator,
    typename BinaryFunction,
    typename T>
OutputIterator
gold_scan_by_key_exclusive(
    InputIterator1 firstKey,
    InputIterator1 lastKey,
    InputIterator2 values,
    OutputIterator result,
    BinaryFunction binary_op,
    T init)
{
    if(std::distance(firstKey,lastKey) < 1)
         return result;
    typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator >::value_type oType;
    static_assert( std::is_convertible< vType, oType >::value,
        "InputIterator2 and OutputIterator's value types are not convertible." );
    // do zeroeth element
    //*result = *values; // assign value
    oType temp = *values;
    *result = (vType)init;
    // scan oneth element and beyond
    for ( InputIterator1 key = (firstKey+1); key != lastKey; key++)
    {
        // move on to next element
        values++;
        result++;
        // load keys
        kType currentKey  = *(key);
        kType previousKey = *(key-1);
        // load value
        oType currentValue = temp; // convertible
        oType previousValue = *(result-1);
        // within segment
        if (currentKey == previousKey)
        {
            temp = *values;
            oType r = binary_op( previousValue, currentValue);
            *result = r;
            
        }
        else // new segment
        {
           // std::cout << "new segment" << std::endl;
             temp = *values;
            *result = (vType)init;
        }
    }

    return result;
}


class scanByKeyStdVectorWithIters:public ::testing::TestWithParam<int>
{
protected:
    int myStdVectSize;
public:
    scanByKeyStdVectorWithIters():myStdVectSize(GetParam()){
    }
};

typedef scanByKeyStdVectorWithIters ScanByKeyOffsetTest;
typedef scanByKeyStdVectorWithIters ScanByKeyCLtypeTest;
/*
class StdVectCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    StdVectCountingIterator():mySize(GetParam()){
    }
};*/

TEST_P (ScanByKeyCLtypeTest, InclTestLong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
   

    // input and output vectors for device and reference

    //bolt::cl::device_vector< cl_long > output( myStdVectSize, 0 );
    std::vector< cl_long > refInput( myStdVectSize);
    //std::vector< cl_long > refOutput( myStdVectSize , 0);

	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;

	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyCLtypeTest, SerialInclTestLong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;

	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

  
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
          
} 
TEST_P (ScanByKeyCLtypeTest, MulticoreInclTestLong)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 
 
TEST_P (ScanByKeyCLtypeTest, InclTestULong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
         
} 

TEST_P (ScanByKeyCLtypeTest, MulticoreInclTestULong)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 

TEST_P (ScanByKeyCLtypeTest, ExclTestLong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
    
	bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);

    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyCLtypeTest, SerialExclTestLong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),  refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
          
} 
TEST_P (ScanByKeyCLtypeTest, MulticoreExclTestLong)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),  refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
          
     
} 
 
TEST_P (ScanByKeyCLtypeTest, ExclTestULong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyCLtypeTest, SerialExclTestULong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
          
} 
TEST_P (ScanByKeyCLtypeTest, MulticoreExclTestULong)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);
     
	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
} 

TEST_P (ScanByKeyCLtypeTest, InclTestShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyCLtypeTest, SerialInclTestShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
          
} 
TEST_P (ScanByKeyCLtypeTest, MulticoreInclTestShort)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 
 
TEST_P (ScanByKeyCLtypeTest, InclTestUShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
         
} 

TEST_P (ScanByKeyCLtypeTest, MulticoreInclTestUShort)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 

TEST_P (ScanByKeyCLtypeTest, ExclTestShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyCLtypeTest, SerialExclTestShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
         
          
} 
TEST_P (ScanByKeyCLtypeTest, MulticoreExclTestShort)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

     
} 
 
TEST_P (ScanByKeyCLtypeTest, ExclTestUShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyCLtypeTest, SerialExclTestUShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);
    
	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
} 
TEST_P (ScanByKeyCLtypeTest, MulticoreExclTestUShort)
{ 
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
     
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

     
} 



TEST_P (ScanByKeyOffsetTest, InclOffsetTestInt)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< int> input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<int> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if (SERIAL_TBB_OFFSET == 1)
TEST_P (ScanByKeyOffsetTest, SerialInclOffsetTestInt)
{        
    std::vector< int > keys( myStdVectSize, 1);

    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());

    std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<int> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, MulticoreInclOffsetTestInt)
{          
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<int> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 


TEST_P (ScanByKeyOffsetTest, SerialExclOffsetTestInt)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<int> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, MulticoreExclOffsetTestInt)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<int> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, SerialInclOffsetTestFloat)
{        
    std::vector< float > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.f;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<float> eq; 
    bolt::cl::plus<float> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, MulticoreInclOffsetTestFloat)
{          
    std::vector< float > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.f;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<float> eq; 
    bolt::cl::plus<float> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, SerialExclOffsetTestFloat)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< float > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.f;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<float> eq; 
    bolt::cl::plus<float> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, MulticoreExclOffsetTestFloat)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< float > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.f;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<float> eq; 
    bolt::cl::plus<float> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

}
#endif

TEST_P (ScanByKeyOffsetTest, ExclOffsetTestInt)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<int> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4),  refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, InclOffsetTestFloat)
{
    std::vector< float > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.f;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector< float > device_keys( keys.begin(), keys.end());

    std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<float> eq; 
    bolt::cl::plus<float> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, ExclOffsetTestFloat)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< float > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.f;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<float> eq; 
    bolt::cl::plus<float> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if (SERIAL_TBB_OFFSET == 1)

TEST_P (ScanByKeyOffsetTest, SerialInclOffsetTestDouble)
{        
    std::vector< double > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< double  > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::cl::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<double> eq; 
    bolt::cl::plus<double> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, MulticoreInclOffsetTestDouble)
{          
    std::vector< double > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< double  > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::cl::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<double> eq; 
    bolt::cl::plus<double> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, SerialExclOffsetTestDouble)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< double > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< double > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::cl::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<double> eq; 
    bolt::cl::plus<double> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, MulticoreExclOffsetTestDouble)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< double > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector<double > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::cl::device_vector< double > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::cl::equal_to<double> eq; 
    bolt::cl::plus<double> mM3; 
    
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

}
#endif
#if(TEST_DOUBLE == 1)
TEST_P (ScanByKeyOffsetTest, InclOffsetTestDouble)
{
    std::vector< double > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector<double > device_keys( keys.begin(), keys.end());
     
    std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::cl::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<double> eq; 
    bolt::cl::plus<double> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, ExclOffsetTestDouble)
{
    //bolt::cl::device_vector< float > input( myStdVectSize, 2.f);
    //std::vector< float > refInput( myStdVectSize, 2.f);
              
    std::vector< double > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<double > device_keys( keys.begin(), keys.end());
      
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::cl::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<double> eq; 
    bolt::cl::plus<double> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

}

#endif

TEST_P (ScanByKeyOffsetTest, InclOffsetTestLong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
      
    
	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 


TEST_P (ScanByKeyOffsetTest, InclOffsetTestULong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
      
    std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, InclOffsetTestShort)
{
    std::vector< int > keys( myStdVectSize, 1);

    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 


TEST_P (ScanByKeyOffsetTest, InclOffsetTestUShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
    
    //bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 


TEST_P (ScanByKeyOffsetTest, ExclOffsetTestLong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, ExclInclOffsetTestULong)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());

	std::vector< cl_ulong > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 


TEST_P (ScanByKeyOffsetTest, ExclOffsetTestShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, ExclOffsetTestUShort)
{
    std::vector< int > keys( myStdVectSize, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < myStdVectSize; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());

	std::vector< cl_ushort > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

INSTANTIATE_TEST_CASE_P(incl_excl_ScanByKeyIterIntLimit, ScanByKeyCLtypeTest, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^12
INSTANTIATE_TEST_CASE_P(incl_excl_ScanByKeyIterIntLimit, ScanByKeyOffsetTest, ::testing::Range(4096, 65536, 555 ) ); //2^12 to 2^16


#if (TEST_DOUBLE == 1)

TEST(InclusiveScanByKey, IncMixedM3increment)
{
    //setup keys
    int length = (1<<24);
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        
        keys[i] = key;
        segmentIndex++;
    }

    // input and output vectors for device and reference
    std::vector< uddtM3 > input(  length, initialMixM3 );
    //std::vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );

    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
   /* bolt::cl::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);*/

	bolt::cl::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);


#if 0
    // print Bolt scan_by_key
    for (int i = 0; i < length; i++)
    {
        if ( !(output[i] == refOutput[i]) ) {
        std::cout << "BOLT: i=" << i << ", ";
        std::cout << "key={" << keys[i].a << ", " << keys[i].b << "}; ";
        std::cout << "val={" << input[i].a << ", " << input[i].b << ", " << input[i].c << "}; ";
        //std::cout << "out={" << output[i].a << ", " << output[i].b << ", " << output[i].c << "};" << std::endl;
    
        std::cout << "GOLD: i=" << i << ", ";
        std::cout << "key={" << keys[i].a << ", " << keys[i].b << "}; ";
        std::cout << "val={" << refInput[i].a << ", " << refInput[i].b << ", " << refInput[i].c << "}; ";
        //std::cout << "out={" << refOutput[i].a << ", " << refOutput[i].b << ", " << refOutput[i].c << "};" <<std::endl;
        }
    }
#endif

    // compare results
    //cmpArrays(refOutput, output);
	cmpArrays(refInput, input);
}


TEST(InclusiveScanByKey, IncMixedM3same)
{
    //setup keys
    int length = (1<<24);
    uddtM2 key = {1, 2.3f};
    std::vector< uddtM2 > keys( length, key);

    // input and output vectors for device and reference
    std::vector< uddtM3 > input(  length, initialMixM3 );
    //std::vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );

    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
   /* bolt::cl::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);*/

	 bolt::cl::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);

#if 0
    std::cout.setf(std::ios_base::scientific);
    std::cout.precision(15);
    // print Bolt scan_by_key
    for (int i = 0; i < length; i++)
    {
        if ( !(output[i] == refOutput[i]) ) {
        std::cout.precision(3);
        std::cout << "BOLT: i=" << i << ", ";
        std::cout << "key={" << keys[i].a << ", " << keys[i].b << "}; ";
        std::cout << "val={" << input[i].a << ", " << input[i].b << ", " << input[i].c << "}; ";
        std::cout.precision(15);
        //std::cout << "out={" << output[i].a << ", " << output[i].b << ", " << output[i].c << "};" << std::endl;
    
        std::cout << "GOLD: i=" << i << ", ";
        std::cout.precision(3);
        std::cout << "key={" << keys[i].a << ", " << keys[i].b << "}; ";
        std::cout << "val={" << refInput[i].a << ", " << refInput[i].b << ", " << refInput[i].c << "}; ";
        std::cout.precision(15);
       // std::cout << "out={" << refOutput[i].a << ", " << refOutput[i].b << ", " << refOutput[i].c << "};" <<std::endl;
        }
    }
#endif

    // compare results
    cmpArrays(refInput, input);
}


TEST(InclusiveScanByKey, IncMixedM3each)
{
    //setup keys
    int length = (1<<24);
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        ++key;
        keys[i] = key;
    }

    // input and output vectors for device and reference
    std::vector< uddtM3 > input(  length, initialMixM3 );
    //std::vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );

    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
    /*bolt::cl::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);*/

	bolt::cl::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);

#if 0
    // print Bolt scan_by_key
    for (int i = 0; i < length; i++)
    {
        if ( !(output[i] == refOutput[i]) ) {
        std::cout << "BOLT: i=" << i << ", ";
        std::cout << "key={" << keys[i].a << ", " << keys[i].b << "}; ";
        std::cout << "val={" << input[i].a << ", " << input[i].b << ", " << input[i].c << "}; ";
        //std::cout << "out={" << output[i].a << ", " << output[i].b << ", " << output[i].c << "};" << std::endl;
    
        std::cout << "GOLD: i=" << i << ", ";
        std::cout << "key={" << keys[i].a << ", " << keys[i].b << "}; ";
        std::cout << "val={" << refInput[i].a << ", " << refInput[i].b << ", " << refInput[i].c << "}; ";
        //std::cout << "out={" << refOutput[i].a << ", " << refOutput[i].b << ", " << refOutput[i].c << "};" <<std::endl;
        }
    }
#endif

    // compare results
    cmpArrays(refInput, input);
}

#endif

TEST( equalValMult, iValues )
{
    int keys[11] = { 7, 0, 0, 3, 3, 3, -5, -5, -5, -5, 3 }; 
    int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 }; 
    int out[11]; 
   
    bolt::cl::equal_to<int> eq; 
    bolt::cl::multiplies<int> mult; 
   
    bolt::cl::inclusive_scan_by_key( keys, keys+11, vals, out, eq, mult ); 
   
    int arrToMatch[11] = { 2, 2, 4, 2, 4, 8, 2, 4, 8, 16, 2 };

    // compare results
    cmpArrays<int,11>( arrToMatch, out );
}

TEST( equalValMult, Serial_iValues )
{
    int keys[11] = { 7, 0, 0, 3, 3, 3, -5, -5, -5, -5, 3 }; 
    int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 }; 
    int out[11]; 
   
    bolt::cl::equal_to<int> eq; 
    bolt::cl::multiplies<int> mult; 
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    bolt::cl::inclusive_scan_by_key(ctl, keys, keys+11, vals, out, eq, mult ); 
   
    int arrToMatch[11] = { 2, 2, 4, 2, 4, 8, 2, 4, 8, 16, 2 };

    // compare results
    cmpArrays<int,11>( arrToMatch, out );
}

TEST( equalValMult, MultiCore_iValues )
{
    int keys[11] = { 7, 0, 0, 3, 3, 3, -5, -5, -5, -5, 3 }; 
    int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 }; 
    int out[11]; 
   
    bolt::cl::equal_to<int> eq; 
    bolt::cl::multiplies<int> mult; 
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    bolt::cl::inclusive_scan_by_key(ctl, keys, keys+11, vals, out, eq, mult ); 
   
    int arrToMatch[11] = { 2, 2, 4, 2, 4, 8, 2, 4, 8, 16, 2 };

    // compare results
    cmpArrays<int,11>( arrToMatch, out );
}

TEST(ExclusiveScanByKey, Serial_OffsetExclFloatInplace)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;

        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    

	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}


TEST(ExclusiveScanByKey, MultiCore_OffsetExclFloat)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    
    bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),  refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}


TEST(InclusiveScanByKey, MultiCore_OffsetIncFloat)
{
    //setup keys
    int length = 1<<16;

    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;//identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());

    std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    /*bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3);
	// compare results
    cmpArrays(refOutput, output);*/

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3);
    // compare results
    cmpArrays(refInput, input);
}



TEST(ExclusiveScanByKey, Serial_OffsetExclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length, 1);

    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    
    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),  refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}


TEST(InclusiveScanByKey, Serial_OffsetIncFloat)
{
    //setup keys
    int length = 1<<16;

    std::vector< int > keys( length, 1);

    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;//identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::cl::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
    bolt::cl::control ctl;
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4),eq,mM3);
    //gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanByKeyCLtype, DeviceExclLong)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());

    std::vector< cl_long > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanByKeyCLtype, DeviceInclLong)
{
    //setup keys
    int length = 1<<14;

    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;//identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
    std::vector< cl_long > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_long > input( refInput.begin(), refInput.end() );
    // call scan

    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_long> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}


TEST(ScanByKeyCLtype, DeviceExclUlong)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ulong > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1+ rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanByKeyCLtype, DeviceInclUlong)
{
    //setup keys
    int length = 1<<14;

    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;//identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ulong > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ulong > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ulong> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	
	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanByKeyCLtype, DeviceExclShort)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(),  refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}


TEST(ScanByKeyCLtype, DeviceInclShort)
{
    //setup keys
    int length = 1<<14;

    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;//identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_short > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_short > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_short> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanByKeyCLtype, DeviceExclUShort)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(),  refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}


TEST(ScanByKeyCLtype, DeviceInclUShort)
{
    //setup keys
    int length = 1<<14;

    std::vector< int > keys( length, 1);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;//identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< cl_ushort > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::cl::device_vector< cl_ushort > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<cl_ushort> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);

}


/////////////////////////Inclusive//////////////////////////////////////////////////
#if (TEST_DOUBLE == 1)


TEST(ExclusiveScanByKey, OffsetExclUdd)
{
    //setup keys
    int length = 1<<16;
    std::vector< uddtM2 > keys( length, identityMixM2);
    
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3 );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length , identityMixM3);
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
    
    //bolt::cl::exclusive_scan_by_key(device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4), initialMixM3,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3, initialMixM3);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan_by_key(device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4), initialMixM3,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3, initialMixM3);
    // compare results
    cmpArrays(refInput, input);

}


TEST(InclusiveScanByKey, OffsetInclUdd)
{
    //setup keys
    int length = 1<<16;

    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    bolt::cl::device_vector< uddtM3 > input(  length, initialMixM3 );
    //bolt::cl::device_vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length,  identityMixM3);
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
    //bolt::cl::inclusive_scan_by_key(device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4),eq,mM3);
    //gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3);
    // compare results
    cmpArrays(refInput, input);
}


TEST(InclusiveScanByKey, Multicore_DeviceVectorInclUdd)
{
    //setup keys
    int length = 1<<16;

    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;

    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }

    bolt::cl::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    std::vector< uddtM2 > refInput( length, initialMixM2 );
    bolt::cl::device_vector< uddtM2 > input(  refInput.begin(), refInput.end() );

    MixM2 mM2;
    uddtM2_equal_to eq;
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also

    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM2);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2);
    // compare resultsScanByKeyCLtype
    cmpArrays(refInput, input);
}

#endif

TEST(InclusiveScanByKey, Multicore_DeviceVectorInclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);

    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    
    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::cl::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

TEST(InclusiveScanByKey, Serial_DeviceVectorInclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::cl::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

TEST(InclusiveScanByKey, MultiCore_DeviceVectorInclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
       std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::cl::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#if (TEST_DOUBLE == 1)
TEST(InclusiveScanByKey, Multicore_DeviceVectorInclDouble)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0 + rand()%3;
    }
	bolt::cl::device_vector< double > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<double> mM3; 
    // MixM3 mM3;
    // uddtM2_equal_to eq;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#endif


/////////////////////////Exclusive//////////////////////////////////////////////////

TEST(ExclusiveScanByKey, Multicore_DeviceVectorExclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	float n = 1.f + rand()%3;

    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::cl::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),  refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScanByKey, Serial_DeviceVectorExclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);

    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::cl::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),
    //                                                                                             4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScanByKey, MultiCore_DeviceVectorExclFloat)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	float n = 1.f + rand()%3;

    // input and output vectors for device and reference
   	std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::cl::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),
    //                                                                                             4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

#if (TEST_DOUBLE == 1)
TEST(ExclusiveScanByKey, Multicore_DeviceVectorExclDouble)
{
    //setup keys
    int length = 1<<16;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is just scan
        segmentIndex++;
    }
    bolt::cl::device_vector< int > device_keys( keys.begin(), keys.end());
    
	double n = 1.0 + rand()%3;

    // input and output vectors for device and reference
   	std::vector< double > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0 + rand()%3;
    }
	bolt::cl::device_vector< double > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<double> mM3; 
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 4.0,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}

TEST(ExclusiveScanByKey, Multicore_DeviceVectorExclUdd)
{
    //setup keys
    int length = 1<<16;
    std::vector< uddtM2 > keys( length, identityMixM2);
    
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    bolt::cl::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    std::vector< uddtM2 > refInput( length, initialMixM2 );
    bolt::cl::device_vector< uddtM2 > input(  refInput.begin(), refInput.end() );
    MixM2 mM2;
    uddtM2_equal_to eq;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), initialMixM3,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, initialMixM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), initialMixM2, eq, mM2);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2, initialMixM2);
    // compare results
    cmpArrays(refInput, input);

}

TEST(InclusiveScanByKey, MulticoreInclUdd)
{
    //setup keys
    int length = 1<<24;
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< uddtM3 > input(  length, initialMixM3 );
    //std::vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    //bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#endif

TEST(InclusiveScanByKey, InclFloat)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
         input[i] = 1.f + rand()%3;//2.0f;
         refInput[i] = input[i];//2.0f;
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
    ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
    bolt::cl::control ctl = bolt::cl::control::getDefault( );

    //bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}


TEST(InclusiveScanByKey,SerialInclFloat)
{
    //setup keys
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    //bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}


TEST(InclusiveScanByKey, MulticoreInclFloat)
{
    //setup keys
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
	for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) 
	{
        refInput[i]  = 1.0f + rand()%3;
    }	
    std::vector< float > input( refInput.begin(), refInput.end());
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);

}

#if (TEST_DOUBLE == 1)
TEST(InclusiveScanByKey, MulticoreInclDouble)
{
    //setup keys
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< double > input( length);
    //std::vector< double > output( length);
    std::vector< double > refInput( length);
    //std::vector< double > refOutput( length);
    for(int i=0; i<length; i++) {
		input[i] = 1.0 + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<double> mM3; 
    // MixM3 mM3;
    // uddtM2_equal_to eq;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#endif

TEST(ExclusiveScanByKey, ExclFloat)
{
    //setup keys
    int length = 1<<14;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);


    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );

    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}

TEST(ExclusiveScanByKey, SerialExclFloat)
{
    //setup keys
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }

    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
         input[i] = 1.f + rand()%3;
         refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}

TEST(APPSDKInclusiveScanByKey, Multicore_InclusiveIntAPPSDK)
{
    bool status = 1;
    int count = 1;
    int height = 1024;
    int width = 1024;
    int DataSize = height * width;
    std::vector<int> HostKeyBuffer(DataSize);
    std::vector<int> HostDataBuffer;
    std::vector<int> deviceOutput(DataSize);
    bolt::cl::device_vector<int> KeyBuffer(DataSize);
    bolt::cl::device_vector<int> dataBuffer(DataSize);
    // Assign key
    for(int i=0; i<height; ++i) {
        for(int j=0; j<width; ++j) {
            HostKeyBuffer[(i*width) + j] = i;
        }
    }

//	bolt::cl::control::e_RunMode specifiedRunMode = bolt::cl::control::SerialCpu;
    bolt::cl::control::e_RunMode specifiedRunMode = bolt::cl::control::MultiCoreCpu;
    bolt::cl::control::getDefault().setForceRunMode(specifiedRunMode);  // dissable this line, works fine with GPU/SerialCpu

while(status) {
    // Assign data
    HostDataBuffer.assign(DataSize, 1);

    // Copy key and data host to device
    {
        bolt::cl::device_vector<int>::pointer pk=KeyBuffer.data();
        memcpy(pk.get(), HostKeyBuffer.data(), DataSize * sizeof(int));

        bolt::cl::device_vector<int>::pointer pd=dataBuffer.data();
        memcpy(pd.get(), HostDataBuffer.data(), DataSize * sizeof(int));
    }

    bolt::cl::inclusive_scan_by_key(KeyBuffer.begin(),
                                    KeyBuffer.end(),
                                    dataBuffer.begin(),
                                    dataBuffer.begin());
    // copy data device to host
    {
        bolt::cl::device_vector<int>::pointer pd=dataBuffer.data();
        memcpy(deviceOutput.data(), pd.get(), DataSize * sizeof(int));
    }

    // CPU implementation
    for(int i=0; i<height; ++i) {
        for(int j=1; j<width; ++j) {
            HostDataBuffer[(i*width) + j] = HostDataBuffer[(i*width) + j] + HostDataBuffer[(i*width) + (j-1)];
        }
    }

    // Verify
    for(int i=0; i<DataSize; ++i) {
        if(HostDataBuffer[i] != deviceOutput[i]) { status = 0; break; }
    }

    //std::cout << "Verify Result : " << ((status == 1) ? "PASS" : "FAIL") << std::endl;
    if(++count > 20) break;
}
    
}



TEST(ExclusiveScanByKey, MulticoreExclFloat)
{
    //setup keys
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }


    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
         input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

#if (TEST_DOUBLE == 1)
TEST(ExclusiveScanByKey, MulticoreExclDouble)
{
    //setup keys
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is just scan
        segmentIndex++;
    }

    // input and output vectors for device and reference
    std::vector< double > input( length);
    //std::vector< double > output( length);
    std::vector< double> refInput( length);
    //std::vector< double > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.0 + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<double> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0] );
    // compare results
    cmpArrays(refInput, input);

}


TEST(ExclusiveScanByKey, MulticoreExclUdd)
{
    //setup keys
    int length = 1<<24;
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< uddtM3 > input(  length, initialMixM3 );
    //std::vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
   
    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM3,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, initialMixM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM3,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, initialMixM3);
    // compare results
    cmpArrays(refInput, input);

}
#endif

/////////////////////////////////////////////////CL Exclusive test Cases after fix ///////////////////////////

TEST(ExclusiveScanByKey, CLscanbykeyExclFloat)
{
    //setup keys
    int length = 1<<18;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }

    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 
  
    //bolt::cl::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScanByKey, SerialCLscanbykeyExclFloat)
{
    //setup keys
    int length = 1<<18;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }


    // input and output vectors for device and reference
    std::vector< float > input( length);
    std::vector< float > output( length);
    std::vector< float > refInput( length);
    std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
  
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

TEST(ExclusiveScanByKey, MultiCoreCLscanbykeyExclFloat)
{
    //setup keys
    int length = 1<<18;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is actually scan
        segmentIndex++;
    }

	//float n = 1.f + rand()%3;

    // input and output vectors for device and reference
    std::vector< float > input( length);
    //std::vector< float > output( length);
    std::vector< float > refInput( length);
    //std::vector< float > refOutput( length);
    for(int i=0; i<length; i++) {
        input[i] = 1.f + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<float> mM3; 

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
  
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}

#if (TEST_DOUBLE == 1)
TEST(ExclusiveScanByKey, CLscanbykeyExclDouble)
{
    //setup keys
    int length = 1<<18;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    for (int i = 0; i < length; i++)
    {
        if (segmentIndex == segmentLength)
        {
              segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key; // tested with key = 1 also which is just scan
        segmentIndex++;
    }

	//float n = 1.f + rand()%3;

    // input and output vectors for device and reference
    std::vector< double > input( length);
    //std::vector< double > output( length);
    std::vector< double> refInput( length);
    //std::vector< double > refOutput( length);
    for(int i=0; i<length; i++) {
         input[i] = 1.0 + rand()%3;
        refInput[i] = input[i];
    }
    // call scan
    bolt::cl::equal_to<int> eq; 
    bolt::cl::plus<double> mM3; 
    //bolt::cl::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), output.begin(), 4.0,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}
#endif

TEST(ExclusiveScanByKey, CLscanbykeyExclUDD)
{
    //setup keys
    int length = 1<<18;
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< uddtM2 > input(  length, initialMixM2 );
    //std::vector< uddtM2 > output( length, identityMixM2 );
    std::vector< uddtM2 > refInput( length, initialMixM2 );
    //std::vector< uddtM2 > refOutput( length );
    // call scan
    MixM2 mM2;
    uddtM2_equal_to eq;
    //bolt::cl::exclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM2,eq, mM2);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM2, initialMixM2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM2,eq, mM2);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2, initialMixM2);
    // compare results
    cmpArrays(refInput, input);

}

TEST(ExclusiveScanByKey, SerialCLscanbykeyExclUDD)
{
    //setup keys
    int length = 1<<18;
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< uddtM2 > input(  length, initialMixM2 );
   // std::vector< uddtM2 > output( length, identityMixM2 );
    std::vector< uddtM2 > refInput( length, initialMixM2 );
    //std::vector< uddtM2 > refOutput( length );
    // call scan
    MixM2 mM2;
    uddtM2_equal_to eq;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu); 
  
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM2,eq,mM2);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM2, initialMixM2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key( ctl, keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM2,eq, mM2);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2, initialMixM2);
    // compare results
    cmpArrays(refInput, input);
}


TEST(ExclusiveScanByKey, MulticoreCLscanbykeyExclUDD)
{
    //setup keys
    int length = 1<<18;
    std::vector< uddtM2 > keys( length, identityMixM2);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddtM2 key = identityMixM2;
    for (int i = 0; i < length; i++)
    {
        // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        segmentIndex++;
    }
    // input and output vectors for device and reference
    std::vector< uddtM2 > input(  length, initialMixM2 );
    //std::vector< uddtM2 > output( length, identityMixM2 );
    std::vector< uddtM2 > refInput( length, initialMixM2 );
    //std::vector< uddtM2 > refOutput( length );
    // call scan
    MixM2 mM2;
    uddtM2_equal_to eq;

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu); 
  
    //bolt::cl::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM2,eq,mM2);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM2, initialMixM2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::cl::exclusive_scan_by_key(ctl,  keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM2,eq, mM2);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2, initialMixM2);
    // compare results
    cmpArrays(refInput, input);
}

// paste from above
#endif

int _tmain(int argc, _TCHAR* argv[])
{
    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

    //  Initialize googletest; this removes googletest specific flags from command line
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    bool print_clInfo = false;
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;

    try
    {
        // Declare supported options below, describe what they do
        boost::program_options::options_description desc( "Scan GoogleTest command line options" );
        desc.add_options()
            ( "help,h",         "produces this help message" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "platform,p",     boost::program_options::value< cl_uint >( &userPlatform )->default_value( 0 ),
                                                                           "Specify the platform under test" )
            ( "device,d",       boost::program_options::value< cl_uint >( &userDevice )->default_value( 0 ),
                                                                           "Specify the device under test" )
            ;


        boost::program_options::variables_map vm;
        boost::program_options::store( boost::program_options::parse_command_line( argc, argv, desc ), vm );
        boost::program_options::notify( vm );

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
