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
#include "bolt/amp/bolt.h"
//#include "bolt/cl/scan.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/scan_by_key.h"
#include "bolt/unicode.h"
#include "bolt/miniDump.h"

#include "bolt/amp/iterator/counting_iterator.h"
#include "bolt/amp/iterator/constant_iterator.h"

#define SERIAL_TBB_OFFSET 1
#define TEST_LARGE_BUFFERS 1

#include <gtest/gtest.h>
//#include <boost/shared_array.hpp>
#include <boost/program_options.hpp>





/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Below are helper routines to compare the results of two arrays for googletest
//  They return an assertion object that googletest knows how to track

/******************************************************************************
 *  Double x4
 *****************************************************************************/


struct uddtD4
{
    float a;
    float b;

    uddtD4() restrict(cpu, amp) {}
    uddtD4(float x, float y) restrict(cpu, amp) : a(x), b(y) {}
    bool operator==(const uddtD4& rhs) const restrict (cpu,amp)
    {
        bool equal = true;
        double th = 0.0000000001f;
        if (rhs.a < th && rhs.a > -th)
            equal = ( (1.0f*a - rhs.a) < th && (1.0f*a - rhs.a) > -th) ? equal : false;
        else
            equal = ( (1.0f*a - rhs.a)/rhs.a < th && (1.0f*a - rhs.a)/rhs.a > -th) ? equal : false;
        if (rhs.b < th && rhs.b > -th)
            equal = ( (1.0f*b - rhs.b) < th && (1.0f*b - rhs.b) > -th) ? equal : false;
        else
            equal = ( (1.0f*b - rhs.b)/rhs.b < th && (1.0f*b - rhs.b)/rhs.b > -th) ? equal : false;
        return equal;
    }
    uddtD4 operator-() const restrict (cpu,amp)
    {
        uddtD4 r;
        r.a = -a;
        r.b = -b;
        return r;
    }
    uddtD4 operator*(const uddtD4& rhs) const restrict (cpu,amp)
    {
        uddtD4 r;
        r.a = a*a;
        r.b = b*b;
        return r;
    }
};
struct MultD4
{
    uddtD4 operator()(const uddtD4 &lhs, const uddtD4 &rhs) const restrict (cpu,amp)
    {
        uddtD4 _result;
        _result.a = lhs.a*rhs.a;
        _result.b = lhs.b*rhs.b;
        return _result;
    };
}; 

uddtD4 identityMultD4 = uddtD4( 1.0f, 1.0f);
uddtD4 initialMultD4  = uddtD4(1.00001f, 1.000003f );


/******************************************************************************
 *  Integer x2
 *****************************************************************************/
struct uddtI2
{
    int a;
    int b;

    uddtI2() restrict(cpu, amp) {}
    uddtI2(int x, int y) restrict(cpu, amp) : a(x), b(y) {}
    bool operator==(const uddtI2& rhs) const restrict (cpu,amp)
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        return equal;
    }
    uddtI2 operator-() const restrict (cpu,amp)
    {
        uddtI2 r;
        r.a = -a;
        r.b = -b;
        return r;
    }
    uddtI2 operator*(const uddtI2& rhs) const restrict (cpu,amp)
    {
        uddtI2 r;
        r.a = a*a;
        r.b = b*b;
        return r;
    }
};
struct AddI2
{
    uddtI2 operator()(const uddtI2 &lhs, const uddtI2 &rhs) const restrict (cpu,amp)
    {
        uddtI2 _result;
        _result.a = lhs.a+rhs.a;
        _result.b = lhs.b+rhs.b;
        return _result;
    };
}; 


uddtI2 identityAddI2 = uddtI2(  0, 0 );
uddtI2 initialAddI2  = uddtI2( -1, 2 );


/******************************************************************************
 *  Mixed float and int
 *****************************************************************************/

struct uddtM3
{
    int		 a;
    int      b;

    uddtM3() restrict(cpu, amp) {}
    uddtM3(int x, int y) restrict(cpu, amp) : a(x), b(y) {}
    bool operator==(const uddtM3& rhs) const restrict (cpu,amp)
    {
        bool equal = true;
        equal = ( a == rhs.a ) ? equal : false;
        equal = ( b == rhs.b ) ? equal : false;
        return equal;
    }
    uddtM3 operator-() const restrict (cpu,amp)
    {
        uddtM3 r;
        r.a = -a;
        r.b = -b;
        return r;
    }
    uddtM3 operator*(const uddtM3& rhs) const restrict (cpu,amp)
    {
        uddtM3 r;
        r.a = a*a;
        r.b = b*b;
        return r;
    }
};
struct MixM3
{
    uddtM3 operator()(const uddtM3 &lhs, const uddtM3 &rhs) const restrict (cpu,amp)
    {
        uddtM3 _result;
        _result.a = lhs.a^rhs.a;
        _result.b = lhs.b+rhs.b;
        return _result;
    };
}; 

uddtM3 identityMixM3 = uddtM3( 0, 0);
uddtM3 initialMixM3  = uddtM3( 2, 1);

struct uddtM2
{
    int a;
    int b;

    uddtM2() restrict(cpu, amp) : a(0),b(0) {}
    uddtM2(int x, int y) restrict(cpu, amp) : a(x),b(y) {}
    bool operator==(const uddtM2& rhs) const restrict (cpu,amp)
    {
        bool equal = true;
        float ths = 1; // thresh hold single(float)
        equal = ( a == rhs.a ) ? equal : false;
        if (rhs.b < ths && rhs.b > -ths)
            equal = ( (1*b - rhs.b) < ths && (1*b - rhs.b) > -ths) ? equal : false;
        else
            equal = ( (1*b - rhs.b)/rhs.b < ths && (1*b - rhs.b)/rhs.b > -ths) ? equal : false;
        return equal;
    }
    uddtM2 operator-() const restrict (cpu,amp)
    {
        uddtM2 r;
        r.a = -a;
        r.b = -b;
        return r;
    }
    uddtM2 operator*(const uddtM2& rhs) const restrict (cpu,amp)
    {
        uddtM2 r;
        r.a = a*a;
        r.b = b*b;
        return r;
    }
    void operator++()  restrict (cpu,amp)
    {
        a += 1;
        b += 2;
    }
};
struct MixM2
{
    uddtM2 operator()(const uddtM2 &lhs, const uddtM2 &rhs) const restrict (cpu,amp)
    {
        uddtM2 _result;
        _result.a = lhs.a^rhs.a;
        _result.b = lhs.b+rhs.b;
        return _result;
    };
}; 

uddtM2 identityMixM2 = uddtM2(0, 3 );
uddtM2 initialMixM2  = uddtM2( 2, 1 );


struct uddtM2_equal_to
{
    bool operator()(const uddtM2& lhs, const uddtM2& rhs) const restrict (cpu,amp)
    {
        return lhs == rhs;
    }
};


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
typedef scanByKeyStdVectorWithIters ScanByKeyAMPtypeTest;
/*
class StdVectCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    StdVectCountingIterator():mySize(GetParam()){
    }
};*/

TEST_P (ScanByKeyAMPtypeTest, InclTestLong)
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
    
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
   

    // input and output vectors for device and reference

    //bolt::amp::device_vector< long > output( myStdVectSize, 0 );
    std::vector< long > refInput( myStdVectSize);
    //std::vector< long > refOutput( myStdVectSize , 0);

	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;

	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyAMPtypeTest, SerialInclTestLong)
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
    
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;

	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

  
    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
          
} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyAMPtypeTest, MulticoreInclTestLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 
 #endif
TEST_P (ScanByKeyAMPtypeTest, InclTestunsignedLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
         
} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyAMPtypeTest, MulticoreInclTestunsignedLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 
#endif
TEST_P (ScanByKeyAMPtypeTest, ExclTestLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
    
	bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);

    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyAMPtypeTest, SerialExclTestLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),  refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
          
} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyAMPtypeTest, MulticoreExclTestLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),  refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
          
     
} 
 #endif
TEST_P (ScanByKeyAMPtypeTest, ExclTestunsignedLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyAMPtypeTest, SerialExclTestunsignedLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
          
} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyAMPtypeTest, MulticoreExclTestunsignedLong)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);
     
	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
} 
#endif
TEST_P (ScanByKeyAMPtypeTest, InclTestShort)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = (float)1 + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
         
} 
TEST_P (ScanByKeyAMPtypeTest, SerialInclTestShort)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
          
} 

TEST_P (ScanByKeyAMPtypeTest, InclTestUShort)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
         
} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyAMPtypeTest, MulticoreInclTestUShort)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector<double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
     
} 
TEST_P (ScanByKeyAMPtypeTest, MulticoreExclTestShort)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = (float)1 + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
     
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

     
} 
 #endif
TEST_P (ScanByKeyAMPtypeTest, ExclTestUShort)
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = (float)1 + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< int> input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<int> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

    std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<int> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if defined( ENABLE_TBB )
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<int> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#endif

TEST_P (ScanByKeyOffsetTest, SerialExclOffsetTestInt)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<int> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyOffsetTest, MulticoreExclOffsetTestInt)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<int> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	 bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#endif
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
    
    bolt::amp::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<float> eq; 
    bolt::amp::plus<float> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if defined( ENABLE_TBB )
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
    
    bolt::amp::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<float> eq; 
    bolt::amp::plus<float> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);
    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#endif
TEST_P (ScanByKeyOffsetTest, SerialExclOffsetTestFloat)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    
    bolt::amp::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<float> eq; 
    bolt::amp::plus<float> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyOffsetTest, MulticoreExclOffsetTestFloat)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    bolt::amp::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<float> eq; 
    bolt::amp::plus<float> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

}
#endif
#endif
TEST_P (ScanByKeyOffsetTest, ExclOffsetTestInt)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< int > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< int > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<int> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4),  refInput[0], eq, mM3);
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
    
    bolt::amp::device_vector< float > device_keys( keys.begin(), keys.end());

    std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<float> eq; 
    bolt::amp::plus<float> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
TEST_P (ScanByKeyOffsetTest, ExclOffsetTestFloat)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    bolt::amp::device_vector< float > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<float> eq; 
    bolt::amp::plus<float> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
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
    bolt::amp::device_vector< double  > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<double> eq; 
    bolt::amp::plus<double> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if defined( ENABLE_TBB )
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
    bolt::amp::device_vector< double  > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<double> eq; 
    bolt::amp::plus<double> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#endif
TEST_P (ScanByKeyOffsetTest, SerialExclOffsetTestDouble)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    bolt::amp::device_vector< double > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<double> eq; 
    bolt::amp::plus<double> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 
#if defined( ENABLE_TBB )
TEST_P (ScanByKeyOffsetTest, MulticoreExclOffsetTestDouble)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    
    bolt::amp::device_vector<double > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::amp::equal_to<double> eq; 
    bolt::amp::plus<double> mM3; 
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

}
#endif
#endif

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
    
    bolt::amp::device_vector<double > device_keys( keys.begin(), keys.end());
     
    std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<double> eq; 
    bolt::amp::plus<double> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, ExclOffsetTestDouble)
{
    //bolt::amp::device_vector< float > input( myStdVectSize, 2.f);
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
    bolt::amp::device_vector<double > device_keys( keys.begin(), keys.end());
      
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1.0 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<double> eq; 
    bolt::amp::plus<double> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

}


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
    
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
      
    
	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
    gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 


TEST_P (ScanByKeyOffsetTest, InclOffsetTestunsignedLong)
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
    
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
      
    std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
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
    
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = (float)1 + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
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
    
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< double> refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    
    //bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), eq, mM3);
    //gold_scan_by_key(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

TEST_P (ScanByKeyOffsetTest, ExclInclOffsetTestunsignedLong)
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());

	std::vector< unsigned long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());

	std::vector< double > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), output.begin() + (myStdVectSize/4), 2, eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refOutput.begin() + (myStdVectSize/4), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() + (myStdVectSize/4), device_keys.end() - (myStdVectSize/4), input.begin() + (myStdVectSize/4), input.begin() + (myStdVectSize/4), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive(keys.begin() + (myStdVectSize/4), keys.end() - (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), refInput.begin() + (myStdVectSize/4), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

    printf("\nPass for size=%d Offset=%d\n",myStdVectSize, myStdVectSize/2);

} 

INSTANTIATE_TEST_CASE_P(incl_excl_ScanByKeyIterIntLimit, ScanByKeyAMPtypeTest, ::testing::Range( 1, 1048576, 5987 )); 
INSTANTIATE_TEST_CASE_P(incl_excl_ScanByKeyIterIntLimit, ScanByKeyOffsetTest, ::testing::Range(1025, 65535, 555)); 


TEST(InclusiveScanByKey, IncMixedM3increment)
{
    //setup keys
    int length = (1<<25);
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
   /* bolt::amp::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);*/

	bolt::amp::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
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
    int length = (1<<20);
    uddtM2 key = uddtM2(1, 2);
    std::vector< uddtM2 > keys( length, key);

    // input and output vectors for device and reference
    std::vector< uddtM3 > input(  length, initialMixM3 );
    //std::vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );

    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
   /* bolt::amp::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);*/

	 bolt::amp::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
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
    int length = (1<<20);
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
    /*bolt::amp::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);*/

	bolt::amp::inclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
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


TEST( equalValMult, iValues )
{
    int keys[11] = { 7, 0, 0, 3, 3, 3, -5, -5, -5, -5, 3 }; 
    int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 }; 
    int out[11]; 
   
    bolt::amp::equal_to<int> eq; 
    bolt::amp::multiplies<int> mult; 
   
    bolt::amp::inclusive_scan_by_key( keys, keys+11, vals, out, eq, mult ); 
   
    int arrToMatch[11] = { 2, 2, 4, 2, 4, 8, 2, 4, 8, 16, 2 };

    // compare results
    cmpArrays<int,11>( arrToMatch, out );
}

TEST( equalValMult, Serial_iValues )
{
    int keys[11] = { 7, 0, 0, 3, 3, 3, -5, -5, -5, -5, 3 }; 
    int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 }; 
    int out[11]; 
   
    bolt::amp::equal_to<int> eq; 
    bolt::amp::multiplies<int> mult; 
   
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    bolt::amp::inclusive_scan_by_key(ctl, keys, keys+11, vals, out, eq, mult ); 
   
    int arrToMatch[11] = { 2, 2, 4, 2, 4, 8, 2, 4, 8, 16, 2 };

    // compare results
    cmpArrays<int,11>( arrToMatch, out );
}
#if defined( ENABLE_TBB )
TEST( equalValMult, MultiCore_iValues )
{
    int keys[11] = { 7, 0, 0, 3, 3, 3, -5, -5, -5, -5, 3 }; 
    int vals[11] = { 2, 2, 2, 2, 2, 2,  2,  2,  2,  2, 2 }; 
    int out[11]; 
   
    bolt::amp::equal_to<int> eq; 
    bolt::amp::multiplies<int> mult; 
   
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    bolt::amp::inclusive_scan_by_key(ctl, keys, keys+11, vals, out, eq, mult ); 
   
    int arrToMatch[11] = { 2, 2, 4, 2, 4, 8, 2, 4, 8, 16, 2 };

    // compare results
    cmpArrays<int,11>( arrToMatch, out );
}
#endif
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    

	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    
    bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

#if defined( ENABLE_TBB )
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    
    bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),  refInput[0],eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());

    std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    /*bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3);
	// compare results
    cmpArrays(refOutput, output);*/

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3);
    // compare results
    cmpArrays(refInput, input);
}
#endif


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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    
    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4), 2.f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3, 2.f);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),  refInput[0],eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1.f + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4),eq,mM3);
    //gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),eq,mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());

    std::vector< long > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(), refInput[0] ,eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
    std::vector< long > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );
    // call scan

    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}


TEST(ScanByKeyCLtype, DeviceExclunsignedLong)
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< unsigned long > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1+ rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}


TEST(ScanByKeyCLtype, DeviceInclunsignedLong)
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< unsigned long > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< unsigned long > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<unsigned long> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	
	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(),  refInput[0],eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = (float)1 + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< double > input( refInput.begin(), refInput.end() );


    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 2,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(device_keys.begin() , device_keys.end(), input.begin(), input.begin(),  refInput[0],eq, mM3);
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
    bolt::amp::device_vector<int> device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
	for(int i=0; i<length; i++)
		refInput[i] = (float)1 + rand()%3;
	bolt::amp::device_vector< float > input( refInput.begin(), refInput.end() );

    // call scan

    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  //  MixM3 mM3;
  //  uddtM2_equal_to eq;
   
    //bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);

}


/////////////////////////Inclusive//////////////////////////////////////////////////


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
    bolt::amp::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    bolt::amp::device_vector< uddtM3 > input(  length, initialMixM3 );
    //bolt::amp::device_vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length , identityMixM3);
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
    
    //bolt::amp::exclusive_scan_by_key(device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4), initialMixM3,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3, initialMixM3);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::amp::exclusive_scan_by_key(device_keys.begin() + (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4), initialMixM3,eq, mM3);
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
    bolt::amp::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    bolt::amp::device_vector< uddtM3 > input(  length, initialMixM3 );
    //bolt::amp::device_vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length,  identityMixM3);
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
    //bolt::amp::inclusive_scan_by_key(device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), output.begin()+ (length/4),eq,mM3);
    //gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refOutput.begin()+ (length/4), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(device_keys.begin()+ (length/4), device_keys.end()- (length/4), input.begin()+ (length/4), input.begin()+ (length/4),eq,mM3);
    gold_scan_by_key(keys.begin()+ (length/4), keys.end()- (length/4), refInput.begin()+ (length/4), refInput.begin()+ (length/4), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#if defined( ENABLE_TBB )
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

    bolt::amp::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    bolt::amp::device_vector< uddtM3 > input(  length, initialMixM3 );
    //bolt::amp::device_vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also

    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare resultsScanByKeyCLtype
    cmpArrays(refInput, input);
}


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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    
    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}
#endif
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
    // input and output vectors for device and reference
    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}
#if defined( ENABLE_TBB )
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
       std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),eq,mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< double > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0 + rand()%3;
    }
	bolt::amp::device_vector< double > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    // MixM3 mM3;
    // uddtM2_equal_to eq;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),eq,mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}



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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	float n = 1.f + rand()%3;

    std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(),  refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3,  refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}
#endif
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),
    //                                                                                             4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}
#if defined( ENABLE_TBB )
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	float n = 1.f + rand()%3;

    // input and output vectors for device and reference
   	std::vector< float > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0f + rand()%3;
    }
	bolt::amp::device_vector< float > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(),
    //                                                                                             4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());
    
	double n = 1.0 + rand()%3;

    // input and output vectors for device and reference
   	std::vector< double > refInput( length);
    for(int i=0; i<length; i++) {
        refInput[i] =  1.0 + rand()%3;
    }
	bolt::amp::device_vector< double > input(  refInput.begin(),  refInput.end());

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), 4.0,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0],eq, mM3);
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
    bolt::amp::device_vector< uddtM2 > device_keys( keys.begin(), keys.end());
    // input and output vectors for device and reference
    bolt::amp::device_vector< uddtM3 > input(  length, initialMixM3 );
    //bolt::amp::device_vector< uddtM3 > output( length, identityMixM3 );
    std::vector< uddtM3 > refInput( length, initialMixM3 );
    //std::vector< uddtM3 > refOutput( length );
    // call scan
    MixM3 mM3;
    uddtM2_equal_to eq;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), output.begin(), initialMixM3,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, initialMixM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), initialMixM3,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, initialMixM3);
    // compare results
    cmpArrays(refInput, input);

}
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
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

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
    bolt::amp::control ctl = bolt::amp::control::getDefault( );

    //bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#if TEST_LARGE_BUFFERS
TEST(InclusiveScanByKey,SerialInclFloat)
{
    //setup keys
    int length = 1<<26;
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}

#if defined( ENABLE_TBB )
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
        keys[i] = 1;
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);

}

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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    // MixM3 mM3;
    // uddtM2_equal_to eq;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), eq, mM3);
    //gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::inclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), eq, mM3);
    gold_scan_by_key(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3);
    // compare results
    cmpArrays(refInput, input);
}
#endif
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  
    bolt::amp::control ctl = bolt::amp::control::getDefault( );

    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}

#if TEST_LARGE_BUFFERS
TEST(ExclusiveScanByKey, SerialExclFloat)
{
    //setup keys
    int length = 1<<26;
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

    bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0] ,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}



#if defined( ENABLE_TBB )
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
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
   
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // tested for serial cpu also
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM3,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, initialMixM3);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM3,eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, initialMixM3);
    // compare results
    cmpArrays(refInput, input);

}
#endif
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 
  
    //bolt::amp::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
  
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}
#if defined( ENABLE_TBB )
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<float> mM3; 

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
  
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), 4.0f,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0f);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);

}
#endif
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
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<double> mM3; 
    //bolt::amp::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), output.begin(), 4.0,eq, mM3);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM3, 4.0);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(keys.begin(), keys.end(), input.begin(), input.begin(), input[0],eq, mM3);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);
    // compare results
    cmpArrays(refInput, input);
}

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
    //bolt::amp::exclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM2,eq, mM2);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM2, initialMixM2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key( keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM2,eq, mM2);
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

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 
  
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM2,eq,mM2);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM2, initialMixM2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key( ctl, keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM2,eq, mM2);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2, initialMixM2);
    // compare results
    cmpArrays(refInput, input);
}

#if defined( ENABLE_TBB )
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

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); 
  
    //bolt::amp::exclusive_scan_by_key(ctl, keys.begin(), keys.end(), input.begin(), output.begin(), initialMixM2,eq,mM2);
    //gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refOutput.begin(), mM2, initialMixM2);
    //// compare results
    //cmpArrays(refOutput, output);

	bolt::amp::exclusive_scan_by_key(ctl,  keys.begin(), keys.end(), input.begin(), input.begin(), initialMixM2,eq, mM2);
    gold_scan_by_key_exclusive(keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM2, initialMixM2);
    // compare results
    cmpArrays(refInput, input);
}
#endif
TEST(sanity_constant_iterator_amp_inclu_scan_by_key_int, inclu_scan_by_key_int){
    
    const int length = 1<<20;
//    const int length = 1<<5;

    int value = 100;
    std::vector< int > svInVec1( length );
    std::vector< int > svInVec2( length );
    
    std::vector< int > svOutVec( length );
    std::vector< int > stlOut( length );
    std::fill( svOutVec.begin( ), svOutVec.end( ), 100 );

    bolt::amp::device_vector< int > dvInVec1( length );
    bolt::amp::device_vector< int > dvOutVec( length );

    bolt::amp::constant_iterator<int> constIter1 (value);
    bolt::amp::constant_iterator<int> constIter2 (10);

    bolt::amp::plus<int> pls;
    bolt::amp::equal_to<int> eql;
    int n = (int) 1 + rand()%10;

            int segmentLength = 0;
            int segmentIndex = 0;
            std::vector<int> key(1);
            key[0] = 0;

            for (int i = 0; i < length; i++)
            {
                // start over, i.e., begin assigning new key
                if (segmentIndex == segmentLength)
                {
                    segmentLength++;
                    segmentIndex = 0;
                    key[0] = key[0]+1 ; // key[0]++  is not working in the device_vector
                }
                svInVec1[i] = key[0];
                dvInVec1[i] = svInVec1[i];
                segmentIndex++;
            }

   bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu); 

       //bolt::amp::inclusive_scan_by_key(constIter1, constIter1 + length, constIter1, svOutVec.begin(), eql, pls);
      bolt::amp::inclusive_scan_by_key( ctl, constIter1, constIter1 + length, constIter1, dvOutVec.begin(), eql, pls);
      gold_scan_by_key( svOutVec.begin( ), svOutVec.end( ), svOutVec.begin( ), stlOut.begin( ), pls );
    
       //	STD_INC_SCAN_BY_KEY
    std::vector<int> const_vector(length, value);

    for(int i =0; i< length; i++)
    {
        EXPECT_EQ( dvOutVec[i], stlOut[i]);
    }
}


TEST (ABB, ExclTestLong)
{
    int myStdVectSize = 257;
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
    bolt::amp::device_vector< int > device_keys( keys.begin(), keys.end());

	std::vector< long > refInput( myStdVectSize);
	for(int i=0; i<myStdVectSize; i++)
		refInput[i] = 1 + rand()%3;
	bolt::amp::device_vector< long > input( refInput.begin(), refInput.end() );

    // call scan
    bolt::amp::equal_to<int> eq; 
    bolt::amp::plus<long> mM3; 
    
    bolt::amp::control ctl;
    ctl.setForceRunMode( bolt::amp::control::SerialCpu );
	bolt::amp::exclusive_scan_by_key( ctl, device_keys.begin(), device_keys.end(), input.begin(), input.begin(), refInput[0], eq, mM3);
    gold_scan_by_key_exclusive( keys.begin(), keys.end(), refInput.begin(), refInput.begin(), mM3, refInput[0]);

    // compare results
    cmpArrays(refInput, input);
}


// paste from above


int _tmain(int argc, _TCHAR* argv[])
{
	
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;
    std::cout << "#######################################################################################" <<std::endl;

    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
    #if defined(_WIN32)
    bolt::miniDumpSingleton::enableMiniDumps( );
    #endif
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
