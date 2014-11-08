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

#define ENABLE_GTEST 1
#define ENABLE_DEBUGGING 0
#define UDD 1
#define TEST_DOUBLE 1
#include "common/stdafx.h"
#include "bolt/amp/reduce_by_key.h"
#include "bolt/amp/fill.h"
#include "bolt/unicode.h"
#include "bolt/amp/functional.h"

#include <gtest/gtest.h>
#include <type_traits>

#include "common/stdafx.h"
#include "common/test_common.h"
//#include "bolt/miniDump.h"

#include <array>
#include <cmath>
#include <algorithm>


#if !ENABLE_GTEST

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryFunction>
std::pair<OutputIterator1, OutputIterator2>
gold_reduce_by_key(
    InputIterator1 keys_first,
    InputIterator1 keys_last,
    InputIterator2 values_first,
    OutputIterator1 keys_output,
    OutputIterator2 values_output,
    BinaryFunction binary_op)
{
    typedef std::iterator_traits< InputIterator1 >::value_type kType;
    typedef std::iterator_traits< InputIterator2 >::value_type vType;
    typedef std::iterator_traits< OutputIterator1 >::value_type koType;
    typedef std::iterator_traits< OutputIterator2 >::value_type voType;
       static_assert( std::is_convertible< vType, voType >::value,
        "InputIterator2 and OutputIterator's value types are not convertible." );

   unsigned int numElements = static_cast< unsigned int >( std::distance( keys_first, keys_last ) );

    // do zeroeth element
    *values_output = *values_first;
    *keys_output = *keys_first;
    unsigned int count = 1;
    // rbk oneth element and beyond

    values_first++;
    for ( InputIterator1 key = (keys_first+1); key != keys_last; key++)
    {
        // load keys
        kType currentKey  = *(key);
        kType previousKey = *(key-1);

        // load value
        voType currentValue = *values_first;
        voType previousValue = *values_output; //Sure?: Damn sure

        previousValue = *values_output;
        // within segment
        if (currentKey == previousKey)
        {
            //std::cout << "continuing segment" << std::endl;
            voType r = binary_op( previousValue, currentValue);
            *values_output = r;
            *keys_output = currentKey;

        }
        else // new segment
        {
            values_output++;
            keys_output++;
            *values_output = currentValue;
            *keys_output = currentKey;
            count++; //To count the number of elements in the output array
        }
        values_first++;
    }

    return std::make_pair(keys_output, values_output);


}

int _tmain(int argc, _TCHAR* argv[])
{

    int length = 10000;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    //std::vector< int > refInput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);
    // call reduce_by_key

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;


    bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                             binary_predictor, binary_operator);
    
    gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(), vrefOutput.begin(),
                        std::plus<int>());

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);

    return 0;
}

#else

template<
    typename InputIterator1,
    typename InputIterator2,
    typename OutputIterator1,
    typename OutputIterator2,
    typename BinaryFunction>
std::pair<OutputIterator1, OutputIterator2>
gold_reduce_by_key( InputIterator1 keys_first,
                    InputIterator1 keys_last,
                    InputIterator2 values_first,
                    OutputIterator1 keys_output,
                    OutputIterator2 values_output,
                    BinaryFunction binary_op )
{
    typedef typename std::iterator_traits< InputIterator1 >::value_type kType;
    typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
    typedef typename std::iterator_traits< OutputIterator1 >::value_type koType;
    typedef typename std::iterator_traits< OutputIterator2 >::value_type voType;
    static_assert( std::is_convertible< vType, voType >::value,
                   "InputIterator2 and OutputIterator's value types are not convertible." );

   unsigned int numElements = static_cast< unsigned int >( std::distance( keys_first, keys_last ) );

    // do zeroeth element
    *values_output = *values_first;
    *keys_output = *keys_first;
    unsigned int count = 1;
    // rbk oneth element and beyond

    values_first++;
    for ( InputIterator1 key = (keys_first+1); key != keys_last; key++)
    {
        // load keys
        kType currentKey  = *(key);
        kType previousKey = *(key-1);

        // load value
        voType currentValue = *values_first;
        voType previousValue = *values_output;

        previousValue = *values_output;
        // within segment
        if (currentKey == previousKey)
        {
            voType r = binary_op( previousValue, currentValue);
            *values_output = r;
            *keys_output = currentKey;

        }
        else // new segment
        {
            values_output++;
            keys_output++;
            *values_output = currentValue;
            *keys_output = currentKey;
            count++; //To count the number of elements in the output array
        }
        values_first++;
    }

    //std::cout<<count<<std::endl;
    return std::make_pair(keys_output+1, values_output+1);
}

class reduceStdVectorWithIters:public ::testing::TestWithParam<int>
{
protected:
    int myStdVectSize;
public:
    reduceStdVectorWithIters():myStdVectSize(GetParam()){
    }
};

typedef reduceStdVectorWithIters ReduceByKeyTest;

TEST_P (ReduceByKeyTest, ReduceByKeyTestFloat)
{

    int length = myStdVectSize;
    std::vector< float > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.0f;
    std::vector< float > refInput( length );
    std::vector< float > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%4 == 1) key++;
        keys[i] = key;
        refInput[i] = (float)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< float > koutput( length );
    std::vector< float > voutput( length );
    std::vector< float > krefOutput( length );
    std::vector< float > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0f);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0f);
    
    bolt::amp::equal_to<float> binary_predictor;
    bolt::amp::plus<float> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<float>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}

TEST_P (ReduceByKeyTest, ReduceByKeyTestFloatIncreasingKeys)
{

    int length = myStdVectSize;
    std::vector< float > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 0.0f;
    std::vector<float>  refInput( length );
    std::vector<float>  input( length );
    //std::vector<int>  input( length );

    for (int i = 0; i < length; i++)
    { // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        //device_keys[i] = key;
        segmentIndex++;

        refInput[i] = 3.0f;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< float > koutput( length );
    std::vector< float > voutput( length );
    std::vector< float > krefOutput( length );
    std::vector< float > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0f);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0f);
    
    bolt::amp::equal_to<float> binary_predictor;
    bolt::amp::plus<float> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<float>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, ReduceByKeyTestFloat3)
{

    int length = myStdVectSize;
    std::vector< float > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.0f;
    std::vector< float > refInput( length );
    std::vector< float > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = (float)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< float > koutput( length );
    std::vector< float > voutput( length );
    std::vector< float > krefOutput( length );
    std::vector< float > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0f);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0f);
    
    bolt::amp::equal_to<float> binary_predictor;
    bolt::amp::plus<float> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<float>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, SameKeyReduceByKeyTestFloat)
{

    int length = myStdVectSize;
    std::vector< float > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.0f;
    std::vector< float > refInput( length );
    std::vector< float > input( length );
    for (int i = 0; i < length; i++)
    {
        keys[i] = 1;
        refInput[i] = (float)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< float > koutput( length );
    std::vector< float > voutput( length );
    std::vector< float > krefOutput( length );
    std::vector< float > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0f);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0f);
    
    bolt::amp::equal_to<float> binary_predictor;
    bolt::amp::plus<float> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<float>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, DifferentKeyReduceByKeyTestFloat)
{

    int length = myStdVectSize;
    std::vector< float > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.0f;
    std::vector< float > refInput( length );
    std::vector< float > input( length );
    for (int i = 0; i < length; i++)
    {
        keys[i] = (float)i;
        refInput[i] = (float)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< float > koutput( length );
    std::vector< float > voutput( length );
    std::vector< float > krefOutput( length );
    std::vector< float > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0f);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0f);
    
    bolt::amp::equal_to<float> binary_predictor;
    bolt::amp::plus<float> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<float>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}

TEST_P (ReduceByKeyTest, DifferentKeyReduceByKeyTestFloatDevice)
{

    int length = myStdVectSize;
    bolt::amp::device_vector< float > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    float key = 1.0f;
    bolt::amp::device_vector< float > refInput( length );
    bolt::amp::device_vector< float > input( length );
    for (int i = 0; i < length; i++)
    {
        keys[i] = (float)i;
        refInput[i] = (float)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    bolt::amp::device_vector< float > koutput( length );
    bolt::amp::device_vector< float > voutput( length );
    bolt::amp::device_vector< float > krefOutput( length );
    bolt::amp::device_vector< float > vrefOutput( length );
    bolt::amp::fill(krefOutput.begin(),krefOutput.end(),0.0f);
    bolt::amp::fill(vrefOutput.begin(),vrefOutput.end(),0.0f);
    
    bolt::amp::equal_to<float> binary_predictor;
    bolt::amp::plus<float> binary_operator;


	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

	auto refPair = bolt::amp::reduce_by_key(ctl, keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(), vrefOutput.begin(),
                                      binary_predictor, binary_operator);

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}




#if(TEST_DOUBLE==1)
TEST_P (ReduceByKeyTest, ReduceByKeyTestDouble)
{

    int length = myStdVectSize;
    std::vector< double > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    std::vector< double > refInput( length );
    std::vector< double > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%4 == 1) key++;
        keys[i] = key;
        refInput[i] = (double)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< double > koutput( length );
    std::vector< double > voutput( length );
    std::vector< double > krefOutput( length );
    std::vector< double > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0);
    
    bolt::amp::equal_to<double> binary_predictor;
    bolt::amp::plus<double> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<double>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, ReduceByKeyTestDoubleIncreasingKeys)
{

    int length = myStdVectSize;
    std::vector< double > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 0.0;
    std::vector<double>  refInput( length );
    std::vector<double>  input( length );
    //std::vector<int>  input( length );

    for (int i = 0; i < length; i++)
    { // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key;
        }
        keys[i] = key;
        //device_keys[i] = key;
        segmentIndex++;

        refInput[i] = 3.0;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< double > koutput( length );
    std::vector< double > voutput( length );
    std::vector< double > krefOutput( length );
    std::vector< double > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0);
    
    bolt::amp::equal_to<double> binary_predictor;
    bolt::amp::plus<double> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<double>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, ReduceByKeyTestDouble3)
{

    int length = myStdVectSize;
    std::vector< double > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0f;
    std::vector< double > refInput( length );
    std::vector< double > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = (double)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< double > koutput( length );
    std::vector< double > voutput( length );
    std::vector< double > krefOutput( length );
    std::vector< double > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0);
    
    bolt::amp::equal_to<double> binary_predictor;
    bolt::amp::plus<double> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<double>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, SameKeyReduceByKeyTestDouble)
{

    int length = myStdVectSize;
    std::vector< double > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    std::vector< double > refInput( length );
    std::vector< double > input( length );
    for (int i = 0; i < length; i++)
    {
        keys[i] = 1;
        refInput[i] = (double)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< double > koutput( length );
    std::vector< double > voutput( length );
    std::vector< double > krefOutput( length );
    std::vector< double > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0);
    
    bolt::amp::equal_to<double> binary_predictor;
    bolt::amp::plus<double> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<double>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}

TEST_P (ReduceByKeyTest, SameKeyReduceByKeyTestDoubleDevice)
{

    int length = myStdVectSize;
    bolt::amp::device_vector< double > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    bolt::amp::device_vector< double > refInput( length );
    bolt::amp::device_vector< double > input( length );
    for (int i = 0; i < length; i++)
    {
        keys[i] = 1;
        refInput[i] = (double)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    bolt::amp::device_vector< double > koutput( length );
    bolt::amp::device_vector< double > voutput( length );
    bolt::amp::device_vector< double > krefOutput( length );
    bolt::amp::device_vector< double > vrefOutput( length );
    bolt::amp::fill(krefOutput.begin(),krefOutput.end(),0.0);
    bolt::amp::fill(vrefOutput.begin(),vrefOutput.end(),0.0);
    
    bolt::amp::equal_to<double> binary_predictor;
    bolt::amp::plus<double> binary_operator;

	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);


    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

	auto refPair = bolt::amp::reduce_by_key(ctl, keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(), vrefOutput.begin(),
                                      binary_predictor, binary_operator);

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}



TEST_P (ReduceByKeyTest, DifferentKeyReduceByKeyTestDouble)
{

    int length = myStdVectSize;
    std::vector< double > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    double key = 1.0;
    std::vector< double > refInput( length );
    std::vector< double > input( length );
    for (int i = 0; i < length; i++)
    {
        keys[i] = (double)i;
        refInput[i] = (double)(std::rand()%4);
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< double > koutput( length );
    std::vector< double > voutput( length );
    std::vector< double > krefOutput( length );
    std::vector< double > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0.0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0.0);
    
    bolt::amp::equal_to<double> binary_predictor;
    bolt::amp::plus<double> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<double>());
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
#endif

struct uddfltint
{
    float x;
    int y;

    bool operator==(const uddfltint& rhs) const restrict(amp, cpu)
    {
        bool equal = true;
        float ths = 0.00001f; // thresh hold single(float)
        equal = ( x == rhs.x ) ? equal : false;
        if (rhs.y < ths && rhs.y > -ths)
            equal = ( (1.0f*y - rhs.y) < ths && (1.0f*y - rhs.y) > -ths) ? equal : false;
        else
            equal = ( (1.0f*y - rhs.y)/rhs.y < ths && (1.0f*y - rhs.y)/rhs.y > -ths) ? equal : false;
        return equal;
    }

	 bool operator&&(const uddfltint& rhs) const restrict(amp, cpu)
    {
        bool res = true;
        res = ( (x && rhs.x ) && (y && rhs.y)) ? res : false;
        return res;
    }

    void operator++(int) restrict(amp, cpu)
    {
        x += (float)1.0f;
        y += 1;
    }

    bool operator>(int rhs) const restrict(amp, cpu)
    {

      bool greater = true;
      greater = ( x > rhs ) ? greater : false;
      greater = ( y > rhs ) ? greater : false;
      return greater;

    }
    void operator=(int rhs) restrict(amp, cpu)
    {
      x = (float)rhs;
      y = rhs;
    }

    bool operator!=(int rhs) const restrict(amp, cpu)
    {
        bool nequal = true;
        float ths = 0.00001f; // thresh hold single(float)
        nequal = ( x != rhs ) ? nequal : false;
        if (rhs < ths && rhs > -ths)
            nequal = ( (1.0f*y - rhs) < ths && (1.0f*y - rhs) > -ths) ? false : nequal;
        else
            nequal = ( (1.0f*y - rhs)/rhs < ths && (1.0f*y - rhs)/rhs > -ths) ? false : nequal;
        return nequal;
    }

    void operator-(int rhs) restrict(amp, cpu)
    {
        x -= (float)rhs;
        y -= rhs;
    }
};


struct uddfltint_equal_to
{
    bool operator()(const uddfltint& lhs, const uddfltint& rhs) const restrict(amp, cpu)
    {
        return lhs == rhs;
    };
};


struct uddfltint_plus
{
    uddfltint operator()(const uddfltint &lhs, const uddfltint &rhs) const restrict(amp, cpu)
    {
        uddfltint _result;
        _result.x = lhs.x+rhs.x;
        _result.y = lhs.y+rhs.y;
        return _result;
    }
};


#if UDD
TEST_P (ReduceByKeyTest, ReduceByKeyTestUDD)
{

    int length = myStdVectSize;

    std::vector< uddfltint > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%5 == 1) key++;
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       binary_operator);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, ReduceByKeyTestUDDIncreasingKeys)
{

    int length = myStdVectSize;

    std::vector< uddfltint > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    { // start over, i.e., begin assigning new key
        if (segmentIndex == segmentLength)
        {
            segmentLength++;
            segmentIndex = 0;
            ++key.x;
            ++key.y;
        }
        keys[i] = key;
        //device_keys[i] = key;
        segmentIndex++;

        refInput[i].x = 3.0f;
        refInput[i].y = 3;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       binary_operator);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, ReduceByKeyTestUDD3)
{

    int length = myStdVectSize;

     std::vector< uddfltint > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

   
    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       binary_operator);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, SameKeyReduceByKeyTestUDD)
{

    int length = myStdVectSize;

    std::vector< uddfltint > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);


    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       binary_operator);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
TEST_P (ReduceByKeyTest, DifferentKeyReduceByKeyTestUDD)
{

    int length = myStdVectSize;

    
    std::vector< uddfltint > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
  
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        keys[i].x = (float)i;
        keys[i].y = i;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;


    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);


    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                        binary_operator);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);


}
#endif

INSTANTIATE_TEST_CASE_P(ReduceByKeyIterLimit, ReduceByKeyTest, ::testing::Range(1024, 262144, 9999)); 

//Test Case Not Executing Anything!

 TEST(ReduceByKeyBasic, DeviceVectorTest)
{
    int length = 10;
    std::vector< int > keys(length);
    

    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    std::vector<int>  refInput( length );
    
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

        refInput[i] = i;
    }

    bolt::amp::device_vector<int>  input( refInput.begin(), refInput.end() );
    bolt::amp::device_vector<int> device_keys(keys.begin(), keys.end());
    
    // input and output vectors for device and reference
	std::vector<int>  std_koutput( length );
	std::vector<int>  std_voutput( length );

    bolt::amp::device_vector<int>  koutput( std_koutput.begin(), std_koutput.end() );
    bolt::amp::device_vector<int>  voutput( std_voutput.begin(), std_voutput.end() );
    std::vector<int>  krefOutput( length );
    std::vector<int>  vrefOutput( length );

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( device_keys.begin(), device_keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                      std::plus<int>());


    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
} 
 TEST(ReduceByKeyBasic, CPUDeviceVectorTest)
{
    int length = 1<<10;
    std::vector< int > keys(length);


    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    std::vector<int>  refInput( length );
    

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

        refInput[i] = i;
    }

    bolt::amp::device_vector<int>  input( refInput.begin(), refInput.end() );
    bolt::amp::device_vector<int> device_keys(keys.begin(), keys.end());
    
    // input and output vectors for device and reference

    std::vector<int>  std_koutput( length );
	std::vector<int>  std_voutput( length );

    bolt::amp::device_vector<int>  koutput( std_koutput.begin(), std_koutput.end() );
    bolt::amp::device_vector<int>  voutput( std_voutput.begin(), std_voutput.end() );

    std::vector<int>  krefOutput( length );
    std::vector<int>  vrefOutput( length );

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( ctl, device_keys.begin(), device_keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                      std::plus<int>());


    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
}
#if defined( ENABLE_TBB )
 TEST(ReduceByKeyBasic, MultiCoreCPUDeviceVectorTest)
{
    //for(int i=15; i<20; i++)
	//{
    //printf("\n i = %d\n", i);

	int length = 1<<15;
    std::vector< int > keys(length);

    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    std::vector<int>  refInput( length );

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

        refInput[i] = i;
    }

    // input and output vectors for device and reference

    bolt::amp::device_vector<int>  input( refInput.begin(), refInput.end() );
    bolt::amp::device_vector<int> device_keys(keys.begin(), keys.end());
    
    std::vector<int>  std_koutput( length );
	std::vector<int>  std_voutput( length );

    bolt::amp::device_vector<int>  koutput( std_koutput.begin(), std_koutput.end() );
    bolt::amp::device_vector<int>  voutput( std_voutput.begin(), std_voutput.end() );

    std::vector<int>  krefOutput( length );
    std::vector<int>  vrefOutput( length );

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( ctl, device_keys.begin(), device_keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                      std::plus<int>());


    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
	//}
}
#endif

TEST(reduce_by_key__bolt_Std_vect, Basic_EPR377067){

int size = 10;
std::vector<int> vectKeyIn(size);
std::vector<int> vectValueIn(size);
std::vector<int> keyBoltAmpDevVectOp(size);
std::vector<int> valueBoltAmpDevVectOp(size);

for (int i = 0; i < std::ceil(size/3.0); i++)
{
    vectKeyIn[i] = (int)2;
}
for (int i = int(std::ceil(size/3.0) + 1); i < std::ceil((2* size)/3.0); i++)
{
vectKeyIn[i] = (int)3;
}
for (int i =  int(std::ceil((2* size)/3.0) + 1); i < size; i++)
{
vectKeyIn[i] = (int)5;
}
//now elemetns in vectKeyIn are as: {2 2 2 2 0 3 3 0 5 5 }

for (int i = 0; i < size; i++){
vectValueIn[i] = (int) i; //elements in vectValueIn are as: {0 1 2 3 4 5 6 7 8 9}
}

bolt::amp::equal_to<int> eq;
bolt::amp::reduce_by_key(vectKeyIn.begin(), vectKeyIn.end(), vectValueIn.begin(), keyBoltAmpDevVectOp.begin(),
                                                                        valueBoltAmpDevVectOp.begin(), eq);

int eleKeyOp_Expexted[5] = {2, 0, 3, 0, 5};
int eleValueOp_Expexted[5] = {6, 4, 11, 7, 17};

for (int i = 0; i < 5; i++){
EXPECT_EQ ( eleKeyOp_Expexted[i], keyBoltAmpDevVectOp[i]);
EXPECT_EQ ( eleValueOp_Expexted[i], valueBoltAmpDevVectOp[i]);
}
}
TEST(reduce_by_key__bolt_Std_vect, SerialBasic_EPR377067){

int size = 10;
std::vector<int> vectKeyIn(size);
std::vector<int> vectValueIn(size);
std::vector<int> keyBoltAmpDevVectOp(size);
std::vector<int> valueBoltAmpDevVectOp(size);

for (int i = 0; i < std::ceil(size/3.0); i++)
{
    vectKeyIn[i] = (int)2;
}
for (int i = int(std::ceil(size/3.0) + 1); i < std::ceil((2* size)/3.0); i++)
{
vectKeyIn[i] = (int)3;
}
for (int i =  int(std::ceil((2* size)/3.0) + 1); i < size; i++)
{
vectKeyIn[i] = (int)5;
}
//now elemetns in vectKeyIn are as: {2 2 2 2 0 3 3 0 5 5 }

for (int i = 0; i < size; i++){
vectValueIn[i] = (int) i; //elements in vectValueIn are as: {0 1 2 3 4 5 6 7 8 9}
}

bolt::amp::control ctl = bolt::amp::control::getDefault( );
ctl.setForceRunMode(bolt::amp::control::SerialCpu);

bolt::amp::equal_to<int> eq;
bolt::amp::reduce_by_key(ctl, vectKeyIn.begin(), vectKeyIn.end(), vectValueIn.begin(), keyBoltAmpDevVectOp.begin(),
                                                                        valueBoltAmpDevVectOp.begin(), eq);

int eleKeyOp_Expexted[5] = {2, 0, 3, 0, 5};
int eleValueOp_Expexted[5] = {6, 4, 11, 7, 17};

for (int i = 0; i < 5; i++){
EXPECT_EQ ( eleKeyOp_Expexted[i], keyBoltAmpDevVectOp[i]);
EXPECT_EQ ( eleValueOp_Expexted[i], valueBoltAmpDevVectOp[i]);
}
}
#if defined( ENABLE_TBB )
TEST(reduce_by_key__bolt_Std_vect, MulticoreBasic_EPR377067){

int size = 10;
std::vector<int> vectKeyIn(size);
std::vector<int> vectValueIn(size);
std::vector<int> keyBoltAmpDevVectOp(size);
std::vector<int> valueBoltAmpDevVectOp(size);

for (int i = 0; i < std::ceil(size/3.0); i++)
{
    vectKeyIn[i] = (int)2;
}
for (int i = int(std::ceil(size/3.0) + 1); i < std::ceil((2* size)/3.0); i++)
{
vectKeyIn[i] = (int)3;
}
for (int i =  int(std::ceil((2* size)/3.0) + 1); i < size; i++)
{
vectKeyIn[i] = (int)5;
}
//now elemetns in vectKeyIn are as: {2 2 2 2 0 3 3 0 5 5 }

for (int i = 0; i < size; i++){
vectValueIn[i] = (int) i; //elements in vectValueIn are as: {0 1 2 3 4 5 6 7 8 9}
}

bolt::amp::control ctl = bolt::amp::control::getDefault( );
ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);


bolt::amp::equal_to<int> eq;
bolt::amp::reduce_by_key(ctl, vectKeyIn.begin(), vectKeyIn.end(), vectValueIn.begin(), keyBoltAmpDevVectOp.begin(),
                                                                        valueBoltAmpDevVectOp.begin(), eq);

int eleKeyOp_Expexted[5] = {2, 0, 3, 0, 5};
int eleValueOp_Expexted[5] = {6, 4, 11, 7, 17};

for (int i = 0; i < 5; i++){
EXPECT_EQ ( eleKeyOp_Expexted[i], keyBoltAmpDevVectOp[i]);
EXPECT_EQ ( eleValueOp_Expexted[i], valueBoltAmpDevVectOp[i]);
}
}
#endif

TEST(reduce_by_key__bolt_Dev_vect, Basic_EPR377067){

int size = 10;
std::vector<int> KeyIn(size);
std::vector<int> ValueIn(size);

std::vector<int> stdvec(size);
bolt::amp::device_vector<int> keyBoltAmpDevVectOp(10);//stdvec.begin(), stdvec.end());
bolt::amp::device_vector<int> valueBoltAmpDevVectOp(10);//stdvec.begin(), stdvec.end());

for (int i = 0; i < std::ceil(size/3.0); i++){
KeyIn[i] = (int)2;
}
for (int i = (int)(std::ceil(size/3.0) + 1); i < std::ceil((2* size)/3.0); i++){
KeyIn[i] = (int)3;
}
for (int i = (int)(std::ceil((2* size)/3.0) + 1); i < size; i++){
KeyIn[i] = (int)5;
}
//now elemetns in vectKeyIn are as: {2 2 2 2 0 3 3 0 5 5 }

for (int i = 0; i < size; i++){
ValueIn[i] = (int) i; //elements in vectValueIn are as: {0 1 2 3 4 5 6 7 8 9}
}

bolt::amp::device_vector<int> vectKeyIn(KeyIn.begin(), KeyIn.end());
bolt::amp::device_vector<int> vectValueIn(ValueIn.begin(), ValueIn.end());

bolt::amp::equal_to<int> eq;
bolt::amp::reduce_by_key(vectKeyIn.begin(), vectKeyIn.end(), vectValueIn.begin(), keyBoltAmpDevVectOp.begin(),
                                                                            valueBoltAmpDevVectOp.begin(), eq);

int eleKeyOp_Expexted[5] = {2, 0, 3, 0, 5};
int eleValueOp_Expexted[5] = {6, 4, 11, 7, 17};

for (int i = 0; i < 5; i++){
EXPECT_EQ ( eleKeyOp_Expexted[i], keyBoltAmpDevVectOp[i]);
EXPECT_EQ ( eleValueOp_Expexted[i], valueBoltAmpDevVectOp[i]);
}
}
TEST(reduce_by_key__bolt_Dev_vect, SerialBasic_EPR377067){

int size = 10;
std::vector<int> KeyIn(size);
std::vector<int> ValueIn(size);

std::vector<int> stdvec(size);
bolt::amp::device_vector<int> keyBoltAmpDevVectOp(10);//stdvec.begin(), stdvec.end());
bolt::amp::device_vector<int> valueBoltAmpDevVectOp(10);//stdvec.begin(), stdvec.end());

for (int i = 0; i < std::ceil(size/3.0); i++){
KeyIn[i] = (int)2;
}
for (int i = (int)(std::ceil(size/3.0) + 1); i < std::ceil((2* size)/3.0); i++){
KeyIn[i] = (int)3;
}
for (int i = (int)(std::ceil((2* size)/3.0) + 1); i < size; i++){
KeyIn[i] = (int)5;
}
//now elemetns in vectKeyIn are as: {2 2 2 2 0 3 3 0 5 5 }

for (int i = 0; i < size; i++){
ValueIn[i] = (int) i; //elements in vectValueIn are as: {0 1 2 3 4 5 6 7 8 9}
}

bolt::amp::device_vector<int> vectKeyIn(KeyIn.begin(), KeyIn.end());
bolt::amp::device_vector<int> vectValueIn(ValueIn.begin(), ValueIn.end());

bolt::amp::control ctl = bolt::amp::control::getDefault( );
ctl.setForceRunMode(bolt::amp::control::SerialCpu);

bolt::amp::equal_to<int> eq;
bolt::amp::reduce_by_key(ctl, vectKeyIn.begin(), vectKeyIn.end(), vectValueIn.begin(), keyBoltAmpDevVectOp.begin(),
                                                                            valueBoltAmpDevVectOp.begin(), eq);

int eleKeyOp_Expexted[5] = {2, 0, 3, 0, 5};
int eleValueOp_Expexted[5] = {6, 4, 11, 7, 17};

for (int i = 0; i < 5; i++){
EXPECT_EQ ( eleKeyOp_Expexted[i], keyBoltAmpDevVectOp[i]);
EXPECT_EQ ( eleValueOp_Expexted[i], valueBoltAmpDevVectOp[i]);
}
}
#if defined( ENABLE_TBB )
TEST(reduce_by_key__bolt_Dev_vect, MulticoreBasic_EPR377067){

int size = 10;
std::vector<int> KeyIn(size);
std::vector<int> ValueIn(size);

std::vector<int> stdvec(size);
bolt::amp::device_vector<int> keyBoltAmpDevVectOp(10);//stdvec.begin(), stdvec.end());
bolt::amp::device_vector<int> valueBoltAmpDevVectOp(10);//stdvec.begin(), stdvec.end());

for (int i = 0; i < std::ceil(size/3.0); i++){
KeyIn[i] = (int)2;
}
for (int i = (int)(std::ceil(size/3.0) + 1); i < std::ceil((2* size)/3.0); i++){
KeyIn[i] = (int)3;
}
for (int i = (int)(std::ceil((2* size)/3.0) + 1); i < size; i++){
KeyIn[i] = (int)5;
}
//now elemetns in vectKeyIn are as: {2 2 2 2 0 3 3 0 5 5 }

for (int i = 0; i < size; i++){
ValueIn[i] = (int) i; //elements in vectValueIn are as: {0 1 2 3 4 5 6 7 8 9}
}

bolt::amp::device_vector<int> vectKeyIn(KeyIn.begin(), KeyIn.end());
bolt::amp::device_vector<int> vectValueIn(ValueIn.begin(), ValueIn.end());

bolt::amp::control ctl = bolt::amp::control::getDefault( );
ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

bolt::amp::equal_to<int> eq;
bolt::amp::reduce_by_key(ctl, vectKeyIn.begin(), vectKeyIn.end(), vectValueIn.begin(), keyBoltAmpDevVectOp.begin(),
                                                                            valueBoltAmpDevVectOp.begin(), eq);

int eleKeyOp_Expexted[5] = {2, 0, 3, 0, 5};
int eleValueOp_Expexted[5] = {6, 4, 11, 7, 17};

for (int i = 0; i < 5; i++){
EXPECT_EQ ( eleKeyOp_Expexted[i], keyBoltAmpDevVectOp[i]);
EXPECT_EQ ( eleValueOp_Expexted[i], valueBoltAmpDevVectOp[i]);
}
}
#endif
TEST(ReduceByKeyBasic, IntegerTest)
{
    int length = 1<<23;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;


    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<int>());

    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}

TEST(ReduceByKeyBasic, DeviceVectorOffsetTest)
{
    int length = 1<<25;
    std::vector< int > keys(length);
    

    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    std::vector<int>  refInput( length );
    
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

        refInput[i] = i;
    }

    bolt::amp::device_vector<int>  input( refInput.begin(), refInput.end() );
    bolt::amp::device_vector<int> device_keys(keys.begin(), keys.end());
    
    // input and output vectors for device and reference
	std::vector<int>  std_koutput( length );
	std::vector<int>  std_voutput( length );

    bolt::amp::device_vector<int>  koutput( std_koutput.begin(), std_koutput.end() );
    bolt::amp::device_vector<int>  voutput( std_voutput.begin(), std_voutput.end() );
    std::vector<int>  krefOutput( length );
    std::vector<int>  vrefOutput( length );

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    // call reduce_by_key

    auto p = bolt::amp::reduce_by_key( device_keys.begin()+10, device_keys.begin()+400, input.begin()+10, koutput.begin()+10, voutput.begin()+10,
                                      binary_predictor, binary_operator);

    auto refPair = gold_reduce_by_key( keys.begin()+10, keys.begin()+400,refInput.begin()+10,krefOutput.begin()+10,vrefOutput.begin()+10,
                                      std::plus<int>());


    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
} 

TEST(ReduceByKeyBasic, SerialDeviceVectorOffsetTest)
{
    int length = 1<<25;
    std::vector< int > keys(length);
    

    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    std::vector<int>  refInput( length );
    
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

        refInput[i] = i;
    }

    bolt::amp::device_vector<int>  input( refInput.begin(), refInput.end() );
    bolt::amp::device_vector<int> device_keys(keys.begin(), keys.end());
    
    // input and output vectors for device and reference
	std::vector<int>  std_koutput( length );
	std::vector<int>  std_voutput( length );

    bolt::amp::device_vector<int>  koutput( std_koutput.begin(), std_koutput.end() );
    bolt::amp::device_vector<int>  voutput( std_voutput.begin(), std_voutput.end() );
    std::vector<int>  krefOutput( length );
    std::vector<int>  vrefOutput( length );

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key

    auto p = bolt::amp::reduce_by_key( ctl, device_keys.begin()+10, device_keys.begin()+400, input.begin()+10, koutput.begin()+10, voutput.begin()+10,
                                      binary_predictor, binary_operator);

    auto refPair = gold_reduce_by_key( keys.begin()+10, keys.begin()+400,refInput.begin()+10,krefOutput.begin()+10,vrefOutput.begin()+10,
                                      std::plus<int>());


    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
} 
#if defined( ENABLE_TBB )
TEST(ReduceByKeyBasic, MultiCoreDeviceVectorOffsetTest)
{
    int length = 1<<25;
    std::vector< int > keys(length);
    

    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 0;
    std::vector<int>  refInput( length );
    
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

        refInput[i] = i;
    }

    bolt::amp::device_vector<int>  input( refInput.begin(), refInput.end() );
    bolt::amp::device_vector<int> device_keys(keys.begin(), keys.end());
    
    // input and output vectors for device and reference
	std::vector<int>  std_koutput( length );
	std::vector<int>  std_voutput( length );

    bolt::amp::device_vector<int>  koutput( std_koutput.begin(), std_koutput.end() );
    bolt::amp::device_vector<int>  voutput( std_voutput.begin(), std_voutput.end() );
    std::vector<int>  krefOutput( length );
    std::vector<int>  vrefOutput( length );

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce_by_key

    auto p = bolt::amp::reduce_by_key( ctl, device_keys.begin()+10, device_keys.begin()+400, input.begin()+10, koutput.begin()+10, voutput.begin()+10,
                                      binary_predictor, binary_operator);

    auto refPair = gold_reduce_by_key( keys.begin()+10, keys.begin()+400,refInput.begin()+10,krefOutput.begin()+10,vrefOutput.begin()+10,
                                      std::plus<int>());


    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
} 
#endif

TEST(ReduceByKeyBasic, IntegerTestOffsetTest)
{
    int length = 1<<25;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;


    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( keys.begin() +10, keys.begin()+400, input.begin()+10, koutput.begin() +10, voutput.begin() +10,
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin() +10, keys.begin() +400, refInput.begin()+10,krefOutput.begin()+10,vrefOutput.begin()+10,
                                       std::plus<int>());

    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}

TEST(ReduceByKeyBasic, SerialIntegerTestOffsetTest)
{
    int length = 1<<25;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key(ctl, keys.begin() +10, keys.begin()+400, input.begin()+10, koutput.begin() +10, voutput.begin() +10,
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin() +10, keys.begin() +400, refInput.begin()+10,krefOutput.begin()+10,vrefOutput.begin()+10,
                                       std::plus<int>());

    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}



TEST(ReduceByKeyBasic, IntegerTestOffsetTestDevice)
{
    int length = 1<<25;
    bolt::amp::device_vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    bolt::amp::device_vector< int > refInput( length );
    bolt::amp::device_vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference


    bolt::amp::device_vector< int > koutput( length );
    bolt::amp::device_vector< int > voutput( length );
    bolt::amp::device_vector< int > krefOutput( length );
    bolt::amp::device_vector< int > vrefOutput( length );

    bolt::amp::fill(krefOutput.begin(),krefOutput.end(),0);
    bolt::amp::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key

    auto p = bolt::amp::reduce_by_key(keys.begin() +10, keys.begin()+400, input.begin()+10, koutput.begin() +10, voutput.begin() +10,
                                      binary_predictor, binary_operator);


#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif
	 auto refPair = bolt::amp::reduce_by_key(ctl, keys.begin() +10, keys.begin()+400, refInput.begin()+10, krefOutput.begin() +10, vrefOutput.begin() +10,
                                      binary_predictor, binary_operator);

  
    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}



#if defined( ENABLE_TBB )
TEST(ReduceByKeyBasic, MultiCoreIntegerTestOffsetTest)
{
    int length = 1<<25;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

	bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key(ctl, keys.begin() +10, keys.begin()+400, input.begin()+10, koutput.begin() +10, voutput.begin() +10,
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin() +10, keys.begin() +400, refInput.begin()+10,krefOutput.begin()+10,vrefOutput.begin()+10,
                                       std::plus<int>());

    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}
#endif

TEST(ReduceByKeyBasic, CPUIntegerTest)
{
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( ctl, keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
     std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                                       vrefOutput.begin(),std::plus<int>());

    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}
#if defined( ENABLE_TBB )
TEST(ReduceByKeyBasic, MultiCoreIntegerTest)
{
    int length = 1<<20;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );
    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);


    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce_by_key
    auto p = bolt::amp::reduce_by_key( ctl, keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                      binary_predictor, binary_operator);

#if 0

    for(unsigned int i = 0; i < 256 ; i++)
    {
     std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    auto refPair = gold_reduce_by_key( keys.begin(), keys.end(),refInput.begin(),krefOutput.begin(),vrefOutput.begin(),
                                       std::plus<int>());

    //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
   // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
}
#endif

TEST(ReduceByKeyPairCheck, IntegerTest2)
{
    int length = 1<<23;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }keys[1] = keys[2] = keys[3] = keys[4] =1;
    keys[5] = keys[6] = keys[7] = keys[8] =2; keys[9] = 3;

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);
    std::fill(koutput.begin(),koutput.end(),0);
    std::fill(voutput.begin(),voutput.end(),0);

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    typedef std::pair<std::vector<int>::iterator,std::vector<int>::iterator> StdPairIterator;
    /*typedef bolt::amp::pair<std::vector<int>::iterator,std::vector<int>::iterator> DevicePairIterator;*/
	typedef bolt::amp::pair<std::vector<int>::iterator,std::vector<int>::iterator> DevicePairIterator;

    // call reduce_by_key
    DevicePairIterator dv_pair =
    bolt::amp::reduce_by_key(
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);

#if 0

    for(unsigned int i = 0; i < length ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    DevicePairIterator gold_pair =
    gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                        vrefOutput.begin(),std::plus<int>());

    size_t sizeAfterCall = gold_pair.first - krefOutput.begin();
    size_t sizeAfterDeviceCall = dv_pair.first - koutput.begin();

    //std::cout<<sizeAfterCall<<" Is the gold key size after call!"<<std::endl;
    //std::cout<<sizeAfterDeviceCall<<" Is the dv key size after call!"<<std::endl;

    krefOutput.resize(sizeAfterCall);
    vrefOutput.resize(sizeAfterCall);
    koutput.resize(sizeAfterDeviceCall);
    voutput.resize(sizeAfterDeviceCall);

#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
}
TEST(ReduceByKeyPairCheck, CPUIntegerTest2)
{
    int length = 1<<24;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }keys[1] = keys[2] = keys[3] = keys[4] =1;
    keys[5] = keys[6] = keys[7] = keys[8] =2; keys[9] = 3;

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);
    std::fill(koutput.begin(),koutput.end(),0);
    std::fill(voutput.begin(),voutput.end(),0);

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    typedef std::pair<std::vector<int>::iterator,std::vector<int>::iterator> StdPairIterator;
    /*typedef bolt::amp::pair<std::vector<int>::iterator,std::vector<int>::iterator> DevicePairIterator;*/
	typedef bolt::amp::pair<std::vector<int>::iterator,std::vector<int>::iterator> DevicePairIterator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key
    DevicePairIterator dv_pair =
    bolt::amp::reduce_by_key(
        ctl,
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);

#if 0

    for(unsigned int i = 0; i < length ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    DevicePairIterator gold_pair =
    gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                        vrefOutput.begin(),std::plus<int>());

    size_t sizeAfterCall = gold_pair.first - krefOutput.begin();
    size_t sizeAfterDeviceCall = dv_pair.first - koutput.begin();

    //std::cout<<sizeAfterCall<<" Is the gold key size after call!"<<std::endl;
    //std::cout<<sizeAfterDeviceCall<<" Is the dv key size after call!"<<std::endl;

    krefOutput.resize(sizeAfterCall);
    vrefOutput.resize(sizeAfterCall);
    koutput.resize(sizeAfterDeviceCall);
    voutput.resize(sizeAfterDeviceCall);

#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
}
#if defined( ENABLE_TBB )
TEST(ReduceByKeyPairCheck, MultiCoreIntegerTest2)
{
    int length = 1<<20;
    std::vector< int > keys( length);
    // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
    int segmentLength = 0;
    int segmentIndex = 0;
    int key = 1;
    std::vector< int > refInput( length );
    std::vector< int > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%3 == 1) key++;
        keys[i] = key;
        refInput[i] = std::rand()%4;
        input[i] = refInput[i];
    }keys[1] = keys[2] = keys[3] = keys[4] =1;
    keys[5] = keys[6] = keys[7] = keys[8] =2; keys[9] = 3;

    // input and output vectors for device and reference

    std::vector< int > koutput( length );
    std::vector< int > voutput( length );
    std::vector< int > krefOutput( length );
    std::vector< int > vrefOutput( length );
    std::fill(krefOutput.begin(),krefOutput.end(),0);
    std::fill(vrefOutput.begin(),vrefOutput.end(),0);
    std::fill(koutput.begin(),koutput.end(),0);
    std::fill(voutput.begin(),voutput.end(),0);

    bolt::amp::equal_to<int> binary_predictor;
    bolt::amp::plus<int> binary_operator;

    typedef std::pair<std::vector<int>::iterator,std::vector<int>::iterator> StdPairIterator;
    /*typedef bolt::amp::pair<std::vector<int>::iterator,std::vector<int>::iterator> DevicePairIterator;*/
	typedef bolt::amp::pair<std::vector<int>::iterator,std::vector<int>::iterator> DevicePairIterator;

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce_by_key
    DevicePairIterator dv_pair =
    bolt::amp::reduce_by_key(
        ctl,
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);

#if 0

    for(unsigned int i = 0; i < length ; i++)
    {
      std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    DevicePairIterator gold_pair =
    gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                        vrefOutput.begin(),std::plus<int>());

    size_t sizeAfterCall = gold_pair.first - krefOutput.begin();
    size_t sizeAfterDeviceCall = dv_pair.first - koutput.begin();

    //std::cout<<sizeAfterCall<<" Is the gold key size after call!"<<std::endl;
    //std::cout<<sizeAfterDeviceCall<<" Is the dv key size after call!"<<std::endl;

    krefOutput.resize(sizeAfterCall);
    vrefOutput.resize(sizeAfterCall);
    koutput.resize(sizeAfterDeviceCall);
    voutput.resize(sizeAfterDeviceCall);

#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i]<<" OValues "<<voutput[i]<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);
}
#endif

TEST(ReduceByKeyBasic, IntegerTestOddSizes)
{
    int length;

    int num,i, count=0;

   for(num=1<<16;count<=50;num+= 555)
   {
      for(i=2;i<num;i++)
      {
         if(num%i==0)
         break;
      }
      if(num==i)
      {
      //  printf("****%d\n", length);
        length = num;
        std::vector< int > keys( length);
        // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
        int segmentLength = 0;
        int segmentIndex = 0;
        int key = 1;
        std::vector< int > refInput( length );
        std::vector< int > input( length );
        for (int i = 0; i < length; i++)
        {
            if(std::rand()%3 == 1) key++;
            keys[i] = key;
            refInput[i] = std::rand()%4;
            input[i] = refInput[i];
        }

        // input and output vectors for device and reference

        std::vector< int > koutput( length );
        std::vector< int > voutput( length );
        std::vector< int > krefOutput( length );
        std::vector< int > vrefOutput( length );
        std::fill(krefOutput.begin(),krefOutput.end(),0);
        std::fill(vrefOutput.begin(),vrefOutput.end(),0);


        bolt::amp::equal_to<int> binary_predictor;
        bolt::amp::plus<int> binary_operator;


        // call reduce_by_key

        auto p = bolt::amp::reduce_by_key( keys.begin(), keys.end(), input.begin(), koutput.begin(), voutput.begin(),
                                          binary_predictor, binary_operator);


    #if 0

        for(unsigned int i = 0; i < 256 ; i++)
        {

            std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i];
            std::cout<<" OValues "<<voutput[i]<<std::endl;

        }

    #endif


        auto refPair = gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                                           vrefOutput.begin(),std::plus<int>());


        //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
        cmpArrays(krefOutput, koutput);
        cmpArrays(vrefOutput, voutput);
        // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
        count++;
      }
   }


}
TEST(ReduceByKeyBasic, CPUIntegerTestOddSizes)
{
    int length;

    int num,i, count=0;

    for(num=1<<16;count<=50;num+= 555)
   {
      for(i=2;i<num;i++)
      {
         if(num%i==0)
         break;
      }
      if(num==i)
      {
      //  printf("****%d\n", length);
        length = num;
        std::vector< int > keys( length);
        // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
        int segmentLength = 0;
        int segmentIndex = 0;
        int key = 1;
        std::vector< int > refInput( length );
        std::vector< int > input( length );
        for (int i = 0; i < length; i++)
        {
            if(std::rand()%3 == 1) key++;
            keys[i] = key;
            refInput[i] = std::rand()%4;
            input[i] = refInput[i];
        }

        // input and output vectors for device and reference

        std::vector< int > koutput( length );
        std::vector< int > voutput( length );
        std::vector< int > krefOutput( length );
        std::vector< int > vrefOutput( length );
        std::fill(krefOutput.begin(),krefOutput.end(),0);
        std::fill(vrefOutput.begin(),vrefOutput.end(),0);


        bolt::amp::equal_to<int> binary_predictor;
        bolt::amp::plus<int> binary_operator;

        bolt::amp::control ctl = bolt::amp::control::getDefault( );
        ctl.setForceRunMode(bolt::amp::control::SerialCpu);

        // call reduce_by_key
        auto p = bolt::amp::reduce_by_key( ctl, keys.begin(), keys.end(), input.begin(), koutput.begin(),
                                          voutput.begin(), binary_predictor, binary_operator);

    #if 0

        for(unsigned int i = 0; i < 256 ; i++)
        {
             std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i];
             std::cout<<" OValues "<<voutput[i]<<std::endl;
        }

    #endif

        auto refPair = gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                                           vrefOutput.begin(),std::plus<int>());

        //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
        cmpArrays(krefOutput, koutput);
        cmpArrays(vrefOutput, voutput);
        // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
        count++;
      }
   }


}
#if defined( ENABLE_TBB )
TEST(ReduceByKeyBasic, MultiCoreIntegerTestOddSizes)
{
    int length;

    int num,i, count=0;

    for(num=1<<16;count<=50;num+= 555)
   {
      for(i=2;i<num;i++)
      {
         if(num%i==0)
         break;
      }
      if(num==i)
      {
      //  printf("****%d\n", length);
        length = num;
        std::vector< int > keys( length);
        // keys = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,...}
        int segmentLength = 0;
        int segmentIndex = 0;
        int key = 1;
        std::vector< int > refInput( length );
        std::vector< int > input( length );
        for (int i = 0; i < length; i++)
        {
            if(std::rand()%3 == 1) key++;
            keys[i] = key;
            refInput[i] = std::rand()%4;
            input[i] = refInput[i];
        }

        // input and output vectors for device and reference

        std::vector< int > koutput( length );
        std::vector< int > voutput( length );
        std::vector< int > krefOutput( length );
        std::vector< int > vrefOutput( length );
        std::fill(krefOutput.begin(),krefOutput.end(),0);
        std::fill(vrefOutput.begin(),vrefOutput.end(),0);


        bolt::amp::equal_to<int> binary_predictor;
        bolt::amp::plus<int> binary_operator;

        bolt::amp::control ctl = bolt::amp::control::getDefault( );
        ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

        // call reduce_by_key
        auto p = bolt::amp::reduce_by_key( ctl, keys.begin(), keys.end(), input.begin(), koutput.begin(),
                                          voutput.begin(), binary_predictor, binary_operator);

    #if 0

        for(unsigned int i = 0; i < 256 ; i++)
        {
            std::cout<<"Ikey "<<keys[i]<<" IValues "<<input[i]<<" -> OKeys "<<koutput[i];
            std::cout<<" OValues "<<voutput[i]<<std::endl;
        }

    #endif

        auto refPair = gold_reduce_by_key( keys.begin(), keys.end(), refInput.begin(), krefOutput.begin(),
                                           vrefOutput.begin(),std::plus<int>());

        //cmpArrays2(krefOutput, koutput, refPair.first, p.first);
        cmpArrays(krefOutput, koutput);
        cmpArrays(vrefOutput, voutput);
        // cmpArrays2(vrefOutput, voutput, refPair.second, p.second);
        count++;
      }
   }


}
#endif

#if UDD
TEST(ReduceByKeyPairUDDTest, UDDFloatIntTest)
{
    int length = 1<<20;
    std::vector< uddfltint > keys( length);
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%5 == 1) key++;
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    // Instead of using fill
    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

    typedef std::pair<std::vector<uddfltint>::iterator,
            std::vector<uddfltint>::iterator> StdPairIterator;
	typedef bolt::amp::pair<std::vector<uddfltint>::iterator,
            std::vector<uddfltint>::iterator> DevicePairIterator;

    DevicePairIterator gold_pair =
    gold_reduce_by_key( keys.begin(),
                        keys.end(),
                        refInput.begin(),
                        krefOutput.begin(),
                        vrefOutput.begin(),
                        binary_operator);

    // call reduce_by_key
    DevicePairIterator dv_pair =
    bolt::amp::reduce_by_key(
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);

    size_t sizeAfterCall = gold_pair.first - krefOutput.begin();
    size_t sizeAfterDeviceCall = dv_pair.first - koutput.begin();

    krefOutput.resize(sizeAfterCall);
    vrefOutput.resize(sizeAfterCall);
    koutput.resize(sizeAfterDeviceCall);
    voutput.resize(sizeAfterDeviceCall);

#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i].x<<" OValues "<<voutput[i].x<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);

}


TEST(ReduceByKeyPairUDDTest, UDDFloatIntTestDevice)
{
    int length = 1<<20;
    bolt::amp::device_vector< uddfltint > keys( length);
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    bolt::amp::device_vector< uddfltint > refInput( length );
    bolt::amp::device_vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%5 == 1) key++;
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    bolt::amp::device_vector< uddfltint > koutput( length );
    bolt::amp::device_vector< uddfltint > voutput( length );
    bolt::amp::device_vector< uddfltint > krefOutput( length );
    bolt::amp::device_vector< uddfltint > vrefOutput( length );

    // Instead of using fill
    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

  
	  bolt::amp::control ctl = bolt::amp::control::getDefault( );
        ctl.setForceRunMode(bolt::amp::control::SerialCpu);

	auto gold_pair =
    bolt::amp::reduce_by_key(ctl,
        keys.begin(),
        keys.end(),
        refInput.begin(),
        krefOutput.begin(),
        vrefOutput.begin(),
        binary_predictor,
        binary_operator);

    // call reduce_by_key
    auto dv_pair =
    bolt::amp::reduce_by_key(
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);


#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i].x<<" OValues "<<voutput[i].x<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);

}












TEST(ReduceByKeyPairUDDTest, CPU_UDDFloatIntTest)
{
    int length = 1<<20;
    std::vector< uddfltint > keys( length);
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%5 == 1) key++;
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    // Instead of using fill
    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

    typedef std::pair<std::vector<uddfltint>::iterator,
            std::vector<uddfltint>::iterator> StdPairIterator;
	typedef bolt::amp::pair<std::vector<uddfltint>::iterator,
            std::vector<uddfltint>::iterator> DevicePairIterator;

    DevicePairIterator gold_pair =
    gold_reduce_by_key( keys.begin(),
                        keys.end(),
                        refInput.begin(),
                        krefOutput.begin(),
                        vrefOutput.begin(),
                        binary_operator);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    // call reduce_by_key
    DevicePairIterator dv_pair =
    bolt::amp::reduce_by_key(
        ctl,
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);

    size_t sizeAfterCall = gold_pair.first - krefOutput.begin();
    size_t sizeAfterDeviceCall = dv_pair.first - koutput.begin();

    krefOutput.resize(sizeAfterCall);
    vrefOutput.resize(sizeAfterCall);
    koutput.resize(sizeAfterDeviceCall);
    voutput.resize(sizeAfterDeviceCall);

#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i].x<<" OValues "<<voutput[i].x<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);

}
#if defined( ENABLE_TBB )
TEST(ReduceByKeyPairUDDTest, MultiCore_UDDFloatIntTest)
{
    int length = 1<<20;
    std::vector< uddfltint > keys( length);
    uddfltint key;
    key.x = 1.0f;
    key.y = 1;
    std::vector< uddfltint > refInput( length );
    std::vector< uddfltint > input( length );

    for (int i = 0; i < length; i++)
    {
        if(std::rand()%5 == 1) key++;
        keys[i] = key;
        refInput[i].x = float(std::rand()%4);
        refInput[i].y = std::rand()%4;
        input[i] = refInput[i];
    }

    std::vector< uddfltint > koutput( length );
    std::vector< uddfltint > voutput( length );
    std::vector< uddfltint > krefOutput( length );
    std::vector< uddfltint > vrefOutput( length );

    // Instead of using fill
    krefOutput.clear();krefOutput.resize(length);
    vrefOutput.clear();vrefOutput.resize(length);
    voutput.clear();voutput.resize(length);
    koutput.clear();koutput.resize(length);

    uddfltint_equal_to binary_predictor;
    uddfltint_plus binary_operator;

    typedef std::pair<std::vector<uddfltint>::iterator,
            std::vector<uddfltint>::iterator> StdPairIterator;
	  typedef bolt::amp::pair<std::vector<uddfltint>::iterator,
            std::vector<uddfltint>::iterator> DevicePairIterator;

    DevicePairIterator gold_pair =
    gold_reduce_by_key( keys.begin(),
                        keys.end(),
                        refInput.begin(),
                        krefOutput.begin(),
                        vrefOutput.begin(),
                        binary_operator);

    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    // call reduce_by_key
    DevicePairIterator dv_pair =
    bolt::amp::reduce_by_key(
        ctl,
        keys.begin(),
        keys.end(),
        input.begin(),
        koutput.begin(),
        voutput.begin(),
        binary_predictor,
        binary_operator);

    size_t sizeAfterCall = gold_pair.first - krefOutput.begin();
    size_t sizeAfterDeviceCall = dv_pair.first - koutput.begin();

    krefOutput.resize(sizeAfterCall);
    vrefOutput.resize(sizeAfterCall);
    koutput.resize(sizeAfterDeviceCall);
    voutput.resize(sizeAfterDeviceCall);

#if 0

    for(unsigned int i = 0; i < sizeAfterDeviceCall ; i++)
    {
        std::cout<<" -> OKeys "<<koutput[i].x<<" OValues "<<voutput[i].x<<std::endl;
    }

#endif

    cmpArrays(krefOutput, koutput);
    cmpArrays(vrefOutput, voutput);

}
#endif

#endif

int _tmain(int argc, _TCHAR* argv[])
{


    //  Initialize googletest; this removes googletest specific flags from command line
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    
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

#endif
