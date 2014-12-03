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

#include "bolt/amp/reduce.h"
#include "bolt/amp/functional.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include "bolt/amp/iterator/counting_iterator.h"
#include "bolt/miniDump.h"

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include "common/test_common.h"
#include <algorithm>
#include <type_traits>
#define TEST_CPU_DEVICE 0
#define TEST_DOUBLE 1
#define TEST_DEVICE_VECTOR 1
#define TEST_LARGE_BUFFERS 1
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process type parameterized tests

//  This class creates a C++ 'TYPE' out of a size_t value
template< size_t N >
class TypeValue
{
public:
    static const size_t value = N;
};
template <typename T>
T generateRandom()
{
    double value = rand();
    static bool negate = true;
    if (negate)
    {
        negate = false;
        return -(T)fmod(value, 10.0);
    }
    else
    {
        negate = true;
        return (T)fmod(value, 10.0);
    }
}
//  Test fixture class, used for the Type-parameterized tests
//  Namely, the tests that use std::array and TYPED_TEST_P macros
template< typename ArrayTuple >
class ReduceArrayTest: public ::testing::Test
{
public:
    ReduceArrayTest( ): m_Errors( 0 )
    {}

    virtual void SetUp( )
    {
        //std::generate(stdInput.begin(), stdInput.end(), generateRandom<ArrayType>);
        std::fill(stdInput.begin(), stdInput.end(), ArrayType(100));
        stdOutput = stdInput;
        boltInput = stdInput;
        boltOutput = stdInput;
    };

    virtual void TearDown( )
    {};

    virtual ~ReduceArrayTest( )
    {}

protected:
    typedef typename std::tuple_element< 0, ArrayTuple >::type ArrayType;
    static const size_t ArraySize = std::tuple_element< 1, ArrayTuple >::type::value;
    typename std::array< ArrayType, ArraySize > stdInput, boltInput, stdOutput, boltOutput;
    int m_Errors;
};

TYPED_TEST_CASE_P( ReduceArrayTest );

TYPED_TEST_P( ReduceArrayTest, Normal )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );
}

TYPED_TEST_P( ReduceArrayTest, SerialNormal )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);
    ArrayType boltReduce = bolt::amp::reduce( ctl, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );
}
#if defined( ENABLE_TBB )
TYPED_TEST_P( ReduceArrayTest, MultiCoreNormal )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce(ctl, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );
}
#endif


TYPED_TEST_P( ReduceArrayTest, GPU_DeviceNormal )
{
    Concurrency::accelerator accel(Concurrency::accelerator::default_accelerator);
    bolt::amp::control c_gpu( accel );  // construct control structure from the queue.

    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;  
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;
    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce(c_gpu, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init, bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );

}

#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( ReduceArrayTest, CPU_DeviceNormal )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize  > ArrayCont;

    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    bolt::amp::control ctl;
    concurrency::accelerator cpuAccelerator = concurrency::accelerator(concurrency::accelerator::cpu_accelerator);
    ctl.setAccelerator(cpuAccelerator);


    ArrayType boltReduce = bolt::amp::reduce( ctl, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ),
                                                       bolt::amp::square<ArrayType>(), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

TYPED_TEST_P( ReduceArrayTest, MultipliesFunction )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}

TYPED_TEST_P( ReduceArrayTest, SerialMultipliesFunction )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType; 
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce(ctl, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}

#if defined( ENABLE_TBB )
TYPED_TEST_P( ReduceArrayTest, MultiCoreMultipliesFunction )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType; 
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize > ArrayCont;

    ArrayType init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce(ctl, ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>( ));

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif


TYPED_TEST_P( ReduceArrayTest, GPU_DeviceMultipliesFunction )
{
    typedef typename ReduceArrayTest< gtest_TypeParam_ >::ArrayType ArrayType;    
    typedef std::array< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize  > ArrayCont;
#if OCL_CONTEXT_BUG_WORKAROUND
  ::cl::Context myContext = bolt::amp::control::getDefault( ).context( );
    bolt::amp::control c_gpu( getQueueFromContext(myContext, CL_DEVICE_TYPE_GPU, 0 ));  
#else
    ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
    bolt::amp::control c_gpu(accel);
#endif


    ArrayType init(0);
    //  Calling the actual functions under test
    ArrayType stlReduce = std::accumulate(ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin(), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce( c_gpu,ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end( ), init,
                                                       bolt::amp::plus<ArrayType>());

    typename ArrayCont::difference_type stdNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::stdInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::stdInput.end() );
    typename ArrayCont::difference_type boltNumElements = std::distance( ReduceArrayTest< gtest_TypeParam_ >::boltInput.begin( ), ReduceArrayTest< gtest_TypeParam_ >::boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ReduceArrayTest< gtest_TypeParam_ >::ArraySize >::cmpArrays( ReduceArrayTest< gtest_TypeParam_ >::stdInput, ReduceArrayTest< gtest_TypeParam_ >::boltInput );    
    // FIXME - releaseOcl(ocl);
}
#if (TEST_CPU_DEVICE == 1)
TYPED_TEST_P( TransformArrayTest, CPU_DeviceMultipliesFunction )
{
    typedef std::array< ArrayType, ArraySize > ArrayCont;

    ArrayType init(0);
    //  Calling the actual functions under test
    bolt::amp::control c_cpu;
    concurrency::accelerator cpuAccelerator = concurrency::accelerator(concurrency::accelerator::cpu_accelerator);
    ctl.setAccelerator(cpuAccelerator);

    ArrayType stlReduce = std::accumulate(stdOutput.begin(), stdOutput.end(), init);

    ArrayType boltReduce = bolt::amp::reduce( c_cpu,boltInput.begin( ), boltInput.end( ),init,
                                                       bolt::amp::plus<ArrayType>());

    ArrayCont::difference_type stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    ArrayCont::difference_type boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpStdArray< ArrayType, ArraySize >::cmpArrays( stdInput, boltInput );    
    // FIXME - releaseOcl(ocl);
}
#endif

#if (TEST_CPU_DEVICE == 1)
REGISTER_TYPED_TEST_CASE_P( TransformArrayTest, Normal, GPU_DeviceNormal, 
                                           MultipliesFunction, GPU_DeviceMultipliesFunction,
                                           CPU_DeviceNormal, CPU_DeviceMultipliesFunction);
#else
REGISTER_TYPED_TEST_CASE_P( ReduceArrayTest, Normal, SerialNormal, GPU_DeviceNormal, 
                                           MultipliesFunction, SerialMultipliesFunction, GPU_DeviceMultipliesFunction );
#if defined( ENABLE_TBB )
REGISTER_TYPED_TEST_CASE_P( MultiCoreNormal,  MultiCoreMultipliesFunction );
#endif
#endif


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fixture classes are now defined to enable googletest to process value parameterized tests
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size

class ReduceIntegerVector: public ::testing::TestWithParam< int >
{
public:

    ReduceIntegerVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                               stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< int > stdInput, boltInput, stdOutput, boltOutput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceFloatVector: public ::testing::TestWithParam< int >
{
public:
    ReduceFloatVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                             stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< float > stdInput, boltInput, stdOutput, boltOutput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceDoubleVector: public ::testing::TestWithParam< int >
{
public:
    ReduceDoubleVector( ): stdInput( GetParam( ) ), boltInput( GetParam( ) ),
                              stdOutput( GetParam( ) ), boltOutput( GetParam( ) )
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<double>);
        boltInput = stdInput;
        stdOutput = stdInput;
        boltOutput = stdInput;
    }

protected:
    std::vector< double > stdInput, boltInput, stdOutput, boltOutput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceIntegerDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceIntegerDeviceVector( ): stdInput( GetParam( ) ), /*boltInput( static_cast<size_t>( GetParam( ) ) ),*/
                                     stdOutput( GetParam( ) )/*, boltOutput( static_cast<size_t>( GetParam( ) ) )*/
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<int>);
        //boltInput = stdInput;      
        //FIXME - The above should work but the below loop is used. 
        for (int i=0; i< GetParam( ); i++)
        {
            //boltInput[i] = stdInput[i];
            //boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    }

protected:
    std::vector< int > stdInput, stdOutput;
    //bolt::amp::device_vector< int > boltInput, boltOutput;
};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceFloatDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceFloatDeviceVector( ): stdInput( GetParam( ) )/*, boltInput( stdInput ), boltOutput( stdInput )*/
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<float>);
        stdOutput = stdInput;

        //FIXME - The above should work but the below loop is used. 
        /*for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
        }*/
    }

protected:
    std::vector< float > stdInput, stdOutput;
    //bolt::amp::device_vector< float, concurrency::array_view > boltInput, boltOutput;
};

#if (TEST_DOUBLE == 1)
//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceDoubleDeviceVector: public ::testing::TestWithParam< int >
{
public:
    // Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceDoubleDeviceVector( ): stdInput( GetParam( ) )/*, boltInput( static_cast<size_t>( GetParam( ) ) ),
                                                        boltOutput( static_cast<size_t>( GetParam( ) ) )*/
    {
        std::generate(stdInput.begin(), stdInput.end(), generateRandom<double>);
        stdOutput = stdInput;

        //FIXME - The above should work but the below loop is used. 
        /* for (int i=0; i< GetParam( ); i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
        } */
    }

protected:
    std::vector< double > stdInput, stdOutput;
    //bolt::amp::device_vector< double > boltInput, boltOutput;
};
#endif

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceIntegerNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceIntegerNakedPointer( ): stdInput( new int[ GetParam( ) ] ), boltInput( new int[ GetParam( ) ] ), 
                                     stdOutput( new int[ GetParam( ) ] ), boltOutput( new int[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<int>);
		for (size_t i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
};

protected:
     int* stdInput;
     int* boltInput;
     int* stdOutput;
     int* boltOutput;

};

//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class ReduceFloatNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    ReduceFloatNakedPointer( ): stdInput( new float[ GetParam( ) ] ), boltInput( new float[ GetParam( ) ] ), 
                                   stdOutput( new float[ GetParam( ) ] ), boltOutput( new float[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        size_t size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<float>);
		for (size_t i = 0; i<size; i++)
        {
            boltInput[i] = stdInput[i];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
    };

protected:
     float* stdInput;
     float* boltInput;
     float* stdOutput;
     float* boltOutput;
};

class ReduceStdVectWithInit :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    ReduceStdVectWithInit():mySize(GetParam()){
    }
};


TEST( ReduceStdVectWithInit, OffsetTest)
{
    int length = 1000;
    std::vector<int> stdInput( length );
    bolt::amp::device_vector<int> boltInput( length );


    for (int i = 0; i < length; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( boltInput.begin( ) + offset, boltInput.end( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#if defined( ENABLE_TBB )
TEST( ReduceStdVectWithInit, OffsetTestMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( stdInput.begin( ) + offset, stdInput.end( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif

TEST( ReduceStdVectWithInit, OffsetTestSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( stdInput.begin( ) + offset, stdInput.end( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( ReduceStdVectWithInit, OffsetTestDeviceVector)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );

    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(  stdInput.begin( ) + offset, stdInput.end( ),
                                               init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce(  dVectorA.begin( ) + offset, dVectorA.end( ),
                                                init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST( ReduceStdVectWithInit, OffsetTestDeviceVectorSerialCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );


    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(  stdInput.begin( ) + offset, stdInput.end( ),
                                               init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce(  ctl, dVectorA.begin( ) + offset, dVectorA.end( ),
                                                init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#if defined( ENABLE_TBB )
TEST( ReduceStdVectWithInit, OffsetTestDeviceVectorMultiCoreCpu)
{
    int length = 1024;
    std::vector<int> stdInput( length );
    std::fill( stdInput.begin(), stdInput.end(), 1024 );

    bolt::amp::device_vector<int> dVectorA( stdInput.begin(), stdInput.end() );

    bolt::amp::control ctl;
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
    
    //  Calling the actual functions under test
    int init = 0, offset = 100;
    int stlTransformReduce = std::accumulate(stdInput.begin( ) + offset, stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( dVectorA.begin( ) + offset, dVectorA.end( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif

TEST_P( ReduceStdVectWithInit, withIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( boltInput.begin( ), boltInput.end( ), init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

TEST_P( ReduceStdVectWithInit, SerialwithIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce(ctl, boltInput.begin( ), boltInput.end( ), init,
                                                                     bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#if defined( ENABLE_TBB )
TEST_P( ReduceStdVectWithInit, MultiCorewithIntWdInit)
{
    std::vector<int> stdInput( mySize );
    std::vector<int> stdOutput( mySize );
    std::vector<int> boltInput( mySize );

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce(ctl, boltInput.begin( ), boltInput.end( ),init,bolt::amp::plus<int>());

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif


TEST_P( ReduceStdVectWithInit, withIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::amp::reduce( boltInput.begin( ), boltInput.end( ), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( ReduceStdVectWithInit, SerialwithIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::amp::reduce(ctl, boltInput.begin( ), boltInput.end(),init,bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#if defined( ENABLE_TBB )
TEST_P( ReduceStdVectWithInit, MultiCorewithIntWdInitWithStdPlus)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);
 
    //  Calling the actual functions under test
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init, std::plus<int>());
    int boltTransformReduce= bolt::amp::reduce(ctl, boltInput.begin( ), boltInput.end(),init,bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#endif


TEST_P( ReduceStdVectWithInit, withIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    //  Calling the actual functions under test
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltTransformReduce= bolt::amp::reduce( boltInput.begin( ), boltInput.end( ), init, bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}

TEST_P( ReduceStdVectWithInit, SerialwithIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltTransformReduce= bolt::amp::reduce( ctl, boltInput.begin( ), boltInput.end(),init,bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#if defined( ENABLE_TBB )
TEST_P( ReduceStdVectWithInit, MultiCorewithIntWdInitWdAnyFunctor)
{
    //int mySize = 10;
    int init = 10;

    std::vector<int> stdInput (mySize);
    std::vector<int> stdOutput (mySize);

    std::vector<int> boltInput (mySize);
    //std::vector<int> boltOutput (mySize);

    for (int i = 0; i < mySize; ++i){
        stdInput[i] = i;
        boltInput[i] = stdInput[i];
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int stlTransformReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltTransformReduce= bolt::amp::reduce( ctl, boltInput.begin( ), boltInput.end(),init,bolt::amp::plus<int>());

    EXPECT_EQ(stlTransformReduce, boltTransformReduce);
}
#endif
class StdVectCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    StdVectCountingIterator():mySize(GetParam()){
    }
};



TEST_P( StdVectCountingIterator, withCountingIterator)
{
    std::vector<int> stdInput( mySize );
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
    }
    
    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( first, last, init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}

#if 0
TEST_P( StdVectCountingIterator, SerialwithCountingIterator)
{
    std::vector<int> stdInput( mySize );
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;
    
    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
    }
    
    bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( ctl, first, last, init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}


TEST_P( StdVectCountingIterator, MultiCorewithCountingIterator)
{
    std::vector<int> stdInput( mySize );
    bolt::amp::counting_iterator<int> first(0);
    bolt::amp::counting_iterator<int> last = first +  mySize;

    for (int i = 0; i < mySize; ++i)
    {
        stdInput[i] = i;
    }
    
     bolt::amp::control ctl = bolt::amp::control::getDefault( );
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int init = 10;
    int stlTransformReduce = std::accumulate(stdInput.begin( ), stdInput.end( ), init, bolt::amp::plus<int>( ) );
    int boltTransformReduce= bolt::amp::reduce( ctl, first, last, init, bolt::amp::plus<int>( ) );

    EXPECT_EQ( stlTransformReduce, boltTransformReduce );
}
#endif

INSTANTIATE_TEST_CASE_P( withIntWithInitValue, ReduceStdVectWithInit, ::testing::Range(1, 100, 1) );
INSTANTIATE_TEST_CASE_P( withCountingIterator, StdVectCountingIterator, ::testing::Range(1, 100, 1) );

class ReduceTestMultFloat: public ::testing::TestWithParam<int>{
protected:
    int arraySize;
public:
    ReduceTestMultFloat( ):arraySize( GetParam( ) )
    {}
};

TEST_P (ReduceTestMultFloat, multiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }

    float stlTransformReduce = std::accumulate(myArray, myArray + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::amp::reduce(myBoltArray, myBoltArray + arraySize,
                                                  1.0f, bolt::amp::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}

TEST_P (ReduceTestMultFloat, SerialmultiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    float stlTransformReduce = std::accumulate(myArray, myArray + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::amp::reduce(ctl, myBoltArray, myBoltArray + arraySize,
                                                  1.0f, bolt::amp::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
#if defined( ENABLE_TBB )
TEST_P (ReduceTestMultFloat, MultiCoremultiplyWithFloats)
{
    float* myArray = new float[ arraySize ];
    float* myArray2 = new float[ arraySize ];
    float* myBoltArray = new float[ arraySize ];

    myArray[ 0 ] = 1.0f;
    myBoltArray[ 0 ] = 1.0f;
    for( int i=1; i < arraySize; i++ )
    {
        myArray[i] = myArray[i-1] + 0.0625f;
        myBoltArray[i] = myArray[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    float stlTransformReduce = std::accumulate(myArray, myArray + arraySize, 1.0f, std::multiplies<float>());
    float boltTransformReduce = bolt::amp::reduce(ctl, myBoltArray, myBoltArray + arraySize,
                                                  1.0f, bolt::amp::multiplies<float>());

    EXPECT_FLOAT_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}
#endif


TEST_P( ReduceTestMultFloat, FloatValues )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    float stdTransformReduceValue = std::accumulate(A.begin(), A.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::amp::reduce(boltVect.begin(), boltVect.end(), 0.0f, bolt::amp::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}

TEST_P( ReduceTestMultFloat, SerialFloatValues )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    float stdTransformReduceValue = std::accumulate(A.begin(), A.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::amp::reduce(ctl,boltVect.begin(),boltVect.end(),0.0f,bolt::amp::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}
#if defined( ENABLE_TBB )
TEST_P( ReduceTestMultFloat, MultiCoreFloatValues )
{
    std::vector<float> A( arraySize );
    std::vector<float> B( arraySize );
    std::vector<float> boltVect( arraySize );
    
    float myFloatValues = 9.0625f;

    for( int i=0; i < arraySize; ++i )
    {
        A[i] = myFloatValues + float(i);
        boltVect[i] = A[i];
    }

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    float stdTransformReduceValue = std::accumulate(A.begin(), A.end(), 0.0f, std::plus<float>());
    float boltClTransformReduce = bolt::amp::reduce(ctl,boltVect.begin(),boltVect.end(),0.0f,bolt::amp::plus<float>());

    //compare these results with each other
    EXPECT_FLOAT_EQ( stdTransformReduceValue, boltClTransformReduce );
}
#endif

INSTANTIATE_TEST_CASE_P(serialValues, ReduceTestMultFloat, ::testing::Range(1, 100, 10));
INSTANTIATE_TEST_CASE_P(multiplyWithFloatPredicate, ReduceTestMultFloat, ::testing::Range(1, 20, 1));
//end of new 2

#if(TEST_DOUBLE == 1)
class ReduceTestMultDouble: public ::testing::TestWithParam<int>{
protected:
        int arraySize;
public:
    ReduceTestMultDouble():arraySize(GetParam()){
    }
};

TEST_P (ReduceTestMultDouble, multiplyWithDouble)
{
    double* myArray = new double[ arraySize ];
    double* myArray2 = new double[ arraySize ];
    double* myBoltArray = new double[ arraySize ];
    
    for (int i=0; i < arraySize; i++)
    {
        myArray[i] = (double)i + 1.25;
        myBoltArray[i] = myArray[i];
    }

    double stlTransformReduce = std::accumulate(myArray, myArray + arraySize, 1.0, std::multiplies<double>());

    double boltTransformReduce = bolt::amp::reduce(myBoltArray, myBoltArray + arraySize, 1.0,
                                                            bolt::amp::multiplies<double>());
    
    EXPECT_DOUBLE_EQ(stlTransformReduce , boltTransformReduce )<<"Values does not match\n";

    delete [] myArray;
    delete [] myArray2;
    delete [] myBoltArray;
}



INSTANTIATE_TEST_CASE_P( multiplyWithDoublePredicate, ReduceTestMultDouble, ::testing::Range(1, 20, 1) );


//  ::testing::TestWithParam< int > means that GetParam( ) returns int values, which i use for array size
class TransformDoubleNakedPointer: public ::testing::TestWithParam< int >
{
public:
    //  Create an std and a bolt vector of requested size, and initialize all the elements to 1
    TransformDoubleNakedPointer( ): stdInput( new double[ GetParam( ) ] ), boltInput( new double[ GetParam( ) ] ),
                                    stdOutput( new double[ GetParam( ) ] ), boltOutput( new double[ GetParam( ) ] )
    {}

    virtual void SetUp( )
    {
        int size = GetParam( );

        std::generate(stdInput, stdInput + size, generateRandom<double>);

        for( int i=0; i < size; i++ )
        {
            boltInput[ i ] = stdInput[ i ];
            boltOutput[i] = stdInput[i];
            stdOutput[i] = stdInput[i];
        }
    };

    virtual void TearDown( )
    {
        delete [] stdInput;
        delete [] boltInput;
        delete [] stdOutput;
        delete [] boltOutput;
    };

protected:
     double* stdInput;
     double* boltInput;
     double* stdOutput;
     double* boltOutput;
};
#endif

TEST_P( ReduceIntegerVector, Normal )
{

    int init(0);
    //  Calling the actual functions under test
    int stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltReduce = bolt::amp::reduce( boltInput.begin( ), boltInput.end( ), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}

TEST_P( ReduceIntegerVector, Serial )
{

    int init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    int stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltReduce = bolt::amp::reduce( ctl, boltInput.begin( ), boltInput.end( ), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}
#if defined( ENABLE_TBB )
TEST_P( ReduceIntegerVector, MultiCore )
{

    int init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    int boltReduce = bolt::amp::reduce( ctl, boltInput.begin( ), boltInput.end( ), init,
                                                       bolt::amp::plus<int>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );

}
#endif

TEST_P( ReduceFloatVector, Normal )
{
    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    float boltReduce = bolt::amp::reduce( boltInput.begin( ), boltInput.end( ),init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}

TEST_P( ReduceFloatVector, Serial )
{
    float init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    float stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    float boltReduce = bolt::amp::reduce( ctl, boltInput.begin( ), boltInput.end( ),init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#if defined( ENABLE_TBB )
TEST_P( ReduceFloatVector, MultiCore )
{
    float init(0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    float stlReduce = std::accumulate(stdInput.begin(), stdInput.end(), init);

    float boltReduce = bolt::amp::reduce( ctl, boltInput.begin( ), boltInput.end( ),init,
                                                       bolt::amp::plus<float>());

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput );
}
#endif

TEST_P( ReduceIntegerNakedPointer, Inplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
#else
    int* wrapStdInput = stdInput;
    int* wrapStdOutput = stdOutput;
    int* wrapBoltInput = boltInput;
#endif

    int init(0);
    //  Calling the actual functions under test
    int stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    int boltReduce = bolt::amp::reduce( wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::amp::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( ReduceIntegerNakedPointer, SerialInplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
#else
    int* wrapStdInput = stdInput;
    int* wrapStdOutput = stdOutput;
    int* wrapBoltInput = boltInput;
#endif

    int init(0);

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
    int stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    int boltReduce = bolt::amp::reduce( ctl, wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::amp::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( ReduceIntegerNakedPointer, MultiCoreInplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< int* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< int* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< int* > wrapBoltInput( boltInput, endIndex );
#else
    int* wrapStdInput = stdInput;
    int* wrapStdOutput = stdOutput;
    int* wrapBoltInput = boltInput;
#endif

    int init(0);

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
    int stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    int boltReduce = bolt::amp::reduce( ctl, wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::amp::plus<int>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif

TEST_P( ReduceFloatNakedPointer, Inplace )
{
    unsigned int endIndex = GetParam( );

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
#else
    float* wrapStdInput = stdInput;
    float* wrapStdOutput = stdOutput;
    float* wrapBoltInput = boltInput;
#endif

    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    float boltReduce = bolt::amp::reduce( wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::amp::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}

TEST_P( ReduceFloatNakedPointer, SerialInplace )
{
    unsigned int endIndex = GetParam( );

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
#else
    float* wrapStdInput = stdInput;
    float* wrapStdOutput = stdOutput;
    float* wrapBoltInput = boltInput;
#endif

    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    float boltReduce = bolt::amp::reduce( ctl, wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::amp::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#if defined( ENABLE_TBB )
TEST_P( ReduceFloatNakedPointer, MultiCoreInplace )
{
    unsigned int endIndex = GetParam( );

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    //  Calling the actual functions under test
#if defined (_WIN32 )
    stdext::checked_array_iterator< float* > wrapStdInput( stdInput, endIndex );
    stdext::checked_array_iterator< float* > wrapStdOutput( stdOutput, endIndex );
    stdext::checked_array_iterator< float* > wrapBoltInput( boltInput, endIndex );
#else
    float* wrapStdInput = stdInput;
    float* wrapStdOutput = stdOutput;
    float* wrapBoltInput = boltInput;
#endif

    float init(0);
    //  Calling the actual functions under test
    float stlReduce = std::accumulate(wrapStdInput,wrapStdInput + endIndex, init);

    float boltReduce = bolt::amp::reduce( ctl, wrapBoltInput, wrapBoltInput + endIndex, init,
                                                       bolt::amp::plus<float>());

    EXPECT_EQ( stlReduce, boltReduce );

    //  Loop through the array and compare all the values with each other
    cmpArrays( stdInput, boltInput, endIndex );
}
#endif


std::array<int, 10> TestValues = {2,4,8,16,32,64,128,256,512,1024};
std::array<int, 5> TestValues1 = {2048,4096,8192,16384,32768};

//Test lots of consecutive numbers, but small range, suitable for integers because they overflow easier
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceIntegerVector, ::testing::Range( 1, 4096, 54 ) ); //   1 to 2^22
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceIntegerVector, ::testing::ValuesIn( TestValues.begin(),TestValues.end()));
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceFloatVector, ::testing::Range( 4096, 65536, 555 ) ); //2^12 to 2^16	
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceFloatVector, ::testing::ValuesIn( TestValues.begin(), TestValues.end()));

//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceIntegerVector, ::testing::ValuesIn( TestValues1.begin(),TestValues1.end()));
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceFloatVector, ::testing::ValuesIn( TestValues1.begin(), TestValues1.end()));
//#endif

#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceDoubleVector, ::testing::Range( 65536, 2097152, 55555 ) ); //2^16 to 2^21
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceDoubleVector, ::testing::ValuesIn( TestValues.begin(), TestValues.end()));
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceDoubleVector, ::testing::ValuesIn( TestValues1.begin(), TestValues1.end()));
//#endif
#endif
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceIntegerDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceIntegerDeviceVector, ::testing::ValuesIn( TestValues.begin(),
                                                                                    TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceFloatDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceFloatDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceIntegerDeviceVector, ::testing::ValuesIn( TestValues1.begin(),
                                                                                    TestValues1.end() ) );
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceFloatDeviceVector, ::testing::ValuesIn( TestValues1.begin(), 
                                                                                    TestValues1.end() ) );
//#endif

#if (TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceDoubleDeviceVector, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceDoubleDeviceVector, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceDoubleDeviceVector, ::testing::ValuesIn( TestValues1.begin(), 
                                                                                    TestValues1.end() ) );
//#endif
#endif
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceIntegerNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceIntegerNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
INSTANTIATE_TEST_CASE_P( ReduceRange, ReduceFloatNakedPointer, ::testing::Range( 1, 32768, 3276 ) ); // 1 to 2^15
INSTANTIATE_TEST_CASE_P( ReduceValues, ReduceFloatNakedPointer, ::testing::ValuesIn( TestValues.begin(), 
                                                                                    TestValues.end() ) );
//#if TEST_LARGE_BUFFERS
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceIntegerNakedPointer, ::testing::ValuesIn( TestValues1.begin(), 
                                                                                    TestValues1.end() ) );
INSTANTIATE_TEST_CASE_P( ReduceValues1, ReduceFloatNakedPointer, ::testing::ValuesIn( TestValues1.begin(), 
                                                                                    TestValues1.end() ) );
//#endif

//#endif

typedef ::testing::Types< 
    std::tuple< int, TypeValue< 1 > >,
    std::tuple< int, TypeValue< 31 > >,
    std::tuple< int, TypeValue< 32 > >,
    std::tuple< int, TypeValue< 63 > >,
    std::tuple< int, TypeValue< 64 > >,
    std::tuple< int, TypeValue< 127 > >,
    std::tuple< int, TypeValue< 128 > >,
    std::tuple< int, TypeValue< 129 > >,
    std::tuple< int, TypeValue< 1000 > >,
    std::tuple< int, TypeValue< 1053 > >,
	/*#if TEST_LARGE_BUFFERS
	,*/
    std::tuple< int, TypeValue< 4096 > >,
    std::tuple< int, TypeValue< 4097 > >,
    std::tuple< int, TypeValue< 65535 > >,
    std::tuple< int, TypeValue< 65536 > >
    //#endif
> IntegerTests;

typedef ::testing::Types< 
    std::tuple< float, TypeValue< 1 > >,
    std::tuple< float, TypeValue< 31 > >,
    std::tuple< float, TypeValue< 32 > >,
    std::tuple< float, TypeValue< 63 > >,
    std::tuple< float, TypeValue< 64 > >,
    std::tuple< float, TypeValue< 127 > >,
    std::tuple< float, TypeValue< 128 > >,
    std::tuple< float, TypeValue< 129 > >,
    std::tuple< float, TypeValue< 1000 > >,
    std::tuple< float, TypeValue< 1053 > >,
	/*#if TEST_LARGE_BUFFERS
	,*/
    std::tuple< float, TypeValue< 4096 > >,
    std::tuple< float, TypeValue< 4097 > >,
    std::tuple< float, TypeValue< 65535 > >,
    std::tuple< float, TypeValue< 65536 > >
    //#endif
> FloatTests;

#if (TEST_DOUBLE == 1)
typedef ::testing::Types< 
    std::tuple< double, TypeValue< 1 > >,
    std::tuple< double, TypeValue< 31 > >,
    std::tuple< double, TypeValue< 32 > >,
    std::tuple< double, TypeValue< 63 > >,
    std::tuple< double, TypeValue< 64 > >,
    std::tuple< double, TypeValue< 127 > >,
    std::tuple< double, TypeValue< 128 > >,
    std::tuple< double, TypeValue< 129 > >,
    std::tuple< double, TypeValue< 1000 > >,
    std::tuple< double, TypeValue< 1053 > >,
	/*#if TEST_LARGE_BUFFERS
	,*/
    std::tuple< double, TypeValue< 4096 > >,
    std::tuple< double, TypeValue< 4097 > >,
    std::tuple< double, TypeValue< 65535 > >,
    std::tuple< double, TypeValue< 65536 > >
    //#endif
> DoubleTests;
#endif 

struct UDD { 
    int a; 
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) const restrict (cpu,amp) {
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    } 
    bool operator < (const UDD& other) const restrict (cpu,amp){
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const restrict (cpu,amp){
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const restrict (cpu,amp){
        return ((a+b) == (other.a+other.b));
    }
    UDD operator + (const UDD &rhs) const restrict (amp,cpu){
                UDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }
    UDD()   restrict (cpu,amp)
        : a(0),b(0) { }
    UDD(int _in)   restrict (cpu,amp)
        : a(_in), b(_in +1)  { }
    // User should provide copy ctor
    UDD(const UDD&other)   restrict (cpu,amp)
        : a(other.a), b(other.b)  { }
    // User should provide copy assign ctor
    UDD& operator = (const UDD&other)   restrict (cpu,amp) {
      a = other.a;
      b = other.b;
      return *this;
    }
}; 

struct UDDplus
{
   UDD operator() (const UDD &lhs, const UDD &rhs) const restrict (cpu,amp)
   {
     UDD _result;
     _result.a = lhs.a + rhs.a;
     _result.b = lhs.b + rhs.b;
     return _result;
   }

};


TEST( ReduceUDD , UDDPlusOperatorInts )
{
    //setup containers
    int length = 1024;
    std::vector< UDD > refInput( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    UDD zero;
    zero.a = 0;
    zero.b = 0;
    // call reduce
    UDDplus plusOp;
    UDD boltReduce = bolt::amp::reduce( refInput.begin(), refInput.end(), zero, plusOp );
    UDD stdReduce =  std::accumulate( refInput.begin(), refInput.end(), zero, plusOp ); // out-of-place scan

    EXPECT_EQ(boltReduce,stdReduce);

}


TEST( ReduceUDD , SerialUDDPlusOperatorInts )
{
    //setup containers
    int length = 1024;
    std::vector< UDD > refInput( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    UDD zero;
    zero.a = 0;
    zero.b = 0;
    // call reduce
    UDDplus plusOp;

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    UDD boltReduce = bolt::amp::reduce( ctl, refInput.begin(), refInput.end(), zero, plusOp );
    UDD stdReduce =  std::accumulate( refInput.begin(), refInput.end(), zero, plusOp ); // out-of-place scan

    EXPECT_EQ(boltReduce,stdReduce);

}
#if defined( ENABLE_TBB )
TEST( ReduceUDD , MultiCoreUDDPlusOperatorInts )
{
    //setup containers
    int length = 1024;
    std::vector< UDD > refInput( length );
    for( int i = 0; i < length ; i++ )
    {
      refInput[i].a = i;
      refInput[i].b = i+1;
    }

    UDD zero;
    zero.a = 0;
    zero.b = 0;
    // call reduce
    UDDplus plusOp;

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    UDD boltReduce = bolt::amp::reduce( ctl, refInput.begin(), refInput.end(), zero, plusOp );
    UDD stdReduce =  std::accumulate( refInput.begin(), refInput.end(), zero, plusOp ); // out-of-place scan

    EXPECT_EQ(boltReduce,stdReduce);

}
#endif

INSTANTIATE_TYPED_TEST_CASE_P( Integer, ReduceArrayTest, IntegerTests );
INSTANTIATE_TYPED_TEST_CASE_P( Float, ReduceArrayTest, FloatTests );
#if (TEST_DOUBLE == 1)
INSTANTIATE_TYPED_TEST_CASE_P( Double, ReduceArrayTest, DoubleTests );
#endif 

TEST( Reducetest, Normal)
{
    const int aSize = 1<<26;
    std::vector<int> stdInput(aSize);
    std::vector<int> tbbInput(aSize);


    for(int i=0; i<aSize; i++) {
        stdInput[i] = 2;
        tbbInput[i] = 2;
    };

    int hSum = std::accumulate(stdInput.begin(), stdInput.end(), 2);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::Gpu); 
    int sum = bolt::amp::reduce(ctl, tbbInput.begin(), tbbInput.end(), 2);
    if(hSum == sum)
        printf ("\nGPU Test case PASSED %d %d\n", hSum, sum);
    else
        printf ("\nGPU Test case FAILED\n");

};
#if defined( ENABLE_TBB )
void testTBB()
{
    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);
    std::vector<int> tbbInput(aSize);


    for(int i=0; i<aSize; i++) {
        stdInput[i] = 2;
        tbbInput[i] = 2;
    };

    int hSum = std::accumulate(stdInput.begin(), stdInput.end(), 2);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // serial path also tested
    int sum = bolt::amp::reduce(ctl, tbbInput.begin(), tbbInput.end(), 2);
    if(hSum == sum)
        printf ("\nTBB Test case PASSED %d %d\n", hSum, sum);
    else
        printf ("\nTBB Test case FAILED\n");

};
#endif

#if (TEST_DOUBLE == 1)
#if defined( ENABLE_TBB )
void testdoubleTBB()
{
  const int aSize = 1<<24;
    std::vector<double> stdInput(aSize);
    std::vector<double> tbbInput(aSize);


    for(int i=0; i<aSize; i++) {
        stdInput[i] = 3.0;
        tbbInput[i] = 3.0;
    };

    double hSum = std::accumulate(stdInput.begin(), stdInput.end(), 1.0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // serial path also tested
    double sum = bolt::amp::reduce(ctl, tbbInput.begin(), tbbInput.end(), 1.0);
    if(hSum == sum)
        printf ("\nTBB Test case PASSED %lf %lf\n", hSum, sum);
    else
        printf ("\nTBB Test case FAILED\n");
}
#endif
#endif

#if defined( ENABLE_TBB )
void testUDDTBB()
{

    const int aSize = 1<<19;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);

    UDD initial;
    initial.a = 1;
    initial.b = 2;

    for(int i=0; i<aSize; i++) {
        stdInput[i].a = 2;
        stdInput[i].b = 3;
        tbbInput[i].a = 2;
        tbbInput[i].b = 3;

    };
    bolt::amp::plus<UDD> add;
    UDD hSum = std::accumulate(stdInput.begin(), stdInput.end(), initial,add);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // serial path also tested
    UDD sum = bolt::amp::reduce(ctl, tbbInput.begin(), tbbInput.end(), initial, add);
    if(hSum == sum)
        printf ("\nUDDTBB Test case PASSED %d %d %d %d\n", hSum.a, sum.a, hSum.b, sum.b);
    else
        printf ("\nUDDTBB Test case FAILED\n");
        
}
#endif

TEST( ReduceFunctor, NormalLambdaFunctor )
{
  
    int init(0);
    //  Calling the actual functions under test
    std::vector<int> stdInput(1026+1010), boltInput(1026+1010);
    std::fill( stdInput.begin(), stdInput.end(), 100 );
    std::copy( stdInput.begin(), stdInput.end(), boltInput.begin() );

    int stlReduce = std::accumulate(stdInput.begin(),
                                    stdInput.end(),
                                    init,
                                    []( int x, int y ) restrict (amp,cpu){return x+y;}
                                   );

    int boltReduce = bolt::amp::reduce( boltInput.begin( ),
                                        boltInput.end( ),
                                        init,
                                        []( int x, int y ) restrict (amp,cpu){return x+y;}
                                      );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    EXPECT_EQ( stlReduce, boltReduce );
}

TEST( ReduceFunctor, SerialLambdaFunctor )
{
  
    int init(0);
    //  Calling the actual functions under test
    std::vector<int> stdInput(1024), boltInput(1024);
    std::fill( stdInput.begin(), stdInput.end(), 100 );
    std::copy( stdInput.begin(), stdInput.end(), boltInput.begin() );

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::SerialCpu);

    int stlReduce = std::accumulate(stdInput.begin(),
                                    stdInput.end(),
                                    init,
                                    []( int x, int y ) restrict (amp,cpu){return x+y;}
                                   );

    int boltReduce = bolt::amp::reduce( ctl, boltInput.begin( ),
                                        boltInput.end( ),
                                        init,
                                        []( int x, int y ) restrict (amp,cpu){return x+y;}
                                      );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    //EXPECT_EQ( stlReduce, boltReduce );
}

#if defined( ENABLE_TBB )
TEST( ReduceFunctor, MultiCoreLambdaFunctor )
{
  
    int init(0);
    //  Calling the actual functions under test
    std::vector<int> stdInput(1024), boltInput(1024);
    std::fill( stdInput.begin(), stdInput.end(), 100 );
    std::copy( stdInput.begin(), stdInput.end(), boltInput.begin() );

    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu);

    int stlReduce = std::accumulate(stdInput.begin(),
                                    stdInput.end(),
                                    init,
                                    []( int x, int y ) restrict (amp,cpu){return x+y;}
                                   );

    int boltReduce = bolt::amp::reduce( ctl, boltInput.begin( ),
                                        boltInput.end( ),
                                        init,
                                        []( int x, int y ) restrict (amp,cpu){return x+y;}
                                      );

    size_t stdNumElements = std::distance( stdInput.begin( ), stdInput.end() );
    size_t boltNumElements = std::distance( boltInput.begin( ), boltInput.end() );

    //  Both collections should have the same number of elements
    EXPECT_EQ( stdNumElements, boltNumElements );
    //EXPECT_EQ( stlReduce, boltReduce );
}
#endif


#if defined( ENABLE_TBB )
void testTBBDevicevector()
{
    int aSize = 1<<16;
    std::vector<int> stdInput(aSize);
 //   bolt::amp::device_vector<int> tbbInput(aSize, 0);
    bolt::amp::device_vector<int> tbbInput(aSize);

	for (int i = 0; i<aSize; i++) {
        stdInput[i] = (int)i;
        tbbInput[i] = (int)i;
    };

    int hSum = std::accumulate(stdInput.begin(), stdInput.end(), 0);
    bolt::amp::control ctl = bolt::amp::control::getDefault();
    ctl.setForceRunMode(bolt::amp::control::MultiCoreCpu); // serial path also tested
    int sum = bolt::amp::reduce(ctl, tbbInput.begin(), tbbInput.end(), 0);
    if(hSum == sum)
        printf ("\nTBBDevicevector Test case PASSED %d %d\n", hSum, sum);
    else
        printf ("\nTBBDevicevector Test case FAILED*********\n");


};
#endif




int main(int argc, char* argv[])
{
    /*
#if defined( ENABLE_TBB )
    testTBB( );
    #if (TEST_DOUBLE == 1)
    testdoubleTBB();
    #endif
    testUDDTBB();
    testTBBDevicevector();
#endif*/
    
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

