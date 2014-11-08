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

#pragma once
#if !defined( BOLT_CL_STABLESORT_BY_KEY_INL )
#define BOLT_CL_STABLESORT_BY_KEY_INL

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/stable_sort_by_key.h"
#endif

#define BOLT_CL_STABLESORT_BY_KEY_CPU_THRESHOLD 256
#include "bolt/cl/sort_by_key.h"

namespace bolt {
namespace cl {

namespace detail
{
    template< typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if<
        !( std::is_same< typename std::iterator_traits<DVKeys >::value_type, unsigned int >::value ||
           std::is_same< typename std::iterator_traits<DVKeys >::value_type, int >::value 
         )
                           >::type
    sort_by_key_enqueue(control &ctl, const DVKeys& keys_first,
                        const DVKeys& keys_last, const DVValues& values_first,
                        const StrictWeakOrdering& comp, const std::string& cl_code);

    template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp, const std::string& cl_code);


    template<typename DVKeys, typename DVValues, typename StrictWeakOrdering>
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVKeys >::value_type,
                                           unsigned int
                                         >::value
                           >::type  /*If enabled then this typename will be evaluated to void*/
    sort_by_key_enqueue( control &ctl,
                         DVKeys keys_first, DVKeys keys_last,
                         DVValues values_first,
                         StrictWeakOrdering comp, const std::string& cl_code);


    enum stableSort_by_keyTypes { stableSort_by_key_KeyType, stableSort_by_key_KeyIterType, stableSort_by_key_ValueType,
        stableSort_by_key_ValueIterType, stableSort_by_key_lessFunction, stableSort_by_key_end };

    class StableSort_by_key_KernelTemplateSpecializer : public KernelTemplateSpecializer
    {
    public:
        StableSort_by_key_KernelTemplateSpecializer() : KernelTemplateSpecializer( )
        {
            addKernelName( "LocalMergeSort" );
            addKernelName( "merge" );
        }

        const ::std::string operator( ) ( const ::std::vector< ::std::string >& typeNames ) const
        {
            const std::string templateSpecializationString =
                "template __attribute__((mangled_name(" + name( 0 ) + "Instantiated)))\n"
                "kernel void " + name( 0 ) + "Template(\n"
                "global " + typeNames[stableSort_by_key_KeyType] + "* data_ptr,\n"
                ""        + typeNames[stableSort_by_key_KeyIterType] + " data_iter,\n"
                "global " + typeNames[stableSort_by_key_ValueType] + "* value_ptr,\n"
                ""        + typeNames[stableSort_by_key_ValueIterType] + " value_iter,\n"
                "const uint vecSize,\n"
                "local "  + typeNames[stableSort_by_key_KeyType] + "* key_lds,\n"
				"local "  + typeNames[stableSort_by_key_KeyType] + "* key_lds2,\n"
                "local "  + typeNames[stableSort_by_key_ValueType] + "* val_lds,\n"
				"local "  + typeNames[stableSort_by_key_ValueType] + "* val_lds2,\n"
                "global " + typeNames[stableSort_by_key_lessFunction] + " * lessOp\n"
                ");\n\n"

                "template __attribute__((mangled_name(" + name( 1 ) + "Instantiated)))\n"
                "kernel void " + name( 1 ) + "Template(\n"
                "global " + typeNames[stableSort_by_key_KeyType] + "* iKey_ptr,\n"
                ""        + typeNames[stableSort_by_key_KeyIterType] + " iKey_iter,\n"
                "global " + typeNames[stableSort_by_key_ValueType] + "* iValue_ptr,\n"
                ""        + typeNames[stableSort_by_key_ValueIterType] + " iValue_iter,\n"
                "global " + typeNames[stableSort_by_key_KeyType] + "* oKey_ptr,\n"
                ""        + typeNames[stableSort_by_key_KeyIterType] + " oKey_iter,\n"
                "global " + typeNames[stableSort_by_key_ValueType] + "* oValue_ptr,\n"
                ""        + typeNames[stableSort_by_key_ValueIterType] + " oValue_iter,\n"
                "const uint srcVecSize,\n"
                "const uint srcBlockSize,\n"
                "local "  + typeNames[stableSort_by_key_KeyType] + "* key_lds,\n"
                "local "  + typeNames[stableSort_by_key_ValueType] + "* val_lds,\n"
                "global " + typeNames[stableSort_by_key_lessFunction] + " * lessOp\n"
                ");\n\n";

            return templateSpecializationString;
        }
    };

    //Serial CPU code path implementation.
    //Class to hold the key value pair. This will be used to zip th ekey and value together in a vector.
    template <typename keyType, typename valueType>
    class std_stable_sort
    {
    public:
        keyType   key;
        valueType value;
    };
    //This is the functor which will sort the std_stable_sort vector.
    template <typename keyType, typename valueType, typename StrictWeakOrdering>
    class std_stable_sort_comp
    {
    public:
        typedef std_stable_sort<keyType, valueType> KeyValueType;
        std_stable_sort_comp(const StrictWeakOrdering &_swo):swo(_swo)
        {}
        StrictWeakOrdering swo;
        bool operator() (const KeyValueType &lhs, const KeyValueType &rhs) const
        {
            return swo(lhs.key, rhs.key);
        }
    };

    //The serial CPU implementation of stable_sort_by_key routine. This routines zips the key value pair and then sorts
    //using the std::stable_sort routine.
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void serialCPU_stable_sort_by_key(const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                      const RandomAccessIterator2 values_first,
                                      const StrictWeakOrdering& comp)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;
        typedef std_stable_sort<keyType, valType> KeyValuePair;
        typedef std_stable_sort_comp<keyType, valType, StrictWeakOrdering> KeyValuePairFunctor;

        int vecSize = static_cast<int>( std::distance( keys_first, keys_last ) );
        std::vector<KeyValuePair> KeyValuePairVector(vecSize);
        KeyValuePairFunctor functor(comp);
        //Zip the key and values iterators into a std_stable_sort vector.
        for (int i=0; i< vecSize; i++)
        {
            KeyValuePairVector[i].key   = *(keys_first + i);
            KeyValuePairVector[i].value = *(values_first + i);
        }
        //Sort the std_stable_sort vector using std::stable_sort
        std::stable_sort(KeyValuePairVector.begin(), KeyValuePairVector.end(), functor);
        //Extract the keys and values from the KeyValuePair and fill the respective iterators.
        for (int i=0; i< vecSize; i++)
        {
            *(keys_first + i)   = KeyValuePairVector[i].key;
            *(values_first + i) = KeyValuePairVector[i].value;
        }
    }


    /**************************************************************/
    /**StableSort_by_key with keys as unsigned int specialization**/
    /**************************************************************/
    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type,
                                           unsigned int
                                         >::value
                           >::type   /*If enabled then this typename will be evaluated to void*/
    stablesort_by_key_enqueue( control& ctrl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code )
    {
		
        bolt::cl::detail::sort_by_key_enqueue(ctrl, keys_first, keys_last, values_first, comp, cl_code);
        return;    
    }

    /**************************************************************/
    /*StableSort_by_key with keys as int specialization*/
    /**************************************************************/
    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    typename std::enable_if< std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type,
                                           int
                                         >::value
                           >::type   /*If enabled then this typename will be evaluated to void*/
    stablesort_by_key_enqueue( control& ctrl,
                               const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                               const DVRandomAccessIterator2 values_first,
                               const StrictWeakOrdering& comp, const std::string& cl_code )
    {
		
        bolt::cl::detail::sort_by_key_enqueue(ctrl, keys_first, keys_last, values_first, comp, cl_code);
        return;    
    }


    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    typename std::enable_if<
    !(std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type, unsigned int >::value || 
      std::is_same< typename std::iterator_traits<DVRandomAccessIterator1 >::value_type, int >::value  )
                       >::type
    stablesort_by_key_enqueue( control& ctrl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code )
    {
        cl_int l_Error;
        cl_uint vecSize = static_cast< cl_uint >( std::distance( keys_first, keys_last ) );

        /**********************************************************************************
         * Type Names - used in KernelTemplateSpecializer
         *********************************************************************************/
        typedef typename std::iterator_traits< DVRandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< DVRandomAccessIterator2 >::value_type valueType;

        std::vector<std::string> typeNames( stableSort_by_key_end );
        typeNames[stableSort_by_key_KeyType] = TypeName< keyType >::get( );
        typeNames[stableSort_by_key_ValueType] = TypeName< valueType >::get( );
        typeNames[stableSort_by_key_KeyIterType] = TypeName< DVRandomAccessIterator1 >::get( );
        typeNames[stableSort_by_key_ValueIterType] = TypeName< DVRandomAccessIterator2 >::get( );
        typeNames[stableSort_by_key_lessFunction] = TypeName< StrictWeakOrdering >::get( );

        /**********************************************************************************
         * Type Definitions - directrly concatenated into kernel string
         *********************************************************************************/
        std::vector< std::string > typeDefinitions;
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< keyType >::get( ) )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< valueType >::get( ) )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVRandomAccessIterator1 >::get( ) )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< DVRandomAccessIterator2 >::get( ) )
        PUSH_BACK_UNIQUE( typeDefinitions, ClCode< StrictWeakOrdering  >::get( ) )

        /**********************************************************************************
         * Compile Options
         *********************************************************************************/
        bool cpuDevice = ctrl.getDevice( ).getInfo< CL_DEVICE_TYPE >( ) == CL_DEVICE_TYPE_CPU;

        /**********************************************************************************
         * Request Compiled Kernels
         *********************************************************************************/
        std::string compileOptions;

        StableSort_by_key_KernelTemplateSpecializer ss_kts;
        std::vector< ::cl::Kernel > kernels = bolt::cl::getKernels(
            ctrl,
            typeNames,
            &ss_kts,
            typeDefinitions,
            stablesort_by_key_kernels,
            compileOptions );
        // kernels returned in same order as added in KernelTemplaceSpecializer constructor

        size_t localRange= BOLT_CL_STABLESORT_BY_KEY_CPU_THRESHOLD;
		

        //  Make sure that globalRange is a multiple of localRange
        size_t globalRange = vecSize;
        size_t modlocalRange = ( globalRange & ( localRange-1 ) );
        if( modlocalRange )
        {
            globalRange &= ~modlocalRange;
            globalRange += localRange;
        }

	    ALIGNED( 256 ) StrictWeakOrdering aligned_comp( comp );
        control::buffPointer userFunctor = ctrl.acquireBuffer( sizeof( aligned_comp ),
                                                              CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, &aligned_comp );

        //  kernels[ 0 ] sorts values within a workgroup, in parallel across the entire vector
        //  kernels[ 0 ] reads and writes to the same vector
        cl_uint keyLdsSize  = static_cast< cl_uint >( localRange * sizeof( keyType ) );
        cl_uint valueLdsSize  = static_cast< cl_uint >( localRange * sizeof( valueType ) );
        typename  DVRandomAccessIterator1::Payload keys_first_payload = keys_first.gpuPayload( );
        typename  DVRandomAccessIterator2::Payload values_first_payload  = values_first.gpuPayload( ) ;
         // Input buffer
        V_OPENCL( kernels[ 0 ].setArg( 0, keys_first.getContainer().getBuffer( ) ),    "Error setting argument for kernels[ 0 ]" );
        V_OPENCL( kernels[ 0 ].setArg( 1, keys_first.gpuPayloadSize( ),&keys_first_payload ),
                                       "Error setting a kernel argument" );
         // Input buffer
        V_OPENCL( kernels[ 0 ].setArg( 2, values_first.getContainer().getBuffer( ) ),
                                       "Error setting argument for kernels[ 0 ]" );

        V_OPENCL( kernels[ 0 ].setArg( 3, values_first.gpuPayloadSize( ),&values_first_payload),
                                       "Error setting a kernel argument" );
         // Size of scratch buffer
        V_OPENCL( kernels[ 0 ].setArg( 4, vecSize ),            "Error setting argument for kernels[ 0 ]" );
         // Scratch buffer
        V_OPENCL( kernels[ 0 ].setArg( 5, keyLdsSize, NULL ),          "Error setting argument for kernels[ 0 ]" );
		V_OPENCL( kernels[ 0 ].setArg( 6, keyLdsSize, NULL ),          "Error setting argument for kernels[ 0 ]" );
		// Scratch buffer
        V_OPENCL( kernels[ 0 ].setArg( 7, valueLdsSize, NULL ),          "Error setting argument for kernels[ 0 ]" );
        V_OPENCL( kernels[ 0 ].setArg( 8, valueLdsSize, NULL ),          "Error setting argument for kernels[ 0 ]" );
         // User provided functor class
        V_OPENCL( kernels[ 0 ].setArg( 9, *userFunctor ),           "Error setting argument for kernels[ 0 ]" );



        ::cl::CommandQueue& myCQ = ctrl.getCommandQueue( );

        ::cl::Event blockSortEvent;
        l_Error = myCQ.enqueueNDRangeKernel( kernels[ 0 ], ::cl::NullRange,
                ::cl::NDRange( globalRange ), ::cl::NDRange( localRange ), NULL, &blockSortEvent );
        V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for perBlockInclusiveScan kernel" );

        //  Early exit for the case of no merge passes, values are already in destination vector
        if( vecSize <= localRange )
        {
            wait( ctrl, blockSortEvent );
            return;
        };

        //  An odd number of elements requires an extra merge pass to sort
        size_t numMerges = 0;

        //  Calculate the log2 of vecSize, taking into account our block size from kernel 1 is 64
        //  this is how many merge passes we want
        size_t log2BlockSize = vecSize >> 8;
        for( ; log2BlockSize > 1; log2BlockSize >>= 1 )
        {
            ++numMerges;
        }

        //  Check to see if the input vector size is a power of 2, if not we will need last merge pass
        size_t vecPow2 = (vecSize & (vecSize-1));
        numMerges += vecPow2? 1: 0;

        //  Allocate a flipflop buffer because the merge passes are out of place
		device_vector< keyType >       tmpKeyBuffer( vecSize);
		device_vector< valueType >     tmpValueBuffer( vecSize);

         // Size of scratch buffer
        V_OPENCL( kernels[ 1 ].setArg( 8, vecSize ),            "Error setting argument for kernels[ 0 ]" );
         // Scratch buffer
        V_OPENCL( kernels[ 1 ].setArg( 10, keyLdsSize, NULL ),          "Error setting argument for kernels[ 0 ]" );
         // Scratch buffer
        V_OPENCL( kernels[ 1 ].setArg( 11, valueLdsSize, NULL ),          "Error setting argument for kernels[ 0 ]" );
         // User provided functor class
        V_OPENCL( kernels[ 1 ].setArg( 12, *userFunctor ),           "Error setting argument for kernels[ 0 ]" );

        ::cl::Event kernelEvent;
        for( size_t pass = 1; pass <= numMerges; ++pass )
        {
			typename DVRandomAccessIterator1::Payload  keys_first_payload = keys_first.gpuPayload( );
			typename DVRandomAccessIterator2::Payload  values_first_payload = values_first.gpuPayload( );
			typename DVRandomAccessIterator1::Payload  keys_first1_payload = tmpKeyBuffer.begin().gpuPayload( );
			typename DVRandomAccessIterator2::Payload  values_first1_payload = tmpValueBuffer.begin().gpuPayload( );

            //  For each pass, flip the input-output buffers
            if( pass & 0x1 )
            {
                 // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 0, keys_first.getContainer().getBuffer() ),"Error setting argument for kernels[ 0 ]");
                V_OPENCL( kernels[ 1 ].setArg( 1, keys_first.gpuPayloadSize( ),&keys_first_payload ),
                                               "Error setting a kernel argument" );
                 // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 2, values_first.getContainer().getBuffer() ),"Error setting argument for kernels[0]");
                V_OPENCL( kernels[ 1 ].setArg( 3, values_first.gpuPayloadSize( ),&values_first_payload  ),
                                               "Error setting a kernel argument" );
                 // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 4, tmpKeyBuffer.begin().getContainer().getBuffer() ),"Error setting argument for kernels[0]");
                V_OPENCL( kernels[ 1 ].setArg( 5, tmpKeyBuffer.begin().gpuPayloadSize( ),&keys_first1_payload),
                                               "Error setting a kernel argument" );
                 // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 6, tmpValueBuffer.begin().getContainer().getBuffer() ),"Error setting argument for kernels[ 0 ]" );
                V_OPENCL( kernels[ 1 ].setArg( 7, tmpValueBuffer.begin().gpuPayloadSize( ),&values_first1_payload ),
                                               "Error setting a kernel argument" );
            }
            else
            {
                 // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 0, tmpKeyBuffer.begin().getContainer().getBuffer() ),    "Error setting argument for kernels[ 0 ]" );
                V_OPENCL( kernels[ 1 ].setArg( 1, tmpKeyBuffer.begin().gpuPayloadSize( ),&keys_first1_payload),
                                               "Error setting a kernel argument" );
                 // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 2, tmpValueBuffer.begin().getContainer().getBuffer()),    "Error setting argument for kernels[ 0 ]" );
                V_OPENCL( kernels[ 1 ].setArg( 3, tmpValueBuffer.begin().gpuPayloadSize( ),&values_first1_payload ),
                                               "Error setting a kernel argument" );
                V_OPENCL( kernels[ 1 ].setArg( 4, keys_first.getContainer().getBuffer() ),
                                               "Error setting argument for kernels[ 0 ]" ); // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 5, keys_first.gpuPayloadSize( ),&keys_first_payload),
                                               "Error setting a kernel argument" );
                V_OPENCL( kernels[ 1 ].setArg( 6, values_first.getContainer().getBuffer() ),
                                               "Error setting argument for kernels[ 0 ]" ); // Input buffer
                V_OPENCL( kernels[ 1 ].setArg( 7, values_first.gpuPayloadSize( ),&values_first_payload ),
                                               "Error setting a kernel argument" );
            }
            //  For each pass, the merge window doubles
            unsigned srcLogicalBlockSize = static_cast< unsigned >( localRange << (pass-1) );
            V_OPENCL( kernels[ 1 ].setArg( 9, static_cast< unsigned >( srcLogicalBlockSize ) ),
                                           "Error setting argument for kernels[ 0 ]" ); // Size of scratch buffer

            if( pass == numMerges )
            {
                //  Grab the event to wait on from the last enqueue call
                l_Error = myCQ.enqueueNDRangeKernel( kernels[ 1 ], ::cl::NullRange, ::cl::NDRange( globalRange ),
                        ::cl::NDRange( localRange ), NULL, &kernelEvent );
                V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for mergeTemplate kernel" );
            }
            else
            {
                l_Error = myCQ.enqueueNDRangeKernel( kernels[ 1 ], ::cl::NullRange, ::cl::NDRange( globalRange ),
                        ::cl::NDRange( localRange ), NULL, &kernelEvent );
                V_OPENCL( l_Error, "enqueueNDRangeKernel() failed for mergeTemplate kernel" );
            }

        }

        //  If there are an odd number of merges, then the output data is sitting in the temp buffer.  We need to copy
        //  the results back into the input array
        if( numMerges & 1 )
        {
			detail::copy_enqueue(ctrl, tmpKeyBuffer.begin(), vecSize, keys_first);
			detail::copy_enqueue(ctrl, tmpValueBuffer.begin(), vecSize, values_first);

            /*::cl::Event copyEvent;

            wait( ctrl, kernelEvent );
            l_Error = myCQ.enqueueCopyBuffer( tmpKeyBuffer.begin().getContainer().getBuffer(), keys_first.getContainer().getBuffer(), 0,
                                               keys_first.m_Index * sizeof( keyType ),
                                               vecSize * sizeof( keyType ), NULL, NULL );
            V_OPENCL( l_Error, "device_vector failed to copy data inside of operator=()" );

            l_Error = myCQ.enqueueCopyBuffer( tmpValueBuffer.begin().getContainer().getBuffer(), values_first.getContainer().getBuffer(), 0,
                                               values_first.m_Index * sizeof( valueType ),
                                               vecSize * sizeof( valueType ), NULL, &copyEvent );
            V_OPENCL( l_Error, "device_vector failed to copy data inside of operator=()" );

            wait( ctrl, copyEvent );*/
        }
        else
        {
            wait( ctrl, kernelEvent );
        }

        return;
    }// END of sort_enqueue

    //Non Device Vector specialization.
    //This implementation creates a cl::Buffer and passes the cl buffer to the sort specialization whichtakes the cl buffer as a parameter.
    //In the future, Each input buffer should be mapped to the device_vector and the specialization specific to device_vector should be called.
    //This implementation creates a cl::Buffer and passes the cl buffer to the sort specialization whichtakes
    //the cl buffer as a parameter.
    //In the future, Each input buffer should be mapped to the device_vector and the specialization specific
    //to device_vector should be called.
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_pick_iterator( control &ctl,
                                const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                const RandomAccessIterator2 values_first,
                                const StrictWeakOrdering& comp, const std::string& cl_code,
                                std::random_access_iterator_tag, std::random_access_iterator_tag )
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;

        int vecSize = static_cast<int>( std::distance( keys_first, keys_last ) );
        if( vecSize < 2 )
            return;

        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();

				

        if( runMode == bolt::cl::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );

        }
        #if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
        if( runMode == bolt::cl::control::SerialCpu )
        { 
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_STABLESORTBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Stable_Sort_By_Key::SERIAL_CPU");
            #endif
            serialCPU_stable_sort_by_key(keys_first, keys_last, values_first, comp);
            return;
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
            #ifdef ENABLE_TBB
                #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_STABLESORTBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Stable_Sort_By_Key::MULTICORE_CPU");
                #endif
                bolt::btbb::stable_sort_by_key(keys_first, keys_last, values_first, comp);
            #else
                throw std::runtime_error("MultiCoreCPU Version of stable_sort_by_key not Enabled! \n");
            #endif
            return;
        }
        else
        {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_STABLESORTBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Stable_Sort_By_Key::OPENCL_GPU");
            #endif
			
            device_vector< keyType > dvKeys( keys_first, keys_last, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, ctl );
            device_vector< valType > dvValues( values_first, vecSize, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, false, ctl );

            //Now call the actual cl algorithm
            stablesort_by_key_enqueue( ctl, dvKeys.begin(), dvKeys.end(), dvValues.begin( ), comp, cl_code );

            //Map the buffer back to the host
            dvKeys.data( );
            dvValues.data( );
            return;
        }
    }

    //Device Vector specialization
    template< typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_pick_iterator( control &ctl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    bolt::cl::device_vector_tag, bolt::cl::device_vector_tag )
    {
        typedef typename std::iterator_traits< DVRandomAccessIterator1 >::value_type keyType;
        typedef typename std::iterator_traits< DVRandomAccessIterator2 >::value_type valueType;
        int vecSize = static_cast<int>( std::distance( keys_first, keys_last ) );
        if( vecSize < 2 )
            return;

        bolt::cl::control::e_RunMode runMode = ctl.getForceRunMode();

        if( runMode == bolt::cl::control::Automatic )
        {
            runMode = ctl.getDefaultPathToRun( );
        }
        #if defined(BOLT_DEBUG_LOG)
        BOLTLOG::CaptureLog *dblog = BOLTLOG::CaptureLog::getInstance();
        #endif
		
        if( runMode == bolt::cl::control::SerialCpu )
        {
		        #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_STABLESORTBYKEY,BOLTLOG::BOLT_SERIAL_CPU,"::Stable_Sort_By_Key::SERIAL_CPU");
                #endif
                typename bolt::cl::device_vector< keyType >::pointer   keysPtr   =  keys_first.getContainer( ).data( );
                typename bolt::cl::device_vector< valueType >::pointer valuesPtr =  values_first.getContainer( ).data( );
                serialCPU_stable_sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                             &valuesPtr[values_first.m_Index], comp);
                return;
        }
        else if( runMode == bolt::cl::control::MultiCoreCpu )
        {
            #ifdef ENABLE_TBB
			    #if defined(BOLT_DEBUG_LOG)
                dblog->CodePathTaken(BOLTLOG::BOLT_STABLESORTBYKEY,BOLTLOG::BOLT_MULTICORE_CPU,"::Stable_Sort_By_Key::MULTICORE_CPU");
                #endif
                typename bolt::cl::device_vector< keyType >::pointer   keysPtr   =  keys_first.getContainer( ).data( );
                typename bolt::cl::device_vector< valueType >::pointer valuesPtr =  values_first.getContainer( ).data( );
                bolt::btbb::stable_sort_by_key(&keysPtr[keys_first.m_Index], &keysPtr[keys_last.m_Index],
                                             &valuesPtr[values_first.m_Index], comp);
                return;
             #else
                throw std::runtime_error("MultiCoreCPU Version of stable_sort_by_key not Enabled! \n");
             #endif
        }
        else
        {
		    #if defined(BOLT_DEBUG_LOG)
            dblog->CodePathTaken(BOLTLOG::BOLT_STABLESORTBYKEY,BOLTLOG::BOLT_OPENCL_GPU,"::Stable_Sort_By_Key::OPENCL_GPU");
            #endif
			
            stablesort_by_key_enqueue( ctl, keys_first, keys_last, values_first, comp, cl_code );
        }
        return;
    }

    //Fancy iterator specialization
    template<typename DVRandomAccessIterator1, typename DVRandomAccessIterator2, typename StrictWeakOrdering>
    void stablesort_by_key_pick_iterator( control &ctl,
                                    const DVRandomAccessIterator1 keys_first, const DVRandomAccessIterator1 keys_last,
                                    const DVRandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code, bolt::cl::fancy_iterator_tag )
    {
        static_assert(std::is_same< DVRandomAccessIterator1, bolt::cl::fancy_iterator_tag >::value  , "It is not possible to output to fancy iterators; they are not mutable! " );
    }


    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    std::random_access_iterator_tag, std::random_access_iterator_tag )
    {
        return stablesort_by_key_pick_iterator( ctl, keys_first, keys_last, values_first,
                                    comp, cl_code,
                                    typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                    typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
    };

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    bolt::cl::fancy_iterator_tag, std::input_iterator_tag )
    {
        static_assert(std::is_same< RandomAccessIterator1, bolt::cl::fancy_iterator_tag>::value, "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert(std::is_same< RandomAccessIterator2,std::input_iterator_tag >::value  , "It is not possible to sort fancy iterators. They are not mutable" );
    }
    // Wrapper that uses default control class, iterator interface
    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    std::input_iterator_tag, std::input_iterator_tag )
    {
        //  \TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
        //  to a temporary buffer.  Should we?
        static_assert(std::is_same< RandomAccessIterator1,std::input_iterator_tag>::value  , "Bolt only supports random access iterator types" );
        static_assert(std::is_same< RandomAccessIterator2,std::input_iterator_tag>::value  , "Bolt only supports random access iterator types" );
    };


    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stablesort_by_key_detect_random_access( control &ctl,
                                    const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last,
                                    const RandomAccessIterator2 values_first,
                                    const StrictWeakOrdering& comp, const std::string& cl_code,
                                    std::input_iterator_tag, bolt::cl::fancy_iterator_tag )
    {


        static_assert(std::is_same< RandomAccessIterator2, bolt::cl::fancy_iterator_tag>::value, "It is not possible to sort fancy iterators. They are not mutable" );
        static_assert(std::is_same< RandomAccessIterator1,std::input_iterator_tag >::value  , "It is not possible to sort fancy iterators. They are not mutable" );

    }



}//namespace bolt::cl::detail


    template< typename RandomAccessIterator1, typename RandomAccessIterator2 >
    void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, const std::string& cl_code )
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type T;

        detail::stablesort_by_key_detect_random_access( control::getDefault( ),
                                           keys_first, keys_last, values_first,
                                           less< T >( ), cl_code,
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stable_sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, StrictWeakOrdering comp, const std::string& cl_code )
    {
        detail::stablesort_by_key_detect_random_access( control::getDefault( ),
                                           keys_first, keys_last, values_first,
                                           comp, cl_code,
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2 >
    void stable_sort_by_key( control &ctl, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, const std::string& cl_code)
    {
        typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type T;

        detail::stablesort_by_key_detect_random_access(ctl,
                                           keys_first, keys_last, values_first,
                                          less< T >( ), cl_code,
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

    template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
    void stable_sort_by_key( control &ctl, RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last,
        RandomAccessIterator2 values_first, StrictWeakOrdering comp, const std::string& cl_code )
    {
        detail::stablesort_by_key_detect_random_access(ctl,
                                           keys_first, keys_last, values_first,
                                          comp, cl_code,
                                           typename std::iterator_traits< RandomAccessIterator1 >::iterator_category( ),
                                           typename std::iterator_traits< RandomAccessIterator2 >::iterator_category( ) );
        return;
    }

}//namespace bolt::cl
}//namespace bolt

#endif
