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

#if !defined( BOLT_AMP_COPY_INL )
#define BOLT_AMP_COPY_INL
#pragma once

#define COPY_WAVEFRONT_SIZE 256 

#include <algorithm>
#include <type_traits>
#include "bolt/amp/bolt.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/iterator_traits.h"
#include <amp.h>

#ifdef ENABLE_TBB
//TBB Includes
#include "bolt/btbb/copy.h"
#endif


namespace bolt {
namespace amp {


namespace detail {

template< typename DVInputIterator, typename Size, typename DVOutputIterator >
 void  copy_enqueue(bolt::amp::control &ctrl, const DVInputIterator& first, const Size& n,
    const DVOutputIterator& result)
{

	  concurrency::accelerator_view av = ctrl.getAccelerator().get_default_view();

      typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
      typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
    
      const int szElements = static_cast< int >( n );
      const unsigned int leng =  szElements + COPY_WAVEFRONT_SIZE - (szElements % COPY_WAVEFRONT_SIZE);

	 concurrency::extent< 1 > inputExtent(leng);

      try
      {

         concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
         {
             int globalId = idx[ 0 ];

            if( globalId >= szElements)
                return;

             result[globalId] = first[globalId];
         });
      }

   
      catch(std::exception &e)
      {
        std::cout << "Exception while calling bolt::amp::copy parallel_for_each " ;
        std::cout<< e.what() << std::endl;
        throw std::exception();
      }	

}


/*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of copy, it already assumes
             *  random access iterators.  This overload is called strictly for non-device_vector iterators
            */
template<typename InputIterator, typename Size, typename OutputIterator>
void copy_pick_iterator( bolt::amp::control &ctrl,  const InputIterator& first, const Size& n,
                         const OutputIterator& result, std::random_access_iterator_tag, std::random_access_iterator_tag )
{

    typedef typename std::iterator_traits<InputIterator>::value_type iType;
    typedef typename std::iterator_traits<OutputIterator>::value_type oType;


     bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );
     if (runMode == bolt::amp::control::Automatic)
     {
         runMode = ctrl.getDefaultPathToRun();
     }


     if( runMode == bolt::amp::control::SerialCpu )
     {
         
         #if defined( _WIN32 )
           std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
         #else
           std::copy_n( first, n, result );
         #endif
     }
     else if( runMode == bolt::amp::control::MultiCoreCpu )
     {
        #ifdef ENABLE_TBB   
           bolt::btbb::copy_n(first, n, &(*result));
        #else
            throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
        #endif
     }
     else
     {
        // A host 2 host copy operation, just fallback on the optimized std:: implementation
        #if defined( _WIN32 )
          std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
        #else
          std::copy_n( first, n, result );
        #endif
     }
}



// This template is called by the non-detail versions of copy, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator( bolt::amp::control &ctrl,  const DVInputIterator& first, const Size& n,
                         const DVOutputIterator& result, bolt::amp::device_vector_tag, bolt::amp::device_vector_tag )
{
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );
     if (runMode == bolt::amp::control::Automatic)
     {
         runMode = ctrl.getDefaultPathToRun();
     }
     
     if( runMode == bolt::amp::control::SerialCpu )
     {
          
            typename bolt::amp::device_vector< iType >::pointer copySrc =  const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));
            typename bolt::amp::device_vector< oType >::pointer copyDest =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
#if defined( _WIN32 )
            std::copy_n( &copySrc[first.m_Index], n, stdext::make_checked_array_iterator( &copyDest[result.m_Index], n) );
#else
            std::copy_n( &copySrc[first.m_Index], n, &copyDest[result.m_Index] );
#endif
            

            return;
     }
     else if( runMode == bolt::amp::control::MultiCoreCpu )
     {

         #ifdef ENABLE_TBB
             typename bolt::amp::device_vector< iType >::pointer copySrc =  const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));
             typename bolt::amp::device_vector< oType >::pointer copyDest =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
             bolt::btbb::copy_n( &copySrc[first.m_Index], n, &copyDest[result.m_Index] );    
            return;
         #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
         #endif
     }
     else
     {	
         copy_enqueue( ctrl, first, n, result);
     }
}

template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator( bolt::amp::control &ctrl,  const DVInputIterator& first, const Size& n,
                         const DVOutputIterator& result, bolt::amp::fancy_iterator_tag, bolt::amp::device_vector_tag )
{
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );
     if (runMode == bolt::amp::control::Automatic)
     {
         runMode = ctrl.getDefaultPathToRun();
     }
     
     if( runMode == bolt::amp::control::SerialCpu )
     {
          
            typename bolt::amp::device_vector< oType >::pointer copyDest =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
#if defined( _WIN32 )
            std::copy_n( first, n, stdext::make_checked_array_iterator( &copyDest[result.m_Index], n) );
#else
            std::copy_n( first, n, &copyDest[result.m_Index] );
#endif
            

            return;
     }
     else if( runMode == bolt::amp::control::MultiCoreCpu )
     {

         #ifdef ENABLE_TBB
             typename bolt::amp::device_vector< oType >::pointer copyDest =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
             bolt::btbb::copy_n( first, n, &copyDest[result.m_Index] );    
            return;
         #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
         #endif
     }
     else
     {	
         copy_enqueue( ctrl, first, n, result);
     }
}

// This template is called by the non-detail versions of copy, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator( bolt::amp::control &ctrl,  const DVInputIterator& first, const Size& n,
                         const DVOutputIterator& result, std::random_access_iterator_tag, bolt::amp::device_vector_tag)
{
    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;
     bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );

     if (runMode == bolt::amp::control::Automatic)
     {
         runMode = ctrl.getDefaultPathToRun();
     }

   
     if( runMode == bolt::amp::control::SerialCpu )
     {
           
            typename bolt::amp::device_vector< oType >::pointer copyDest =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
#if defined( _WIN32 )
            std::copy_n( first, n, stdext::make_checked_array_iterator( &copyDest[result.m_Index], n) );
#else
            std::copy_n( first, n, &copyDest[result.m_Index] );
#endif
           
            return;
     }
     else if( runMode == bolt::amp::control::MultiCoreCpu )
     {

         #ifdef ENABLE_TBB
              typename bolt::amp::device_vector< oType >::pointer copyDest =  const_cast<typename bolt::amp::device_vector< oType >::pointer>(result.getContainer( ).data( ));
              bolt::btbb::copy_n( first, n, &copyDest[result.m_Index] );

            return;
         #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
         #endif
     }
     else
     {
       
        device_vector< iType, concurrency::array_view> dvInput( first, n, false, ctrl );
        //Now call the actual algorithm
        copy_enqueue( ctrl, dvInput.begin(), n, result );
        //Map the buffer back to the host
#if 0
        dvInput.data( );
#endif
     }
}

// This template is called by the non-detail versions of copy, it already assumes random access iterators
// This is called strictly for iterators that are derived from device_vector< T >::iterator
template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(bolt::amp::control &ctrl,  const DVInputIterator& first, const Size& n,
                        const DVOutputIterator& result, bolt::amp::device_vector_tag, std::random_access_iterator_tag)
{

    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

    bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );
    if (runMode == bolt::amp::control::Automatic)
    {
        runMode = ctrl.getDefaultPathToRun();
    }
     
     if( runMode == bolt::amp::control::SerialCpu )
     {
         #if defined( _WIN32 )
           std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
         #else
           std::copy_n( first, n, result );
         #endif
           return;
     }
     else if( runMode == bolt::amp::control::MultiCoreCpu )
     {

           #ifdef ENABLE_TBB
               typename bolt::amp::device_vector< iType >::pointer copySrc =  const_cast<typename bolt::amp::device_vector< iType >::pointer>(first.getContainer( ).data( ));
               bolt::btbb::copy_n( &copySrc[first.m_Index], n, result );
           #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
           #endif
     }
     else
     {

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        // Map the output iterator to a device_vector
        device_vector< oType, concurrency::array_view> dvOutput( result, n, true, ctrl );
        copy_enqueue( ctrl, first, n, dvOutput.begin( ));
        dvOutput.data();
     }
}


template<typename DVInputIterator, typename Size, typename DVOutputIterator>
void copy_pick_iterator(bolt::amp::control &ctrl,  const DVInputIterator& first, const Size& n,
                        const DVOutputIterator& result, bolt::amp::fancy_iterator_tag, std::random_access_iterator_tag)
{

    typedef typename std::iterator_traits<DVInputIterator>::value_type iType;
    typedef typename std::iterator_traits<DVOutputIterator>::value_type oType;

    bolt::amp::control::e_RunMode runMode = ctrl.getForceRunMode( );
    if (runMode == bolt::amp::control::Automatic)
    {
        runMode = ctrl.getDefaultPathToRun();
    }
     
     if( runMode == bolt::amp::control::SerialCpu )
     {
         #if defined( _WIN32 )
           std::copy_n( first, n, stdext::checked_array_iterator<oType*>(&(*result), n ) );
         #else
           std::copy_n( first, n, result );
         #endif
           return;
     }
     else if( runMode == bolt::amp::control::MultiCoreCpu )
     {

           #ifdef ENABLE_TBB
               bolt::btbb::copy_n( first, n, result );
           #else
                throw std::runtime_error( "The MultiCoreCpu version of Copy is not enabled to be built." );
           #endif
     }
     else
     {

        // Use host pointers memory since these arrays are only read once - no benefit to copying.
        // Map the output iterator to a device_vector
        device_vector< oType, concurrency::array_view> dvOutput( result, n, true, ctrl );
        copy_enqueue( ctrl, first, n, dvOutput.begin( ));
        dvOutput.data();
     }
}


template<typename InputIterator, typename Size, typename OutputIterator >
OutputIterator copy_detect_random_access( bolt::amp::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, std::random_access_iterator_tag )
{
    if (n < 0)
    {
      std::cout<<"\n Number of elements to copy cannot be negative! "<< std::endl;
    }
    if (n > 0)
    {
      copy_pick_iterator( ctrl, first, n, result, typename std::iterator_traits< InputIterator >::iterator_category( ),
                          typename std::iterator_traits< OutputIterator >::iterator_category( ));
    }
    return (result+n);
};

template<typename InputIterator, typename Size, typename OutputIterator >
OutputIterator copy_detect_random_access( bolt::amp::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, bolt::amp::device_vector_tag )
{
    if (n < 0)
    {
      std::cout<<"\n Number of elements to copy cannot be negative! "<< std::endl;
    }
    if (n > 0)
    {

      copy_pick_iterator( ctrl, first, n, result, typename std::iterator_traits< InputIterator >::iterator_category( ),
                          typename std::iterator_traits< OutputIterator >::iterator_category( )); 
    }
    return (result+n);
};

template<typename InputIterator, typename Size, typename OutputIterator >
OutputIterator copy_detect_random_access( bolt::amp::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, bolt::amp::fancy_iterator_tag )
{
    if (n < 0)
    {
      std::cout<<"\n Number of elements to copy cannot be negative! "<< std::endl;
    }
    if (n > 0)
    {

      copy_pick_iterator( ctrl, first, n, result, typename std::iterator_traits< InputIterator >::iterator_category( ),
                          typename std::iterator_traits< OutputIterator >::iterator_category( )); 
    }
    return (result+n);
};


// Wrapper that uses default control class, iterator interface
template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_detect_random_access( bolt::amp::control& ctrl, const InputIterator& first, const Size& n,
                const OutputIterator& result, std::input_iterator_tag )
{
    static_assert( std::is_same< InputIterator, /*bolt::amp*/std::input_iterator_tag  >::value, "Bolt only supports random access iterator types" );
    return NULL;
};

}//End of detail namespace



// user control
template<typename InputIterator, typename OutputIterator>
OutputIterator copy(bolt::amp::control &ctrl,  InputIterator first, InputIterator last, OutputIterator result)
{
    int n = static_cast<int>( std::distance( first, last ) );
    return detail::copy_detect_random_access( ctrl, first, n, result,
         typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

// default control
template<typename InputIterator, typename OutputIterator>
OutputIterator copy( InputIterator first, InputIterator last, OutputIterator result)
{

    int n = static_cast<int>( std::distance( first, last ) );
            return detail::copy_detect_random_access( control::getDefault(), first, n, result,
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

// default control
template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(InputIterator first, Size n, OutputIterator result)
{
            return detail::copy_detect_random_access( control::getDefault(), first, n, result,
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

// user control
template<typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(bolt::amp::control &ctrl, InputIterator first, Size n, OutputIterator result)
{
    return detail::copy_detect_random_access( ctrl, first, n, result,
                    typename std::iterator_traits< InputIterator >::iterator_category( ) );
}

}//end of amp namespace
};//end of bolt namespace

#endif
