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

///////////////////////////////////////////////////////////////////////////////
// AMP Transform
//////////////////////////////////////////////////////////////////////////////

#pragma once
#if !defined( BOLT_AMP_TRANSFORM_INL )
#define BOLT_AMP_TRANSFORM_INL
#define TRANSFORM_WAVEFRNT_SIZE 256

#ifdef BOLT_ENABLE_PROFILING
#include "bolt/AsyncProfiler.h"
//AsyncProfiler aProfiler("transform");
#endif

#include <algorithm>
#include <type_traits>
#include "bolt/amp/bolt.h"
#include "bolt/amp/device_vector.h"
#include "bolt/amp/iterator/iterator_traits.h"

#ifdef ENABLE_TBB
    #include "bolt/btbb/transform.h"
#endif



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace bolt
{
    namespace amp
    {
        namespace detail
        {
            namespace serial
            {

                template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
                void binary_transform( const InputIterator1& first1, const InputIterator1& last1,
                                    const InputIterator2& first2, const OutputIterator& result, const BinaryFunction& f)
                {
                    size_t sz = (last1 - first1);
                    if (sz == 0)
                        return;
                    for(int index=0; index < (int)(sz); index++)
                    {
                        *(result + index) = f( *(first1+index), *(first2+index) );
                    }
                }

                template<typename Iterator, typename OutputIterator, typename UnaryFunction>
                void unary_transform( Iterator& first, Iterator& last,
                                OutputIterator& result, UnaryFunction& f )
                {
                    size_t sz = (last - first);
                    if (sz == 0)
                        return;
                    for(int index=0; index < (int)(sz); index++)
                    {
                        *(result + index) = f( *(first+index) );
                    }
        
                    return;
                }
            }

           
            template< typename DVInputIterator1, typename DVInputIterator2, typename DVOutputIterator, typename BinaryFunction >
            void transform_enqueue( bolt::amp::control &ctl,
                                    const DVInputIterator1& first1,
                                    const DVInputIterator1& last1,
                                    const DVInputIterator2& first2,
                                    const DVOutputIterator& result,
                                    const BinaryFunction& f)
            {
                concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();

                typedef typename std::iterator_traits< DVInputIterator1 >::value_type iType1;
                typedef typename std::iterator_traits< DVInputIterator2 >::value_type iType2;
                typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

                const int szElements =  static_cast< int >( std::distance( first1, last1 ) );

                const unsigned int leng =  szElements + TRANSFORM_WAVEFRNT_SIZE - (szElements % TRANSFORM_WAVEFRNT_SIZE);

                concurrency::extent< 1 > inputExtent(leng);

                try
                {

                    concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                    {
                        int globalId = idx[ 0 ];

                        if( globalId >= szElements)
                        return;

                        result[globalId] = f(first1[globalId],first2[globalId]);
                    });
                }

			    catch(std::exception &e)
                {

                      std::cout << "Exception while calling bolt::amp::transform parallel_for_each"<<e.what()<<std::endl;

                      return;
                }
            };

            template< typename DVInputIterator, typename DVOutputIterator, typename UnaryFunction >
            void transform_unary_enqueue(bolt::amp::control &ctl,
                                         const DVInputIterator& first,
                                         const DVInputIterator& last,
                                         const DVOutputIterator& result,
                                         const UnaryFunction& f)
            {

               typedef typename std::iterator_traits< DVInputIterator >::value_type iType;
               typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;


                const int szElements =  static_cast< int >( std::distance( first, last ) );
                concurrency::accelerator_view av = ctl.getAccelerator().get_default_view();

                const unsigned int leng =  szElements + TRANSFORM_WAVEFRNT_SIZE - (szElements % TRANSFORM_WAVEFRNT_SIZE);

                concurrency::extent< 1 > inputExtent(leng);

                try
                {

                    concurrency::parallel_for_each(av,  inputExtent, [=](concurrency::index<1> idx) restrict(amp)
                    {
                        int globalId = idx[ 0 ];

                        if( globalId >= szElements)
                        return;

                        result[globalId] = f(first[globalId]);
                    });
                }

			    catch(std::exception &e)
                {

                      std::cout << "Exception while calling bolt::amp::transform parallel_for_each"<<e.what()<<std::endl;

                      return;
                }
            }

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
             void transform_pick_iterator(bolt::amp::control &ctl,
              const InputIterator1& first1,
              const InputIterator1& last1,
              const InputIterator2& first2,
              const OutputIterator& result,
              const BinaryFunction& f,
              bolt::amp::fancy_iterator_tag,
              bolt::amp::device_vector_tag)
            {
                typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
                typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
                typedef typename std::iterator_traits<OutputIterator>::value_type oType;
                int sz = static_cast<int>(last1 - first1);
                if (sz == 0)
                  return;
                // Use host pointers memory since these arrays are only read once - no benefit to copying.
                const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if (runMode == bolt::amp::control::SerialCpu)
                {
                  std::transform(first1, last1, first2, result, f);
                  return;
                }
                else if (runMode == bolt::amp::control::MultiCoreCpu)
                {
#if defined( ENABLE_TBB )

                  bolt::btbb::transform(first1, last1, first2, result, f);
#else
                  throw Concurrency::runtime_exception("The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                  return;
                }
                else
                {
                  // Use host pointers memory since these arrays are only read once - no benefit to copying.
                  transform_enqueue(ctl, first1, last1, first2, result, f);
                }
              }

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
             void transform_pick_iterator(bolt::amp::control &ctl,
              const InputIterator1& first1,
              const InputIterator1& last1,
              const InputIterator2& first2,
              const OutputIterator& result,
              const BinaryFunction& f,
              bolt::amp::device_vector_tag,
              bolt::amp::fancy_iterator_tag)
            {
                typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
                typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
                typedef typename std::iterator_traits<OutputIterator>::value_type oType;
                int sz = static_cast<int>(last1 - first1);
                if (sz == 0)
                  return;
                // Use host pointers memory since these arrays are only read once - no benefit to copying.
                const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if (runMode == bolt::amp::control::SerialCpu)
                {
                   typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  first1.getContainer( ).data( );
                   typename bolt::amp::device_vector< oType >::pointer resPtr =  result.getContainer( ).data( );

#if defined( _WIN32 )

                  std::transform( &firstPtr[ first1.m_Index ], &firstPtr[first1.m_Index +  sz ], first2,
                  stdext::make_checked_array_iterator( &resPtr[ result.m_Index ], sz ), f );
#else
                   std::transform( &firstPtr[ first1.m_Index ], &firstPtr[ first1.m_Index + sz ],
                                    first2, &resPtr[ result.m_Index ], f );
#endif
                   return;
                }
                else if (runMode == bolt::amp::control::MultiCoreCpu)
                {
#if defined( ENABLE_TBB )
                  typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  first1.getContainer( ).data( );
                  typename bolt::amp::device_vector< oType >::pointer resPtr =  result.getContainer( ).data( );
                  bolt::btbb::transform(&firstPtr[ first1.m_Index ],&firstPtr[ first1.m_Index + sz ],
                                        first2, &resPtr[ result.m_Index ],f);

#else
                 throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                 return;
                }
                else
                {
                  // Use host pointers memory since these arrays are only read once - no benefit to copying.
                  transform_enqueue(ctl, first1, last1, first2, result, f);
                }
              }


             /*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of transform, it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
            */
            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_pick_iterator( bolt::amp::control &ctl,
                                     const InputIterator1& first1,
                                     const InputIterator1& last1,
                                     const InputIterator2& first2,
                                     const OutputIterator& result,
                                     const BinaryFunction& f,
                                     std::random_access_iterator_tag,
                                     std::random_access_iterator_tag)
            {
                typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
                typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
                typedef typename std::iterator_traits<OutputIterator>::value_type oType;
                int sz = static_cast<int>(last1 - first1);
                if (sz == 0)
                    return;
                // Use host pointers memory since these arrays are only read once - no benefit to copying.
               const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
               if( runMode == bolt::amp::control::SerialCpu )
               {
                    std::transform( first1, last1, first2, result, f );
                    return;
               }
               else if( runMode == bolt::amp::control::MultiCoreCpu )
               {
#if defined( ENABLE_TBB )

                    bolt::btbb::transform(first1,last1,first2,result,f);
#else
                    throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                    return;
               }
               else
               {
                    // Use host pointers memory since these arrays are only read once - no benefit to copying.
                    // Map the input iterator to a device_vector
                    //device_vector< iType > dvInput( first1, last1, ctl );
                    device_vector< iType1, concurrency::array_view > dvInput( first1, last1, false, ctl );
                    //device_vector< iType > dvInput2( first2, sz, true, ctl );
                    device_vector< iType2, concurrency::array_view > dvInput2( first2, sz, false, ctl );
                    // Map the output iterator to a device_vector
                    //device_vector< oType > dvOutput( result, sz, false, ctl );
                    device_vector< oType, concurrency::array_view > dvOutput( result, sz, true, ctl );
                    transform_enqueue( ctl, dvInput.begin( ), dvInput.end( ), dvInput2.begin( ), dvOutput.begin( ), f  );
                    // This should immediately map/unmap the buffer
                    dvOutput.data( );
               }
            }

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_pick_iterator( bolt::amp::control &ctl,
                                     const InputIterator1& first1,
                                     const InputIterator1& last1,
                                     const InputIterator2& first2,
                                     const OutputIterator& result,
                                     const BinaryFunction& f,
                                     bolt::amp::fancy_iterator_tag,
                                     std::random_access_iterator_tag)
            {
                typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
                typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
                typedef typename std::iterator_traits<OutputIterator>::value_type oType;
                int sz = static_cast<int>(last1 - first1);
                if (sz == 0)
                    return;
                // Use host pointers memory since these arrays are only read once - no benefit to copying.
               const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
               if( runMode == bolt::amp::control::SerialCpu )
               {
                    std::transform( first1, last1, first2, result, f );
                    return;
               }
               else if( runMode == bolt::amp::control::MultiCoreCpu )
               {
#if defined( ENABLE_TBB )

                    bolt::btbb::transform(first1,last1,first2,result,f);
#else
                    throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                    return;
               }
               else
               {
                    // Use host pointers memory since these arrays are only read once - no benefit to copying.
                    // Map the input iterator to a device_vector
                    device_vector< iType2, concurrency::array_view > dvInput2( first2, sz, false, ctl );
                    // Map the output iterator to a device_vector
                    device_vector< oType, concurrency::array_view > dvOutput( result, sz, true, ctl );
                    transform_enqueue( ctl, first1, last1, dvInput2.begin( ), dvOutput.begin( ), f  );
                    // This should immediately map/unmap the buffer
                    dvOutput.data( );
               }
            }

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_pick_iterator( bolt::amp::control &ctl,
                                     const InputIterator1& first1,
                                     const InputIterator1& last1,
                                     const InputIterator2& first2,
                                     const OutputIterator& result,
                                     const BinaryFunction& f,
                                     std::random_access_iterator_tag,
                                     bolt::amp::fancy_iterator_tag)
            {
                typedef typename std::iterator_traits<InputIterator1>::value_type iType1;
                typedef typename std::iterator_traits<InputIterator2>::value_type iType2;
                typedef typename std::iterator_traits<OutputIterator>::value_type oType;
                int sz = static_cast<int>(last1 - first1);
                if (sz == 0)
                    return;
                // Use host pointers memory since these arrays are only read once - no benefit to copying.
               const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
               if( runMode == bolt::amp::control::SerialCpu )
               {
                    std::transform( first1, last1, first2, result, f );
                    return;
               }
               else if( runMode == bolt::amp::control::MultiCoreCpu )
               {
#if defined( ENABLE_TBB )

                    bolt::btbb::transform(first1,last1,first2,result,f);
#else
                    throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0 );
#endif
                    return;
               }
               else
               {
                    // Use host pointers memory since these arrays are only read once - no benefit to copying.
                    // Map the input iterator to a device_vector
                    //device_vector< iType > dvInput( first1, last1, ctl );
                    device_vector< iType1, concurrency::array_view > dvInput( first1, last1, false, ctl );
                    // Map the output iterator to a device_vector
                    device_vector< oType, concurrency::array_view > dvOutput( result, sz, true, ctl );
                    transform_enqueue( ctl, dvInput.begin( ), dvInput.end( ), first2, dvOutput.begin( ), f  );
                    // This should immediately map/unmap the buffer
                    dvOutput.data( );
               }
            }


            // This template is called by the non-detail versions of transform, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVInputIterator1, typename DVInputIterator2, typename DVOutputIterator, typename BinaryFunction>
            void transform_pick_iterator( bolt::amp::control &ctl,
                                     const DVInputIterator1& first1,
                                     const DVInputIterator1& last1,
                                     const DVInputIterator2& first2,
                                     const DVOutputIterator& result,
                                     const BinaryFunction& f,
                                     bolt::amp::device_vector_tag,
                                     bolt::amp::device_vector_tag)
            {
               typedef typename std::iterator_traits< DVInputIterator1 >::value_type iType1;
               typedef typename std::iterator_traits< DVInputIterator2 >::value_type iType2;
               typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

               int sz = static_cast<int>(std::distance( first1, last1 ));
               if( sz == 0 )
                    return;

               const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

               if( runMode == bolt::amp::control::SerialCpu )
               {
                   typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  first1.getContainer( ).data( );
                   typename bolt::amp::device_vector< iType2 >::pointer secPtr =  first2.getContainer( ).data( );
                   typename bolt::amp::device_vector< oType >::pointer resPtr =  result.getContainer( ).data( );

#if defined( _WIN32 )
                  std::transform( &firstPtr[ first1.m_Index ], &firstPtr[first1.m_Index +  sz ], &secPtr[ first2.m_Index ],
                  stdext::make_checked_array_iterator( &resPtr[ result.m_Index ], sz ), f );
#else
                   std::transform( &firstPtr[ first1.m_Index ], &firstPtr[ first1.m_Index + sz ], &secPtr[ first2.m_Index ], &resPtr[ result.m_Index ], f );
#endif
                   return;
              }
              else if( runMode == bolt::amp::control::MultiCoreCpu )
              {

#if defined( ENABLE_TBB )
                  typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  first1.getContainer( ).data( );
                  typename bolt::amp::device_vector< iType2 >::pointer secPtr =  first2.getContainer( ).data( );
                  typename bolt::amp::device_vector< oType >::pointer resPtr =  result.getContainer( ).data( );
                  bolt::btbb::transform(&firstPtr[ first1.m_Index ],&firstPtr[ first1.m_Index + sz ],&secPtr[ first2.m_Index ],&resPtr[ result.m_Index ],f);

#else
                 throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                 return;
              }
              else
              {
                  transform_enqueue( ctl, first1, last1, first2, result, f  );
              }

            }


            /*! \brief This template function overload is used to seperate device_vector iterators from all other iterators
                \detail This template is called by the non-detail versions of transform, it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
            */
            template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
            void transform_unary_pick_iterator( bolt::amp::control &ctl,
                                           const InputIterator& first,
                                           const InputIterator& last,
                                           const OutputIterator& result,
                                           const UnaryFunction& f,
                                           std::random_access_iterator_tag)
            {
                typedef typename std::iterator_traits<InputIterator>::value_type iType;
                typedef typename std::iterator_traits<OutputIterator>::value_type oType;
                int sz = static_cast<int>(last - first);
                if (sz == 0)
                    return;
                const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();
                if( runMode == bolt::amp::control::SerialCpu )
                {
                   std::transform( first, last, result, f );
                   return;
                }
                else if( runMode == bolt::amp::control::MultiCoreCpu )
                {
#if defined( ENABLE_TBB )

                    bolt::btbb::transform(first, last, result, f);

#else
                   throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                  return;
                }
                else
                {
                   // Use host pointers memory since these arrays are only read once - no benefit to copying.

                   // Map the input iterator to a device_vector
                   //device_vector< iType > dvInput( first, last, ctl );
                   device_vector< iType, concurrency::array_view > dvInput( first, last, false, ctl );

                   // Map the output iterator to a device_vector
                   //device_vector< oType > dvOutput( result, sz, false, ctl );
                   device_vector< oType, concurrency::array_view > dvOutput( result, sz, true, ctl );

                   transform_unary_enqueue( ctl, dvInput.begin( ), dvInput.end( ), dvOutput.begin( ), f );

                   // This should immediately map/unmap the buffer
                   dvOutput.data( );
                }
            }

            // This template is called by the non-detail versions of transform, it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVInputIterator, typename DVOutputIterator, typename UnaryFunction>
            void transform_unary_pick_iterator( bolt::amp::control &ctl,
                                           const DVInputIterator& first,
                                           const DVInputIterator& last,
                                           const DVOutputIterator& result,
                                           const UnaryFunction& f,
                                           bolt::amp::device_vector_tag)
            {

              typedef typename std::iterator_traits< DVInputIterator >::value_type iType;
              typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

              int sz = static_cast<int>(std::distance( first, last ));
              if( sz == 0 )
                  return;

              const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

              //  TBB does not have an equivalent for two input iterator std::transform
             if( (runMode == bolt::amp::control::SerialCpu) )
             {
                 typename bolt::amp::device_vector< iType >::pointer firstPtr = first.getContainer( ).data( );
                 typename bolt::amp::device_vector< oType >::pointer resPtr = result.getContainer( ).data( );

#if defined( _WIN32 )

                std::transform( &firstPtr[ first.m_Index ], &firstPtr[ first.m_Index + sz ],
                stdext::make_checked_array_iterator( &resPtr[ result.m_Index ], sz ), f );
#else
                std::transform( &firstPtr[ first.m_Index ], &firstPtr[ first.m_Index + sz ], &resPtr[ result.m_Index ], f );
#endif
                return;
             }
             else if( (runMode == bolt::amp::control::MultiCoreCpu) )
             {

#if defined( ENABLE_TBB )
                typename bolt::amp::device_vector< iType >::pointer firstPtr = first.getContainer( ).data( );
                typename bolt::amp::device_vector< oType >::pointer resPtr = result.getContainer( ).data( );

                bolt::btbb::transform( &firstPtr[ first.m_Index ],  &firstPtr[ first.m_Index  + sz ], &resPtr[ result.m_Index], f);
#else
                throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                return;
             }
             else
             {
                transform_unary_enqueue( ctl, first, last, result, f);
             }
         };


         template<typename DVInputIterator, typename DVOutputIterator, typename UnaryFunction>
         void transform_unary_pick_iterator( bolt::amp::control &ctl,
                                           const DVInputIterator& first,
                                           const DVInputIterator& last,
                                           const DVOutputIterator& result,
                                           const UnaryFunction& f,
                                           bolt::amp::fancy_iterator_tag)
            {

              typedef typename std::iterator_traits< DVInputIterator >::value_type iType;
              typedef typename std::iterator_traits< DVOutputIterator >::value_type oType;

              int sz = std::distance( first, last );
              if( sz == 0 )
                  return;

              const bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.

              //  TBB does not have an equivalent for two input iterator std::transform
             if( (runMode == bolt::amp::control::SerialCpu) )
             {
                serial::unary_transform( first, last, result, f );
                return;
             }
             else if( (runMode == bolt::amp::control::MultiCoreCpu) )
             {

#if defined( ENABLE_TBB )

                bolt::btbb::transform( first, last, result, f);
#else
                throw Concurrency::runtime_exception(  "The MultiCoreCpu version of transform is not enabled to be built.", 0);
#endif
                return;
             }
             else
             {
                transform_unary_enqueue( ctl, first, last, result, f);
             }
         };

#ifdef _WIN32
             // Wrapper that uses default control class, iterator interface
            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_detect_random_access( bolt::amp::control& ctl,
                                                 const InputIterator1& first1,
                                                 const InputIterator1& last1,
                                                 const InputIterator2& first2,
                                                 const OutputIterator& result,
                                                 const BinaryFunction& f,
                                                 std::input_iterator_tag,
                                                 std::input_iterator_tag)
            {
                //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
                //  to a temporary buffer.  Should we?
                static_assert( false, "Bolt only supports random access iterator types" );
            }

            // Wrapper that uses default control class, iterator interface
            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_detect_random_access( bolt::amp::control& ctl,
                                                 const InputIterator1& first1,
                                                 const InputIterator1& last1,
                                                 const InputIterator2& first2,
                                                 const OutputIterator& result,
                                                 const BinaryFunction& f,
                                                 std::input_iterator_tag,
                                                 std::random_access_iterator_tag)
            {
                //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
                //  to a temporary buffer.  Should we?
                static_assert( false, "Bolt only supports random access iterator types" );
            }

            // Wrapper that uses default control class, iterator interface
            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_detect_random_access( bolt::amp::control& ctl,
                                                 const InputIterator1& first1,
                                                 const InputIterator1& last1,
                                                 const InputIterator2& first2,
                                                 const OutputIterator& result,
                                                 const BinaryFunction& f,
                                                 std::random_access_iterator_tag,
                                                 std::input_iterator_tag)
            {
                //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
                //  to a temporary buffer.  Should we?
                static_assert( false, "Bolt only supports random access iterator types" );
            }
#endif

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_detect_random_access( bolt::amp::control& ctl,
                                                 const InputIterator1& first1,
                                                 const InputIterator1& last1,
                                                 const InputIterator2& first2,
                                                 const OutputIterator& result,
                                                 const BinaryFunction& f,
                                                 std::random_access_iterator_tag,
                                                 std::random_access_iterator_tag)
            {
                transform_pick_iterator( ctl, first1, last1, first2, result, f,
                                         typename std::iterator_traits< InputIterator1 >::iterator_category(),
                                         typename std::iterator_traits< InputIterator2 >::iterator_category());
            }

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_detect_random_access(bolt::amp::control& ctl,
              const InputIterator1& first1,
              const InputIterator1& last1,
              const InputIterator2& first2,
              const OutputIterator& result,
              const BinaryFunction& f,
              bolt::amp::fancy_iterator_tag,
              std::random_access_iterator_tag)
            {
              transform_pick_iterator( ctl, first1, last1, first2, result, f,
                                       typename std::iterator_traits< InputIterator1 >::iterator_category(),
                                       typename std::iterator_traits< InputIterator2 >::iterator_category());
            }

            template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
            void transform_detect_random_access(bolt::amp::control& ctl,
              const InputIterator1& first1,
              const InputIterator1& last1,
              const InputIterator2& first2,
              const OutputIterator& result,
              const BinaryFunction& f,
              std::random_access_iterator_tag,
              bolt::amp::fancy_iterator_tag)
            {
              transform_pick_iterator( ctl, first1, last1, first2, result, f,
                                       typename std::iterator_traits< InputIterator1 >::iterator_category(),
                                       typename std::iterator_traits< InputIterator2 >::iterator_category());
            }
// FIXME: it is not valid on Linux
#ifdef _WIN32
            // Wrapper that uses default control class, iterator interface
            template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
            void transform_unary_detect_random_access( bolt::amp::control& ctl,
                                                       const InputIterator& first1,
                                                       const InputIterator& last1,
                                                       const OutputIterator& result,
                                                       const UnaryFunction& f,
                                                       std::input_iterator_tag )
            {
                //  TODO:  It should be possible to support non-random_access_iterator_tag iterators, if we copied the data
                //  to a temporary buffer.  Should we?
                static_assert( false, "Bolt only supports random access iterator types" );
            }
#endif
            template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
            void transform_unary_detect_random_access( bolt::amp::control& ctl,
                                                       const InputIterator& first1,
                                                       const InputIterator& last1,
                                                       const OutputIterator& result,
                                                       const UnaryFunction& f,
                                                       std::random_access_iterator_tag )
            {
                transform_unary_pick_iterator( ctl, first1, last1, result, f,
                                               typename std::iterator_traits< InputIterator >::iterator_category() );
            }

            template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
            void transform_unary_detect_random_access( bolt::amp::control& ctl,
                                                       const InputIterator& first1,
                                                       const InputIterator& last1,
                                                       const OutputIterator& result,
                                                       const UnaryFunction& f,
                                                       bolt::amp::fancy_iterator_tag )
            {
                transform_unary_pick_iterator( ctl, first1, last1, result, f,
                                               typename std::iterator_traits< InputIterator >::iterator_category() );
            }


        };//end of namespace detail


        //////////////////////////////////////////
        //  Transform overloads
        //////////////////////////////////////////
        // default control, two-input transform, std:: iterator
        template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
        void transform( bolt::amp::control& ctl,
                       InputIterator1 first1,
                       InputIterator1 last1,
                       InputIterator2 first2,
                       OutputIterator result,
                       BinaryFunction f )
        {
            detail::transform_detect_random_access( ctl,
                                                   first1,
                                                   last1,
                                                   first2,
                                                   result,
                                                   f,
                                                   typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                                   typename std::iterator_traits< InputIterator2 >::iterator_category( ));
        };


        // default control, two-input transform, std:: iterator
        template<typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
        void transform( InputIterator1 first1,
                        InputIterator1 last1,
                        InputIterator2 first2,
                        OutputIterator result,
                        BinaryFunction f )
        {
            detail::transform_detect_random_access( control::getDefault(),
                                                    first1,
                                                    last1,
                                                    first2,
                                                    result,
                                                    f,
                                                    typename std::iterator_traits< InputIterator1 >::iterator_category( ),
                                                    typename std::iterator_traits< InputIterator2 >::iterator_category( ) );
        };

        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
        void transform( bolt::amp::control& ctl,
                        InputIterator first1,
                        InputIterator last1,
                        OutputIterator result,
                        UnaryFunction f )
        {
            detail::transform_unary_detect_random_access( ctl,
                                                          first1,
                                                          last1,
                                                          result,
                                                          f,
                                                          typename std::iterator_traits< InputIterator >::iterator_category( ) );
        };

        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
        void transform( InputIterator first1,
                        InputIterator last1,
                        OutputIterator result,
                        UnaryFunction f )
        {
            detail::transform_unary_detect_random_access( control::getDefault(),
                                                          first1,
                                                          last1,
                                                          result,
                                                          f,
                                                          typename std::iterator_traits< InputIterator >::iterator_category( ) );
        };

    }; //end of namespace amp
}; //end of namespace bolt

#endif // AMP_TRANSFORM_INL
