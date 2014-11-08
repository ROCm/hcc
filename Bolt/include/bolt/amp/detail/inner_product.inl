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

/*
TODO:
1. Optimize the code. In this version transform and reduce are called directly which performs better.
2. Found a caveat in Multi-GPU scenario (Evergreen+Tahiti). Which basically applies to most of the routines.
*/

#if !defined( BOLT_AMP_INNERPRODUCT_INL )
#define BOLT_AMP_INNERPRODUCT_INL


#pragma once


#include <type_traits>
#include <bolt/amp/detail/reduce.inl>
#include <bolt/amp/detail/transform.inl>
#include "bolt/amp/device_vector.h"
#include "bolt/amp/bolt.h"

//TBB Includes
#ifdef ENABLE_TBB
#include "bolt/btbb/inner_product.h"
#endif

namespace bolt {
    namespace amp {

namespace detail {

            namespace serial
            {

                template< typename InputIterator1,
                          typename InputIterator2,
                          typename OutputType,
                          typename BinaryFunction1,
                          typename BinaryFunction2>
                OutputType inner_product( const InputIterator1& first1,
                                          const InputIterator1& last1,
                                          const InputIterator2& first2,
                                          const OutputType& val,
                                          const BinaryFunction1& f1,
                                          const BinaryFunction2& f2 )
                {
                    size_t sz = (last1 - first1);
                    if (sz == 0)
                        return val;
                    OutputType accumulator = val;
                    for(int index=0; index < (int)(sz); index++)
                    {
                        accumulator = f1( accumulator, f2(*(first1+index), *(first2+index)) );
                    }
                    return accumulator;
                }
            }

            template< typename DVInputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
            OutputType inner_product_enqueue(bolt::amp::control &ctl, const DVInputIterator& first1,
                const DVInputIterator& last1, const DVInputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2)
            {

                typedef typename std::iterator_traits<DVInputIterator>::value_type iType;

                const int distVec = static_cast< int >( std::distance( first1, last1 ) );

                if( distVec == 0 )
                    return init;

                device_vector< iType> tempDV( distVec, iType(), false, ctl);

                detail::transform_enqueue( ctl, first1, last1, first2, tempDV.begin() ,f2);
                return detail::reduce_enqueue( ctl, tempDV.begin(), tempDV.end(), init, f1);

            };



            /*! \brief This template function overload is used to seperate device_vector iterators from all
                other iterators
                \detail This template is called by the non-detail versions of inner_product,
                it already assumes random access
             *  iterators.  This overload is called strictly for non-device_vector iterators
            */
            template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
            OutputType
            inner_product_pick_iterator( bolt::amp::control &ctl,  const InputIterator& first1,
                                       const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                                       const BinaryFunction1& f1,
                                       const BinaryFunction2& f2,
                                       std::random_access_iterator_tag )
            {
                typedef typename std::iterator_traits<InputIterator>::value_type iType;
                int sz = static_cast<int>((last1 - first1));
                if (sz == 0)
                    return init;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
                
                if( runMode == bolt::amp::control::SerialCpu)
                {
                    #if defined( _WIN32 )
                           return std::inner_product(first1, last1, stdext::checked_array_iterator<iType*>(&(*first2), sz ), init, f1, f2);
                    #else
                           return std::inner_product(first1, last1, first2, init, f1, f2);
                    #endif
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
                           return bolt::btbb::inner_product(first1, last1, first2, init, f1, f2);
                    #else
                           throw std::runtime_error("MultiCoreCPU Version of inner_product not Enabled! \n");
                    #endif
                }
                else
                {

                    // Use host pointers memory since these arrays are only read once - no benefit to copying.

                    // Map the input iterator to a device_vector

                    device_vector< iType, concurrency::array_view> dvInput( first1, last1, false, ctl);
                    device_vector< iType, concurrency::array_view> dvInput2( first2, sz, false, ctl);

                    return inner_product_enqueue( ctl, dvInput.begin( ), dvInput.end( ), dvInput2.begin( ),
                                                   init, f1, f2);

                }
            }

            // This template is called by the non-detail versions of inner_product,
            // it already assumes random access iterators
            // This is called strictly for iterators that are derived from device_vector< T >::iterator
            template<typename DVInputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
            OutputType
            inner_product_pick_iterator( bolt::amp::control &ctl,  const DVInputIterator& first1,
                                         const DVInputIterator& last1,const DVInputIterator& first2,
                                         const OutputType& init, const BinaryFunction1&f1, const BinaryFunction2& f2,
                                         bolt::amp::device_vector_tag )
            {
                 int sz = static_cast< int >(last1 - first1);
                 if (sz == 0)
                    return init;

                typedef typename std::iterator_traits< DVInputIterator >::value_type iType1;
                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
                
                if( runMode == bolt::amp::control::SerialCpu)
                {
                    
                    typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  first1.getContainer( ).data( );
                    typename bolt::amp::device_vector< iType1 >::pointer first2Ptr =  first2.getContainer( ).data( );

                    #if defined( _WIN32 )
                       return std::inner_product(  &firstPtr[ first1.m_Index ],
                                                &firstPtr[ last1.m_Index ],
                                                stdext::make_checked_array_iterator( &first2Ptr[ first2.m_Index ], sz),
                                                init, f1, f2);
                    #else
                       return std::inner_product(  &firstPtr[ first1.m_Index ],
                                                &firstPtr[ last1.m_Index ],
                                                &first2Ptr[ first2.m_Index ], init, f1, f2);
                    #endif
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                #ifdef ENABLE_TBB
                    typename bolt::amp::device_vector< iType1 >::pointer firstPtr =  first1.getContainer( ).data( );
                    typename bolt::amp::device_vector< iType1 >::pointer first2Ptr =  first2.getContainer( ).data( );
                    return bolt::btbb::inner_product(  &firstPtr[ first1.m_Index ],  &firstPtr[ last1.m_Index ],
                                                &first2Ptr[ first2.m_Index ], init, f1, f2);
                #else
                           throw std::runtime_error("MultiCoreCPU Version of inner_product not Enabled! \n");
                #endif
                }
                else
                {
                    return inner_product_enqueue( ctl, first1, last1, first2, init, f1, f2 );
                }
            }

            template<typename InputIterator, typename OutputType, typename BinaryFunction1,typename BinaryFunction2>
            OutputType
            inner_product_pick_iterator( bolt::amp::control &ctl,  const InputIterator& first1,
                                       const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                                       const BinaryFunction1& f1,
                                       const BinaryFunction2& f2,
                                       bolt::amp::fancy_iterator_tag )
            {
                typedef typename std::iterator_traits<InputIterator>::value_type iType;
                int sz = static_cast< int >(last1 - first1);
                if (sz == 0)
                    return init;

                bolt::amp::control::e_RunMode runMode = ctl.getForceRunMode();  // could be dynamic choice some day.
                if (runMode == bolt::amp::control::Automatic)
                {
                    runMode = ctl.getDefaultPathToRun();
                }
                
                if( runMode == bolt::amp::control::SerialCpu)
                {

                    return serial::inner_product(first1, last1, first2, init, f1, f2);
                }
                else if(runMode == bolt::amp::control::MultiCoreCpu)
                {
                    #ifdef ENABLE_TBB
                           return bolt::btbb::inner_product(first1, last1, first2, init, f1, f2);
                    #else
                           throw std::runtime_error("MultiCoreCPU Version of inner_product not Enabled! \n");
                    #endif
                }
                else
                {

                    // Use host pointers memory since these arrays are only read once - no benefit to copying.
                    return inner_product_enqueue( ctl, first1, last1, first2, init, f1, f2);

                }
            }

          

            template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
            OutputType inner_product_detect_random_access( bolt::amp::control& ctl, const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1,
                const BinaryFunction2& f2, std::random_access_iterator_tag )
            {
                return inner_product_pick_iterator( ctl, first1, last1, first2, init, f1, f2,
                                                 typename std::iterator_traits< InputIterator >::iterator_category( ));
            };


            template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
            OutputType inner_product_detect_random_access( bolt::amp::control& ctl, const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1,
                const BinaryFunction2& f2, bolt::amp::fancy_iterator_tag )
            {
                return inner_product_pick_iterator( ctl, first1, last1, first2, init, f1, f2,
                                                 typename std::iterator_traits< InputIterator >::iterator_category( ));
            };

            // Wrapper that uses default control class, iterator interface
            template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
            OutputType inner_product_detect_random_access( bolt::amp::control& ctl, const InputIterator& first1,
                const InputIterator& last1, const InputIterator& first2, const OutputType& init,
                const BinaryFunction1& f1, const BinaryFunction2& f2,
                std::input_iterator_tag )
            {
                //  TODO:  It should be possible to support non-random_access_iterator_tag iterators,
                //  if we copied the data
                //  to a temporary buffer.  Should we?
                static_assert(std::is_same< InputIterator, std::input_iterator_tag >::value  , "Bolt only supports random access iterator types" );
            };



        }//End OF detail namespace


        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
         OutputType inner_product(bolt::amp::control& ctl, InputIterator first1, InputIterator last1,
         InputIterator first2, OutputType init, BinaryFunction1 f1, BinaryFunction2 f2)
        {
            return detail::inner_product_detect_random_access( ctl, first1, last1, first2, init, f1, f2, 
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
        }

        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            BinaryFunction1 f1, BinaryFunction2 f2)
        {
            return detail::inner_product_detect_random_access( control::getDefault(), first1, last1, first2, init, f1,
                f2, typename std::iterator_traits< InputIterator >::iterator_category( ) );
        }
        template<typename InputIterator, typename OutputType>
        OutputType inner_product(bolt::amp::control& ctl,InputIterator first1,InputIterator last1,InputIterator first2,
            OutputType init )
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            return detail::inner_product_detect_random_access(ctl, first1,last1,first2,init,bolt::amp::plus< iType >( ),
                bolt::amp::multiplies< iType >( ), typename std::iterator_traits<InputIterator>::iterator_category());
        }

        // default control, two-input transform, std:: iterator
        template<typename InputIterator, typename OutputType>
        OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init)
        {
            typedef typename std::iterator_traits<InputIterator>::value_type iType;
            return detail::inner_product_detect_random_access( control::getDefault(), first1, last1, first2, init,
                bolt::amp::plus< iType >( ), bolt::amp::multiplies< iType >( ),
                typename std::iterator_traits< InputIterator >::iterator_category( ) );
        }


    }//end of amp namespace
};//end of bolt namespace



#endif
