/***************************************************************************
*   Copyright 2012 Advanced Micro Devices, Inc.
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

/******************************************************************************
 * OpenCL Scan
 *****************************************************************************/

#if !defined( BOLT_BTBB_SCAN_INL )
#define BOLT_BTBB_SCAN_INL
#pragma once

//#include <thread>
#include "tbb/partitioner.h"

namespace bolt {
namespace   btbb {

      template <typename InputIterator, typename OutputIterator, typename BinaryFunction,typename T>
      struct Scan_tbb{
          T sum;
          T start;
          InputIterator& x;
          OutputIterator& y;
          BinaryFunction scan_op;
          bool inclusive, flag;
          public:
          Scan_tbb() : sum(0) {}
          Scan_tbb( InputIterator&  _x,
                    OutputIterator& _y,
                    const BinaryFunction &_opr,
                    const bool &_incl ,const T &init) : x(_x), y(_y), scan_op(_opr),inclusive(_incl),start(init),flag(true){}
          T get_sum() const {return sum;}
          template<typename Tag>
          void operator()( const tbb::blocked_range<int>& r, Tag ) {
             T temp = sum, temp1;
			 int rend = r.end();
             for(int i=r.begin(); i<rend; ++i ) {
                 if(Tag::is_final_scan()){
                     if(!inclusive){
                        if(i==0 ) {
                            temp1 = *(x+i);
                            *(y+i) = start;
                            temp = scan_op(start, temp1);
                         }
                         else{
                           temp1 = *(x+i);
                           *(y+i) = temp;
                            temp = scan_op(temp, temp1);
                         }
                         continue;
                     }
                     else if(i == 0){
                        temp = *(x+i);
                     }
                     else{
                        temp = scan_op(temp, *(x+i));
                     }
                     *(y+i) = temp;
                  }
                  else{
                     if(flag){
                       temp = *(x+i);
                       flag = false;
                     }
                     else
                        temp = scan_op(temp, *(x+i));
                  }
             }
             sum = temp;
          }
          Scan_tbb( Scan_tbb& b, tbb::split):y(b.y),x(b.x),inclusive(b.inclusive),start(b.start),flag(true){
          }
          void reverse_join( Scan_tbb& a ) {
               sum = scan_op(a.sum, sum);
          }
          void assign( Scan_tbb& b ) {
             sum = b.sum;
          }
       };




template< typename InputIterator, typename OutputIterator, typename BinaryFunction >
OutputIterator
inclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result,
    BinaryFunction binary_op)
    {

               unsigned int numElements = static_cast< unsigned int >( std::distance( first, last ) );
               typedef typename std::iterator_traits< InputIterator >::value_type iType;
               //tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

			   //Gets the number of concurrent threads supported by the underlying platform
               //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
			   unsigned int concurentThreadsSupported = tbb::task_scheduler_init::default_num_threads();

			   //Explicitly setting the number of threads to spawn
               tbb::task_scheduler_init((int) concurentThreadsSupported);

               Scan_tbb<InputIterator, OutputIterator, BinaryFunction, iType> tbb_scan((InputIterator &)first,(OutputIterator &)
                                                                         result,binary_op,true,iType());

               tbb::parallel_scan( tbb::blocked_range<int>(  0, static_cast< int >( std::distance( first, last )), 12500), tbb_scan, tbb::simple_partitioner() );
               return result + numElements;
    }

template< typename InputIterator, typename OutputIterator >
OutputIterator
inclusive_scan(
    InputIterator first,
    InputIterator last,
    OutputIterator result)
    {
		typedef typename std::iterator_traits< InputIterator >::value_type iType;
		inclusive_scan(first,last,result,std::plus< iType >( ));
    }


template< typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction >
OutputIterator
    exclusive_scan( InputIterator first, InputIterator last, OutputIterator result, T init, BinaryFunction binary_op)
    {

               unsigned int numElements = static_cast< unsigned int >( std::distance( first, last ) );
               typedef typename std::iterator_traits< InputIterator >::value_type iType;
               //tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

			   //Gets the number of concurrent threads supported by the underlying platform
               //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
			   unsigned int concurentThreadsSupported = tbb::task_scheduler_init::default_num_threads();

			   //Explicitly setting the number of threads to spawn
               tbb::task_scheduler_init((int) concurentThreadsSupported);
			   
               Scan_tbb<InputIterator, OutputIterator, BinaryFunction, iType> tbb_scan((InputIterator &)first,(OutputIterator &)
                                                                         result,binary_op,false,init);

               tbb::parallel_scan( tbb::blocked_range<int>(  0, static_cast< int >( std::distance( first, last )), 12500), tbb_scan, tbb::simple_partitioner() );
               return result + numElements;
    }

    }



template< typename InputIterator, typename OutputIterator, typename T >
OutputIterator
    exclusive_scan( InputIterator first, InputIterator last, OutputIterator result, T init )
    {
	typedef typename std::iterator_traits< InputIterator >::value_type iType;
	exclusive_scan( first, last, result, init,std::plus< iType >( ));
    }

template< typename InputIterator, typename OutputIterator >
OutputIterator
    exclusive_scan( InputIterator first, InputIterator last, OutputIterator result )
    {
		typedef typename std::iterator_traits< InputIterator >::value_type iType;
		exclusive_scan( first, last, result, iType());
    }

}


#endif // BTBB_SCAN_INL
