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

#if !defined( BOLT_BTBB_REDUCE_BY_KEY_INL)
#define BOLT_BTBB_REDUCE_BY_KEY_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include <iterator>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include <iostream>

#include "bolt/btbb/scan.h"

using namespace std;

namespace bolt
{
    namespace btbb 
    {

		  template<
                 typename InputIterator1,
                 typename InputIterator2,
                 typename OutputIterator1,
                 typename OutputIterator2,
                 typename BinaryPredicate,
                 typename BinaryFunction>

          struct reduce_by_key_tbb { 
                           
		  typedef typename std::iterator_traits< OutputIterator2 >::value_type oType;
          typedef typename std::iterator_traits< OutputIterator1 >::value_type key_oType;

          key_oType sum_key;
		  oType sum;
		  oType start;
		  InputIterator1& first_key;
		  InputIterator2& first_value;
          OutputIterator1& key_result;
		  OutputIterator2& result;
		  unsigned int numElements, strt_indx, end_indx;
		  const BinaryFunction binary_op;
		  const BinaryPredicate binary_pred;
		  bool flag, pre_flag, next_flag;
          std::vector<int> & t_key_array;
		  public:
		  reduce_by_key_tbb() : sum(0), sum_key(0){}
		  reduce_by_key_tbb( InputIterator1&  _first,
			InputIterator2& first_val,
            OutputIterator1& _key_result,
			OutputIterator2& _result,
		    unsigned int _numElements,
			const BinaryPredicate &_pred,
            const BinaryFunction &_opr,
            std::vector<int> & _t_key_array) : first_key(_first), first_value(first_val), key_result(_key_result), result(_result), numElements(_numElements), binary_op(_opr), binary_pred(_pred),
							 flag(false), pre_flag(true),next_flag(false), t_key_array(_t_key_array){}
		  oType get_sum() const {return sum;}
          key_oType get_sum_key() const {return sum_key;}
		  template<typename Tag>
		  void operator()( const tbb::blocked_range<unsigned int>& r, Tag ) 
          {

			  oType temp = sum;
              key_oType temp_key = sum_key;
             
			  next_flag = flag = false;
              unsigned int i;
			  strt_indx = r.begin();
              end_indx = r.end();
			  unsigned int rend = r.end();


			  for( i=r.begin(); i<rend; ++i ) 
              {

				 if( Tag::is_final_scan() )
                 {		 
					 if(i == 0 )
					 {
                        temp_key = *(first_key + i);
						temp = *(first_value+i);
					 }
					 else if(binary_pred(*(first_key+i), *(first_key +i- 1))) 
					 {
						temp = binary_op(temp, *(first_value+i));
					 }
					 else
					 {		

                        *(key_result + t_key_array[i-1]) = temp_key;
						*(result + t_key_array[i-1]) = temp;

						temp = *(first_value+i);
                        temp_key = *(first_key + i);
						flag = true; 
					 }

				 }
				 else if(pre_flag)
				 {
                     temp_key = *(first_key + i);
					 temp = *(first_value+i);
					 pre_flag = false;
				 }
				 else if(binary_pred(*(first_key+i), *(first_key +i - 1)))
                 {
					 temp = binary_op(temp, *(first_value+i));
                 }
				 else 
				 {
                     *(key_result + t_key_array[i-1]) = temp_key;
					 *(result + t_key_array[i-1]) = temp;

		   		     temp = *(first_value+i);
                     temp_key = *(first_key + i);
					 flag = true; 

				 }

                 if(i == (numElements - 1))
                 {     
                        *(key_result + t_key_array[i]) = temp_key;
					    *(result + t_key_array[i]) = temp;
                 }

			 }

			 if(i<numElements && !binary_pred(*(first_key+i-1), *(first_key +i )))
			 {
		        *(key_result + t_key_array[i-1]) = temp_key;
			    *(result + t_key_array[i-1]) = temp;

				next_flag = true;     // this will check the key change at boundaries
			 } 

			 sum = temp;
             sum_key = temp_key;

		  }

		  reduce_by_key_tbb( reduce_by_key_tbb& b, tbb::split):first_key(b.first_key),key_result(b.key_result), result(b.result),
             first_value(b.first_value),numElements(b.numElements),pre_flag(true),binary_op(b.binary_op), binary_pred(b.binary_pred), t_key_array(b.t_key_array){}

		  void reverse_join( reduce_by_key_tbb& a )
		  {
			if(!a.next_flag && !flag && binary_pred(*(a.first_key +  a.end_indx),*(first_key+strt_indx))) 
            {
                  sum = binary_op(a.sum,sum);
                  sum_key = binary_op(a.sum_key,sum_key);
            }
		  }

		  void assign( reduce_by_key_tbb& b ) 
		  {
			 sum = b.sum;
             sum_key = b.sum_key;
		  }
   };

template<
           typename InputIterator1,
           typename InputIterator2,
           typename OutputIterator1,
           typename OutputIterator2,
           typename BinaryPredicate,
           typename BinaryFunction>

           unsigned int reduce_by_key( 
                            InputIterator1  keys_first,
	                        InputIterator1  keys_last,
	                        InputIterator2  vals_first,
                            OutputIterator1  keys_result,
	                        OutputIterator2  vals_result,
                            BinaryPredicate binary_pred,
                            BinaryFunction binary_op )

	{
		unsigned int numElements = static_cast< unsigned int >( std::distance( keys_first, keys_last ) );
		typedef typename std::iterator_traits< InputIterator2 >::value_type vType;

		//tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

		//Gets the number of concurrent threads supported by the underlying platform
        //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
		unsigned int concurentThreadsSupported = tbb::task_scheduler_init::default_num_threads();
	    //Explicitly setting the number of threads to spawn
        tbb::task_scheduler_init((int) concurentThreadsSupported);

        std::vector<int> t_key_array(numElements);


		tbb::parallel_for (tbb::blocked_range<int>(0,numElements),[&](const tbb::blocked_range<int>& r)
        {
					int rend = r.end();
                    for(int iter = r.begin(); iter!=rend; iter++)
                    {   
						    if(iter == 0)
                            {  
                                t_key_array[iter] = 0;

                            }
                            else if(binary_pred( keys_first[iter], keys_first[iter-1]))
                            
                                t_key_array[iter] = 0;
                            else 
                                t_key_array[iter] = 1;
                    }
       }); 
                    
	   std::vector<int>::iterator it;
	   it = t_key_array.begin();

       bolt::btbb::inclusive_scan(it,  it + numElements , it, std::plus<int>() );

        /*int val = 0;
        for( unsigned int i=0; i<numElements; ++i ) 
        {
               
                  if(i == 0)
                        t_key_array[i] = val;
                  else if(binary_pred(*(keys_first+i), *(keys_first +i- 1))) 
                        t_key_array[i] = val;
                  else
                  {
                         val++;
                         t_key_array[i] = val;
                  }
        }*/

		reduce_by_key_tbb<InputIterator1, InputIterator2, OutputIterator1, OutputIterator2, BinaryPredicate, BinaryFunction> tbbkey_scan((InputIterator1 &) keys_first,
			(InputIterator2&) vals_first, (OutputIterator1 &)keys_result, (OutputIterator2 &)vals_result, numElements,  binary_pred, binary_op, t_key_array);
		tbb::parallel_scan( tbb::blocked_range<unsigned int>(  0, numElements, 6250), tbbkey_scan, tbb::simple_partitioner());

		return numElements;

	}



template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator1,
	typename OutputIterator2,
	typename BinaryPredicate>
unsigned int
reduce_by_key(
	InputIterator1  keys_first,
	InputIterator1  keys_last,
	InputIterator2  vals_first,
    OutputIterator1  keys_result,
	OutputIterator2  vals_result,
	BinaryPredicate binary_pred)
	{
		typedef typename std::iterator_traits<OutputIterator2>::value_type oType;
		reduce_by_key(keys_first,keys_last,vals_first, keys_result,vals_result,binary_pred,plus<oType>());
	}



template<
	typename InputIterator1,
	typename InputIterator2,
    typename OutputIterator1,
	typename OutputIterator2>
unsigned int
reduce_by_key(
	InputIterator1  keys_first,
	InputIterator1  keys_last,
	InputIterator2  vals_first,
    OutputIterator1  keys_result,
	OutputIterator2  vals_result)
	{
		typedef typename std::iterator_traits<InputIterator1>::value_type kType;
		reduce_by_key(keys_first,keys_last,vals_first, keys_result,vals_result,equal_to<kType>());
	}
        
    } //tbb
} // bolt

#endif //BTBB_REDUCE_BY_KEY_INL
