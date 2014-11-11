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

#if !defined( BOLT_BTBB_SCAN_BY_KEY_INL )
#define BOLT_BTBB_SCAN_BY_KEY_INL
#pragma once

//#include <thread>
#include "tbb/partitioner.h"

namespace bolt
{
	namespace btbb
	{


	  template <typename InputIterator1, typename InputIterator2, typename OutputIterator,
				 typename BinaryFunction, typename BinaryPredicate,typename T>
	  struct ScanKey_tbb{
		  typedef typename std::iterator_traits< OutputIterator >::value_type oType;
		  oType sum;
		  oType start;
		  InputIterator1& first_key;
		  InputIterator2& first_value;
		  OutputIterator& result;
		  unsigned int numElements, strt_indx, end_indx;
		  const BinaryFunction binary_op;
		  const BinaryPredicate binary_pred;
		  const bool inclusive;
		  bool flag, pre_flag, next_flag;
		  public:
		  ScanKey_tbb() : sum(0) {}
		  ScanKey_tbb( InputIterator1&  _first,
			InputIterator2& first_val,
			OutputIterator& _result,
		    unsigned int _numElements,
			const BinaryFunction &_opr,
			const BinaryPredicate &_pred,
			const bool& _incl,
			const oType &init) : first_key(_first), first_value(first_val), result(_result), numElements(_numElements), binary_op(_opr), binary_pred(_pred),
							 inclusive(_incl), start(init), flag(false), pre_flag(true),next_flag(false){}
		  oType get_sum() const {return sum;}
		  template<typename Tag>
		  void operator()( const tbb::blocked_range<unsigned int>& r, Tag ) {
			  oType temp = sum, temp1;
			  next_flag = flag = false;
              unsigned int i;
			  strt_indx = r.begin();
              end_indx = r.end();
			  unsigned int rend = r.end();
			  for( i=r.begin(); i<rend; ++i ) {
				 if( Tag::is_final_scan() ) {
					 if(!inclusive){
						  if( i==0){
							 temp1 = *(first_value + i);
							 *(result + i) = start;
							 temp = binary_op(start, temp1);
						  }
						  else if(binary_pred(*(first_key+i), *(first_key +i- 1))){
							 temp1 = *(first_value + i);
							 *(result + i) = temp;
							 temp = binary_op(temp, temp1);
						  }
						  else{
							 temp1 = *(first_value + i);
							 *(result + i) = start;
							 temp = binary_op(start, temp1);
							 flag = true; 
						  }
						  continue;
					 }
					 else if(i == 0 ){
						temp = *(first_value+i);
					 }
					 else if(binary_pred(*(first_key+i), *(first_key +i- 1))) {
						temp = binary_op(temp, *(first_value+i));
					 }
					 else{
						temp = *(first_value+i);
						flag = true; 
					 }
					 *(result + i) = temp;
				 }
				 else if(pre_flag){
					 temp = *(first_value+i);
					 pre_flag = false;
				 }
				 else if(binary_pred(*(first_key+i), *(first_key +i - 1)))
					 temp = binary_op(temp, *(first_value+i));
				 else if (!inclusive){
					 temp = binary_op(start, *(first_value+i));
					 flag = true; 
				 }
				 else {
					 temp = *(first_value+i);
					 flag = true; 
				 }
			 }
			 if(i<numElements && !binary_pred(*(first_key+i-1), *(first_key +i ))){
				next_flag = true;     // this will check the key change at boundaries
			 }
			 sum = temp;
		  }
		  ScanKey_tbb( ScanKey_tbb& b, tbb::split):first_key(b.first_key),result(b.result),
first_value(b.first_value),numElements(b.numElements),inclusive(b.inclusive),start(b.start),pre_flag(true),binary_op(b.binary_op), binary_pred(b.binary_pred){}
		  void reverse_join( ScanKey_tbb& a ) {
			if(!a.next_flag && !flag && binary_pred(*(a.first_key +  a.end_indx),*(first_key+strt_indx))) 
                  sum = binary_op(a.sum,sum);
		  }
		  void assign( ScanKey_tbb& b ) {
			 sum = b.sum;
		  }
   };

template<typename T>
struct equal_to
{
	bool operator()(const T &lhs, const T &rhs) const  {return lhs == rhs;}
};

template<typename T>
struct plus
{
	T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
};




template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator,
	typename BinaryPredicate,
	typename BinaryFunction>
OutputIterator
inclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result,
	BinaryPredicate binary_pred,
	BinaryFunction  binary_funct)
	{
		unsigned int numElements = static_cast< unsigned int >( std::distance( first1, last1 ) );
		typedef typename std::iterator_traits< InputIterator2 >::value_type vType;

		//tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

		//Gets the number of concurrent threads supported by the underlying platform
        //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
		unsigned int concurentThreadsSupported = tbb::task_scheduler_init::default_num_threads();
	    //Explicitly setting the number of threads to spawn
        tbb::task_scheduler_init((int) concurentThreadsSupported);

		ScanKey_tbb<InputIterator1, InputIterator2, OutputIterator, BinaryFunction, BinaryPredicate,vType> tbbkey_scan((InputIterator1 &)first1,
			(InputIterator2&) first2,(OutputIterator &)result, numElements, binary_funct, binary_pred, true, vType());
		tbb::parallel_scan( tbb::blocked_range<unsigned int>(  0, static_cast< unsigned int >( std::distance( first1, last1 )), 6250), tbbkey_scan, tbb::simple_partitioner());

		return result + numElements;

	}



template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator,
	typename BinaryPredicate>
OutputIterator
inclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result,
	BinaryPredicate binary_pred)
	{
		typedef typename std::iterator_traits<OutputIterator>::value_type oType;
		inclusive_scan_by_key(first1,last1,first2,result,binary_pred,plus<oType>());
	}



template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator>
OutputIterator
inclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result)
	{
		typedef typename std::iterator_traits<InputIterator1>::value_type kType;
		inclusive_scan_by_key(first1,last1,first2,result,equal_to<kType>());
	}


template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator,
	typename T,
	typename BinaryPredicate,
	typename BinaryFunction>
OutputIterator
exclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result,
	T               init,
	BinaryPredicate binary_pred,
	BinaryFunction  binary_funct)
	{
		unsigned int numElements = static_cast< unsigned int >( std::distance( first1, last1 ) );

		//tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

		//Gets the number of concurrent threads supported by the underlying platform
        //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
		unsigned int concurentThreadsSupported = tbb::task_scheduler_init::default_num_threads();
	    //Explicitly setting the number of threads to spawn
        tbb::task_scheduler_init((int) concurentThreadsSupported);

		ScanKey_tbb<InputIterator1, InputIterator2, OutputIterator, BinaryFunction, BinaryPredicate,T> tbbkey_scan((InputIterator1 &)first1,
			(InputIterator2&) first2,(OutputIterator &)result, numElements, binary_funct, binary_pred, false, init);
		tbb::parallel_scan( tbb::blocked_range<unsigned int>(  0, static_cast< unsigned int >( std::distance( first1, last1 )), 6250), tbbkey_scan, tbb::simple_partitioner());
		return result + numElements;

	}

template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator,
	typename T,
	typename BinaryPredicate>
OutputIterator
exclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result,
	T               init,
	BinaryPredicate binary_pred)
	{

		typedef typename std::iterator_traits<OutputIterator>::value_type oType;		
		exclusive_scan_by_key(first1,last1, first2, result, init,binary_pred, plus<oType>());
	}


template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator,
	typename T>
OutputIterator
exclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result,
	T               init)
	{

		typedef typename std::iterator_traits<InputIterator1>::value_type kType;
		exclusive_scan_by_key(first1,last1, first2, result, init,equal_to<kType>());
	}


template<
	typename InputIterator1,
	typename InputIterator2,
	typename OutputIterator>
OutputIterator
exclusive_scan_by_key(
	InputIterator1  first1,
	InputIterator1  last1,
	InputIterator2  first2,
	OutputIterator  result)
	{

		typedef typename std::iterator_traits< InputIterator2 >::value_type vType;
		exclusive_scan_by_key(first1,last1, first2, result, vType());


	}



	}
}


#endif
