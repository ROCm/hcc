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

#if !defined( BOLT_BTBB_MIN_ELEMENT_INL)
#define BOLT_BTBB_MIN_ELEMENT_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include <iterator>

#include<iostream>
namespace bolt{
    namespace btbb {

            template<typename ForwardIterator, typename BinaryPredicate>
            struct Min_Element_comp 
            {    
                ForwardIterator value;
                BinaryPredicate op;
                bool flag;

               
                Min_Element_comp ( ForwardIterator &_val, BinaryPredicate &_op): op(_op), value(_val) {}
                Min_Element_comp( Min_Element_comp& s, tbb::split ) : flag(true), op(s.op), value(s.value) {}
                void operator()( const tbb::blocked_range<ForwardIterator>& r ) {
                    ForwardIterator temp = value;
                    
                    for( ForwardIterator a=r.begin(); a!=r.end(); ++a ) {
                      if(flag){
                        temp = a;
                        flag = false;
                      }
                      else{
                         if(op(*a, *temp))
                           temp = a;
                      }
                    }
                    value = temp;
                }
                void join( Min_Element_comp& rhs )
                {
                       if(op( *rhs.value, *value))
                           value = rhs.value;
                }
            };

            template<typename ForwardIterator, typename BinaryPredicate>
            struct Max_Element_comp 
            {    
                ForwardIterator value;
                BinaryPredicate op;
                bool flag;

               
                Max_Element_comp ( ForwardIterator &_val, BinaryPredicate &_op): op(_op), value(_val) {}
                Max_Element_comp( Max_Element_comp& s, tbb::split ) : flag(true), op(s.op), value(s.value) {}
                void operator()( const tbb::blocked_range<ForwardIterator>& r ) {
                    ForwardIterator temp = value;
                    
                    for( ForwardIterator a=r.begin(); a!=r.end(); ++a ) {
                      if(flag){
                        temp = a;
                        flag = false;
                      }
                      else{
                         if(op(*temp, *a))
                           temp = a;
                      }
                    }
                    value = temp;
                }
                void join( Max_Element_comp& rhs )
                {
                       if(op( *value, *rhs.value))
                           value = rhs.value;
                }
            };

            template<typename ForwardIterator,typename BinaryPredicate>
            ForwardIterator min_element(ForwardIterator first, ForwardIterator last, BinaryPredicate binary_op)
            {

               tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
               Min_Element_comp<ForwardIterator, BinaryPredicate> min_element_op(first, binary_op);
               tbb::parallel_reduce( tbb::blocked_range<ForwardIterator>( first, last), min_element_op );
               return min_element_op.value;
             
            }

            template<typename ForwardIterator,typename BinaryPredicate>
            ForwardIterator max_element(ForwardIterator first, ForwardIterator last, BinaryPredicate binary_op)
            {

              tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);
              Max_Element_comp<ForwardIterator, BinaryPredicate> max_element_op(first, binary_op);
              tbb::parallel_reduce( tbb::blocked_range<ForwardIterator>( first, last), max_element_op );
              return max_element_op.value;  
            }


    } //tbb
} // bolt

#endif //BTBB_MIN_ELEMENT_INL
