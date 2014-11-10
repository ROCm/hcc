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

#if !defined( BOLT_BTBB_INNER_PRODUCT_INL)
#define BOLT_BTBB_INNER_PRODUCT_INL
#pragma once

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
//#include <thread>
#include <iterator>

namespace bolt{
    namespace btbb {
       
             template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
             struct Inner_Product_Op 
             {

               OutputType result;
               Inner_Product_Op () {result = 0;}
               Inner_Product_Op (OutputType _x ): result(_x) {}

               void operator() (InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
               BinaryFunction1 f1, BinaryFunction2 f2)
               {
                  
                    if (first1 == last1) 
                        result = init;
                    else
                    {  
                      int n = (int) std::distance(first1, last1);
                      std::vector<OutputType> res_vector(n);
                      typename std::vector<OutputType>::iterator res = res_vector.begin();

                      tbb::parallel_for(  tbb::blocked_range<int>(0, n) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                        {
                              for(int i = r.begin(); i!=r.end(); ++i)
                              { 
                                      //Stores the result of applying f2 to the two input vectors
                                      *(res + i) = f2(*(first1 + i), *(first2 + i));  
                                      
                              }   
                          
                        });
                       
                       //Applies reduce with f1 on the result Vector
                       result = bolt::btbb::reduce(res_vector.begin(),res_vector.end(),init, f1);

                    }
               }    

            };

            template<typename InputIterator, typename OutputType, typename BinaryFunction1, typename BinaryFunction2>
            OutputType inner_product( InputIterator first1, InputIterator last1, InputIterator first2, OutputType init,
            BinaryFunction1 f1, BinaryFunction2 f2 )
            {
              //Gets the number of concurrent threads supported by the underlying platform
              //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();

              //This allows TBB to choose the number of threads to spawn.
              tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

              //Explicitly setting the number of threads to spawn
              //tbb::task_scheduler_init((int) concurentThreadsSupported);

              Inner_Product_Op <InputIterator, OutputType,BinaryFunction1, BinaryFunction2 > inner_prod_op;
              inner_prod_op(first1, last1, first2, init, f1, f2);

              return inner_prod_op.result;
           }

       
    } //tbb
} // bolt

#endif //BTBB_INNER_PRODUCT_INL
