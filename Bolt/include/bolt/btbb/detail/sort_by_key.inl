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

#if !defined( BOLT_BTBB_SORT_BY_KEY_INL)
#define BOLT_BTBB_SORT_BY_KEY_INL
#pragma once

#include "tbb/task_scheduler_init.h"
//#include <thread>
#include <iterator>

#include "bolt/btbb/sort.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

namespace bolt{
    namespace btbb {
       
              template <typename keyType, typename valueType>
              class tbb_sort
              {
                  public:
                    keyType   key;
                    valueType value;
              };

              //This is the functor which will sort the tbb_sort vector. 
              template <typename keyType, typename valueType, typename StrictWeakOrdering>
              class tbb_sort_comp
              {
                  public:
                     typedef tbb_sort<keyType, valueType> KeyValueType;
                     tbb_sort_comp(const StrictWeakOrdering &_swo):swo(_swo) {}
                     StrictWeakOrdering swo;
                     bool operator() (const KeyValueType &lhs, const KeyValueType &rhs) const
                     {
                       return swo(lhs.key, rhs.key);
                     }
              };

              template< typename RandomAccessIterator1, typename RandomAccessIterator2>
              void Parallel_sort_by_key(const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last, 
                                      const RandomAccessIterator2 values_first)
              {
                     typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
                     typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;
                     typedef tbb_sort<keyType, valType> KeyValuePair;
       
                     size_t vecSize = std::distance( keys_first, keys_last ); 
                     std::vector<KeyValuePair> KeyValuePairVector(vecSize);

                     //Zip the key and values iterators into a tbb_sort vector.
                     tbb::parallel_for(  tbb::blocked_range<int>(0, (int) vecSize) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                     {
                              
                              for(int i = r.begin(); i!=r.end(); i++)
                              {
                                 
                                   KeyValuePairVector[i].key   = *(keys_first + i);
                                   KeyValuePairVector[i].value = *(values_first + i);
                              }   
                          
                     });

                     //Sort the tbb_sort vector using TBB sort
                     bolt::btbb::sort(KeyValuePairVector.begin(), KeyValuePairVector.end());

                     //Extract the keys and values from the KeyValuePair and fill the respective iterators. 
                     tbb::parallel_for(  tbb::blocked_range<int>(0, (int) vecSize) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                     {
                              
                              for(int i = r.begin(); i!=r.end(); i++)
                              {
                                 
                                   *(keys_first + i)   = KeyValuePairVector[i].key;
                                   *(values_first + i) = KeyValuePairVector[i].value;
                              }   
                          
                     });
              }

              template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering >
              void Parallel_sort_by_key_comp(const RandomAccessIterator1 keys_first, const RandomAccessIterator1 keys_last, 
                                      const RandomAccessIterator2 values_first, StrictWeakOrdering comp )
              {
                     typedef typename std::iterator_traits< RandomAccessIterator1 >::value_type keyType;
                     typedef typename std::iterator_traits< RandomAccessIterator2 >::value_type valType;
                     typedef tbb_sort<keyType, valType> KeyValuePair;
                     typedef tbb_sort_comp<keyType, valType, StrictWeakOrdering> KeyValuePairFunctor;
       
                     size_t vecSize = std::distance( keys_first, keys_last ); 
                     std::vector<KeyValuePair> KeyValuePairVector(vecSize);
                     KeyValuePairFunctor functor(comp);

                     //Zip the key and values iterators into a tbb_sort vector.
                     tbb::parallel_for(  tbb::blocked_range<int>(0, (int) vecSize) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                     {
                              
                              for(int i = r.begin(); i!=r.end(); i++)
                              {
                                 
                                   KeyValuePairVector[i].key   = *(keys_first + i);
                                   KeyValuePairVector[i].value = *(values_first + i);
                              }   
                          
                     });

                     //Sort the tbb_sort vector using TBB sort
                     bolt::btbb::sort(KeyValuePairVector.begin(), KeyValuePairVector.end(), functor);

                     //Extract the keys and values from the KeyValuePair and fill the respective iterators.
                     tbb::parallel_for(  tbb::blocked_range<int>(0, (int) vecSize) ,
                        [&] (const tbb::blocked_range<int> &r) -> void
                     {
                              
                              for(int i = r.begin(); i!=r.end(); i++)
                              {
                                 
                                   *(keys_first + i)   = KeyValuePairVector[i].key;
                                   *(values_first + i) = KeyValuePairVector[i].value;
                              }   
                          
                     });
             }

             template< typename RandomAccessIterator1, typename RandomAccessIterator2 > 
             struct SortByKey
             {

               SortByKey () {}  

               void operator() ( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first)
               {
                    int n = (int) std::distance(keys_first, keys_last);

                    if(n == 1)  // Only one element
                         return; // Nothing to Sort!
                    else
                         Parallel_sort_by_key(keys_first, keys_last, values_first);
                    
               }          

           };

           template<typename RandomAccessIterator1,typename RandomAccessIterator2, typename StrictWeakOrdering>
           struct SortByKey_comp
           {

               SortByKey_comp () {}  

               void operator() (RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, StrictWeakOrdering comp )
               {
                    int n = (int) std::distance(keys_first, keys_last);

                    if(n == 1)  // Only one element
                         return; // Nothing to Sort!
                    else   
                         Parallel_sort_by_key_comp(keys_first, keys_last, values_first,comp);
                    
               }    

           };

           template< typename RandomAccessIterator1, typename RandomAccessIterator2 > 
           void sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, 
           RandomAccessIterator2 values_first)
           {
                //Gets the number of concurrent threads supported by the underlying platform
                //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();

                //This allows TBB to choose the number of threads to spawn.
                tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

                //Explicitly setting the number of threads to spawn
                //tbb::task_scheduler_init((int) concurentThreadsSupported);

                SortByKey <RandomAccessIterator1, RandomAccessIterator2 > sort_by_key_op;
                sort_by_key_op(keys_first, keys_last, values_first);
           }

           template< typename RandomAccessIterator1, typename RandomAccessIterator2, typename StrictWeakOrdering> 
           void sort_by_key( RandomAccessIterator1 keys_first, RandomAccessIterator1 keys_last, RandomAccessIterator2 values_first, 
           StrictWeakOrdering comp)
           {
                //Gets the number of concurrent threads supported by the underlying platform
                //unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();

                //This allows TBB to choose the number of threads to spawn.
                tbb::task_scheduler_init initialize(tbb::task_scheduler_init::automatic);

                //Explicitly setting the number of threads to spawn
                //tbb::task_scheduler_init((int) concurentThreadsSupported);

                SortByKey_comp <RandomAccessIterator1, RandomAccessIterator2, StrictWeakOrdering >sort_by_key_op;
                sort_by_key_op(keys_first, keys_last, values_first, comp);
          }
       
    } //tbb
} // bolt

#endif //BTBB_SORT_BY_KEY_INL