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
#if !defined( BOLT_CL_DISTANCE_INL )
#define BOLT_CL_DISTANCE_INL

#include <bolt/cl/iterator/iterator_categories.h>
namespace bolt
{
namespace cl
{
namespace detail
{


    template<typename InputIterator>
    inline typename bolt::cl::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last, bolt::cl::incrementable_traversal_tag)
    {
        typename bolt::cl::iterator_traits<InputIterator>::difference_type result(0);

        while(first != last)
        {
        ++first;
        ++result;
        } 
        return result;
    } 


    template<typename InputIterator>
    inline typename bolt::cl::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last, bolt::cl::random_access_traversal_tag)
    {
        return last - first;
    } 


} // namespace detail


    template<typename InputIterator>
    inline typename bolt::cl::iterator_traits<InputIterator>::difference_type
        distance(InputIterator first, InputIterator last)
    {
      // dispatch on iterator traversal
      return bolt::cl::detail::distance(first, last, typename bolt::cl::iterator_traversal<InputIterator>::type());
    } 


} //  namespace cl
} //  namespace bolt

#endif