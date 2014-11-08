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

// Copyright David Abrahams 2003. Use, modification and distribution is
// subject to the Boost Software License, Version 1.0. (See accompanying
// file BOOST_LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOLT_FACADE_ITERATOR_CATEGORY_H
# define BOLT_FACADE_ITERATOR_CATEGORY_H



#include <bolt/cl/iterator/iterator_categories.h>

namespace bolt {
namespace cl { 
    struct use_default; 
} 
}

namespace bolt { 
namespace cl { 
namespace detail  {

struct input_output_iterator_tag
  : std::input_iterator_tag
{
    // Using inheritance for only input_iterator_tag helps to avoid
    // ambiguities when a stdlib implementation dispatches on a
    // function which is overloaded on both input_iterator_tag and
    // output_iterator_tag, as STLPort does, in its __valid_range
    // function.  I claim it's better to avoid the ambiguity in these
    // cases.
    operator std::output_iterator_tag() const
    {
        return std::output_iterator_tag();
    }
};

//
// Convert an iterator_facade's traversal category, Value parameter,
// and ::reference type to an appropriate old-style category.
//
// If writability has been disabled per the above metafunction, the
// result will not be convertible to output_iterator_tag.
//
// Otherwise, if Traversal == single_pass_traversal_tag, the following
// conditions will result in a tag that is convertible both to
// input_iterator_tag and output_iterator_tag:
//
//    1. Reference is a reference to non-const
//    2. Reference is not a reference and is convertible to Value
//
template <typename Traversal, typename ValueParam, typename Reference>
struct iterator_facade_default_category
  : bolt::cl::detail::eval_if<
        bolt::cl::detail::and_<
            std::is_reference<Reference>
          , std::is_convertible<Traversal,forward_traversal_tag>
        >
      , bolt::cl::detail::eval_if<
            std::is_convertible<Traversal,random_access_traversal_tag>
          , bolt::cl::detail::identity_<std::random_access_iterator_tag>
          , bolt::cl::detail::if_<
                std::is_convertible<Traversal,bidirectional_traversal_tag>
              , std::bidirectional_iterator_tag
              , std::forward_iterator_tag
            >
        >
      , bolt::cl::detail::eval_if<
            bolt::cl::detail::and_<
                std::is_convertible<Traversal, single_pass_traversal_tag>
                // check for readability
              , std::is_convertible<Reference, ValueParam>
            >
          , bolt::cl::detail::identity_<std::input_iterator_tag>
          , bolt::cl::detail::identity_<Traversal>
        >
    >
{
};

// True iff T is convertible to an old-style iterator category.
template <class T>
struct is_iterator_category
  : bolt::cl::detail::or_<
        std::is_convertible<T,std::input_iterator_tag>
      , std::is_convertible<T,std::output_iterator_tag>
    >
{
};

template <class T>
struct is_iterator_traversal
  : std::is_convertible<T,incrementable_traversal_tag>
{};

//
// A composite iterator_category tag convertible to Category (a pure
// old-style category) and Traversal (a pure traversal tag).
// Traversal must be a strict increase of the traversal power given by
// Category.
//
template <class Category, class Traversal>
struct iterator_category_with_traversal
  : Category, Traversal
{};

// Computes an iterator_category tag whose traversal is Traversal and
// which is appropriate for an iterator
template <class Traversal, class ValueParam, class Reference>
struct facade_iterator_category_impl
{  
    typedef typename iterator_facade_default_category<
        Traversal,ValueParam,Reference
    >::type category;
    
    typedef typename bolt::cl::detail::if_<
        std::is_same<
            Traversal
          , typename bolt::cl::iterator_category_to_traversal<category>::type
        >
      , category
      , iterator_category_with_traversal<category,Traversal>
    >::type type;
};

//
// Compute an iterator_category for iterator_facade
//
template <class CategoryOrTraversal, class ValueParam, class Reference>
struct facade_iterator_category
  : bolt::cl::detail::eval_if<
        is_iterator_category<CategoryOrTraversal>
      , bolt::cl::detail::identity_<CategoryOrTraversal> // old-style categories are fine as-is
      , facade_iterator_category_impl<CategoryOrTraversal,ValueParam,Reference>
    >
{
};

}}} // namespace bolt::cl::detail


#endif // BOLT_FACADE_ITERATOR_CATEGORY_H
