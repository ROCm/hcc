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

// (C) Copyright Jeremy Siek 2002.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file BOOST_LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOLT_ITERATOR_CATEGORIES_H
# define BOLT_ITERATOR_CATEGORIES_H


#include <type_traits>
#include <bolt/cl/detail/type_traits.h>
#include <bolt/cl/iterator/iterator_traits.h>

namespace bolt {
namespace cl {
//
// Traversal Categories
//

struct no_traversal_tag {};

struct incrementable_traversal_tag 
  : no_traversal_tag
{
//     incrementable_traversal_tag() {}
//     incrementable_traversal_tag(std::output_iterator_tag const&) {};
};
  
struct single_pass_traversal_tag
  : incrementable_traversal_tag
{
//     single_pass_traversal_tag() {}
//     single_pass_traversal_tag(std::input_iterator_tag const&) {};
};
  
struct forward_traversal_tag
  : single_pass_traversal_tag
{
//     forward_traversal_tag() {}
//     forward_traversal_tag(std::forward_iterator_tag const&) {};
};
  
struct bidirectional_traversal_tag
  : forward_traversal_tag
{
//     bidirectional_traversal_tag() {};
//     bidirectional_traversal_tag(std::bidirectional_iterator_tag const&) {};
};
  
struct random_access_traversal_tag
  : bidirectional_traversal_tag
{
//     random_access_traversal_tag() {};
//     random_access_traversal_tag(std::random_access_iterator_tag const&) {};
};

namespace detail
{  
  //
  // Convert a "strictly old-style" iterator category to a traversal
  // tag.  This is broken out into a separate metafunction to reduce
  // the cost of instantiating iterator_category_to_traversal, below,
  // for new-style types.
  //
  template <class Cat>
  struct old_category_to_traversal
    : bolt::cl::detail::eval_if<
          std::is_convertible<Cat,std::random_access_iterator_tag>
        , bolt::cl::detail::identity_<random_access_traversal_tag>
        , bolt::cl::detail::eval_if<
              std::is_convertible<Cat,std::bidirectional_iterator_tag>
            , bolt::cl::detail::identity_<bidirectional_traversal_tag>
            , bolt::cl::detail::eval_if<
                  std::is_convertible<Cat,std::forward_iterator_tag>
                , bolt::cl::detail::identity_<forward_traversal_tag>
                , bolt::cl::detail::eval_if<
                      std::is_convertible<Cat,std::input_iterator_tag>
                    , bolt::cl::detail::identity_<single_pass_traversal_tag>
                    , bolt::cl::detail::eval_if<
                          std::is_convertible<Cat,std::output_iterator_tag>
                        , bolt::cl::detail::identity_<incrementable_traversal_tag>
                        , void
                      >
                  >
              >
          >
      >
  {};


} // namespace detail


//
// Convert an iterator category into a traversal tag
//
template <class Cat>
struct iterator_category_to_traversal
  : bolt::cl::detail::eval_if< // if already convertible to a traversal tag, we're done.
        std::is_convertible<Cat,incrementable_traversal_tag>
      , bolt::cl::detail::identity_<Cat>
      , bolt::cl::detail::old_category_to_traversal<Cat>
    >
{};

// Trait to get an iterator's traversal category
template <class Iterator >
struct iterator_traversal
  : iterator_category_to_traversal<
        typename bolt::cl::iterator_traits<Iterator>::iterator_category
    >
{};

} } // namespace bolt::cl


#endif // BOLT_ITERATOR_CATEGORIES_H
