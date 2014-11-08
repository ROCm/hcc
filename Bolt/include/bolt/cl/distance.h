

#pragma once


#include <bolt/cl/iterator/iterator_traits.h>

namespace bolt
{
namespace cl
{


template<typename InputIterator>
typename bolt::cl::iterator_traits<InputIterator>::difference_type
    distance(InputIterator first, InputIterator last);


} } //namespace bolt::cl
#include <bolt/cl/detail/distance.inl>


