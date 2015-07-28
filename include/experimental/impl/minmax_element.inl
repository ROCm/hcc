
#include <utility>

//
// minmax_element is implemented by reduce
//
template< typename ForwardIt, typename Compare >
std::pair<ForwardIt, ForwardIt> minmax_element( ForwardIt first, ForwardIt last, Compare cmp) {
  return {
    std::experimental::parallel::min_element(first, last, cmp),
    std::experimental::parallel::max_element(first, last, cmp)
  };
}

//
// minmax_element is implemented by reduce
//
template< typename ForwardIt >
std::pair<ForwardIt, ForwardIt> minmax_element( ForwardIt first, ForwardIt last) {
  typedef typename std::iterator_traits<ForwardIt>::value_type T;
  return std::experimental::parallel::minmax_element(first, last, std::less<T>());
}

