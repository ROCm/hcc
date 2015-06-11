
//
// min_element is implemented by reduce
//
template< typename ForwardIt, typename Compare >
ForwardIt min_element( ForwardIt first, ForwardIt last, Compare cmp) {
  ForwardIt result = first;
  result = __reduce(first, last, result, [&](const ForwardIt& a, const ForwardIt& b) {
    return cmp(*a, *b) ? a : b;
  });
  return result;
}

//
// min_element is implemented by reduce
//
template< typename ForwardIt >
ForwardIt min_element( ForwardIt first, ForwardIt last) {
  typedef typename std::iterator_traits<ForwardIt>::value_type T;
  return std::experimental::parallel::min_element(first, last, std::less<T>());
}

