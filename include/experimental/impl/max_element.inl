
//
// max_element is implemented by reduce
//
template< typename ForwardIt, typename Compare >
ForwardIt max_element( ForwardIt first, ForwardIt last, Compare cmp) {
  ForwardIt result = first;
  result = __reduce(first, last, result, [&](const ForwardIt& a, const ForwardIt& b) {
    return cmp(*a, *b) ? b : a;
  });
  return result;
}

//
// max_element is implemented by reduce
//
template< typename ForwardIt >
ForwardIt max_element( ForwardIt first, ForwardIt last) {
  typedef typename std::iterator_traits<ForwardIt>::value_type T;
  return std::experimental::parallel::max_element(first, last, std::less<T>());
}

