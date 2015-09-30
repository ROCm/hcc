
//
// count_if is implemented by reduce
//
template< typename InputIt, typename UnaryPredicate >
typename std::iterator_traits<InputIt>::difference_type
count_if( InputIt first, InputIt last, UnaryPredicate p ) {
  typedef typename std::iterator_traits<InputIt>::value_type T;

  return reduce(first, last, 0, [&](const T& a, const T& b) {
    return p(b) ? (a + 1) : a;
  });
}

//
// count is implemented by reduce
//
template< typename InputIt, typename T >
typename std::iterator_traits<InputIt>::difference_type
count( InputIt first, InputIt last, const T& value ) {
  return reduce(first, last, 0, [&](const T& a, const T& b) {
    return (b == value) ? (a + 1) : a;
  });
}

