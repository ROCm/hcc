
//
// any_of is implemented by transform_reduce
//
template< typename InputIt, typename UnaryPredicate >
bool all_of( InputIt first, InputIt last, UnaryPredicate p ) {
  typedef typename std::iterator_traits<InputIt>::value_type T;

  return transform_reduce(first, last, p, true,
                          [](bool a, bool b) { return a && b; });
}

//
// any_of is implemented by transform_reduce
//
template< typename InputIt, typename UnaryPredicate >
bool any_of( InputIt first, InputIt last, UnaryPredicate p ) {
  typedef typename std::iterator_traits<InputIt>::value_type T;

  return transform_reduce(first, last, p, false,
                          [](bool a, bool b) { return a || b; });
}

//
// none_of is implemented by transform_reduce
//
template< typename InputIt, typename UnaryPredicate >
bool none_of( InputIt first, InputIt last, UnaryPredicate p ) {
  return any_of(first, last, p) == false;
}

