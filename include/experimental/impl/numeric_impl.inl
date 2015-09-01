//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

//
// count_if is implemented by reduce
//
template<typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
typename std::iterator_traits<InputIt>::difference_type
count_if(InputIt first, InputIt last, UnaryPredicate p) {
  typedef typename std::iterator_traits<InputIt>::value_type T;
  typedef typename std::iterator_traits<InputIt>::difference_type DT;

  const size_t N = static_cast<size_t>(std::distance(first, last));
  if (N <= details::PARALLELIZE_THRESHOLD) {
    return std::count_if(first, last, p);
  }

  return transform_reduce(first, last,
                          [p](const T &v) -> DT { return DT(p(v)); },
                          DT{},
                          std::plus<DT>());
}

template<typename ExecutionPolicy,
         typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
typename std::iterator_traits<InputIt>::difference_type
count_if(ExecutionPolicy&& exec, InputIt first, InputIt last, UnaryPredicate p) {
  if (utils::isParallel(exec)) {
    return count_if(first, last, p);
  } else {
    return std::count_if(first, last, p);
  }
}


//
// count is implemented by transform_reduce
//
template<typename InputIt, typename T,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
typename std::iterator_traits<InputIt>::difference_type
count(InputIt first, InputIt last, const T& value) {
  return count_if(first, last,
                  [&value](const T &v) -> bool { return v == value; });
}

template<typename ExecutionPolicy,
         typename InputIt, typename T,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
typename std::iterator_traits<InputIt>::difference_type
count(ExecutionPolicy&& exec, InputIt first, InputIt last, const T& value) {
  if (utils::isParallel(exec)) {
    return count(first, last, value);
  } else {
    return std::count(first, last, value);
  }
}


//
// max_element is implemented by reduce
//
template<typename ForwardIt, typename Compare,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt max_element(ForwardIt first, ForwardIt last, Compare cmp) {
  ForwardIt result = first;
  result = __reduce(first, last, result, [&](const ForwardIt& a, const ForwardIt& b) {
    return cmp(*a, *b) ? b : a;
  });
  return result;
}

template<typename ExecutionPolicy,
         typename ForwardIt, typename Compare,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt max_element(ExecutionPolicy&& exec, ForwardIt first, ForwardIt last, Compare cmp) {
  if (utils::isParallel(exec)) {
    return max_element(first, last, cmp);
  } else {
    return std::max_element(first, last, cmp);
  }
}


//
// max_element is implemented by reduce
//
template<typename ForwardIt,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt max_element(ForwardIt first, ForwardIt last) {
  typedef typename std::iterator_traits<ForwardIt>::value_type T;
  return std::experimental::parallel::max_element(first, last, std::less<T>());
}

template<typename ExecutionPolicy,
         typename ForwardIt,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt max_element(ExecutionPolicy&& exec, ForwardIt first, ForwardIt last) {
  if (utils::isParallel(exec)) {
    return max_element(first, last);
  } else {
    return std::max_element(first, last);
  }
}


//
// min_element is implemented by reduce
//
template<typename ForwardIt, typename Compare,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt min_element(ForwardIt first, ForwardIt last, Compare cmp) {
  ForwardIt result = first;
  result = __reduce(first, last, result, [&](const ForwardIt& a, const ForwardIt& b) {
    return cmp(*a, *b) ? a : b;
  });
  return result;
}

template<typename ExecutionPolicy,
         typename ForwardIt, typename Compare,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt min_element(ExecutionPolicy&& exec, ForwardIt first, ForwardIt last, Compare cmp) {
  if (utils::isParallel(exec)) {
    return min_element(first, last, cmp);
  } else {
    return std::min_element(first, last, cmp);
  }
}


//
// min_element is implemented by reduce
//
template<typename ForwardIt,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt min_element(ForwardIt first, ForwardIt last) {
  typedef typename std::iterator_traits<ForwardIt>::value_type T;
  return std::experimental::parallel::min_element(first, last, std::less<T>());
}

template<typename ExecutionPolicy,
         typename ForwardIt,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
ForwardIt min_element(ExecutionPolicy&& exec, ForwardIt first, ForwardIt last) {
  if (utils::isParallel(exec)) {
    return min_element(first, last);
  } else {
    return std::min_element(first, last);
  }
}


//
// minmax_element is implemented by reduce
//
template<typename ForwardIt, typename Compare,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
std::pair<ForwardIt, ForwardIt>
minmax_element(ForwardIt first, ForwardIt last, Compare cmp) {
  return {
    std::experimental::parallel::min_element(first, last, cmp),
    std::experimental::parallel::max_element(first, last, cmp)
  };
}

template<typename ExecutionPolicy,
         typename ForwardIt, typename Compare,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
std::pair<ForwardIt, ForwardIt>
minmax_element(ExecutionPolicy&& exec, ForwardIt first, ForwardIt last, Compare cmp) {
  if (utils::isParallel(exec)) {
    return minmax_element(first, last, cmp);
  } else {
    return std::minmax_element(first, last, cmp);
  }
}


//
// minmax_element is implemented by reduce
//
template<typename ForwardIt,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
std::pair<ForwardIt, ForwardIt>
minmax_element(ForwardIt first, ForwardIt last) {
  typedef typename std::iterator_traits<ForwardIt>::value_type T;
  return std::experimental::parallel::minmax_element(first, last, std::less<T>());
}

template<typename ExecutionPolicy,
         typename ForwardIt,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isForwardIt<ForwardIt>> = nullptr>
std::pair<ForwardIt, ForwardIt>
minmax_element(ExecutionPolicy&& exec, ForwardIt first, ForwardIt last) {
  if (utils::isParallel(exec)) {
    return minmax_element(first, last);
  } else {
    return std::minmax_element(first, last);
  }
}


//
// any_of is implemented by transform_reduce
//
template<typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
bool all_of(InputIt first, InputIt last, UnaryPredicate p) {
  typedef typename std::iterator_traits<InputIt>::value_type T;

  return transform_reduce(first, last, p, true,
                          std::logical_and<bool>());
}

template<typename ExecutionPolicy,
         typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
bool all_of(ExecutionPolicy&& exec, InputIt first, InputIt last, UnaryPredicate p) {
  if (utils::isParallel(exec)) {
    return all_of(first, last, p);
  } else {
    return std::all_of(first, last, p);
  }
}


//
// any_of is implemented by transform_reduce
//
template<typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
bool any_of(InputIt first, InputIt last, UnaryPredicate p) {
  typedef typename std::iterator_traits<InputIt>::value_type T;

  return transform_reduce(first, last, p, false,
                          std::logical_or<bool>());
}

template<typename ExecutionPolicy,
         typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
bool any_of(ExecutionPolicy&& exec, InputIt first, InputIt last, UnaryPredicate p) {
  if (utils::isParallel(exec)) {
    return any_of(first, last, p);
  } else {
    return std::any_of(first, last, p);
  }
}


//
// none_of is implemented by transform_reduce
//
template<typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
bool none_of( InputIt first, InputIt last, UnaryPredicate p ) {
  return any_of(first, last, p) == false;
}

template<typename ExecutionPolicy,
         typename InputIt, typename UnaryPredicate,
         utils::EnableIf<utils::isExecutionPolicy<ExecutionPolicy>> = nullptr,
         utils::EnableIf<utils::isInputIt<InputIt>> = nullptr>
bool none_of(ExecutionPolicy&& exec, InputIt first, InputIt last, UnaryPredicate p ) {
  if (utils::isParallel(exec)) {
    return none_of(first, last, p);
  } else {
    return std::none_of(first, last, p);
  }
}

