//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// sequential versions of algorithms are implemented inside this  file
// FIXME: gradually move them to SPMD-version of algorithms
#include <algorithm>

namespace std {
namespace experimental {
namespace parallel {
inline namespace v1 {

// mismatch
template <typename ExecutionPolicy, typename InputIt1, typename InputIt2>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, std::pair<InputIt1, InputIt2>>::type
mismatch(ExecutionPolicy&& exec,
         InputIt1 first1, InputIt1 last1,
         InputIt2 first2) {
    return std::mismatch(first1, last1, first2);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, std::pair<InputIt1, InputIt2>>::type
mismatch(ExecutionPolicy&& exec,
         InputIt1 first1, InputIt1 last1,
         InputIt2 first2,
         BinaryPredicate p) {
    return std::mismatch(first1, last1, first2, p);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, std::pair<InputIt1, InputIt2>>::type
mismatch(ExecutionPolicy&& exec,
         InputIt1 first1, InputIt1 last1,
         InputIt2 first2, InputIt2 last2) {
    return std::mismatch(first1, last1, first2, last2);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, std::pair<InputIt1, InputIt2>>::type
mismatch(ExecutionPolicy&& exec,
         InputIt1 first1, InputIt1 last1,
         InputIt2 first2, InputIt2 last2,
         BinaryPredicate p) {
    return std::mismatch(first1, last1, first2, last2, p);
}


// equal
template <typename ExecutionPolicy, typename InputIt1, typename InputIt2>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, bool>::type
equal(ExecutionPolicy&& exec,
      InputIt1 first1, InputIt1 last1, 
      InputIt2 first2) {
    return std::equal(first1, last1, first2);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, bool>::type
equal(ExecutionPolicy&& exec,
      InputIt1 first1, InputIt1 last1, 
      InputIt2 first2, BinaryPredicate p) {
    return std::equal(first1, last1, first2, p);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, bool>::type
equal(ExecutionPolicy&& exec,
      InputIt1 first1, InputIt1 last1, 
      InputIt2 first2, InputIt2 last2) {
    return std::equal(first1, last1, first2, last2);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, bool>::type
equal(ExecutionPolicy&& exec,
      InputIt1 first1, InputIt1 last1, 
      InputIt2 first2, InputIt2 last2,
      BinaryPredicate p) {
    return std::equal(first1, last1, first2, last2, p);
}


// find
template <typename ExecutionPolicy, typename InputIt, typename T>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, InputIt>::type
find(ExecutionPolicy&& exec,
     InputIt first, InputIt last, const T& value) {
    return std::find(first, last, value);
}


// find_if
template <typename ExecutionPolicy, typename InputIt, typename UnaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, InputIt>::type
find_if(ExecutionPolicy&& exec,
        InputIt first, InputIt last, 
        UnaryPredicate p) {
    return std::find_if(first, last, p);
}


// find_if_not
template <typename ExecutionPolicy, typename InputIt, typename UnaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, InputIt>::type
find_if_not(ExecutionPolicy&& exec,
            InputIt first, InputIt last, 
            UnaryPredicate p) {
    return std::find_if_not(first, last, p);
}


// find_end
template <typename ExecutionPolicy, typename ForwardIt1, typename ForwardIt2>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt1>::type
find_end(ExecutionPolicy&& exec,
         ForwardIt1 first, ForwardIt1 last,
         ForwardIt2 s_first, ForwardIt2 s_last) {
    return std::find_end(first, last, s_first, s_last);
}

template <typename ExecutionPolicy, typename ForwardIt1, typename ForwardIt2, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt1>::type
find_end(ExecutionPolicy&& exec,
         ForwardIt1 first, ForwardIt1 last,
         ForwardIt2 s_first, ForwardIt2 s_last, BinaryPredicate p) {
    return std::find_end(first, last, s_first, s_last, p);
}


// find_first_of
template <typename ExecutionPolicy, typename InputIt, typename ForwardIt>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, InputIt>::type
find_first_of(ExecutionPolicy&& exec,
              InputIt first, InputIt last,
              ForwardIt s_first, ForwardIt s_last) {
    return std::find_first_of(first, last, s_first, s_last);
}

template <typename ExecutionPolicy, typename InputIt, typename ForwardIt, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, InputIt>::type
find_first_of(ExecutionPolicy&& exec,
              InputIt first, InputIt last,
              ForwardIt s_first, ForwardIt s_last, BinaryPredicate p) {
    return std::find_first_of(first, last, s_first, s_last, p);
}


// adjacent_find
template <typename ExecutionPolicy, typename ForwardIt >
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt>::type
adjacent_find(ExecutionPolicy&& exec,
              ForwardIt first, ForwardIt last) {
    return std::adjacent_find(first, last);
}

template <typename ExecutionPolicy, typename ForwardIt, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt>::type
adjacent_find(ExecutionPolicy&& exec,
              ForwardIt first, ForwardIt last, BinaryPredicate p) {
    return std::adjacent_find(first, last, p);
}


// search
template <typename ExecutionPolicy, typename ForwardIt1, typename ForwardIt2>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt1>::type
search(ExecutionPolicy&& exec,
       ForwardIt1 first, ForwardIt1 last,
       ForwardIt2 s_first, ForwardIt2 s_last) {
    return std::search(first, last, s_first, s_last);
}

template <typename ExecutionPolicy, typename ForwardIt1, typename ForwardIt2, typename BinaryPredicate>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt1>::type
search(ExecutionPolicy&& exec,
       ForwardIt1 first, ForwardIt1 last,
       ForwardIt2 s_first, ForwardIt2 s_last, BinaryPredicate p) {
    return std::search(first, last, s_first, s_last, p);
}


// search_n
template <typename ExecutionPolicy, typename ForwardIt, typename Size, typename T >
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt>::type
search_n(ExecutionPolicy&& exec,
         ForwardIt first, ForwardIt last, Size count, const T& value) {
    return std::search_n(first, last, count, value);
}

template <typename ExecutionPolicy, typename ForwardIt, typename Size, typename T, typename BinaryPredicate >
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, ForwardIt>::type
search_n(ExecutionPolicy&& exec,
         ForwardIt first, ForwardIt last, Size count, const T& value, BinaryPredicate p) {
    return std::search_n(first, last, count, value, p);
}


} // inline namespace v1
} // namespace parallel
} // namespace experimental 
} // namespace std

