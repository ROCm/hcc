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

} // inline namespace v1
} // namespace parallel
} // namespace experimental 
} // namespace std

