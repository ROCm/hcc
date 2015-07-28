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

} // inline namespace v1
} // namespace parallel
} // namespace experimental 
} // namespace std

