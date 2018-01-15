//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

// sequential versions of algorithms are implemented inside this  file
// FIXME: gradually move them to SPMD-version of algorithms
#include <numeric>

namespace std {
namespace experimental {
namespace parallel {
inline namespace v1 {

// inner_product
template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename T>
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, T>::type
inner_product(ExecutionPolicy&& exec,
              InputIt1 first1, InputIt1 last1,
              InputIt2 first2,
              T value) {
    return std::inner_product(first1, last1, first2, value);
}

template <typename ExecutionPolicy, typename InputIt1, typename InputIt2, typename T, typename BinaryOperation1, typename BinaryOperation2> 
typename std::enable_if<is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value, T>::type
inner_product(ExecutionPolicy&& exec,
              InputIt1 first1, InputIt1 last1,
              InputIt2 first2,
              T value,
              BinaryOperation1 op1,
              BinaryOperation2 op2) {
    return std::inner_product(first1, last1, first2, value, op1, op2);
}


} // inline namespace v1
} // namespace parallel
} // namespace experimental 
} // namespace std

