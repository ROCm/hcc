#pragma once

namespace utils {
// type traits utils
template<class It>
using tag = typename std::iterator_traits<It>::iterator_category;

template<class Condition, class T = void>
using EnableIf = typename std::enable_if<Condition::value, T>::type *;

template<class It>
using isInputIt = std::is_base_of<std::input_iterator_tag,
                                  tag<It>>;
template<class It>
using isForwardIt = std::is_base_of<std::forward_iterator_tag,
                                    tag<It>>;

template<class It>
using isRandomAccessIt = std::is_base_of<std::random_access_iterator_tag,
                                         tag<It>>;

template<class ExecutionPolicy>
using isExecutionPolicy =
        is_execution_policy<typename std::decay<ExecutionPolicy>::type>;

template<class ExecutionPolicy>
inline bool isParallel(ExecutionPolicy &&exec) {
  typedef typename std::decay<decltype(exec)>::type Tp;
  if (std::is_base_of<parallel_execution_policy, Tp>::value ||
      std::is_base_of<parallel_vector_execution_policy, Tp>::value) {
    return true;
  }
  return false;
}

// get raw pointer from an iterator
template<typename T>
inline typename std::iterator_traits<T>::pointer
get_pointer(T it) { return &(*it); }

// FIXME: there will be a wrapper on std::array_view implementation,
//        so that operator * works properly
//
//        Reference: N4512
// for array_view
template<size_t N>
inline std::bounds_iterator<N>
get_pointer(std::bounds_iterator<N> it) { return it; }


} // namespace utils

