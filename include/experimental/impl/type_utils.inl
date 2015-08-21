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

} // namespace utils

