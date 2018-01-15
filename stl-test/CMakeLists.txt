set(RUNTESTSDIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.pl.in
  ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_one.pl.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_one.pl @ONLY)

add_custom_target(stl-check
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl ${CMAKE_CURRENT_SOURCE_DIR}/Tests
  COMMENT "Running all C++AMP STL tests")
