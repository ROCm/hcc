include_directories(${CMAKE_CURRENT_SOURCE_DIR}/amp_test_lib/inc)
if (USE_LIBCXX)
  include_directories(${LIBCXX_HEADER})
endif(USE_LIBCXX)

add_mcwamp_library(amptest
amp_test_lib/src/context.cpp amp_test_lib/src/device.cpp amp_test_lib/src/logging.cpp
amp_test_lib/src/main.cpp amp_test_lib/src/runall.cpp amp_test_lib/src/string_utils.cpp
)

target_link_libraries(amptest c++)

set(AMP_CONFORMANCE_NUM_THREADS 1 CACHE INT "The number of threads to use when running amp conformance")

set(AMPTESTINC ${CMAKE_CURRENT_SOURCE_DIR}/amp_test_lib/inc)
if(POLICY CMP0026)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0026 OLD)
endif()
get_property(AMPTESTLIB TARGET amptest PROPERTY LOCATION)
if(POLICY CMP0026)
  cmake_policy(POP)
endif()
set(RUNTESTSDIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.pl.in
  ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_one.pl.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_one.pl @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/output-to-junit.pl
  ${CMAKE_CURRENT_BINARY_DIR}/output-to-junit.pl COPYONLY)

add_custom_target(conformance-all
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl ${CMAKE_CURRENT_SOURCE_DIR}/Tests
  DEPENDS amptest
  COMMENT "Running all C++AMP conformance tests")

add_custom_target(conformance-all-hw
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl --skip-build-only ${CMAKE_CURRENT_SOURCE_DIR}/Tests
  DEPENDS amptest
  COMMENT "Running all C++AMP conformance tests skipping build only tests")

# obsolete
#set(AMP_PASSLIST passlist.txt)
set(AMP_PASSLIST amdgpu-passlist.txt)

add_custom_target(conformance
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl ${CMAKE_CURRENT_SOURCE_DIR}/Tests ${CMAKE_CURRENT_SOURCE_DIR}/${AMP_PASSLIST}
  DEPENDS amptest
  COMMENT "Running C++AMP conformance tests in ${AMP_PASSLIST}")

add_custom_target(conformance-hw
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl --skip-build-only ${CMAKE_CURRENT_SOURCE_DIR}/Tests ${CMAKE_CURRENT_SOURCE_DIR}/${AMP_PASSLIST}
  DEPENDS amptest
  COMMENT "Running C++AMP conformance tests in ${AMP_PASSLIST} skipping build only tests")
