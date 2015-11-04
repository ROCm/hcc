#OpenCL
include_directories(${OPENCL_HEADER}/../ ${CMAKE_CURRENT_SOURCE_DIR}/amp_test_lib/inc)

add_mcwamp_library(amptest
amp_test_lib/src/context.cpp amp_test_lib/src/device.cpp amp_test_lib/src/logging.cpp
amp_test_lib/src/main.cpp amp_test_lib/src/runall.cpp amp_test_lib/src/string_utils.cpp
)

set(AMPTESTINC ${CMAKE_CURRENT_SOURCE_DIR}/amp_test_lib/inc)
get_property(AMPTESTLIB TARGET amptest PROPERTY LOCATION)
set(RUNTESTSDIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.pl.in
  ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_one.pl.in
  ${CMAKE_CURRENT_BINARY_DIR}/test_one.pl @ONLY)

add_custom_target(conformance-all
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl ${CMAKE_CURRENT_SOURCE_DIR}/Tests
  DEPENDS amptest
  COMMENT "Running all C++AMP conformance tests")

add_custom_target(conformance-all-hw
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl --skip-build-only ${CMAKE_CURRENT_SOURCE_DIR}/Tests
  DEPENDS amptest
  COMMENT "Running all C++AMP conformance tests skipping build only tests")

set(AMP_PASSLIST passlist.txt)

if (HSA_USE_AMDGPU_BACKEND)
  set(AMP_PASSLIST amdgpu-passlist.txt)
endif()

add_custom_target(conformance
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl ${CMAKE_CURRENT_SOURCE_DIR}/Tests ${CMAKE_CURRENT_SOURCE_DIR}/${AMP_PASSLIST}
  DEPENDS amptest
  COMMENT "Running C++AMP conformance tests in ${AMP_PASSLIST}")

add_custom_target(conformance-hw
  COMMAND perl ${CMAKE_CURRENT_BINARY_DIR}/run_tests.pl --skip-build-only ${CMAKE_CURRENT_SOURCE_DIR}/Tests ${CMAKE_CURRENT_SOURCE_DIR}/${AMP_PASSLIST}
  DEPENDS amptest
  COMMENT "Running C++AMP conformance tests in ${AMP_PASSLIST} skipping build only tests")
