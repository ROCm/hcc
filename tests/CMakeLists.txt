set(LLVM_SOURCE_DIR "${LLVM_SRC}")
set(LLVM_BINARY_DIR "${LLVM_ROOT}")
set(LLVM_TOOLS_DIR "${LLVM_ROOT}/bin")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib")
set(CPPAMP_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(CPPAMP_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CPPAMP_OBJ_DIR "${CMAKE_BINARY_DIR}")

# libcxx
set(LIBCXX_SRC_DIR "${PROJECT_SOURCE_DIR}/libc++/libcxx")
set(LIBCXX_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxx/lib")
set(LIBCXXRT_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxxrt/lib")

# gtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/utils")
set(GTEST_LIB_DIR "${PROJECT_BINARY_DIR}/utils/gtest")
set(CPPAMP_GTEST_LIB "${GTEST_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main_amp${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(MAKE_DIRECTORY ${GTEST_LIB_DIR})

set(AMPCL_LIB_DIR "${PROJECT_BINARY_DIR}/build/${CMAKE_BUILD_TYPE}/lib")

# opencl_math
set(OPENCL_MATH_DIR "${PROJECT_SOURCE_DIR}/include")

# MCWAMP
set(MCWAMP_LIB_DIR "${LIBRARY_OUTPUT_PATH}")
set(MCWAMP_TOOL_DIR "${PROJECT_BINARY_DIR}/lib")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-config.hxx.in
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/clamp-config.hxx ESCAPE_QUOTES)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-spirify.in
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/clamp-spirify @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-device.in
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/clamp-device @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-embed.in
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/clamp-embed @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-assemble.in
  ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-assemble @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-link.in
  ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-link @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-preprocess.in
  ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-preprocess @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-hsatools.in
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/clamp-hsatools @ONLY)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/clamp-hsail.in
  ${CMAKE_CURRENT_BINARY_DIR}/../lib/clamp-hsail @ONLY)

add_custom_target(test
  COMMAND python ${LLVM_ROOT}/bin/llvm-lit -j 8 --path ${LLVM_TOOLS_DIR} ${LLVM_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
  # DEPENDS ${CPPAMP_GTEST_LIB}
  COMMENT "Running C++AMP regression tests")
