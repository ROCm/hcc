set(LLVM_SOURCE_DIR "${LLVM_SRC}")
set(LLVM_BINARY_DIR "${LLVM_ROOT}")
set(LLVM_TOOLS_DIR "${LLVM_ROOT}/bin")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib")
set(HCC_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(HCC_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HCC_OBJ_DIR "${CMAKE_BINARY_DIR}")

# gtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/utils")
set(GTEST_LIB_DIR "${PROJECT_BINARY_DIR}/utils/gtest")
set(HCC_GTEST_LIB "${GTEST_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main_amp${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(MAKE_DIRECTORY ${GTEST_LIB_DIR})

# MCWAMP
set(MCWAMP_LIB_DIR "${LIBRARY_OUTPUT_PATH}")
set(MCWAMP_TOOL_DIR "${PROJECT_BINARY_DIR}/compiler/bin")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../hcc_config/hcc_config.hxx.in
  ${CMAKE_CURRENT_BINARY_DIR}/../hcc_config/hcc_config.hxx ESCAPE_QUOTES)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/extractkernel.in
  ${CMAKE_CURRENT_BINARY_DIR}/../bin/extractkernel @ONLY)

if(POLICY CMP0037)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0037 OLD)
endif()

include(ProcessorCount)
ProcessorCount(N)

add_custom_target(test
  COMMAND python ${LLVM_ROOT}/bin/llvm-lit -j ${N} --path ${LLVM_TOOLS_DIR} -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS clamp-scripts
  COMMENT "Running HCC regression tests")

if(POLICY CMP0037)
  cmake_policy(POP)
endif()
