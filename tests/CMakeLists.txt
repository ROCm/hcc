set(LLVM_SOURCE_DIR "${LLVM_SRC}")
set(LLVM_BINARY_DIR "${LLVM_ROOT}")
set(LLVM_TOOLS_DIR "${LLVM_ROOT}/bin")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/")
set(CPPAMP_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(CPPAMP_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CPPAMP_OBJ_DIR "${CMAKE_BINARY_DIR}")

# libcxx
set(LIBCXX_SRC_DIR "${PROJECT_SOURCE_DIR}/libc++/libcxx")
set(LIBCXX_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxx")
set(LIBCXXRT_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxxrt")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

  add_custom_target(test
    COMMAND python 
                ${LLVM_ROOT}/bin/llvm-lit
    --path ${LLVM_TOOLS_DIR}
    ${LLVM_LIT_ARGS}
                ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Running C++AMP regression tests")


