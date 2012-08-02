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

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

add_custom_command(
  OUTPUT ${CPPAMP_GTEST_LIB}
  COMMAND ${LLVM_TOOLS_DIR}/clang++ -std=c++amp -stdlib=libc++ -DGTEST_HAS_TR1_TUPLE=0 -c -I${LIBCXX_SRC_DIR}/include -I${GTEST_SRC_DIR} ${GTEST_SRC_DIR}/gtest/gtest_main.cc -o ${GTEST_LIB_DIR}/gtest_main.o
  COMMAND ${LLVM_TOOLS_DIR}/clang++ -std=c++amp -stdlib=libc++ -DGTEST_HAS_TR1_TUPLE=0 -c -I${LIBCXX_SRC_DIR}/include -I${GTEST_SRC_DIR} ${GTEST_SRC_DIR}/gtest/gtest-all.cc -o ${GTEST_LIB_DIR}/gtest-all.o
  COMMAND ${CMAKE_AR} cr ${CPPAMP_GTEST_LIB} ${GTEST_LIB_DIR}/gtest_main.o ${GTEST_LIB_DIR}/gtest-all.o
  DEPENDS clang ${GTEST_SRC_DIR}/gtest/gtest_main.cc ${GTEST_SRC_DIR}/gtest/gtest-all.cc ${CPPAMP_SOURCE_DIR}/include/amp.h
)

add_custom_target(gtest_amp
  DEPENDS ${CPPAMP_GTEST_LIB} 
)


add_custom_target(test
  COMMAND python ${LLVM_ROOT}/bin/llvm-lit --path ${LLVM_TOOLS_DIR} ${LLVM_LIT_ARGS} ${CMAKE_CURRENT_BINARY_DIR}
  # DEPENDS ${CPPAMP_GTEST_LIB} 
  COMMENT "Running C++AMP regression tests")


