cmake_minimum_required( VERSION 2.8 )
project (CPPAMP)
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release") 
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")
#include (MacroEnsureOutOfSourceBuild)
#include (ExportFile)

include (EnsureLLVMisPresent)
include (EnsureCLANGisPresent)
include (EnsureLIBCXXisPresent)
include (EnsureLIBCXXRTisPresent)
include (MCWAMP)

ensure_llvm_is_present(${PROJECT_SOURCE_DIR} compiler)
ensure_clang_is_present(${PROJECT_SOURCE_DIR} compiler)
ensure_libcxx_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxx)
ensure_libcxxrt_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxxrt)

#MACRO_ENSURE_OUT_OF_SOURCE_BUILD("Please build ${CMAKE_PROJECT_NAME} in a different directory than the source directory")

set(LLVM_TARGETS_TO_BUILD "X86")
#set(LLVM_BUILD_TOOLS off)
set(LLVM_INCLUDE_TOOLS off)
set(LLVM_INCLUDE_EXAMPLES off)
set(LLVM_INCLUDE_TESTS on)

# Regression test
set(LLVM_SRC "${PROJECT_SOURCE_DIR}/compiler")
set(LLVM_ROOT "${PROJECT_BINARY_DIR}/compiler")

# obtain specific information about llvm setup
SET(LOCAL_LLVM_INCLUDE compiler/include)

# setup compilation environment
if (UNIX)
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build/${CMAKE_CFG_INTDIR}/${CMAKE_BUILD_TYPE}/bin" )
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build/${CMAKE_CFG_INTDIR}/${CMAKE_BUILD_TYPE}/lib" )
else (UNIX)
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build/${CMAKE_BUILD_TYPE}/bin" )
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build/${CMAKE_BUILD_TYPE}/lib" )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH})
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH})

SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH})
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH})

SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH})
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
MESSAGE("(DEBUG|RELEASE) output changed to path:", "${EXECUTABLE_OUTPUT_PATH}")

endif (UNIX)

SET(PROJ_SEARCH_PATH "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/${LOCAL_LLVM_INCLUDE}" "${PROJECT_BINARY_DIR}/${LOCAL_LLVM_INCLUDE}") #  "${PROJECT_SOURCE_DIR}/compiler/utils/unittest/googletest/include")
include_directories( ${PROJ_SEARCH_PATH} )

LINK_DIRECTORIES( ${LLVM_LIB_DIR} )

# Note: this may be too Gcc-specific. Check with other compilers and set flags accordingly
#if(UNIX)
#SET(LLVM_CPPFLAGS "-DNDEBUG -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
#SET(LLVM_CFLAGS "-Os -fomit-frame-pointer -fPIC")
#SET(LLVM_CXXFLAGS "-Os -fomit-frame-pointer  -fPIC -Woverloaded-virtual -Wcast-qual -fno-rtti -fno-exceptions")
#add_definitions("-Wall -fno-strict-aliasing -fPIC")
#else(UNIX)
#SET(LLVM_CPPFLAGS "-D_CRT_SECURE_NO_WARNINGS -D_MBCS")
#SET(LLVM_CFLAGS "-Os -fomit-frame-pointer -fPIC")
#SET(LLVM_CXXFLAGS "-Os -Gm -EHsc -errorReport:prompt")
#endif(UNIX)

set(LLVM_INCLUDE_TESTS ON)

set(CLANG_CC1 "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
set(LIBCXX_SRC_DIR "${PROJECT_SOURCE_DIR}/libc++")
set(LIBCXX_BIN_DIR "${PROJECT_BINARY_DIR}/libc++")
file(MAKE_DIRECTORY ${LIBCXX_BIN_DIR})
add_custom_target(libc++
  COMMAND ${CMAKE_COMMAND} ${LIBCXX_SRC_DIR} -DCMAKE_CXX_COMPILER=${CLANG_CC1}
  COMMAND make  # not portable, but this is what it is.
  WORKING_DIRECTORY ${LIBCXX_BIN_DIR}
  DEPENDS clang
)

# Turn on by default for now.
option(CXXAMP_ENABLE_GMAC "Build GMAC as backend" ON)

if(CXXAMP_ENABLE_GMAC)
set(OPENCL_HEADER_DIR "/usr/local/cuda/include" CACHE PATH "OpenCL Header Directory")
set(OPENCL_LIBRARY_DIR "/usr/local/cuda/lib64" CACHE PATH "OpenCL Library Directory")

find_path(OPENCL_HEADER cl.h ${OPENCL_HEADER_DIR}/CL)
if (NOT OPENCL_HEADER)
  MESSAGE(FATAL_ERROR "OpenCL header not found. Use -DOPENCL_HEADER_DIR=<path_to_cl.h>.")
endif(NOT OPENCL_HEADER)

find_library(OPENCL_LIBRARY OpenCL ${OPENCL_LIBRARY_DIR}/)
if (NOT OPENCL_LIBRARY)
  MESSAGE(FATAL_ERROR "OpenCL library not found. Use -DOPENCL_LIBRARY_DIR=<path_to_libOpenCL.so>.")
endif(NOT OPENCL_LIBRARY)

endif(CXXAMP_ENABLE_GMAC)

MESSAGE("")
MESSAGE("OPENCL INFORMATION:")
MESSAGE("OPENCL_HEADER_DIR = ${OPENCL_HEADER_DIR}, actually found at: ${OPENCL_HEADER}")
MESSAGE("OPENCL_LIBRARY_DIR = ${OPENCL_LIBRARY_DIR}, actually found at: ${OPENCL_LIBRARY}")
MESSAGE("")

add_subdirectory(compiler)
add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)

add_custom_target(gmac
  # Find python first.
  COMMAND python ${PROJECT_SOURCE_DIR}/scripts/build/gmac.py ${PROJECT_SOURCE_DIR}/gmac ${OPENCL_HEADER_DIR} ${OPENCL_LIBRARY_DIR} ${CXXAMP_ENABLE_GMAC}
)

MESSAGE("COMMAND python ${PROJECT_SOURCE_DIR}/scripts/build/gmac.py ${PROJECT_SOURCE_DIR}/gmac ${OPENCL_HEADER_DIR} ${OPENCL_LIBRARY_DIR} ${CXXAMP_ENABLE_GMAC}")

add_custom_target(world
  DEPENDS clang libc++ gmac
)

MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make world' to build llvm, clang, libc++ and library for testing.")
MESSAGE("")

