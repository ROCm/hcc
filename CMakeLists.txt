cmake_minimum_required( VERSION 2.8 )
project (CPPAMP)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")
#include (MacroEnsureOutOfSourceBuild)
#include (ExportFile)

set(GMAC_URL "." CACHE STRING "GMAC URL")
set(CLANG_URL "." CACHE STRING "CLANG URL")
set(LLVM_REVISION 162227)
include (EnsureGMACisPresent)

include (EnsureLLVMisPresent)
include (EnsureCLANGisPresent)
include (EnsureLIBCXXisPresent)
include (EnsureLIBCXXRTisPresent)
include (SetupCBE)
include (MCWAMP)

ensure_llvm_is_present(${PROJECT_SOURCE_DIR} compiler)
ensure_clang_is_present(${PROJECT_SOURCE_DIR} compiler ${CLANG_URL})
ensure_libcxx_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxx)
ensure_libcxxrt_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxxrt)
setup_CBackend(${PROJECT_SOURCE_DIR}/compiler/lib/Target CBackend)

set(LLVM_TARGETS_TO_BUILD X86)
set(LLVM_INCLUDE_EXAMPLES off)

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



# Turn on by default for now.
option(CXXAMP_ENABLE_GMAC "Build GMAC as backend" ON)

if(CXXAMP_ENABLE_GMAC)

ensure_gmac_is_present(${PROJECT_SOURCE_DIR} gmac ${GMAC_URL})

set(DEPS "gmac")

set(OPENCL_HEADER_DIR "/usr/local/cuda/include" CACHE PATH "OpenCL Header Directory")
set(OPENCL_LIBRARY_DIR "/usr/local/cuda/lib64" CACHE PATH "OpenCL Library Directory")

set(USE_OPENCL ON)
set(USE_LITE OFF)
set(USE_CUDA OFF)

find_path(OPENCL_HEADER cl.h ${OPENCL_HEADER_DIR}/CL)
if (NOT OPENCL_HEADER)
  MESSAGE(FATAL_ERROR "OpenCL header not found. Use -DOPENCL_HEADER_DIR=<path_to_cl.h>.")
endif(NOT OPENCL_HEADER)

find_library(OPENCL_LIBRARY OpenCL ${OPENCL_LIBRARY_DIR}/)
if (NOT OPENCL_LIBRARY)
  MESSAGE(FATAL_ERROR "OpenCL library not found. Use -DOPENCL_LIBRARY_DIR=<path_to_libOpenCL.so>.")
endif(NOT OPENCL_LIBRARY)

add_subdirectory(gmac)

endif(CXXAMP_ENABLE_GMAC)

MESSAGE("")
MESSAGE("OPENCL INFORMATION:")
MESSAGE("OPENCL_HEADER_DIR = ${OPENCL_HEADER_DIR}, actually found at: ${OPENCL_HEADER}")
MESSAGE("OPENCL_LIBRARY_DIR = ${OPENCL_LIBRARY_DIR}, actually found at: ${OPENCL_LIBRARY}")
MESSAGE("")


add_subdirectory(compiler)
# libc++
set(CLANG_CC1 "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
set(LIBCXX_SRC_DIR "${PROJECT_SOURCE_DIR}/libc++")
set(LIBCXX_BIN_DIR "${PROJECT_BINARY_DIR}/libc++")
file(MAKE_DIRECTORY ${LIBCXX_BIN_DIR})
add_custom_target(libc++
  COMMAND ${CMAKE_COMMAND} ${LIBCXX_SRC_DIR} -DCMAKE_CXX_COMPILER=${CLANG_CC1} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  COMMAND make  # not portable, but this is what it is.
  WORKING_DIRECTORY ${LIBCXX_BIN_DIR}
  DEPENDS clang
)

add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(axtor)
add_subdirectory(include)

add_custom_target(world
    DEPENDS clang libc++ ${DEPS}
)

MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make world' to build clang, libc++ and library for testing.")
MESSAGE("   'make' to build the rest of LLVM tools")
MESSAGE("")

