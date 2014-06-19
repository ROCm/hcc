cmake_minimum_required( VERSION 2.8 )
project (CLAMP)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")
#include (MacroEnsureOutOfSourceBuild)
#include (ExportFile)
execute_process(COMMAND git describe --abbrev=4 --dirty --always --tags
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GMAC_URL "." CACHE STRING "GMAC URL")
set(CLANG_URL "." CACHE STRING "CLANG URL")
set(CMAKE_MACOSX_RPATH 1)
include (EnsureGMACisPresent)

include (EnsureLLVMisPresent)
include (EnsureCLANGisPresent)
include (EnsureLIBCXXisPresent)
include (EnsureLIBCXXRTisPresent)
include (SetupCBE)
include (SetupSPIRify)
include (SetupEraseNonkernel)
include (MCWAMP)

ensure_llvm_is_present(${PROJECT_SOURCE_DIR} compiler)
ensure_clang_is_present(${PROJECT_SOURCE_DIR} compiler ${CLANG_URL})
ensure_libcxx_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxx)
if (NOT APPLE)
ensure_libcxxrt_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxxrt)
endif (NOT APPLE)
setup_CBackend(${PROJECT_SOURCE_DIR}/compiler/lib/Target CBackend)
setup_SPIRify(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms SPIRify)
setup_EraseNonkernel(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms EraseNonkernel)
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
MESSAGE("(DEBUG|RELEASE) output changed to path:" "${EXECUTABLE_OUTPUT_PATH}")

endif (UNIX)

SET(PROJ_SEARCH_PATH "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/${LOCAL_LLVM_INCLUDE}" "${PROJECT_BINARY_DIR}/${LOCAL_LLVM_INCLUDE}") #  "${PROJECT_SOURCE_DIR}/compiler/utils/unittest/googletest/include")
include_directories( ${PROJ_SEARCH_PATH} )

LINK_DIRECTORIES( ${LLVM_LIB_DIR} )



# Turn on by default for now.
option(CXXAMP_ENABLE_GMAC "Build GMAC as backend" ON)
option(CXXAMP_ENABLE_HSA_OKRA "Use HSA Okra runtime as backend" OFF)
if (CXXAMP_ENABLE_HSA_OKRA)
# A bunch of external dependencies
set (AMDOCLTOOLS_ROOT "/opt/amd/"
        CACHE PATH "AMD OpenCL bin release path")
set (OKRA_ROOT "/opt/hsa/"
        CACHE PATH "Okra bin release path")
find_program (HSAILASM NAME hsailasm PATHS
        ${AMDOCLTOOLS_ROOT}/bin
        NO_DEFAULT_PATH)
# clc.exe and AMD OpenCL library builtins
find_program (AMD_CLC NAME clc PATHS
        "${AMDOCLTOOLS_ROOT}/bin"
        NO_DEFAULT_PATH)
# Okra headers needs JNI
find_package(JNI REQUIRED)
endif (CXXAMP_ENABLE_HSA_OKRA)

if(CXXAMP_ENABLE_GMAC)

ensure_gmac_is_present(${PROJECT_SOURCE_DIR} gmac ${GMAC_URL})

set(DEPS "gmac")

set(OPENCL_HEADER_DIR "/usr/local/cuda/include" CACHE PATH "OpenCL Header Directory")
set(OPENCL_LIBRARY_DIR "/usr/local/cuda/lib64" CACHE PATH "OpenCL Library Directory")

SET(FOO_BUILD_SHARED OFF CACHE BOOL "Build libfoo shared library")
set(USE_OPENCL ON CACHE BOOL "GMAC: Enable OpenCL")
set(USE_LITE OFF CACHE BOOL "GMAC: Disable Lite")
set(USE_CUDA OFF CACHE BOOL "GMAC: Disable CUDA")

find_path(OPENCL_HEADER cl.h ${OPENCL_HEADER_DIR} PATHS /usr/include/CL /opt/AMDAPP/include/CL )
if (NOT APPLE AND NOT OPENCL_HEADER)
  MESSAGE(FATAL_ERROR "OpenCL header not found. Use -DOPENCL_HEADER_DIR=<path_to_cl.h>.")
endif(NOT APPLE AND NOT OPENCL_HEADER)

find_library(OPENCL_LIBRARY OpenCL ${OPENCL_LIBRARY_DIR}/ PATHS /usr/lib)
if (NOT APPLE AND NOT OPENCL_LIBRARY)
  MESSAGE(FATAL_ERROR "OpenCL library not found. Use -DOPENCL_LIBRARY_DIR=<path_to_libOpenCL.so>.")
endif(NOT APPLE AND NOT OPENCL_LIBRARY)
# Pass OpenCL header dir to GMAC..
set(OPENCL_INCLUDE ${OPENCL_HEADER}/.. CACHE PATH "OpenCL header for GMAC")
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
if (APPLE)
add_custom_target(libc++
  COMMAND ${CMAKE_COMMAND} ${LIBCXX_SRC_DIR}
      -DLIBCXX_CXX_ABI="libcxxabi" -DLIBCXX_LIBCXXABI_INCLUDE_PATHS="/usr/include/"
      -DCMAKE_CXX_COMPILER=${CLANG_CC1}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  COMMAND make  # not portable, but this is what it is.
  WORKING_DIRECTORY ${LIBCXX_BIN_DIR}
  DEPENDS clang
)
else (APPLE)
# use libcxxrt as ABI instead
add_custom_target(libc++
  COMMAND ${CMAKE_COMMAND} ${LIBCXX_SRC_DIR} -DCMAKE_CXX_COMPILER=${CLANG_CC1} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  COMMAND make  # not portable, but this is what it is.
  WORKING_DIRECTORY ${LIBCXX_BIN_DIR}
  DEPENDS clang
)
endif (APPLE)

add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(include)
add_subdirectory(amp-conformance)

add_custom_target(world
    DEPENDS clang libc++ ${DEPS}
)

set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX "/opt/clamp")
set(CPACK_PACKAGE_NAME "clamp")
set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
set(CPACK_PACKAGE_VERSION "0.2.0-${GIT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "clamp: a C++AMP to OpenCL compiler")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Ray I-Jui Sung <ray@multicorewareinc.com>")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcxxamp (>= 0.2.0)")
if (CXXAMP_ENABLE_HSA_OKRA)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS 
            "${CPACK_DEBIAN_PACKAGE_DEPENDS}, okra-dev (>= 0.1.1)")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS 
            "${CPACK_DEBIAN_PACKAGE_DEPENDS}, amd-ocl-tools-dev (>= 0.1.1)")
endif (CXXAMP_ENABLE_HSA_OKRA)
set(CPACK_GENERATOR "DEB;TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")
include (CPack)
MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make world' to build clang, libc++ and library for testing.")
MESSAGE("   'make' to build the rest of LLVM tools")
MESSAGE("")

