cmake_minimum_required( VERSION 2.8 )
project (CLAMP)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")
execute_process(COMMAND git describe --abbrev=4 --dirty --always --tags
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# set as release build by default
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Release Debug" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# set default cppamp-ng URL
IF (NOT CLANG_URL)
  SET(CLANG_URL "https://bitbucket.org/multicoreware/cppamp-ng.git" CACHE STRING "CLANG URL" FORCE)
ENDIF (NOT CLANG_URL)
set(CMAKE_MACOSX_RPATH 1)

include (EnsureLLVMisPresent)
include (EnsureCLANGisPresent)
include (EnsureLIBCXXisPresent)
include (EnsureLIBCXXRTisPresent)
include (SetupCBE)
include (SetupSPIRify)
include (SetupEraseNonkernel)
include (SetupTileUniform)
include (MCWAMP)
include (PatchSimplifyCFG)

ensure_llvm_is_present(${PROJECT_SOURCE_DIR} compiler)
ensure_clang_is_present(${PROJECT_SOURCE_DIR} compiler ${CLANG_URL})
ensure_libcxx_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxx)
if (NOT APPLE)
ensure_libcxxrt_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxxrt)
endif (NOT APPLE)
setup_CBackend(${PROJECT_SOURCE_DIR}/compiler/lib/Target CBackend)
setup_SPIRify(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms SPIRify)
setup_EraseNonkernel(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms EraseNonkernel)
setup_TileUniform(${PROJECT_SOURCE_DIR}/compiler/lib/Analysis TileUniform)
patch_SimplifyCFG(TileUniform)
set(LLVM_TARGETS_TO_BUILD X86 CACHE STRING "LLVM host target")
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD CBackend CACHE STRING "set this to 'CBackend' to enable generating OpenCL-C")
set(LLVM_INCLUDE_EXAMPLES off)

# Regression test
set(LLVM_SRC "${PROJECT_SOURCE_DIR}/compiler")
set(LLVM_ROOT "${PROJECT_BINARY_DIR}/compiler")

# obtain specific information about llvm setup
SET(LOCAL_LLVM_INCLUDE compiler/include)

# setup compilation environment
set(CMAKE_INSTALL_PREFIX "/opt/clamp")
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
option(CXXAMP_ENABLE_BOLT "Enable Bolt as external dependency" ON)

SET(FOO_BUILD_SHARED OFF CACHE BOOL "Build libfoo shared library")

#################
# Detect OpenCL
#################

find_path(OPENCL_HEADER cl.h PATHS ${OPENCL_HEADER_DIR} /usr/include/CL /opt/AMDAPP/include/CL /usr/local/cuda/include/CL )
if (NOT APPLE AND NOT OPENCL_HEADER)
  MESSAGE("OpenCL header not found. Use -DOPENCL_HEADER_DIR=<path_to_cl.h>.")
endif(NOT APPLE AND NOT OPENCL_HEADER)

find_library(OPENCL_LIBRARY OpenCL PATHS ${OPENCL_LIBRARY_DIR} /usr/lib /opt/AMDAPP/lib/x86_64 /usr/local/cuda/lib64 )
if (NOT APPLE AND NOT OPENCL_LIBRARY)
  MESSAGE("OpenCL library not found. Use -DOPENCL_LIBRARY_DIR=<path_to_libOpenCL.so>.")
endif(NOT APPLE AND NOT OPENCL_LIBRARY)

# display OpenCL information
MESSAGE("")
MESSAGE("OPENCL INFORMATION:")
MESSAGE("OPENCL_HEADER_DIR = ${OPENCL_HEADER_DIR}, actually found at: ${OPENCL_HEADER}")
MESSAGE("OPENCL_LIBRARY_DIR = ${OPENCL_LIBRARY_DIR}, actually found at: ${OPENCL_LIBRARY}")
MESSAGE("")

if ((NOT OPENCL_HEADER) OR (NOT OPENCL_LIBRARY))
  set(HAS_OPENCL 0)
else ((NOT OPENCL_HEADER) OR (NOT OPENCL_LIBRARY))
  set(HAS_OPENCL 1)

  # find detailed OpenCL version information
  file(STRINGS "${OPENCL_HEADER}/cl.h" lines11 REGEX "CL_VERSION_1_1[ \t]*1")
  if (lines11)
    set (HAS_OPENCL11 1)
  else (lines11)
    set (HAS_OPENCL11 0)
  endif (lines11)

  file(STRINGS "${OPENCL_HEADER}/cl.h" lines12 REGEX "CL_VERSION_1_2[ \t]*1")
  if (lines12)
    set (HAS_OPENCL12 1)
  else (lines12)
    set (HAS_OPENCL12 0)
  endif (lines12)

endif ((NOT OPENCL_HEADER) OR (NOT OPENCL_LIBRARY))

#################
# Detect HSA
#################

# HSA external dependencies
set (HSAIL_ROOT "/opt/amd"
        CACHE PATH "AMD HSAIL compiler path")
set (HSA_ROOT "/opt/hsa"
        CACHE PATH "HSA runtime path")

find_path(HSA_HEADER hsa.h PATHS ${HSA_HEADER_DIR} ${HSA_ROOT}/include)
if (NOT HSA_HEADER)
  MESSAGE("HSA header not found. Use -DHSA_HEADER_DIR=<path_to_hsa.h>.")
endif (NOT HSA_HEADER)

find_library(HSA_LIBRARY hsa-runtime64 PATHS ${HSA_LIBRARY_DIR} ${HSA_ROOT}/lib)
if (NOT HSA_LIBRARY)
  MESSAGE("HSA runtime library not found. Use -DHSA_LIBRARY_DIR=<path_to_libhsaruntime64.so>.")
endif (NOT HSA_LIBRARY)

find_library(HSA_KMT_LIBRARY libhsakmt.so.1 PATHS ${HSA_KMT_LIBRARY_DIR} ${HSA_ROOT}/lib)
if (NOT HSA_KMT_LIBRARY)
  MESSAGE("HSA Kernel library not found. Use -DHSA_KMT_LIBRARY_DIR=<path_to_libhsakmt.so.1>.")
endif (NOT HSA_KMT_LIBRARY)

find_path(LIBHSAIL_HEADER Brig_new.hpp PATHS ${LIBHSAIL_HEADER_DIR} ${HSA_ROOT}/include)
if (NOT LIBHSAIL_HEADER)
  MESSAGE("libHSAIL header not found. Use -DLIBHSAIL_HEADER_DIR=<path_to_Brig_new.hpp>.")
endif (NOT LIBHSAIL_HEADER)

find_library(LIBHSAIL_LIBRARY hsail PATHS ${LIBHSAIL_LIBRARY_DIR} ${HSA_ROOT}/lib)
if (NOT LIBHSAIL_LIBRARY)
  MESSAGE("libHSAIL not found. Use -DLIBHSAIL_LIBRARY_DIR=<path_to_libhsail.a>.")
endif (NOT LIBHSAIL_LIBRARY)

find_program(HSAIL_COMPILER hsailasm PATHS ${HSAIL_COMPILER_DIR} ${HSAIL_ROOT}/bin)
if (NOT HSAIL_COMPILER)
  MESSAGE("HSAIL compiler not found. Use -DHSAIL_COMPILER_DIR=<path_to_hsailasm>.")
endif (NOT HSAIL_COMPILER)


# display HSA information
MESSAGE("")
MESSAGE("HSA INFORMATION:")
MESSAGE("HSA_HEADER_DIR = ${HSA_HEADER_DIR}, actually found at: ${HSA_HEADER}")
MESSAGE("HSA_LIBRARY_DIR = ${HSA_LIBRARY_DIR}, actually found at: ${HSA_LIBRARY}")
MESSAGE("HSA_KMT_LIBRARY_DIR = ${HSA_KMT_LIBRARY_DIR}, actually found at: ${HSA_KMT_LIBRARY}")
MESSAGE("LIBHSAIL_HEADER_DIR = ${LIBHSAIL_HEADER_DIR}, actually found at: ${LIBHSAIL_HEADER}")
MESSAGE("LIBHSAIL_LIBRARY_DIR = ${LIBHSAIL_LIBRARY_DIR}, actually found at: ${LIBHSAIL_LIBRARY}")
MESSAGE("HSAIL_COMPILER_DIR = ${HSAIL_COMPILER_DIR}, actually found at: ${HSAIL_COMPILER}")
MESSAGE("")

if ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT HSA_KMT_LIBRARY) OR (NOT LIBHSAIL_HEADER) OR (NOT LIBHSAIL_LIBRARY) OR (NOT HSAIL_COMPILER))
  set(HAS_HSA 0)
else ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT HSA_KMT_LIBRARY) OR (NOT LIBHSAIL_HEADER) OR (NOT LIBHSAIL_LIBRARY) OR (NOT HSAIL_COMPILER))
  set(HAS_HSA 1)
  get_filename_component(HSAIL_BIN ${HSAIL_COMPILER} DIRECTORY)
endif ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT HSA_KMT_LIBRARY) OR (NOT LIBHSAIL_HEADER) OR (NOT LIBHSAIL_LIBRARY) OR (NOT HSAIL_COMPILER))

#################
# Detect if we have at least one of OpenCL or HSA available
#################
if ((HAS_OPENCL EQUAL 0) AND (HAS_HSA EQUAL 0))
  MESSAGE(FATAL_ERROR "Neither OpenCL nor HSA is available on the system!")
endif ((HAS_OPENCL EQUAL 0) AND (HAS_HSA EQUAL 0))


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

# Must build after 'world' target
if(CXXAMP_ENABLE_BOLT)
  # TODO: add an independent repo. for Bolt ported on clamp

  ##### Pass clamp paths to Bolt project #####
  # Note that with 'DIRS' suffix is for include/link directory and 'DIR' is for cxxflags/linkflags
  SET( CLAMP_C_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang"  )
  SET( CLAMP_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++"  )
  SET( CLAMP_LIBCXX_INC_DIR "${PROJECT_SOURCE_DIR}/libc++/libcxx/include" )
  SET( CLAMP_LIBCXX_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxx/lib" )
  SET( CLAMP_LIBCXXRT_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxxrt/lib" )

  SET( CLAMP_LIBRARIES mcwamp c++ cxxrt dl )
  SET( CLAMP_INC_DIRS ${PROJECT_SOURCE_DIR}/include 
     ${CLAMP_LIBCXX_INC_DIR} )

  SET( CLAMP_LIB_DIRS ${PROJECT_BINARY_DIR}/build/${CMAKE_BUILD_TYPE}/lib
     ${CLAMP_LIBCXX_LIB_DIR} 
     ${CLAMP_LIBCXXRT_LIB_DIR} )

  # This is the only source code folder we can build as a generator
  add_subdirectory( Bolt/superbuild )

  # Searched paths for BoltVersion.h and Boost headers
  # These are used in building clang-plugin and demo sample in lib/clang-plugin folder
  # Generally they shall be ouput from Bolt project. However we manually specify them in here
  
  SET( BOLT_INC_DIRS ${PROJECT_BINARY_DIR}/Bolt/superbuild/Bolt-build/include
    ${PROJECT_BINARY_DIR}/Bolt/superbuild/external/boost/src/Boost 
    ${PROJECT_SOURCE_DIR}/Bolt/include )
  
  SET( BOLT_LIB_DIRS ${PROJECT_BINARY_DIR}/Bolt/superbuild/Bolt-build/bolt/amp
    ${PROJECT_BINARY_DIR}/Bolt/superbuild/external/boost/src/Boost/stage/lib )
  
  ADD_DEPENDENCIES( Bolt world )
  
endif(CXXAMP_ENABLE_BOLT)

set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX "/opt/clamp")
set(CPACK_PACKAGE_NAME "clamp")
set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
set(CPACK_PACKAGE_VERSION "0.4.0-${GIT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "4")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "clamp: a C++AMP to OpenCL/HSA compiler")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jack Chung <jack@multicorewareinc.com>")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcxxamp (>= 0.4.0)")
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

