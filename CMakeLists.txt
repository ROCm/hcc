cmake_minimum_required( VERSION 2.8 )
project (Kalmar)

# set default installation path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
  set(CMAKE_INSTALL_PREFIX "/opt/kalmar")
else ()
  set(CMAKE_INSTALL_PREFIX "/opt/kalmar")
endif ()
MESSAGE("Package installation path: ${CMAKE_INSTALL_PREFIX}")

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")

# set as release build by default
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Release Debug" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# set default cppamp-ng URL
IF (NOT CLANG_URL)
  SET(CLANG_URL "https://bitbucket.org/multicoreware/cppamp-ng-35.git" CACHE STRING "CLANG URL" FORCE)
ENDIF (NOT CLANG_URL)
set(CMAKE_MACOSX_RPATH 1)

include (EnsureLLVMisPresent)
include (EnsureCLANGisPresent)
include (EnsureLIBCXXisPresent)
include (EnsureLIBCXXRTisPresent)
include (EnsureHLCisPresent)
include (EnsureHSAILASMisPresent)
include (SetupCBE)
include (SetupSPIRify)
include (SetupEraseNonkernel)
include (SetupCpuRename)
include (SetupTileUniform)
include (SetupUtils)
include (MCWAMP)
include (PatchAsmParser)
include (SetupGridLaunch)

ensure_llvm_is_present(${PROJECT_SOURCE_DIR} compiler)
ensure_clang_is_present(${PROJECT_SOURCE_DIR} compiler ${CLANG_URL})
ensure_libcxx_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxx)
if (NOT APPLE)
ensure_libcxxrt_is_present("${PROJECT_SOURCE_DIR}/libc++" libcxxrt)
endif (NOT APPLE)
setup_CBackend(${PROJECT_SOURCE_DIR}/compiler/lib/Target CBackend)
setup_SPIRify(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms SPIRify)
setup_EraseNonkernel(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms EraseNonkernel)
setup_CpuRename(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms CpuRename)
setup_TileUniform(${PROJECT_SOURCE_DIR}/compiler/lib/Analysis TileUniform)
setup_Utils(utils)
patch_AsmParser(SPIRify)
setup_GridLaunch(${PROJECT_SOURCE_DIR}/compiler GridLaunch)
set(LLVM_TARGETS_TO_BUILD X86 CACHE STRING "LLVM host target")
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD CBackend CACHE STRING "set this to 'CBackend' to enable generating OpenCL-C")
set(LLVM_INCLUDE_EXAMPLES off)

# Regression test
set(LLVM_SRC "${PROJECT_SOURCE_DIR}/compiler")
set(LLVM_ROOT "${PROJECT_BINARY_DIR}/compiler")

# obtain specific information about llvm setup
SET(LOCAL_LLVM_INCLUDE compiler/include)

# setup HLC in Kalmar
ensure_HLC_is_present(${PROJECT_SOURCE_DIR} hlc)

# setup HSAILasm in Kalmar
ensure_HSAILASM_is_present(${PROJECT_SOURCE_DIR} HSAILasm)

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


# Turn off Bolt by default for now.
option(CXXAMP_ENABLE_BOLT "Enable Bolt as external dependency" OFF)

SET(FOO_BUILD_SHARED OFF CACHE BOOL "Build libfoo shared library")

#################
# Detect OpenCL
#################

find_path(OPENCL_HEADER cl.h PATHS ${OPENCL_HEADER_DIR} /usr/include/CL /opt/AMDAPP/include/CL /usr/local/cuda/include/CL NO_DEFAULT_PATH)
find_path(OPENCL_HEADER cl.h)
if (NOT APPLE AND NOT OPENCL_HEADER)
  MESSAGE("OpenCL header not found. Use -DOPENCL_HEADER_DIR=<path_to_cl.h>.")
endif(NOT APPLE AND NOT OPENCL_HEADER)

find_library(OPENCL_LIBRARY OpenCL PATHS ${OPENCL_LIBRARY_DIR} /usr/lib /opt/AMDAPP/lib/x86_64 /usr/local/cuda/lib64 NO_DEFAULT_PATH)
find_library(OPENCL_LIBRARY OpenCL)
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
endif ((NOT OPENCL_HEADER) OR (NOT OPENCL_LIBRARY))

#################
# Detect HSA
#################

# HSA external dependencies
set (HSAIL_ROOT "/opt/amd"
        CACHE PATH "AMD HSAIL compiler path")
set (HSA_ROOT "/opt/hsa"
        CACHE PATH "HSA runtime path")


find_path(HSA_HEADER hsa.h PATHS ${HSA_HEADER_DIR} ${HSA_ROOT}/include NO_DEFAULT_PATH)
find_path(HSA_HEADER hsa.h)
if (NOT HSA_HEADER)
  MESSAGE("HSA header not found. Use -DHSA_HEADER_DIR=<path_to_hsa.h>.")
endif (NOT HSA_HEADER)


find_library(HSA_LIBRARY hsa-runtime64 PATHS ${HSA_LIBRARY_DIR} ${HSA_ROOT}/lib NO_DEFAULT_PATH)
find_library(HSA_LIBRARY hsa-runtime64)
if (NOT HSA_LIBRARY)
  MESSAGE("HSA runtime library not found. Use -DHSA_LIBRARY_DIR=<path_to_libhsa-runtime64.so>.")
endif (NOT HSA_LIBRARY)


find_library(HSA_KMT_LIBRARY libhsakmt.so.1 PATHS ${HSA_KMT_LIBRARY_DIR} ${HSA_ROOT}/lib NO_DEFAULT_PATH)
find_library(HSA_KMT_LIBRARY libhsakmt.so.1)
if (NOT HSA_KMT_LIBRARY)
  MESSAGE("HSA Kernel library not found. Use -DHSA_KMT_LIBRARY_DIR=<path_to_libhsakmt.so.1>.")
endif (NOT HSA_KMT_LIBRARY)

find_program(HSAIL_COMPILER llc PATHS ${HSAIL_COMPILER_DIR} ${HSAIL_ROOT}/bin NO_DEFAULT_PATH)
find_program(HSAIL_COMPILER llc)
if (NOT HSAIL_COMPILER)
  MESSAGE("HSAIL compiler not found. Use -DHSAIL_COMPILER_DIR=<path_to_llc>.")
endif (NOT HSAIL_COMPILER)

find_program(HSAIL_ASSEMBLER HSAILasm PATHS ${HSAIL_ASSEMBLER_DIR} ${HSAIL_ROOT}/bin NO_DEFAULT_PATH)
find_program(HSAIL_ASSEMBLER HSAILasm)
if (NOT HSAIL_ASSEMBLER)
  MESSAGE("HSAIL assembler not found. Use -DHSAIL_ASSEMBLER_DIR=<path_to_HSAILasm>.")
endif (NOT HSAIL_ASSEMBLER)

find_program(HSA_HOF hof PATHS ${HSA_HOF_DIR} ${HSAIL_ROOT}/bin NO_DEFAULT_PATH)
find_program(HSA_HOF hof)
if (NOT HSA_HOF)
  MESSAGE("HSA offline finalizer (HOF) not found.  Use -DHSA_HOF_DIR=<path_to_hof>.")
endif (NOT HSA_HOF)


# display HSA information
MESSAGE("")
MESSAGE("HSA INFORMATION:")
MESSAGE("HSA_HEADER_DIR = ${HSA_HEADER_DIR}, actually found at: ${HSA_HEADER}")
MESSAGE("HSA_LIBRARY_DIR = ${HSA_LIBRARY_DIR}, actually found at: ${HSA_LIBRARY}")
MESSAGE("HSA_KMT_LIBRARY_DIR = ${HSA_KMT_LIBRARY_DIR}, actually found at: ${HSA_KMT_LIBRARY}")
MESSAGE("HSAIL_COMPILER_DIR = ${HSAIL_COMPILER_DIR}, actually found at: ${HSAIL_COMPILER}")
MESSAGE("HSAIL_ASSEMBLER_DIR = ${HSAIL_ASSEMBLER_DIR}, actually found at: ${HSAIL_ASSEMBLER}")
MESSAGE("HSA_HOF_DIR = ${HSA_HOF_DIR}, actually found at: ${HSA_HOF}")
MESSAGE("")

# determine if we have HSA stack available
# at least we need the followings:
# - HSA headers (installed by HSA runtime)
# - HSA runtime libraries (installed by HSA runtime)
# - HSA ext64 library (installed by HSA runtime)
# - HSA KMT library (installed by HSA KFD driver)
# - HSAIL compiler (installed by HLC)
# - HSAIL assembler (installed by HSAIL-Tools)
if ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT HSA_KMT_LIBRARY) OR (NOT HSAIL_COMPILER) OR (NOT HSAIL_ASSEMBLER))
  set(HAS_HSA 0)
else ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT HSA_KMT_LIBRARY) OR (NOT HSAIL_COMPILER) OR (NOT HSAIL_ASSEMBLER))
  set(HAS_HSA 1)
  get_filename_component(HSAIL_BIN ${HSAIL_COMPILER} DIRECTORY)
endif ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT HSA_KMT_LIBRARY) OR (NOT HSAIL_COMPILER) OR (NOT HSAIL_ASSEMBLER))

# determine if HSA offline finalization is possible
# at least we need the followings:
# - a working HSA stack (HAS_HSA is 1)
# - HOF executable (install by HSA HOF)
if (HAS_HSA)
  if (NOT HSA_HOF)
    set (HAS_HSA_HOF 0)
  else (NOT HSA_HOF)
    set (HAS_HSA_HOF 1)
    get_filename_component(HOF_BIN ${HSA_HOF} DIRECTORY)
  endif (NOT HSA_HOF)
endif (HAS_HSA)

#################
# Detect if we have at least one of OpenCL or HSA available
#################
if ((HAS_OPENCL EQUAL 0) AND (HAS_HSA EQUAL 0))
  MESSAGE(FATAL_ERROR "Neither OpenCL nor HSA is available on the system!")
endif ((HAS_OPENCL EQUAL 0) AND (HAS_HSA EQUAL 0))

#################
# Set up version information
#################

# define Kalmar version information
SET(KALMAR_VERSION_MAJOR "0")
SET(KALMAR_VERSION_MINOR "7")

# get date information
# use the last two digits of year + week number as KALMAR_VERSION_PATCH
execute_process(COMMAND date +%y%W
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE KALMAR_VERSION_PATCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# get commit information
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE KALMAR_DRIVER_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/compiler/tools/clang
                OUTPUT_VARIABLE KALMAR_COMPILER_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# set Kalmar version string
set(KALMAR_VERSION_STRING "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}-${KALMAR_DRIVER_COMMIT}-${KALMAR_COMPILER_COMMIT}")

# show Kalmar version string
MESSAGE("========================================")
MESSAGE("Kalmar version: ${KALMAR_VERSION_STRING}")
MESSAGE("========================================")


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
  COMMAND ${MAKE}
  WORKING_DIRECTORY ${LIBCXX_BIN_DIR}
  DEPENDS clang
)
else (APPLE)
# use libcxxrt as ABI instead
add_custom_target(libc++
  COMMAND ${CMAKE_COMMAND} ${LIBCXX_SRC_DIR} -DCMAKE_CXX_COMPILER=${CLANG_CC1} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  COMMAND ${MAKE}
  WORKING_DIRECTORY ${LIBCXX_BIN_DIR}
  DEPENDS clang
)
add_custom_target(
    libcpprt
    DEPENDS ${CMAKE_BINARY_DIR}/libc++/libcxxrt/lib/libcxxrt.so
    )
endif (APPLE)

add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(include)
add_subdirectory(amp-conformance)
add_subdirectory(stl-test)

# build HLC
set(HLC_SRC_DIR "${PROJECT_SOURCE_DIR}/hlc")
set(HLC_BIN_DIR "${PROJECT_BINARY_DIR}/hlc")
file(MAKE_DIRECTORY ${HLC_BIN_DIR})
add_custom_target(hlc
  COMMAND ${CMAKE_COMMAND} ${HLC_SRC_DIR}
          -DLLVM_ENABLE_EH=ON
          -DLLVM_ENABLE_RTTI=ON
          -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=HSAIL
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  COMMAND ${MAKE}
  WORKING_DIRECTORY ${HLC_BIN_DIR}
)

# build HSAILasm
set(HSAILASM_SRC_DIR "${PROJECT_SOURCE_DIR}/HSAILasm")
set(HSAILASM_BIN_DIR "${PROJECT_BINARY_DIR}/HSAILasm")
file(MAKE_DIRECTORY ${HSAILASM_BIN_DIR})
add_custom_target(HSAILasm
  COMMAND ${CMAKE_COMMAND} ${HSAILASM_SRC_DIR}/libHSAIL
  COMMAND ${MAKE}
  WORKING_DIRECTORY ${HSAILASM_BIN_DIR}
)

add_custom_target(world
    DEPENDS clang libc++ hlc HSAILasm ${DEPS}
)

# create documentation
configure_file(
  ${PROJECT_SOURCE_DIR}/lib/doxygen_config.in
  ${PROJECT_BINARY_DIR}/lib/doxygen_config @ONLY)

add_custom_target(docs
    COMMAND doxygen "${PROJECT_BINARY_DIR}/lib/doxygen_config"
)

# Must build after 'world' target
if(CXXAMP_ENABLE_BOLT)
  # TODO: add an independent repo. for Bolt ported on clamp

  ##### Pass clamp paths to Bolt project #####
  # Note that with 'DIRS' suffix is for include/link directory and 'DIR' is for cxxflags/linkflags
  SET( CLAMP_C_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang"  )
  SET( CLAMP_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++"  )
  SET( CLAMP_LIBCXX_INC_DIR "${PROJECT_SOURCE_DIR}/libc++/libcxx/include" )
  SET( CLAMP_LIBCXX_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/lib" )
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
set(CPACK_INSTALL_PREFIX "/opt/kalmar")
set(CPACK_PACKAGE_NAME "kalmar")
set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
set(CPACK_PACKAGE_VERSION ${KALMAR_VERSION_STRING})
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Kalmar: a Heterogeneous C++ to OpenCL/HSA compiler")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jack Chung <jack@multicorewareinc.com>")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcxxamp (>= 0.6.0)")
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
MESSAGE("   'make world' to build clang, libc++, HLC, and library for testing.")
MESSAGE("   'make' to build the rest of LLVM tools")
MESSAGE("   'make docs' to build the HTML API reference")
MESSAGE("")

