cmake_minimum_required( VERSION 2.8 )
project (HCC)

# set default installation path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
  set(CMAKE_INSTALL_PREFIX "/opt/rocm/hcc-lc" CACHE PATH "Default installation path of hcc" FORCE)
endif ()
MESSAGE("Package installation path: ${CMAKE_INSTALL_PREFIX}")

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")

# set as release build by default
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Release Debug" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

IF (NOT NUM_BUILD_THREADS)
  include(ProcessorCount)
  ProcessorCount(NUM_BUILD_THREADS)
ENDIF(NOT NUM_BUILD_THREADS)

# Accepted values for DISTRO: ubuntu, fedora
IF (NOT DISTRO)
  SET(DISTRO "ubuntu")
ENDIF(NOT DISTRO)
MESSAGE("Distro: ${DISTRO}")

include (MCWAMP)

execute_process(COMMAND ln -sf ../../lld WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools)
execute_process(COMMAND ln -sf ../../clang WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools)

# Regression test
set(LLVM_SRC "${PROJECT_SOURCE_DIR}/compiler")
set(LLVM_ROOT "${PROJECT_BINARY_DIR}/compiler")

# obtain specific information about llvm setup
SET(LOCAL_LLVM_INCLUDE compiler/include)

# setup compilation environment
if (UNIX)
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )
else (UNIX)
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH})
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH})

SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH})
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH})

SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH})
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
MESSAGE("(DEBUG|RELEASE) output changed to path:" "${EXECUTABLE_OUTPUT_PATH}")

endif (UNIX)

SET(PROJ_SEARCH_PATH "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/${LOCAL_LLVM_INCLUDE}" "${PROJECT_BINARY_DIR}/${LOCAL_LLVM_INCLUDE}") #  "${PROJECT_SOURCE_DIR}/compiler/utils/unittest/googletest/include")
include_directories( ${PROJ_SEARCH_PATH} )

LINK_DIRECTORIES( ${LLVM_LIB_DIR} )


#################
# Detect ROCm
#################

# ROCm external dependencies
set (ROCM_ROOT "/opt/rocm" CACHE PATH "ROCM runtime path")

find_path(HSA_HEADER hsa/hsa.h PATHS ${HSA_HEADER_DIR} ${ROCM_ROOT}/include NO_DEFAULT_PATH)
find_path(HSA_HEADER hsa/hsa.h)

if (NOT HSA_HEADER)
  MESSAGE("HSA header not found. Use -DHSA_HEADER_DIR=<path_to_hsa.h>.")
endif (NOT HSA_HEADER)

find_library(HSA_LIBRARY hsa-runtime64 PATHS ${HSA_LIBRARY_DIR} ${ROCM_ROOT}/lib NO_DEFAULT_PATH)
find_library(HSA_LIBRARY hsa-runtime64)
if (NOT HSA_LIBRARY)
  MESSAGE("HSA runtime library not found. Use -DHSA_LIBRARY_DIR=<path_to_libhsa-runtime64.so>.")
endif (NOT HSA_LIBRARY)

find_path(ROCM_DEVICE_LIB ocml.amdgcn.bc PATHS ${ROCM_DEVICE_LIB_DIR} ${ROCM_ROOT}/lib NO_DEFAULT_PATH)
find_path(ROCM_DEVICE_LIB ocml.amdgcn.bc)

if (NOT ROCM_DEVICE_LIB)
  MESSAGE("ROCM Device Lib not found. Use -DROCM_DEVICE_LIB_DIR=<path_to_ocml.amdgcn.bc>.")
endif (NOT ROCM_DEVICE_LIB)

# display ROCm information
MESSAGE("")
MESSAGE("ROCm INFORMATION:")
MESSAGE("HSA_HEADER_DIR = ${HSA_HEADER_DIR}, actually found at: ${HSA_HEADER}")
MESSAGE("HSA_LIBRARY_DIR = ${HSA_LIBRARY_DIR}, actually found at: ${HSA_LIBRARY}")
MESSAGE("ROCM_DEVICE_LIB_DIR = ${ROCM_DEVICE_LIB_DIR}, actually found at: ${ROCM_DEVICE_LIB}")
MESSAGE("")

#################
# Detect libc++
#################
find_path(LIBCXX_HEADER random PATHS /usr/local/include/c++/v1 /usr/include/c++/v1 NO_DEFAULT_PATH)
MESSAGE("libc++ headers found at ${LIBCXX_HEADER}")

#################
# Set up version information
#################

# define HCC version information
SET(KALMAR_VERSION_MAJOR "1")
SET(KALMAR_VERSION_MINOR "0")

# get date information based on UTC
# use the last two digits of year + week number + day in the week as KALMAR_VERSION_PATCH
# use the commit date, instead of build date
# add xargs to remove strange trailing newline character
execute_process(COMMAND git show -s --format=@%ct
                COMMAND xargs
                COMMAND date -f - --utc +%y%U%w
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE KALMAR_VERSION_PATCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# get commit information
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE KALMAR_SDK_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/clang
                OUTPUT_VARIABLE KALMAR_FRONTEND_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/compiler
                OUTPUT_VARIABLE KALMAR_BACKEND_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set HCC version string. The rule for version string is:
# KALMAR_VERSION_MAJOR . KALMAR_VERSION_MINOR . KALMAR_VERSION_PATCH-KALMAR_SDK_COMIT-KALMAR_FRONTEND_COMMIT-KALMAR_BACKEND_COMMIT
set(KALMAR_VERSION_STRING "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}-${KALMAR_SDK_COMMIT}-${KALMAR_FRONTEND_COMMIT}-${KALMAR_BACKEND_COMMIT}")

# show HCC version string
MESSAGE("=============================================")
MESSAGE("HCC version: ${KALMAR_VERSION_STRING}")
MESSAGE("=============================================")

#################
# Detect if we have ROCm stack available
#################

# at least we need the followings:
# - HSA headers (installed by HSA runtime)
# - HSA runtime libraries (installed by HSA runtime)
# - ROCm Device Library
if ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT ROCM_DEVICE_LIB))
  set(HAS_ROCM 0)
else ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT ROCM_DEVICE_LIB))
  set(HAS_ROCM 1)
endif ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY) OR (NOT ROCM_DEVICE_LIB))

if (HAS_ROCM EQUAL 0)
  MESSAGE(FATAL_ERROR "ROCm is NOT available on the system!")
endif (HAS_ROCM EQUAL 0)

#################
# Detect AMDGPU backend for native codegen
#################

set(HSA_AMDGPU_GPU_TARGET "AMD:AMDGPU:8:0:3" CACHE STRING "Target AMDGPU ISA version")

# Tokenize HSA_AMDGPU_GPU_TARGET
foreach(AMDGPU_GPU_TARGET IN LISTS HSA_AMDGPU_GPU_TARGET)
  if (NOT (AMDGPU_GPU_TARGET STREQUAL "carrizo" OR
           AMDGPU_GPU_TARGET STREQUAL "fiji" OR
           AMDGPU_GPU_TARGET STREQUAL "hawaii" OR
           AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:7:0:1" OR
           AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:1" OR
           AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:3"))
    MESSAGE(FATAL_ERROR "${AMDGPU_GPU_TARGET} is not a valid AMDGPU target")
  endif()

  # AMDGPU_TARGET_LIST would hold values used by compiler scripts
  # Map ISA version to GPU family
  if (AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:7:0:1")
    list(APPEND AMDGPU_TARGET_LIST "hawaii")
  endif()

  if (AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:1")
    list(APPEND AMDGPU_TARGET_LIST "carrizo")
  endif()

  if (AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:3")
    list(APPEND AMDGPU_TARGET_LIST "fiji")
  endif()
endforeach(AMDGPU_GPU_TARGET)

# join list
string(REPLACE ";" " " AMDGPU_TARGET "${AMDGPU_TARGET_LIST}")
MESSAGE(STATUS "HCC configured with AMDGPU targets: ${AMDGPU_TARGET}")

#################
# Setup HCC backend macro string
# Possible values:
# - AMDGPU : for HSA systems configured with Lightning backend
#################

set(KALMAR_BACKEND "HCC_BACKEND_AMDGPU")

#########################
# build target: world
#########################

# build Clang frontend + AMDGPU backend
set(CLANG_SRC_DIR "${PROJECT_SOURCE_DIR}/compiler")
set(CLANG_BIN_DIR "${PROJECT_BINARY_DIR}/compiler")
file(MAKE_DIRECTORY ${CLANG_BIN_DIR})
add_custom_target(clang
  COMMAND ${CMAKE_COMMAND} ${CLANG_SRC_DIR}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DKALMAR_VERSION_STRING=${KALMAR_VERSION_STRING}
          -DKALMAR_VERSION_MAJOR=${KALMAR_VERSION_MAJOR}
          -DKALMAR_VERSION_MINOR=${KALMAR_VERSION_MINOR}
          -DKALMAR_VERSION_PATCH=${KALMAR_VERSION_PATCH}
          -DKALMAR_SDK_COMMIT=${KALMAR_SDK_COMMIT}
          -DKALMAR_FRONTEND_COMMIT=${KALMAR_FRONTEND_COMMIT}
          -DKALMAR_BACKEND_COMMIT=${KALMAR_BACKEND_COMMIT}
          -DKALMAR_BACKEND=${KALMAR_BACKEND}
          -DLLVM_TARGETS_TO_BUILD="AMDGPU\;X86"
          -DLLVM_INCLUDE_EXAMPLES=off
  COMMAND make -j ${NUM_BUILD_THREADS} # not portable, but we need it this way
  WORKING_DIRECTORY ${CLANG_BIN_DIR}
)

# create symlink of mostly used commands in clang to bin/
# - hcc
# - clang++
# - clang

add_custom_command(TARGET clang POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/hcc ${PROJECT_BINARY_DIR}/bin/hcc
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/clang++ ${PROJECT_BINARY_DIR}/bin/clang++
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/clang ${PROJECT_BINARY_DIR}/bin/clang
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../llvm/bin/llvm-mc ${PROJECT_BINARY_DIR}/bin/llvm-mc
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../llvm/bin/llvm-objdump ${PROJECT_BINARY_DIR}/bin/llvm-objdump
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../llvm/bin/lld ${PROJECT_BINARY_DIR}/bin/lld
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
install(FILES ${PROJECT_BINARY_DIR}/bin/hcc
              ${PROJECT_BINARY_DIR}/bin/clang++
              ${PROJECT_BINARY_DIR}/bin/clang
              ${PROJECT_BINARY_DIR}/bin/llvm-mc
              ${PROJECT_BINARY_DIR}/bin/llvm-objdump
              ${PROJECT_BINARY_DIR}/bin/lld
        DESTINATION bin)

# install clang along with HCC
# stored under compiler/
install(DIRECTORY ${CLANG_BIN_DIR}/bin
        DESTINATION compiler
        COMPONENT compiler
        PATTERN * PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(DIRECTORY ${CLANG_BIN_DIR}/lib
        DESTINATION compiler
        COMPONENT compiler
        PATTERN CMakeFiles EXCLUDE
        PATTERN lib*
        PATTERN LLVM*
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(DIRECTORY ${CLANG_BIN_DIR}/include
        DESTINATION compiler
        COMPONENT compiler
        PATTERN CMakeFiles EXCLUDE
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

install(DIRECTORY ${CLANG_SRC_DIR}/include
        DESTINATION compiler
        COMPONENT compiler
        PATTERN CMakeFiles EXCLUDE
        PATTERN CMakeLists* EXCLUDE
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

add_custom_target(world DEPENDS clang)

add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(include)
add_subdirectory(amp-conformance)
add_subdirectory(stl-test)

# create a "hcc" symlink in the include directory
# that points to itself so we could support
# include <hcc.hpp> and include<hcc/hcc.hpp>
execute_process(COMMAND ln -sf ../include hcc
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/include)

# create documentation
configure_file(
  ${PROJECT_SOURCE_DIR}/lib/doxygen_config.in
  ${PROJECT_BINARY_DIR}/lib/doxygen_config @ONLY)

add_custom_target(docs
    COMMAND doxygen "${PROJECT_BINARY_DIR}/lib/doxygen_config"
)

# install documentation
# optionally stored under doc/
install(DIRECTORY ${PROJECT_BINARY_DIR}/docs/html
        DESTINATION doc
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT doc
        OPTIONAL)

# generate Debian control scripts postinst and postrm
set(CPACK_PACKAGE_NAME "hcc_lc")

set(DEBIAN_EXTRA_SCRIPT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/packaging/debian")
set(DEBIAN_EXTRA_SCRIPT_DEST_DIR "${PROJECT_BINARY_DIR}/packaging/debian")
file(MAKE_DIRECTORY ${DEBIAN_EXTRA_SCRIPT_DEST_DIR})

execute_process(COMMAND ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/process_packaging_script.bsh ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/postinst.in  ${DEBIAN_EXTRA_SCRIPT_DEST_DIR}/postinst ${CPACK_PACKAGE_NAME} ${CMAKE_INSTALL_PREFIX}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

execute_process(COMMAND ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/process_packaging_script.bsh ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/prerm.in  ${DEBIAN_EXTRA_SCRIPT_DEST_DIR}/prerm ${CPACK_PACKAGE_NAME} ${CMAKE_INSTALL_PREFIX}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
set(CPACK_PACKAGE_VERSION "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})


set(PACKAGE_DESCRIPTION "HCC: An Open Source, Optimizing C++ Compiler for Heterogeneous Compute")

set(OFFICIAL_RELEASE_BUILD 0)

if ("${DISTRO}" STREQUAL "ubuntu")

  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION ${PACKAGE_DESCRIPTION})
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Siu Chi Chan <siuchi.chan@amd.com>")
  set(CPACK_DEBIAN_PACKAGE_SECTION "devel")

  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${PROJECT_BINARY_DIR}/packaging/debian/postinst;${PROJECT_BINARY_DIR}/packaging/debian/prerm")

  set(HCC_GENERAL_DEBIAN_DEP "libstdc++-4.8-dev, libc++1, libc++-dev, libc++abi1, libc++abi-dev, elfutils")

  # control the list of package dependency depending on whether this is an official release build.
  # for non-official release build, we want to relax the dependency on rocr runtime
  if (OFFICIAL_RELEASE_BUILD)
    set(HCC_ROCR_DEBIAN_DEP "hsa-rocr-dev, hsa-ext-rocr-dev")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${HCC_ROCR_DEBIAN_DEP}, ${HCC_GENERAL_DEBIAN_DEP}")
  else (OFFICIAL_RELEASE_BUILD)
    # dependencies for any local build
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "${HCC_GENERAL_DEBIAN_DEP}")
  endif (OFFICIAL_RELEASE_BUILD)

  set(CPACK_GENERATOR "DEB;TGZ")

elseif ("${DISTRO}" STREQUAL "fedora")

  set(CPACK_RPM_PACKAGE_DESCRIPTION ${PACKAGE_DESCRIPTION})

  set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/packaging/debian/postinst")
  set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/packaging/debian/prerm")

  # disable automatic shared libraries dependency detection
  set(CPACK_RPM_PACKAGE_AUTOREQ 0)
  set(CPACK_RPM_PACKAGE_REQUIRES "hsakmt-roct-dev, hsa-rocr-dev, libstdc++-devel, elfutils-libelf-devel, glibc-devel")
  set(CPACK_GENERATOR "RPM;TGZ")

endif ("${DISTRO}" STREQUAL "ubuntu")


set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")

set(CPACK_COMPONENTS_ALL compiler)

include (CPack)
MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make world' to build HCC Clang, and library for testing.")
MESSAGE("   'make' to build the rest of LLVM tools")
MESSAGE("   'make docs' to build the HTML API reference")
MESSAGE("")

