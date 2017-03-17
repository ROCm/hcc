cmake_minimum_required( VERSION 3.0 )
project (HCC)

include(GNUInstallDirs)

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")

# set as release build by default
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Release Debug" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Use two different methods to determine host distribution: lsb_release and if that fails grep /etc/os-release
find_program( lsb_executable lsb_release )

if( lsb_executable )
  execute_process( COMMAND ${lsb_executable} -is OUTPUT_VARIABLE DISTRO_ID OUTPUT_STRIP_TRAILING_WHITESPACE )
  execute_process( COMMAND ${lsb_executable} -rs OUTPUT_VARIABLE DISTRO_RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE )
else()
  if( EXISTS "/etc/os-release" )
    file( STRINGS "/etc/os-release" DISTRO_ID REGEX "^ID=" )
    file( STRINGS "/etc/os-release" DISTRO_RELEASE REGEX "^VERSION_ID=" )
    string( REPLACE "ID=" "" DISTRO_ID ${DISTRO_ID} )
    string( REPLACE "VERSION_ID=" "" DISTRO_RELEASE ${DISTRO_RELEASE} )
  endif( )
endif( )

# Accepted values for DISTRO_ID: trusty (Ubuntu 14.04), xenial (Ubuntu 16.06), fd23 (Fedora 23)
string(TOLOWER "${DISTRO_ID}" DISTRO_ID )
if( DISTRO_ID MATCHES "ubuntu" OR DISTRO_ID MATCHES "fedora" )
  message( STATUS "Detected distribution: ${DISTRO_ID}:${DISTRO_RELEASE}" )
else()
  message( "This cmakefile does not natively support ${DISTRO_ID}:${DISTRO_RELEASE}.  Continuing with Ubuntu logic" )
  set( DISTRO_ID "ubuntu" )
  set( DISTRO_RELEASE "16.04" )
endif()

include (MCWAMP)

execute_process(COMMAND ln -sf ../../lld WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools)
execute_process(COMMAND ln -sf ../../clang WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools)
execute_process(COMMAND ln -sf ../../compiler-rt WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/projects)

# Regression test
set(LLVM_SRC "${PROJECT_SOURCE_DIR}/compiler")
set(LLVM_ROOT "${PROJECT_BINARY_DIR}/compiler")

# obtain specific information about llvm setup
SET(LOCAL_LLVM_INCLUDE compiler/include)

# setup compilation environment
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )

SET(PROJ_SEARCH_PATH "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/${LOCAL_LLVM_INCLUDE}" "${PROJECT_BINARY_DIR}/${LOCAL_LLVM_INCLUDE}") #  "${PROJECT_SOURCE_DIR}/compiler/utils/unittest/googletest/include")
include_directories( ${PROJ_SEARCH_PATH} )

LINK_DIRECTORIES( ${LLVM_LIB_DIR} )


#################
# Detect ROCm
#################

# ROCm external dependencies
set (ROCM_ROOT "/opt/rocm" CACHE PATH "ROCM runtime path")

find_path(HSA_HEADER hsa/hsa.h
  HINTS
    ${HSA_HEADER_DIR}
  PATHS
    ${ROCM_ROOT}/include
)

if (NOT HSA_HEADER)
  MESSAGE("HSA header not found. Use -DHSA_HEADER=<path_to_hsa.h>.")
endif (NOT HSA_HEADER)

find_library(HSA_LIBRARY hsa-runtime64
  HINTS
    ${HSA_LIBRARY_DIR}
  PATHS
    ${ROCM_ROOT}/lib
)

if (NOT HSA_LIBRARY)
  MESSAGE("HSA runtime library not found. Use -DHSA_LIBRARY=<path_to_libhsa-runtime64.so>.")
endif (NOT HSA_LIBRARY)


################
# Detect ROCm Device Libs
################

find_path(ROCM_DEVICE_LIB ocml.amdgcn.bc
  PATH_SUFFIXES
    lib
  HINTS
    ${ROCM_DEVICE_LIB_DIR}
  PATHS
    ${ROCM_ROOT}/lib
)

if (NOT ROCM_DEVICE_LIB)
  MESSAGE(FATAL_ERROR "ROCm Device Libs not found. Use -DROCM_DEVICE_LIB=<path_to_ocml.amdgcn.bc>.")
endif (NOT ROCM_DEVICE_LIB)

set(ROCM_DEVICE_LIB_PATHS ${ROCM_DEVICE_LIB})
list(APPEND ROCM_DEVICE_LIB_PATHS ${ROCM_DEVICE_LIB_DIR} ${ROCM_ROOT}/lib)

# display ROCm information
MESSAGE("")
MESSAGE("ROCm INFORMATION:")
MESSAGE("HSA_HEADER_DIR = ${HSA_HEADER_DIR}, actually found at: ${HSA_HEADER}")
MESSAGE("HSA_LIBRARY_DIR = ${HSA_LIBRARY_DIR}, actually found at: ${HSA_LIBRARY}")
MESSAGE("ROCM_DEVICE_LIB_DIR = ${ROCM_DEVICE_LIB_DIR}, actually found at: ${ROCM_DEVICE_LIB}")

################
# Detect CodeXL Activity Logger
################
if (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

find_path(CODEXL_ACTIVITY_LOGGER_HEADER NAMES CXLActivityLogger.h PATHS ${CODEXL_ACTIVITY_LOGGER_HEADER_DIR} ${ROCM_ROOT}/profiler/CXLActivityLogger/include NO_DEFAULT_PATH)
find_path(CODEXL_ACTIVITY_LOGGER_HEADER NAMES CXLActivityLogger.h)

if (NOT CODEXL_ACTIVITY_LOGGER_HEADER)
  MESSAGE("CodeXL header not found. use -DCODEXL_ACTIVITY_LOGGER_HEADER_DIR=<path_to_CXLActivityLogger.h>.")
endif (NOT CODEXL_ACTIVITY_LOGGER_HEADER)

find_path(CODEXL_ACTIVITY_LOGGER_LIBRARY NAMES libCXLActivityLogger.so PATHS ${CODEXL_ACTIVITY_LOGGER_LIBRARY_DIR} ${ROCM_ROOT}/profiler/CXLActivityLogger/bin/x86_64 NO_DEFAULT_PATH)
find_path(CODEXL_ACTIVITY_LOGGER_LIBRARY NAMES libCXLActivityLogger.so)

if (NOT CODEXL_ACTIVITY_LOGGER_LIBRARY)
  MESSAGE("CodeXL Activity Logger library not found. Use -DCODEXL_ACTIVITY_LOGGER_LIBRARY_DIR=<path_to_libCXLActivityLogger.so>.")
endif (NOT CODEXL_ACTIVITY_LOGGER_LIBRARY)

# display CodeXL information
MESSAGE("")
MESSAGE("CodeXL INFORMATION:")
MESSAGE("CODEXL_ACTIVITY_LOGGER_HEADER_DIR = ${CODEXL_ACTIVITY_LOGGER_HEADER_DIR}, actually found at: ${CODEXL_ACTIVITY_LOGGER_HEADER}")
MESSAGE("CODEXL_ACTIVITY_LOGGER_LIBRARY_DIR = ${CODEXL_ACTIVITY_LOGGER_LIBRARY_DIR}, actually found at: ${CODEXL_ACTIVITY_LOGGER_LIBRARY}")
MESSAGE("")

endif (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

#################
# Configure which C++ runtime to use
# hcc will use libc++ if USE_LIBCXX is set to ON; otherwise, it will use libstdc++
#################

# if USE_LIBCXX is not explicitly set
if( NOT DEFINED USE_LIBCXX )
  # default to libstdc++
  set( USE_LIBCXX "OFF" )

  if( CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0) )
    # If host compiler is g++, use libc++ if version is less than 5.0
    set( USE_LIBCXX "ON" )
  elseif( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    # If host compiler is clang, detect version of system g++ and use libc++ if less than 5.0
    execute_process( COMMAND g++ -dumpversion OUTPUT_VARIABLE gplusplus_version )
    if( gplusplus_version VERSION_LESS 5.0 )
      set( USE_LIBCXX "ON" )
    endif( )
  endif( )
endif( )

# Detect libc++
if (USE_LIBCXX)
  find_path(LIBCXX_HEADER random PATHS /usr/local/include/c++/v1 /usr/include/c++/v1 NO_DEFAULT_PATH)
  message( "libc++ headers found at ${LIBCXX_HEADER}" )
endif( )

#################
# Set up version information
#################

# define HCC version information
SET(KALMAR_VERSION_MAJOR "1")
SET(KALMAR_VERSION_MINOR "0")

include(GenerateVersionFromGit)

if (NOT KALMAR_VERSION_STRING)
  set(KALMAR_VERSION_STRING
    "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}")
endif()
# Set HCC version string. The rule for version string is:
# KALMAR_VERSION_MAJOR . KALMAR_VERSION_MINOR . KALMAR_VERSION_PATCH-KALMAR_SDK_COMIT-KALMAR_FRONTEND_COMMIT-KALMAR_BACKEND_COMMIT
add_version_info_from_git(KALMAR_VERSION_STRING
  KALMAR_VERSION_PATCH KALMAR_SDK_COMMIT KALMAR_FRONTEND_COMMIT KALMAR_BACKEND_COMMIT)

# set default installation path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
  set(INSTALL_DIR_NAME "hcc-${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}")
  set(CMAKE_INSTALL_PREFIX "/opt/rocm/${INSTALL_DIR_NAME}" CACHE PATH "Default installation path of hcc" FORCE)
endif ()
MESSAGE("Package installation path: ${CMAKE_INSTALL_PREFIX}")


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

set(HSA_AMDGPU_GPU_TARGET "gfx803" CACHE STRING "Target AMDGPU ISA version")

# Tokenize HSA_AMDGPU_GPU_TARGET

# Temporary workaround for use of old long-form GFX_ip names, to be removed once
# we fully deprecate them.
string(REGEX MATCHALL "AMD:AMDGPU:(7:0:0|7:0:1|8:0:1|8:0:2|8:0:3|9:0:0|9:0:1)"
                      tmp
                      ${HSA_AMDGPU_GPU_TARGET})
foreach(long_form IN LISTS tmp)
  string(REPLACE "AMD:AMDGPU:" "gfx" short_form ${long_form})
  string(REPLACE ":" "" short_form ${short_form})
  message(DEPRECATION "Be advised that use of long-form GFX_ip names, such as \
                       ${long_form}, is deprecated. Please use the short-form: \
                       ${short_form}.")
  list(REMOVE_ITEM HSA_AMDGPU_GPU_TARGET ${long_form})
  list(APPEND HSA_AMDGPU_GPU_TARGET ${short_form})
endforeach()

# Map to LC-consumable name.
foreach(AMDGPU_GPU_TARGET IN LISTS HSA_AMDGPU_GPU_TARGET)
  # AMDGPU_TARGET_LIST holds the values which are used by the compiler scripts.
  if (AMDGPU_GPU_TARGET STREQUAL "gfx700")
    list(APPEND AMDGPU_TARGET_LIST "kaveri")
  elseif (AMDGPU_GPU_TARGET STREQUAL "gfx701")
    list(APPEND AMDGPU_TARGET_LIST "hawaii")
  elseif (AMDGPU_GPU_TARGET STREQUAL "gfx801")
    list(APPEND AMDGPU_TARGET_LIST "carrizo")
  elseif (AMDGPU_GPU_TARGET STREQUAL "gfx802")
    list(APPEND AMDGPU_TARGET_LIST "tonga")
  elseif (AMDGPU_GPU_TARGET STREQUAL "gfx803")
    list(APPEND AMDGPU_TARGET_LIST "fiji")
  elseif (AMDGPU_GPU_TARGET STREQUAL "gfx900")
    list(APPEND AMDGPU_TARGET_LIST "gfx900")
  elseif (AMDGPU_GPU_TARGET STREQUAL "gfx901")
    list(APPEND AMDGPU_TARGET_LIST "gfx901")
  else ()
    message(FATAL_ERROR "${AMDGPU_GPU_TARGET} is not a valid AMDPU target!")
  endif ()
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

set(LLVM_TARGETS_TO_BUILD "AMDGPU" "X86"
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\"." FORCE)
option(LLVM_INCLUDE_EXAMPLES "Generate build targets for the LLVM examples" OFF)
option(LLVM_ENABLE_OCAMLDOC "Build OCaml bindings documentation." OFF)
add_subdirectory(${CLANG_SRC_DIR} ${CLANG_BIN_DIR})

# create symlink of mostly used commands in clang to bin/
# - hcc
# - clang++
# - clang
add_custom_target(clang_links DEPENDS clang)
add_custom_command(TARGET clang_links POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/hcc ${PROJECT_BINARY_DIR}/bin/hcc
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/clang++ ${PROJECT_BINARY_DIR}/bin/clang++
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/clang ${PROJECT_BINARY_DIR}/bin/clang
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/llvm-mc ${PROJECT_BINARY_DIR}/bin/llvm-mc
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/llvm-objdump ${PROJECT_BINARY_DIR}/bin/llvm-objdump
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/lld ${PROJECT_BINARY_DIR}/bin/lld
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

add_custom_command(TARGET clang_links POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CLANG_BIN_DIR}/lib/clang/5.0.0/lib/linux/libclang_rt.builtins-x86_64.a ${PROJECT_BINARY_DIR}/lib/libclang_rt.builtins-x86_64.a
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
        PATTERN *tmp EXCLUDE
        PATTERN *inc EXCLUDE
        PATTERN Makefile EXCLUDE
        PATTERN *.cmake EXCLUDE
        PATTERN *.a EXCLUDE
        PATTERN lib*
        PATTERN LLVM*
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

install(FILES ${CLANG_BIN_DIR}/lib/clang/5.0.0/lib/linux/libclang_rt.builtins-x86_64.a
        DESTINATION  ${CMAKE_INSTALL_LIBDIR}
        COMPONENT compiler)

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

add_custom_target(world DEPENDS clang_links)

add_subdirectory(hcc_config)
add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(include)
add_subdirectory(amp-conformance)
add_subdirectory(stl-test)
add_subdirectory(cmake-tests)

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
set(CPACK_PACKAGE_NAME "hcc")

set(DEBIAN_EXTRA_SCRIPT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/packaging/debian")
set(DEBIAN_EXTRA_SCRIPT_DEST_DIR "${PROJECT_BINARY_DIR}/packaging/debian")
file(MAKE_DIRECTORY ${DEBIAN_EXTRA_SCRIPT_DEST_DIR})

execute_process(COMMAND ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/process_packaging_script.bsh ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/postinst.in  ${DEBIAN_EXTRA_SCRIPT_DEST_DIR}/postinst ${CPACK_PACKAGE_NAME} ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_LIBDIR}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

execute_process(COMMAND ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/process_packaging_script.bsh ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/prerm.in  ${DEBIAN_EXTRA_SCRIPT_DEST_DIR}/prerm ${CPACK_PACKAGE_NAME} ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_LIBDIR}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
set(CPACK_PACKAGE_VERSION "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${DISTRO_ID})

if( USE_LIBCXX )
  set( CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-libc++" )
else( )
  set( CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-libstdc++" )
endif( )

set(PACKAGE_DESCRIPTION "HCC: An Open Source, Optimizing C++ Compiler for Heterogeneous Compute")

set(OFFICIAL_RELEASE_BUILD 0)

if( "${DISTRO_ID}" MATCHES "ubuntu" )

  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION ${PACKAGE_DESCRIPTION})
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Siu Chi Chan <siuchi.chan@amd.com>")
  set(CPACK_DEBIAN_PACKAGE_SECTION "devel")

  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${PROJECT_BINARY_DIR}/packaging/debian/postinst;${PROJECT_BINARY_DIR}/packaging/debian/prerm")

  set(HCC_GENERAL_DEBIAN_DEP "g++-multilib, gcc-multilib, findutils, libelf1, libpci3, file")

  if ( USE_LIBCXX )
    set(HCC_LIBCXX_DEBIAN_DEP ", libc++1, libc++-dev, libc++abi1, libc++abi-dev")
  else ( USE_LIBCXX )
    set(HCC_LIBCXX_DEBIAN_DEP "")
  endif( USE_LIBCXX )

  # control the list of package dependency depending on whether this is an official release build.
  # for non-official release build, we want to relax the dependency on rocr runtime
  if (OFFICIAL_RELEASE_BUILD)
    set(HCC_ROCR_DEBIAN_DEP ", hsa-rocr-dev, hsa-ext-rocr-dev, rocm-device-libs")
  else (OFFICIAL_RELEASE_BUILD)
    set(HCC_ROCR_DEBIAN_DEP "")
  endif (OFFICIAL_RELEASE_BUILD)

  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${HCC_GENERAL_DEBIAN_DEP} ${HCC_ROCR_DEBIAN_DEP} ${HCC_LIBCXX_DEBIAN_DEP}")

  set(CPACK_GENERATOR "DEB")
  set(CPACK_BINARY_DEB "ON")
elseif ("${DISTRO_ID}" MATCHES "fedora")

  set(CPACK_RPM_PACKAGE_DESCRIPTION ${PACKAGE_DESCRIPTION})
  set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/packaging/debian/postinst")
  set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/packaging/debian/prerm")

  # disable automatic shared libraries dependency detection
  set(CPACK_RPM_PACKAGE_AUTOREQ 0)

  set(HCC_GENERAL_RPM_DEP "findutils, elfutils-libelf, pciutils-libs, file, pth")

  if (OFFICIAL_RELEASE_BUILD)
    set(HCC_ROCR_RPM_DEP ", hsa-rocr-dev, hsa-ext-rocr-dev, rocm-device-libs")
  else (OFFICIAL_RELEASE_BUILD)
    set(HCC_ROCR_RPM_DEP "")
  endif (OFFICIAL_RELEASE_BUILD)

    set(CPACK_RPM_PACKAGE_REQUIRES "${HCC_GENERAL_RPM_DEP} ${HCC_ROCR_RPM_DEP}" )

  set(CPACK_GENERATOR "RPM")
else()
  # generate a tarball for unknown DISTRO_ID
  set(CPACK_GENERATOR "TGZ")
endif ()

set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")

set(CPACK_COMPONENTS_ALL compiler)

include (CPack)
MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make' to build the rest of LLVM tools")
MESSAGE("   'make docs' to build the HTML API reference")
MESSAGE("")

