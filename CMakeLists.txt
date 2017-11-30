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
    string( REPLACE "ID=" "" DISTRO_ID "${DISTRO_ID}" )
    string( REPLACE "VERSION_ID=" "" DISTRO_RELEASE "${DISTRO_RELEASE}" )
  endif( )
endif( )

# Accepted values for DISTRO_ID: trusty (Ubuntu 14.04), xenial (Ubuntu 16.06), fd23 (Fedora 23)
string(TOLOWER "${DISTRO_ID}" DISTRO_ID )
if( DISTRO_ID MATCHES "ubuntu" OR DISTRO_ID MATCHES "fedora"
   OR DISTRO_ID MATCHES "centos" OR DISTRO_ID MATCHES "redhatenterpriseserver"
   OR DISTRO_ID MATCHES "opensuse")
  message( STATUS "Detected distribution: ${DISTRO_ID}:${DISTRO_RELEASE}" )
else()
  message( "This cmakefile does not natively support ${DISTRO_ID}:${DISTRO_RELEASE}.  Continuing with Ubuntu logic" )
  set( DISTRO_ID "ubuntu" )
  set( DISTRO_RELEASE "16.04" )
endif()

# Need special handling for RHEL 7 or CentOS 7
if (DISTRO_ID MATCHES "centos" OR DISTRO_ID MATCHES "redhatenterpriseserver")
  set(HCC_TOOLCHAIN_RHEL ON)
else()
  set(HCC_TOOLCHAIN_RHEL OFF)
endif()

include (MCWAMP)

# Regression test
set(LLVM_SRC "${PROJECT_SOURCE_DIR}/compiler")
set(LLVM_ROOT "${PROJECT_BINARY_DIR}/compiler")

# obtain specific information about llvm setup
SET(LOCAL_LLVM_INCLUDE compiler/include)

# setup compilation environment
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )

SET(PROJ_SEARCH_PATH "${PROJECT_BINARY_DIR}/include" "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/${LOCAL_LLVM_INCLUDE}") #  "${PROJECT_SOURCE_DIR}/compiler/utils/unittest/googletest/include" "${PROJECT_BINARY_DIR}/${LOCAL_LLVM_INCLUDE}"
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

add_library(hsa-runtime64 SHARED IMPORTED GLOBAL)

set_target_properties(hsa-runtime64 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${HSA_HEADER}"
  IMPORTED_LOCATION "${HSA_LIBRARY}"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HSA_HEADER}"
)

################
# Detect ROCm Device Libs
################

option(HCC_INTEGRATE_ROCDL "Build and use the ROCm Device Library integrated into HCC (Recommended)" ON)

if (NOT HCC_INTEGRATE_ROCDL)

  # If not using the integrated ROCm Device Library,
  # then the one installed on the system
  # or provided by a user
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
endif(NOT HCC_INTEGRATE_ROCDL)

# display ROCm information
MESSAGE("")
MESSAGE("ROCm INFORMATION:")
MESSAGE("HSA_HEADER_DIR = ${HSA_HEADER_DIR}, actually found at: ${HSA_HEADER}")
MESSAGE("HSA_LIBRARY_DIR = ${HSA_LIBRARY_DIR}, actually found at: ${HSA_LIBRARY}")

if (HCC_INTEGRATE_ROCDL)
  MESSAGE("ROCM_DEVICE_LIB_DIR = ${PROJECT_SOURCE_DIR}/rocdl")
else (HCC_INTEGRATE_ROCDL)
  MESSAGE("ROCM_DEVICE_LIB_DIR = ${ROCM_DEVICE_LIB_DIR}, actually found at: ${ROCM_DEVICE_LIB}")
endif (HCC_INTEGRATE_ROCDL)

################
# Detect CodeXL Activity Logger
################
if (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

find_path(CODEXL_ACTIVITY_LOGGER_HEADER NAMES CXLActivityLogger.h PATHS ${CODEXL_ACTIVITY_LOGGER_HEADER_DIR} ${ROCM_ROOT}/profiler/CXLActivityLogger/include NO_DEFAULT_PATH)
find_path(CODEXL_ACTIVITY_LOGGER_HEADER NAMES CXLActivityLogger.h)

if (NOT CODEXL_ACTIVITY_LOGGER_HEADER)
  MESSAGE("CodeXL header not found. use -DCODEXL_ACTIVITY_LOGGER_HEADER_DIR=<path_to_CXLActivityLogger.h>.")
endif (NOT CODEXL_ACTIVITY_LOGGER_HEADER)

find_path(CODEXL_ACTIVITY_LOGGER_LIBRARY NAMES libCXLActivityLogger.so PATHS ${CODEXL_ACTIVITY_LOGGER_LIBRARY_DIR} ${ROCM_ROOT}/profiler/CXLActivityLogger/bin/${CMAKE_SYSTEM_PROCESSOR} NO_DEFAULT_PATH)
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
SET(HCC_VERSION_MAJOR "1")
SET(HCC_VERSION_MINOR "2")

include(GenerateVersionFromGit)

if (NOT HCC_VERSION_STRING)
  set(HCC_VERSION_STRING
    "${HCC_VERSION_MAJOR}.${HCC_VERSION_MINOR}")
endif()

# Set HCC version string. The rule for version string is:
# HCC_VERSION_MAJOR . HCC_VERSION_MINOR . HCC_VERSION_PATCH-KALMAR_SDK_COMIT-KALMAR_FRONTEND_COMMIT-KALMAR_BACKEND_COMMIT
add_version_info_from_git(HCC_VERSION_STRING
  HCC_VERSION_PATCH KALMAR_SDK_COMMIT KALMAR_FRONTEND_COMMIT KALMAR_BACKEND_COMMIT)

# set default installation path
set(INSTALL_DIR_NAME "hcc-${HCC_VERSION_MAJOR}.${HCC_VERSION_MINOR}")
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
  set(CMAKE_INSTALL_PREFIX "/opt/rocm/${INSTALL_DIR_NAME}" CACHE PATH "Default installation path of hcc" FORCE)
endif ()
MESSAGE("Installation path: ${CMAKE_INSTALL_PREFIX}")

# set the installation path for the installer package
set(CPACK_SET_DESTDIR ON CACHE BOOL "Installer package will install hcc to CMAKE_INSTALL_PREFIX instead of CPACK_PACKAGING_INSTALL_PREFIX")
if (CPACK_SET_DESTDIR)
  set(HCC_PACKAGE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
else (CPACK_SET_DESTDIR)
  set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm/${INSTALL_DIR_NAME}" CACHE PATH "Default installation path of hcc installer package")
  set(HCC_PACKAGE_INSTALL_PREFIX ${CPACK_PACKAGING_INSTALL_PREFIX})
endif (CPACK_SET_DESTDIR)
MESSAGE("Installer package installation path: ${HCC_PACKAGE_INSTALL_PREFIX}")

# show HCC version string
MESSAGE("=============================================")
MESSAGE("HCC version: ${HCC_VERSION_STRING}")
MESSAGE("=============================================")

#################
# Detect if we have ROCm stack available
#################

# at least we need the followings:
# - HSA headers (installed by HSA runtime)
# - HSA runtime libraries (installed by HSA runtime)
# - ROCm Device Library
if ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY))
  set(HAS_ROCM 0)
else ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY))
  set(HAS_ROCM 1)
endif ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY))

if (HAS_ROCM EQUAL 0)
  MESSAGE(FATAL_ERROR "ROCm is NOT available on the system!")
endif (HAS_ROCM EQUAL 0)

#################
# Detect AMDGPU backend for native codegen
#################

set(HSA_AMDGPU_GPU_TARGET "auto" CACHE STRING "Target AMDGPU ISA version")

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

# join list
string(REPLACE ";" " " AMDGPU_TARGET "${HSA_AMDGPU_GPU_TARGET}")
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

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(LLVM_ENABLE_ASSERTIONS "On" CACHE BOOL "Compile with assertion checks enabled")
else ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(LLVM_ENABLE_ASSERTIONS "Off" CACHE BOOL "Compile with assertion checks enabled")
endif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

set(LLVM_EXTERNAL_CLANG_SOURCE_DIR "${PROJECT_SOURCE_DIR}/clang")
set(LLVM_EXTERNAL_COMPILER_RT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/compiler-rt")
set(LLVM_EXTERNAL_LLD_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lld")

# determine LLVM targets to be built
# AMDGPU is always on
# For host target, x86_64, aarch64, ppc64, ppc64le are currently supported
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(LLVM_HOST_TARGET "X86")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(LLVM_HOST_TARGET "AArch64")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64")
  set(LLVM_HOST_TARGET "PowerPC")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
  set(LLVM_HOST_TARGET "PowerPC")
endif()
set(LLVM_TARGETS_TO_BUILD "AMDGPU" ${LLVM_HOST_TARGET} CACHE STRING "" FORCE)
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "" FORCE)

if(EXISTS ${PROJECT_SOURCE_DIR}/clang-tools-extra/CMakeLists.txt)
  set(LLVM_EXTERNAL_CLANG_TOOLS_EXTRA_SOURCE_DIR "${PROJECT_SOURCE_DIR}/clang-tools-extra")
endif()

add_subdirectory(${CLANG_SRC_DIR})

install(PROGRAMS $<TARGET_FILE:llvm-as>
                 $<TARGET_FILE:llvm-dis>
                 $<TARGET_FILE:llvm-link>
                 $<TARGET_FILE:llvm-lto>
                 $<TARGET_FILE:llvm-objdump>
                 $<TARGET_FILE:llvm-mc>
                 $<TARGET_FILE:opt>
                 $<TARGET_FILE:llc>
        DESTINATION bin
        COMPONENT compiler)

add_custom_target(clang_links DEPENDS clang)
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
  set(RT_BUILTIN_SUFFIX "powerpc64le")
else()
  set(RT_BUILTIN_SUFFIX ${CMAKE_SYSTEM_PROCESSOR})
endif()
add_custom_command(TARGET clang_links POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CLANG_BIN_DIR}/lib/clang/6.0.0/lib/linux/libclang_rt.builtins-${RT_BUILTIN_SUFFIX}.a ${PROJECT_BINARY_DIR}/lib/libclang_rt.builtins-${RT_BUILTIN_SUFFIX}.a
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CLANG_BIN_DIR}/bin/clang ${PROJECT_BINARY_DIR}/bin/clang
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CLANG_BIN_DIR}/bin/clang++ ${PROJECT_BINARY_DIR}/bin/clang++
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CLANG_BIN_DIR}/bin/hcc ${PROJECT_BINARY_DIR}/bin/hcc
)

# install certain LLVM libraries needed by HIP
install(PROGRAMS $<TARGET_FILE:LLVMAMDGPUDesc>
                 $<TARGET_FILE:LLVMAMDGPUUtils>
                 $<TARGET_FILE:LLVMMC>
                 $<TARGET_FILE:LLVMCore>
                 $<TARGET_FILE:LLVMSupport>
                 $<TARGET_FILE:LLVMCpuRename>
                 $<TARGET_FILE:LLVMDirectFuncCall>
                 $<TARGET_FILE:LLVMSelectAcceleratorCode>
                 $<TARGET_FILE:LLVMHello>
                 $<TARGET_FILE:LLVMTileUniform>
                 $<TARGET_FILE:LLVMWrapperGen>
        DESTINATION lib
        COMPONENT compiler
)

# force library install path to lib
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

install(FILES ${CLANG_BIN_DIR}/lib/clang/6.0.0/lib/linux/libclang_rt.builtins-${RT_BUILTIN_SUFFIX}.a
        DESTINATION  ${CMAKE_INSTALL_LIBDIR}
        COMPONENT compiler)

install( FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake/ImportedTargets.cmake
         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hcc )

add_custom_target(world DEPENDS clang_links)

# move headers to build dir before building rocdl and hcc lib
add_subdirectory(include)

# build the integrated ROCm Device Library
set(AMDHSACOD ${ROCM_ROOT}/bin/amdhsacod CACHE FILEPATH "Specify the amdhsacod tool")
if (HCC_INTEGRATE_ROCDL)

  set(ROCDL_SRC_DIR "${PROJECT_SOURCE_DIR}/rocdl")
  set(ROCDL_BUILD_DIR "${PROJECT_BINARY_DIR}/rocdl")
  file(MAKE_DIRECTORY ${ROCDL_BUILD_DIR})

  set(LLVM_DIR ${CLANG_BIN_DIR})
  set(AMDHSACOD ${AMDHSACOD})
  # override default option value in ROCDL
  set(GENERIC_IS_ZERO ON CACHE BOOL ON FORCE)
  set(BUILD_HC_LIB ON)
  set(AMDGCN_TARGETS_LIB_LIST "AMDGCN_LIB_TARGETS")
  find_package(LLVM REQUIRED CONFIG PATHS ${CMAKE_BINARY_DIR}/compiler NO_DEFAULT_PATH)

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  add_definitions(${LLVM_DEFINITIONS})

  # build rocdl
  add_subdirectory(${ROCDL_SRC_DIR})

  file(MAKE_DIRECTORY ${ROCDL_BUILD_DIR}/lib)
  add_custom_target(rocdl_links DEPENDS ${AMDGCN_LIB_TARGETS})
  add_custom_command(TARGET rocdl_links POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../irif/irif.amdgcn.bc                              irif.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../opencl/opencl.amdgcn.bc                          opencl.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../ockl/ockl.amdgcn.bc                              ockl.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../hc/hc.amdgcn.bc                                  hc.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_701.amdgcn.bc              oclc_isa_version_701.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_802.amdgcn.bc              oclc_isa_version_802.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_810.amdgcn.bc              oclc_isa_version_810.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_finite_only_on.amdgcn.bc               oclc_finite_only_on.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_900.amdgcn.bc              oclc_isa_version_900.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_finite_only_off.amdgcn.bc              oclc_finite_only_off.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_801.amdgcn.bc              oclc_isa_version_801.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_700.amdgcn.bc              oclc_isa_version_700.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_daz_opt_on.amdgcn.bc                   oclc_daz_opt_on.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_unsafe_math_off.amdgcn.bc              oclc_unsafe_math_off.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_800.amdgcn.bc              oclc_isa_version_800.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_daz_opt_off.amdgcn.bc                  oclc_daz_opt_off.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_unsafe_math_on.amdgcn.bc               oclc_unsafe_math_on.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_correctly_rounded_sqrt_off.amdgcn.bc   oclc_correctly_rounded_sqrt_off.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_803.amdgcn.bc              oclc_isa_version_803.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_correctly_rounded_sqrt_on.amdgcn.bc    oclc_correctly_rounded_sqrt_on.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_901.amdgcn.bc              oclc_isa_version_901.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../oclc/oclc_isa_version_804.amdgcn.bc              oclc_isa_version_804.amdgcn.bc
    COMMAND ${CMAKE_COMMAND} -E create_symlink ../ocml/ocml.amdgcn.bc                              ocml.amdgcn.bc
    WORKING_DIRECTORY ${ROCDL_BUILD_DIR}/lib
  )
  # install ROCm Device Library in hcc
  install(DIRECTORY ${ROCDL_BUILD_DIR}/hc
                    ${ROCDL_BUILD_DIR}/irif
                    ${ROCDL_BUILD_DIR}/lib
                    ${ROCDL_BUILD_DIR}/ockl
                    ${ROCDL_BUILD_DIR}/oclc
                    ${ROCDL_BUILD_DIR}/ocml
                    ${ROCDL_BUILD_DIR}/opencl
          DESTINATION rocdl
          PATTERN CMake* EXCLUDE
          PATTERN CPack* EXCLUDE
          PATTERN *.cmake EXCLUDE
          PATTERN Makefile EXCLUDE
          PATTERN test EXCLUDE
          PATTERN cuda2gcn EXCLUDE
          PATTERN *.lib.bc EXCLUDE
          PATTERN *.optout.bc EXCLUDE
          PATTERN src EXCLUDE
          PATTERN CMakeFiles EXCLUDE
          PATTERN transformed_src EXCLUDE
          PATTERN utils EXCLUDE
  )

  # create search paths for ROCDL
  list(APPEND ROCM_DEVICE_LIB_PATHS ${ROCDL_BUILD_DIR}/lib)
  if (NOT CPACK_SET_DESTDIR)
    list(APPEND ROCM_DEVICE_LIB_PATHS ${CPACK_PACKAGING_INSTALL_PREFIX}/rocdl/lib)
  endif(NOT CPACK_SET_DESTDIR)
  list(APPEND ROCM_DEVICE_LIB_PATHS ${CMAKE_INSTALL_PREFIX}/rocdl/lib)

endif(HCC_INTEGRATE_ROCDL)

add_subdirectory(hcc_config)
add_subdirectory(lib)
add_subdirectory(utils)
add_subdirectory(tests)
add_subdirectory(amp-conformance)
add_subdirectory(stl-test)
add_subdirectory(cmake-tests)

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

execute_process(COMMAND ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/process_packaging_script.bsh ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/postinst.in  ${DEBIAN_EXTRA_SCRIPT_DEST_DIR}/postinst ${CPACK_PACKAGE_NAME} ${HCC_PACKAGE_INSTALL_PREFIX} ${CMAKE_INSTALL_LIBDIR}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

execute_process(COMMAND ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/process_packaging_script.bsh ${DEBIAN_EXTRA_SCRIPT_SOURCE_DIR}/prerm.in  ${DEBIAN_EXTRA_SCRIPT_DEST_DIR}/prerm ${CPACK_PACKAGE_NAME} ${HCC_PACKAGE_INSTALL_PREFIX} ${CMAKE_INSTALL_LIBDIR}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
set(CPACK_PACKAGE_VERSION "${HCC_VERSION_MAJOR}.${HCC_VERSION_MINOR}.${HCC_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR ${HCC_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${HCC_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${HCC_VERSION_PATCH})

set(PACKAGE_DESCRIPTION "HCC: An Open Source, Optimizing C++ Compiler for Heterogeneous Compute")

set(OFFICIAL_RELEASE_BUILD 1)

# Default to generate a package appropriate for a distro; users override using command line with -D
# -D writes variables into the cmake cache, and setting CACHE variables won't overwrite variables already in the CACHE
if( "${DISTRO_ID}" MATCHES "ubuntu" )
  set( CPACK_GENERATOR "DEB" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
elseif ("${DISTRO_ID}" MATCHES "fedora")
  set( CPACK_GENERATOR "RPM" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
else()
  # generate a tarball for unknown DISTRO_ID
  set( CPACK_GENERATOR "TGZ" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP" )
endif ()

# DEB package settings
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION ${PACKAGE_DESCRIPTION})
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Siu Chi Chan <siuchi.chan@amd.com>")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
  "${PROJECT_BINARY_DIR}/packaging/debian/postinst;${PROJECT_BINARY_DIR}/packaging/debian/prerm")

set(HCC_GENERAL_DEBIAN_DEP "g++-multilib, gcc-multilib, findutils, libelf1, libpci3, file, libunwind8, libunwind-dev")

if ( USE_LIBCXX )
  set(HCC_LIBCXX_DEBIAN_DEP ", libc++1, libc++-dev, libc++abi1, libc++abi-dev")
else ( USE_LIBCXX )
  set(HCC_LIBCXX_DEBIAN_DEP "")
endif( USE_LIBCXX )

# control the list of package dependency depending on whether this is an official release build.
# for non-official release build, we want to relax the dependency on rocr runtime
if (OFFICIAL_RELEASE_BUILD)
  if (HCC_INTEGRATE_ROCDL)
    set(HCC_ROCR_DEP ", hsa-rocr-dev, hsa-ext-rocr-dev, rocm-utils")
  else (HCC_INTEGRATE_ROCDL)
    set(HCC_ROCR_DEP ", hsa-rocr-dev, hsa-ext-rocr-dev, rocm-device-libs, rocm-utils")
  endif(HCC_INTEGRATE_ROCDL)
else (OFFICIAL_RELEASE_BUILD)
  set(HCC_ROCR_DEP "")
endif (OFFICIAL_RELEASE_BUILD)

set(CPACK_DEBIAN_PACKAGE_DEPENDS "${HCC_GENERAL_DEBIAN_DEP} ${HCC_ROCR_DEP} ${HCC_LIBCXX_DEBIAN_DEP}")

# RPM package settings
set(CPACK_RPM_PACKAGE_DESCRIPTION ${PACKAGE_DESCRIPTION})
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/packaging/debian/postinst")
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${PROJECT_BINARY_DIR}/packaging/debian/prerm")

# disable automatic shared libraries dependency detection
set(CPACK_RPM_PACKAGE_AUTOREQ 0)

set(HCC_GENERAL_RPM_DEP "findutils, elfutils-libelf, pciutils-libs, file, pth, libunwind, libunwind-devel")
if ( USE_LIBCXX )
  set(HCC_LIBCXX_RPM_DEP ", libcxx, libcxxabi")
else ( USE_LIBCXX )
  set(HCC_LIBCXX_RPM_DEP "")
endif( USE_LIBCXX )

set(CPACK_RPM_PACKAGE_REQUIRES "${HCC_GENERAL_RPM_DEP} ${HCC_ROCR_DEP} ${HCC_LIBCXX_RPM_DEP}" )

set(CPACK_COMPONENTS_ALL compiler)

include (CPack)
MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make' to build all")
MESSAGE("   'make docs' to build the HTML API reference")
MESSAGE("")
