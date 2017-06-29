cmake_minimum_required( VERSION 3.0 )
project (HCC)

include(GNUInstallDirs)

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
set(INSTALL_DIR_NAME "hcc-${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}")
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
MESSAGE("HCC version: ${KALMAR_VERSION_STRING}")
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

file(MAKE_DIRECTORY ${CLANG_BIN_DIR})
add_custom_target(clang
  COMMAND ${CMAKE_COMMAND} ${CLANG_SRC_DIR}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS}
          -DKALMAR_VERSION_STRING=${KALMAR_VERSION_STRING}
          -DKALMAR_VERSION_MAJOR=${KALMAR_VERSION_MAJOR}
          -DKALMAR_VERSION_MINOR=${KALMAR_VERSION_MINOR}
          -DKALMAR_VERSION_PATCH=${KALMAR_VERSION_PATCH}
          -DKALMAR_SDK_COMMIT=${KALMAR_SDK_COMMIT}
          -DKALMAR_FRONTEND_COMMIT=${KALMAR_FRONTEND_COMMIT}
          -DKALMAR_BACKEND_COMMIT=${KALMAR_BACKEND_COMMIT}
          -DKALMAR_BACKEND=${KALMAR_BACKEND}
          -DAMDGPU_TARGET=${AMDGPU_TARGET}
          -DLLVM_TARGETS_TO_BUILD="AMDGPU\;X86"
          -DLLVM_INCLUDE_EXAMPLES=off
          -DLLVM_EXTERNAL_PROJECTS="clang;compiler-rt;lld"
          -DLLVM_EXTERNAL_CLANG_SOURCE_DIR="${PROJECT_SOURCE_DIR}/clang"
          -DLLVM_EXTERNAL_COMPILER_RT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/compiler-rt"
          -DLLVM_EXTERNAL_LLD_SOURCE_DIR="${PROJECT_SOURCE_DIR}/lld"
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
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/llvm-mc ${PROJECT_BINARY_DIR}/bin/llvm-mc
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/llvm-objdump ${PROJECT_BINARY_DIR}/bin/llvm-objdump
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/lld ${PROJECT_BINARY_DIR}/bin/lld
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

add_custom_command(TARGET clang POST_BUILD
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
        PATTERN *.a EXCLUDE
        PATTERN lib*
        PATTERN LLVM*
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# install certain LLVM libraries needed by HIP
install(FILES  ${CLANG_BIN_DIR}/lib/libLLVMAMDGPUDesc.a
               ${CLANG_BIN_DIR}/lib/libLLVMAMDGPUUtils.a
               ${CLANG_BIN_DIR}/lib/libLLVMMC.a
               ${CLANG_BIN_DIR}/lib/libLLVMCore.a
               ${CLANG_BIN_DIR}/lib/libLLVMSupport.a
        DESTINATION compiler/lib
        COMPONENT compiler
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

install( FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmake/ImportedTargets.cmake
         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hcc )

add_custom_target(world DEPENDS clang)


# build the integrated ROCm Device Library
set(AMDHSACOD ${ROCM_ROOT}/bin/amdhsacod CACHE FILEPATH "Specify the amdhsacod tool")
if (HCC_INTEGRATE_ROCDL)

  set(ROCDL_SRC_DIR "${PROJECT_SOURCE_DIR}/rocdl")
  set(ROCDL_BUILD_DIR "${PROJECT_BINARY_DIR}/rocdl")
  file(MAKE_DIRECTORY ${ROCDL_BUILD_DIR})

  # custom commands to build rocdl
  add_custom_target(rocdl
    COMMAND CC=${CLANG_BIN_DIR}/bin/clang  ${CMAKE_COMMAND} ${ROCDL_SRC_DIR}
            -DLLVM_DIR=${CLANG_BIN_DIR}
            -DAMDHSACOD=${AMDHSACOD}
            -DGENERIC_IS_ZERO=ON
            -DBUILD_HC_LIB=ON
    COMMAND make -j ${NUM_BUILD_THREADS} # not portable, but we need it this way
    WORKING_DIRECTORY ${ROCDL_BUILD_DIR}
    DEPENDS clang
  )

  file(MAKE_DIRECTORY ${ROCDL_BUILD_DIR}/lib)
  add_custom_command(TARGET rocdl POST_BUILD
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
add_subdirectory(include)
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
set(CPACK_PACKAGE_VERSION "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})

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

set(HCC_GENERAL_RPM_DEP "findutils, elfutils-libelf, pciutils-libs, file, pth, libunwind")
set(CPACK_RPM_PACKAGE_REQUIRES "${HCC_GENERAL_RPM_DEP} ${HCC_ROCR_DEP}" )

set(CPACK_COMPONENTS_ALL compiler)

include (CPack)
MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make' to build all")
MESSAGE("   'make docs' to build the HTML API reference")
MESSAGE("")

