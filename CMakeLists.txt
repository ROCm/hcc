cmake_minimum_required( VERSION 2.8 )
project (HCC)

option(HSA_USE_AMDGPU_BACKEND "Use AMDGPU LLVM backend as compiler for HSA" OFF)

# set default installation path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
  if (HSA_USE_AMDGPU_BACKEND)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/hcc-lc" CACHE PATH "Default installation path of hcc" FORCE)
  else  (HSA_USE_AMDGPU_BACKEND)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/hcc-hsail" CACHE PATH "Default installation path of hcc" FORCE)
  endif()
endif ()
MESSAGE("Package installation path: ${CMAKE_INSTALL_PREFIX}")

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")

# set as release build by default
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Release Debug" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# set default HCC-clang URL
IF (NOT CLANG_URL)
  SET(CLANG_URL "git@github.com:RadeonOpenCompute/hcc-clang-upgrade.git" CACHE STRING "CLANG URL" FORCE)
ENDIF (NOT CLANG_URL)
set(CMAKE_MACOSX_RPATH 1)

IF (NOT NUM_BUILD_THREADS)
  SET(NUM_BUILD_THREADS 8)
ENDIF(NOT NUM_BUILD_THREADS)

include (SetupPromote)
include (SetupEraseNonkernel)
include (SetupCpuRename)
include (SetupTileUniform)
include (SetupHC)
include (SetupUtils)
include (MCWAMP)
include (SetupRemoveSpecialSection)
include (PatchLLVM)

execute_process(COMMAND ln -sf ../../lld WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools)
execute_process(COMMAND ln -sf ../../clang WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiler/tools)
setup_Promote(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms Promote)
setup_EraseNonkernel(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms EraseNonkernel)
setup_CpuRename(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms CpuRename)
setup_TileUniform(${PROJECT_SOURCE_DIR}/compiler/lib/Analysis TileUniform)
setup_HC(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms HC)
setup_Utils(utils)
setup_RemoveSpecialSection(${PROJECT_SOURCE_DIR}/compiler/lib/Transforms RemoveSpecialSection)
patch_LLVM(utils)

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
# Detect HSA
#################

# HSA external dependencies
set (HSA_ROOT "/opt/hsa"
        CACHE PATH "HSA runtime path")

set (ROCM_ROOT "/opt/rocm"
        CACHE PATH "ROCM runtime path")

set (HSA_LLVM_BIN_DIR CACHE PATH "Path to llvm tools with hsail/amdgpu support")

find_path(HSA_HEADER hsa/hsa.h PATHS ${HSA_HEADER_DIR} ${ROCM_ROOT}/include ${HSA_ROOT}/include NO_DEFAULT_PATH)
find_path(HSA_HEADER hsa/hsa.h)

if (NOT HSA_HEADER)
  MESSAGE("HSA header not found. Use -DHSA_HEADER_DIR=<path_to_hsa.h>.")
endif (NOT HSA_HEADER)


find_library(HSA_LIBRARY hsa-runtime64 PATHS ${HSA_LIBRARY_DIR} ${ROCM_ROOT}/lib ${HSA_ROOT}/lib NO_DEFAULT_PATH)
find_library(HSA_LIBRARY hsa-runtime64)
if (NOT HSA_LIBRARY)
  MESSAGE("HSA runtime library not found. Use -DHSA_LIBRARY_DIR=<path_to_libhsa-runtime64.so>.")
endif (NOT HSA_LIBRARY)

# find offline finalization tool for APU
find_program(HSA_HOF_APU hof PATHS ${HSA_HOF_DIR} ${ROCM_ROOT}/bin ${HSA_ROOT}/bin NO_DEFAULT_PATH)
find_program(HSA_HOF_APU hof)

# find offline finalization tool for DGPU
find_program(HSA_HOF_DGPU amdhsafin PATHS ${HSA_HOF_DIR} ${ROCM_ROOT}/bin ${HSA_ROOT}/bin NO_DEFAULT_PATH)
find_program(HSA_HOF_DGPU amdhasfin)

if (HSA_HOF_APU)
  set(HSA_HOF ${HSA_HOF_APU})
else (HSA_HOF_APU)
  if (HSA_HOF_DGPU)
    set(HSA_HOF ${HSA_HOF_DGPU})
  endif (HSA_HOF_DGPU)
endif (HSA_HOF_APU)

if (NOT HSA_HOF)
  MESSAGE("HSA offline finalizer (HOF) not found.  Use -DHSA_HOF_DIR=<path_to_hof>.")
endif (NOT HSA_HOF)


# display HSA information
MESSAGE("")
MESSAGE("HSA INFORMATION:")
MESSAGE("HSA_HEADER_DIR = ${HSA_HEADER_DIR}, actually found at: ${HSA_HEADER}")
MESSAGE("HSA_LIBRARY_DIR = ${HSA_LIBRARY_DIR}, actually found at: ${HSA_LIBRARY}")
MESSAGE("HSA_HOF_DIR = ${HSA_HOF_DIR}, actually found at: ${HSA_HOF}")
MESSAGE("")

#################
# Set up version information
#################

# define HCC version information
SET(KALMAR_VERSION_MAJOR "0")
SET(KALMAR_VERSION_MINOR "10")

# get date information based on UTC
# use the last two digits of year + week number + day in the week as KALMAR_VERSION_PATCH
# use the commit date, instead of build date
# add xargs to remove strange trailing newline character
execute_process(COMMAND git show -s --format=@%ct
                COMMAND xargs
                COMMAND date -f - --utc +%y%W%w
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

# set HCC version string
if (HAS_HSA)
  # For HSA build, the rule for version string is:
  # KALMAR_VERSION_MAJOR . KALMAR_VERSION_MINOR . KALMAR_VERSION_PATCH-KALMAR_DRIVER_COMIT-KALMAR_COMPILER_COMMIT-KALMAR_HLC_COMMIT
  set(KALMAR_VERSION_STRING "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}-${KALMAR_DRIVER_COMMIT}-${KALMAR_COMPILER_COMMIT}-${KALMAR_HLC_COMMIT}")
else (HAS_HSA)
  # For non-HSA build, the rule for version string is:
  # KALMAR_VERSION_MAJOR . KALMAR_VERSION_MINOR . KALMAR_VERSION_PATCH-KALMAR_DRIVER_COMIT-KALMAR_COMPILER_COMMIT
  set(KALMAR_VERSION_STRING "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}-${KALMAR_DRIVER_COMMIT}-${KALMAR_COMPILER_COMMIT}")
endif (HAS_HSA)

# show HCC version string
MESSAGE("=============================================")
MESSAGE("HCC version: ${KALMAR_VERSION_STRING}")
MESSAGE("=============================================")

# determine if we have HSA stack available
# at least we need the followings:
# - HSA headers (installed by HSA runtime)
# - HSA runtime libraries (installed by HSA runtime)
# - HSA ext64 library (installed by HSA runtime)
if ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY))
  set(HAS_HSA 0)
else ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY))
  set(HAS_HSA 1)
endif ((NOT HSA_HEADER) OR (NOT HSA_LIBRARY))

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
# Detect if we have HSA available
#################
if (HAS_HSA EQUAL 0)
  MESSAGE(FATAL_ERROR "HSA is NOT available on the system!")
endif (HAS_HSA EQUAL 0)

#################
# Detect AMDGPU backend for native codegen
#################

if (HSA_USE_AMDGPU_BACKEND)
  add_definitions(-DHSA_USE_AMDGPU_BACKEND)
  set(HAS_HSA_HOF 0)
endif ()

set(HSA_AMDGPU_GPU_TARGET "fiji" CACHE STRING "Target AMDGPU ISA version")

if (NOT (HSA_AMDGPU_GPU_TARGET STREQUAL "kaveri" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "carrizo" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "fiji" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "hawaii" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "tonga" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:7:0:0" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:7:0:1" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:1" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:2" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:3" OR
         HSA_AMDGPU_GPU_TARGET STREQUAL "auto"))
  MESSAGE(FATAL_ERROR "${HSA_AMDGPU_GPU_TARGET} is not a valid AMDGPU target")
endif()

# AMDGPU_TARGET would hold the value used by compiler scripts
# But by default, auto means fiji
if (HSA_AMDGPU_GPU_TARGET STREQUAL "auto")
  SET(AMDGPU_TARGET "fiji")
else (HSA_AMDGPU_GPU_TARGET STREQUAL "auto")
  SET(AMDGPU_TARGET ${HSA_AMDGPU_GPU_TARGET})
endif (HSA_AMDGPU_GPU_TARGET STREQUAL "auto")

# Map ISA version to GPU family
if (HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:7:0:0")
  SET(AMDGPU_TARGET "kaveri")
endif()

if (HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:7:0:1")
  SET(AMDGPU_TARGET "hawaii")
endif()

if (HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:1")
  SET(AMDGPU_TARGET "carrizo")
endif()

if (HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:2")
  SET(AMDGPU_TARGET "tonga")
endif()

if (HSA_AMDGPU_GPU_TARGET STREQUAL "AMD:AMDGPU:8:0:3")
  SET(AMDGPU_TARGET "fiji")
endif()

#################
# Setup HCC backend macro string
# Possible values:
# - CL : for non-HSA systems
# - HSAIL : for HSA systems configured with HLC backend
# - AMDGPU : for HSA systems configured with Lightning backend
#################

if (HSA_USE_AMDGPU_BACKEND)
  set(KALMAR_BACKEND "HCC_BACKEND_AMDGPU")
else(HSA_USE_AMDGPU_BACKEND)
  if (HAS_HSA)
    set(KALMAR_BACKEND "HCC_BACKEND_HSAIL")
  else (HAS_HSA)
    set(KALMAR_BACKEND "HCC_BACKEND_CL")
  endif (HAS_HSA)
endif(HSA_USE_AMDGPU_BACKEND)

message(STATUS "HCC configured with default backend: ${KALMAR_BACKEND}")

#########################
# build target: world
# enable HLC and HSAILasm only if HSA stack is available
#########################

if (HAS_HSA)

# get HLC commit information
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/hlc
                OUTPUT_VARIABLE KALMAR_HLC_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

endif (HAS_HSA)

# build Clang front-end
set(CLANG_SRC_DIR "${PROJECT_SOURCE_DIR}/compiler")
set(CLANG_BIN_DIR "${PROJECT_BINARY_DIR}/compiler")
file(MAKE_DIRECTORY ${CLANG_BIN_DIR})
if (HAS_HSA)
add_custom_target(clang
  COMMAND ${CMAKE_COMMAND} -Wno-dev ${CLANG_SRC_DIR}
          -DHAS_HSA=${HAS_HSA}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DKALMAR_VERSION_STRING=${KALMAR_VERSION_STRING}
          -DKALMAR_VERSION_MAJOR=${KALMAR_VERSION_MAJOR}
          -DKALMAR_VERSION_MINOR=${KALMAR_VERSION_MINOR}
          -DKALMAR_VERSION_PATCH=${KALMAR_VERSION_PATCH}
          -DKALMAR_DRIVER_COMMIT=${KALMAR_DRIVER_COMMIT}
          -DKALMAR_COMPILER_COMMIT=${KALMAR_COMPILER_COMMIT}
          -DKALMAR_HLC_COMMIT=${KALMAR_HLC_COMMIT}
          -DKALMAR_BACKEND=${KALMAR_BACKEND}
          -DLLVM_TARGETS_TO_BUILD=X86
          -DLLVM_INCLUDE_EXAMPLES=off
  COMMAND make -j ${NUM_BUILD_THREADS} # not portable, but we need it this way
  WORKING_DIRECTORY ${CLANG_BIN_DIR}
)
else (HAS_HSA)
add_custom_target(clang
  COMMAND ${CMAKE_COMMAND} -Wno-dev ${CLANG_SRC_DIR}
          -DHAS_HSA=${HAS_HSA}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DKALMAR_VERSION_STRING=${KALMAR_VERSION_STRING}
          -DKALMAR_VERSION_MAJOR=${KALMAR_VERSION_MAJOR}
          -DKALMAR_VERSION_MINOR=${KALMAR_VERSION_MINOR}
          -DKALMAR_VERSION_PATCH=${KALMAR_VERSION_PATCH}
          -DKALMAR_DRIVER_COMMIT=${KALMAR_DRIVER_COMMIT}
          -DKALMAR_COMPILER_COMMIT=${KALMAR_COMPILER_COMMIT}
          -DKALMAR_BACKEND=${KALMAR_BACKEND}
          -DLLVM_TARGETS_TO_BUILD=X86
          -DLLVM_INCLUDE_EXAMPLES=off
  COMMAND make -j ${NUM_BUILD_THREADS} # not portable, but we need it this way
  WORKING_DIRECTORY ${CLANG_BIN_DIR}
)
endif (HAS_HSA)

# create symlink of mostly used commands in clang to bin/
# - hcc
# - clang++
# - clang

if (HSA_USE_AMDGPU_BACKEND AND HSA_LLVM_BIN_DIR)

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

else (HSA_USE_AMDGPU_BACKEND AND HSA_LLVM_BIN_DIR)

add_custom_command(TARGET clang POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/hcc ${PROJECT_BINARY_DIR}/bin/hcc
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/clang++ ${PROJECT_BINARY_DIR}/bin/clang++
  COMMAND ${CMAKE_COMMAND} -E create_symlink ../compiler/bin/clang ${PROJECT_BINARY_DIR}/bin/clang
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
install(FILES ${PROJECT_BINARY_DIR}/bin/hcc
              ${PROJECT_BINARY_DIR}/bin/clang++
              ${PROJECT_BINARY_DIR}/bin/clang
        DESTINATION bin)

endif (HSA_USE_AMDGPU_BACKEND AND HSA_LLVM_BIN_DIR)



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

if (HAS_HSA)

# build HLC

if (NOT HSA_LLVM_BIN_DIR)

set(HLC_SRC_DIR "${PROJECT_SOURCE_DIR}/hlc")
set(HLC_BIN_DIR "${PROJECT_BINARY_DIR}/hlc")
file(MAKE_DIRECTORY ${HLC_BIN_DIR})
add_custom_target(hlc
  COMMAND ${CMAKE_COMMAND} -Wno-dev ${HLC_SRC_DIR}
          -DLLVM_ENABLE_EH=ON
          -DLLVM_ENABLE_RTTI=ON
          -DLLVM_TARGETS_TO_BUILD=
          -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=HSAIL
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  COMMAND make -j ${NUM_BUILD_THREADS} # not portable, but we need it this way
  WORKING_DIRECTORY ${HLC_BIN_DIR}
  DEPENDS clang
)

# install HLC along with HCC
# stored under hlc/
install(DIRECTORY ${HLC_BIN_DIR}/bin
        DESTINATION hlc
        COMPONENT hlc
        PATTERN * PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# build HSAILasm
set(HSAILASM_SRC_DIR "${PROJECT_SOURCE_DIR}/HSAILasm")
set(HSAILASM_BIN_DIR "${PROJECT_BINARY_DIR}/HSAILasm")
file(MAKE_DIRECTORY ${HSAILASM_BIN_DIR})
add_custom_target(HSAILasm
  COMMAND ${CMAKE_COMMAND} -Wno-dev -DBUILD_LIBBRIGDWARF=1 ${HSAILASM_SRC_DIR}
  COMMAND make -j4 # not portable, but we need it this way
  WORKING_DIRECTORY ${HSAILASM_BIN_DIR}
  DEPENDS hlc
)

# stored under HSAILasm/
install(FILES ${HSAILASM_BIN_DIR}/HSAILAsm/HSAILasm
        DESTINATION HSAILasm
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT HSAILasm)

# Copy binary file to directory matching the install directory
add_custom_target(HSAILasm_setup
    COMMAND ${CMAKE_COMMAND} -E copy
    "${HSAILASM_BIN_DIR}/HSAILAsm/HSAILasm" "${HSAILASM_BIN_DIR}"
    DEPENDS HSAILasm)

endif(NOT HSA_LLVM_BIN_DIR)

if (HSA_USE_AMDGPU_BACKEND AND HSA_LLVM_BIN_DIR)

  # create a symlink to the LLVM binaries for the amdgpu backend in the project directory
  add_custom_target(lightning_bin
    COMMAND mkdir -p "${PROJECT_BINARY_DIR}/llvm/bin"
    COMMAND cp -rp "${HSA_LLVM_BIN_DIR}/*" "${PROJECT_BINARY_DIR}/llvm/bin/"
  )

  # include the amdgpu backend binaries in the hcc package
  install(DIRECTORY ${PROJECT_BINARY_DIR}/llvm/bin
          DESTINATION llvm
          COMPONENT lightning_bin
          PATTERN * PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  # do NOT depend on HLC or HSAILasm in case we use AMDGPU backend
  add_custom_target(world DEPENDS lightning_bin clang)
  find_program(OPT opt PATHS ${HSA_LLVM_BIN_DIR} NO_DEFAULT_PATH)
  add_custom_target(opt DEPENDS ${OPT})
  find_program(LLVM_LINK llvm-link PATHS ${HSA_LLVM_BIN_DIR} NO_DEFAULT_PATH)
  add_custom_target(llvm-link DEPENDS ${LLVM_LINK})
else (HSA_USE_AMDGPU_BACKEND AND HSA_LLVM_BIN_DIR)
  # build hlc and HSAILasm in case HSA_USE_AMDGPU_BACKEND 
  # or HSA_LLVM_BIN_DIR is not set
  add_custom_target(world DEPENDS clang hlc HSAILasm_setup)
endif (HSA_USE_AMDGPU_BACKEND AND HSA_LLVM_BIN_DIR)

else (HAS_HSA)

# do NOT depend on HLC or HSAILasm in case HSA is not present
add_custom_target(world
    DEPENDS clang ${DEPS}
)

endif (HAS_HSA)

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
if (HSA_USE_AMDGPU_BACKEND)
  set(CPACK_PACKAGE_NAME "hcc_lc")
else (HSA_USE_AMDGPU_BACKEND)
  set(CPACK_PACKAGE_NAME "hcc_hsail")
endif(HSA_USE_AMDGPU_BACKEND)

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
set(CPACK_PACKAGE_VERSION ${KALMAR_VERSION_STRING})
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "HCC: a Heterogeneous C++ compiler")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Siu Chi Chan <siuchi.chan@amd.com>")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")


set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
  "${PROJECT_BINARY_DIR}/packaging/debian/postinst;${PROJECT_BINARY_DIR}/packaging/debian/prerm")

set(HCC_GENERAL_DEBIAN_DEP "libstdc++-4.8-dev, libc++1, libc++-dev, libc++abi1, libc++abi-dev, elfutils")

# control the list of package dependency depending on whether this is an official release build.
# for non-official release build, we want to relax the dependency on rocr runtime
set(OFFICIAL_RELEASE_BUILD 0)
if (OFFICIAL_RELEASE_BUILD)
  set(HCC_ROCR_DEBIAN_DEP "hsa-rocr-dev, hsa-ext-rocr-dev")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${HCC_ROCR_DEBIAN_DEP}, ${HCC_GENERAL_DEBIAN_DEP}")
else (OFFICIAL_RELEASE_BUILD)
  # dependencies for any local build
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${HCC_GENERAL_DEBIAN_DEP}")
endif (OFFICIAL_RELEASE_BUILD)

set(CPACK_GENERATOR "DEB;TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")

if (HSA_USE_AMDGPU_BACKEND)
  # don't include HSAILasm and hlc if using AMDGPU backend
  set(CPACK_COMPONENTS_ALL compiler)
else (HSA_USE_AMDGPU_BACKEND)
  # put Clang front-end, HLC and HSAILasm in the package as well
  set(CPACK_COMPONENTS_ALL HSAILasm hlc compiler)
endif (HSA_USE_AMDGPU_BACKEND)

include (CPack)
MESSAGE("")
MESSAGE("** For the first time:")
MESSAGE("   'make world' to build clang, HLC, HSAILasm and library for testing.")
MESSAGE("   'make' to build the rest of LLVM tools")
MESSAGE("   'make docs' to build the HTML API reference")
MESSAGE("")

