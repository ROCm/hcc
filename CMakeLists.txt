cmake_minimum_required( VERSION 2.8 )
project (CPPAMP)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release") 
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake")
MESSAGE("Module path: ${CMAKE_MODULE_PATH}")
#include (MacroEnsureOutOfSourceBuild)
#include (ExportFile)
include (EnsureLLVMisPresent)
MESSAGE("${PROJECT_SOURCE_DIR}")
ensure_llvm_is_present(${PROJECT_SOURCE_DIR} compiler)

#MACRO_ENSURE_OUT_OF_SOURCE_BUILD("Please build ${CMAKE_PROJECT_NAME} in a different directory than the source directory")

set(LLVM_TARGETS_TO_BUILD "X86")
#set(LLVM_BUILD_TOOLS off)
set(LLVM_INCLUDE_TOOLS off)
set(LLVM_INCLUDE_EXAMPLES off)
set(LLVM_INCLUDE_TESTS on)

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

SET(PROJ_SEARCH_PATH "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/${LOCAL_LLVM_INCLUDE}" "${PROJECT_BINARY_DIR}/${LOCAL_LLVM_INCLUDE}" "${PROJECT_SOURCE_DIR}/compiler/utils/unittest/googletest/include")
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
#
#endif(UNIX)

#add_subdirectory(include)
add_subdirectory(compiler)
#add_subdirectory(lib)
#add_subdirectory(tools)
#add_subdirectory(test)
#add_subdirectory(icd)
#add_subdirectory(data)

MESSAGE("INCLUDE SEARCH DIRECTORIES: ${PROJ_SEARCH_PATH}")

