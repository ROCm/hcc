if(POLICY CMP0046)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0046 OLD)
endif()

# gtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/utils")
set(GTEST_INC_DIR "${PROJECT_SOURCE_DIR}/utils")

# MCWAMP
set(MCWAMP_INC_DIR "${PROJECT_SOURCE_DIR}/include")

# Imported targets
include(ImportedTargets)

# Additional compile-time options for HCC runtime could be set via:
# - HCC_RUNTIME_CFLAGS
#
# For example: cmake -DHCC_RUNTIME_CFLAGS=-g would configure HCC runtime be built
# with debug information while other parts are not.

macro(amp_target name )
  set(CMAKE_CXX_COMPILER "${PROJECT_BINARY_DIR}/llvm-project/llvm/bin/clang++")
  add_compile_options(-std=c++11)
  add_definitions(-DHCC_MAJOR_VERSION=${HCC_VERSION_MAJOR})
  add_definitions(-DHCC_MINOR_VERSION=${HCC_VERSION_MINOR})

  target_compile_definitions(${name} PRIVATE "GTEST_HAS_TR1_TUPLE=0")
  target_include_directories(${name} SYSTEM PRIVATE ${GTEST_INC_DIR} ${LIBCXX_INC_DIR})
  target_include_directories(${name} PRIVATE ${MCWAMP_INC_DIR})
  target_include_directories(${name} SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)
  target_compile_options(${name} PUBLIC -hc -fPIC)

  # Enable debug line info only if it's a release build and HCC_RUNTIME_DEBUG is OFF
  # Otherwise, -gline-tables-only would override other existing debug flags
  if ((NOT HCC_RUNTIME_DEBUG) AND ("${CMAKE_BUILD_TYPE}" STREQUAL "Release"))
	  target_compile_options(${name} PRIVATE -gline-tables-only)
  endif ((NOT HCC_RUNTIME_DEBUG) AND ("${CMAKE_BUILD_TYPE}" STREQUAL "Release"))

  set_target_properties(${name} PROPERTIES
    SOVERSION "${HCC_VERSION_MAJOR}"
    VERSION   "${HCC_VERSION_MAJOR}.${HCC_VERSION_MINOR}"
  )
endmacro(amp_target name )

####################
# C++AMP runtime interface (mcwamp) 
####################
macro(add_mcwamp_library name )
  add_library( ${name} ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang rocdl)
endmacro(add_mcwamp_library name )

macro(add_mcwamp_shared_library name )
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  target_link_libraries(${name} PUBLIC hsa-runtime64 dl hc_am)
  target_link_libraries(${name} PRIVATE pthread)
  add_dependencies(${name} llvm-link opt clang rocdl hc_am)
endmacro(add_mcwamp_shared_library name )

####################
# C++AMP runtime (CPU implementation)
####################
macro(add_mcwamp_library_cpu name )
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang rocdl)
endmacro(add_mcwamp_library_cpu name )

####################
# C++AMP runtime (HSA implementation) 
####################

macro(add_mcwamp_library_hc_am name )
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang rocdl)
  # add HSA libraries
  target_link_libraries(${name} PUBLIC hsa-runtime64)
  target_link_libraries(${name} PRIVATE pthread)
endmacro(add_mcwamp_library_hc_am name )

if(POLICY CMP0046)
  cmake_policy(POP)
endif()
