include (CMakeForceCompiler)

if(POLICY CMP0046)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0046 OLD)
endif()

# gtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/utils")
set(GTEST_INC_DIR "${PROJECT_SOURCE_DIR}/utils")

# MCWAMP
set(MCWAMP_INC_DIR "${PROJECT_SOURCE_DIR}/include")

# Additional compile-time options for HCC runtime could be set via:
# - HCC_RUNTIME_CFLAGS
#
# For example: cmake -DHCC_RUNTIME_CFLAGS=-g would configure HCC runtime be built
# with debug information while other parts are not.

macro(amp_target name )
	target_compile_definitions(${name} PRIVATE "GTEST_HAS_TR1_TUPLE=0")
  target_include_directories(${name} SYSTEM PRIVATE ${GTEST_INC_DIR} ${LIBCXX_INC_DIR})
	target_include_directories(${name} PRIVATE ${MCWAMP_INC_DIR})
  target_include_directories(${name} SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)
	target_compile_options(${name} PUBLIC -stdlib=libc++ -std=c++amp -fPIC)
endmacro(amp_target name )

####################
# C++AMP runtime interface (mcwamp) 
####################
macro(add_mcwamp_library name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  add_library( ${name} ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
endmacro(add_mcwamp_library name )

####################
# C++AMP runtime (CPU implementation)
####################
macro(add_mcwamp_library_cpu name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
  target_link_libraries(${name} c++)
  target_link_libraries(${name} c++abi)
endmacro(add_mcwamp_library_cpu name )

####################
# C++AMP runtime (HSA implementation) 
####################
macro(add_mcwamp_library_hsa name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  # add HSA headers
  add_library( ${name} SHARED ${ARGN} )
  target_include_directories(${name} SYSTEM PUBLIC ${HSA_HEADER})
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang hc_am)
  # add HSA libraries
  target_link_libraries(${name} ${HSA_LIBRARY})
  target_link_libraries(${name} pthread)
  target_link_libraries(${name} c++)
  target_link_libraries(${name} c++abi)
  target_link_libraries(${name} hc_am)
endmacro(add_mcwamp_library_hsa name )

macro(add_mcwamp_library_hc_am name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  # add HSA headers
  add_library( ${name} SHARED ${ARGN} )
  target_include_directories(${name} PRIVATE ${HSA_HEADER})
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
  # add HSA libraries
  target_link_libraries(${name} ${HSA_LIBRARY})
  target_link_libraries(${name} pthread)
  target_link_libraries(${name} c++)
  target_link_libraries(${name} c++abi)
endmacro(add_mcwamp_library_hc_am name )

####################
# C++AMP config (clamp-config)
####################

macro(add_mcwamp_executable name )
  link_directories(${LIBCXX_LIB_DIR} ${LIBCXXRT_LIB_DIR})
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  add_executable( ${name} ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
  if (APPLE)
    target_link_libraries( ${name} mcwamp c++ c++abi)
  else (APPLE)
    target_link_libraries( ${name} mcwamp dl pthread c++ c++abi)
  endif (APPLE)
endmacro(add_mcwamp_executable name )

if(POLICY CMP0046)
  cmake_policy(POP)
endif()
