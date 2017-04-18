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
 
  if (USE_LIBCXX)
    target_include_directories(${name} SYSTEM PUBLIC ${LIBCXX_HEADER})
    target_compile_options(${name} PUBLIC -stdlib=libc++)
  endif (USE_LIBCXX)
	target_compile_options(${name} PUBLIC -std=c++amp -fPIC)
endmacro(amp_target name )

####################
# C++AMP runtime interface (mcwamp) 
####################
macro(add_mcwamp_library name )
  set(CMAKE_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
  add_compile_options(-std=c++11)
  add_library( ${name} ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang rocdl)
endmacro(add_mcwamp_library name )

####################
# C++AMP runtime (CPU implementation)
####################
macro(add_mcwamp_library_cpu name )
  set(CMAKE_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
  add_compile_options(-std=c++11)
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang rocdl)

  if (USE_LIBCXX)  
    target_include_directories(${name} SYSTEM PUBLIC ${LIBCXX_HEADER})
    target_link_libraries(${name} c++)
    target_link_libraries(${name} c++abi)
  endif (USE_LIBCXX)

endmacro(add_mcwamp_library_cpu name )

####################
# C++AMP runtime (HSA implementation) 
####################
macro(add_mcwamp_library_hsa name )
  set(CMAKE_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
  add_compile_options(-std=c++11)
  # add HSA headers
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang hc_am rocdl)
  # add HSA libraries
  target_link_libraries(${name} hsa-runtime64)
  target_link_libraries(${name} pthread)

  if (USE_LIBCXX)
    target_include_directories(${name} SYSTEM PUBLIC ${LIBCXX_HEADER})
    target_link_libraries(${name} c++)
    target_link_libraries(${name} c++abi)
  endif (USE_LIBCXX)

  target_link_libraries(${name} hc_am)
endmacro(add_mcwamp_library_hsa name )

macro(add_mcwamp_library_hc_am name )
  set(CMAKE_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
  # add HSA headers
  add_library( ${name} SHARED ${ARGN} )
  amp_target(${name})
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang rocdl)
  # add HSA libraries
  target_link_libraries(${name} hsa-runtime64)
  target_link_libraries(${name} pthread)

  if (USE_LIBCXX)
    target_include_directories(${name} SYSTEM PUBLIC ${LIBCXX_HEADER})
    target_link_libraries(${name} c++)
    target_link_libraries(${name} c++abi)
  endif (USE_LIBCXX)

endmacro(add_mcwamp_library_hc_am name )

if(POLICY CMP0046)
  cmake_policy(POP)
endif()
