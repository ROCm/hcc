include (CMakeForceCompiler)

# gtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/utils")
set(GTEST_INC_DIR "${PROJECT_SOURCE_DIR}/utils")

# MCWAMP
set(MCWAMP_INC_DIR "${PROJECT_SOURCE_DIR}/include")

# CXXAMPFLAGS
set(CXXAMP_FLAGS "-I${GTEST_INC_DIR} -I${LIBCXX_INC_DIR} -I${MCWAMP_INC_DIR} -stdlib=libc++ -std=c++amp -DGTEST_HAS_TR1_TUPLE=0 -fPIC")

# Additional compile-time options for HCC runtime could be set via:
# - HCC_RUNTIME_CFLAGS
#
# For example: cmake -DHCC_RUNTIME_CFLAGS=-g would configure HCC runtime be built
# with debug information while other parts are not.

####################
# C++AMP runtime interface (mcwamp) 
####################
macro(add_mcwamp_library name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXAMP_FLAGS} ${HCC_RUNTIME_CFLAGS}")
  add_library( ${name} ${ARGN} )
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
endmacro(add_mcwamp_library name )

####################
# C++AMP runtime (CPU implementation)
####################
macro(add_mcwamp_library_cpu name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXAMP_FLAGS} ${HCC_RUNTIME_CFLAGS}")
  add_library( ${name} SHARED ${ARGN} )
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
endmacro(add_mcwamp_library_cpu name )

####################
# C++AMP runtime (OpenCL implementation) 
####################
macro(add_mcwamp_library_opencl name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXAMP_FLAGS} ${HCC_RUNTIME_CFLAGS}")
  # add OpenCL headers
  include_directories("${OPENCL_HEADER}/..")
  add_library( ${name} SHARED ${ARGN} )
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
  # add OpenCL libraries
  target_link_libraries(${name} ${OPENCL_LIBRARY})
endmacro(add_mcwamp_library_opencl name )

####################
# C++AMP runtime (HSA implementation) 
####################
macro(add_mcwamp_library_hsa name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXAMP_FLAGS} ${HCC_RUNTIME_CFLAGS}")
  # add HSA headers
  include_directories(${HSA_HEADER})
  add_library( ${name} SHARED ${ARGN} )
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
  # add HSA libraries
  target_link_libraries(${name} ${HSA_LIBRARY})
  target_link_libraries(${name} pthread)
endmacro(add_mcwamp_library_hsa name )

####################
# C++AMP config (clamp-config)
####################

macro(add_mcwamp_executable name )
  link_directories(${LIBCXX_LIB_DIR} ${LIBCXXRT_LIB_DIR})
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${GTEST_INC_DIR} -I${LIBCXX_INC_DIR} -I${MCWAMP_INC_DIR} -stdlib=libc++ -std=c++amp" )
  add_executable( ${name} ${ARGN} )
  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
  if (APPLE)
    target_link_libraries( ${name} mcwamp c++abi)
  else (APPLE)
    target_link_libraries( ${name} mcwamp dl pthread c++abi)
  endif (APPLE)
endmacro(add_mcwamp_executable name )


macro(add_config_executable name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_CXX_FLAGS "-std=c++11" )
  add_executable( ${name} ${ARGN} )

  # LLVM and Clang shall be compiled beforehand
  add_dependencies(${name} llvm-link opt clang)
endmacro(add_mcwamp_executable name )
