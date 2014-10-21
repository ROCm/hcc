include (CMakeForceCompiler)

# libc++
set(LIBCXX_SRC_DIR "${PROJECT_SOURCE_DIR}/libc++/libcxx")
set(LIBCXX_INC_DIR "${LIBCXX_SRC_DIR}/include")
set(LIBCXX_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxx/lib")

if (NOT APPLE)
# libcxxrt
set(LIBCXXRT_LIB_DIR "${PROJECT_BINARY_DIR}/libc++/libcxxrt/lib")
endif (NOT APPLE)
# gtest
set(GTEST_SRC_DIR "${PROJECT_SOURCE_DIR}/utils")
set(GTEST_INC_DIR "${PROJECT_SOURCE_DIR}/utils")

# MCWAMP
seT(MCWAMP_INC_DIR "${PROJECT_SOURCE_DIR}/include")

# CXXAMPFLAGS
set(CXXAMP_FLAGS "-I${GTEST_INC_DIR} -I${LIBCXX_INC_DIR} -I${MCWAMP_INC_DIR} -stdlib=libc++ -std=c++amp -DGTEST_HAS_TR1_TUPLE=0")

# STATIC ONLY FOR NOW.
macro(add_mcwamp_library name )
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXAMP_FLAGS} -fPIC -DCXXAMP_NV=1")
  if (CXXAMP_ENABLE_HSA)
    # add HSA headers
    include_directories(${HSA_ROOT}/include)
    add_definitions("-DCXXAMP_ENABLE_HSA=1")
  else (CXXAMP_ENABLE_HSA)
    # add OpenCL headers
    include_directories("${OPENCL_HEADER}/..")
  endif (CXXAMP_ENABLE_HSA)
  add_library( ${name} ${ARGN} )
endmacro(add_mcwamp_library name )

macro(add_mcwamp_executable name )
  if (CXXAMP_ENABLE_HSA)
    add_definitions("-DCXXAMP_ENABLE_HSA=1")
  endif (CXXAMP_ENABLE_HSA)
  link_directories(${LIBCXX_LIB_DIR} ${LIBCXXRT_LIB_DIR})
  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${GTEST_INC_DIR} -I${LIBCXX_INC_DIR} -I${MCWAMP_INC_DIR} -stdlib=libc++ -std=c++amp" )
  add_executable( ${name} ${ARGN} )
  if (APPLE)
    target_link_libraries( ${name} mcwamp c++abi)
  else (APPLE)
    target_link_libraries( ${name} mcwamp cxxrt dl pthread)
  endif (APPLE)
endmacro(add_mcwamp_executable name )
