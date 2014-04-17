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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXAMP_FLAGS}")
  if (CXXAMP_ENABLE_HSA_OKRA)
    include_directories( ${OKRA_ROOT}/include ${JNI_INCLUDE_DIRS})
    add_definitions("-DCXXAMP_ENABLE_HSA_OKRA=1")
  endif (CXXAMP_ENABLE_HSA_OKRA)
  add_library( ${name} ${ARGN} )
endmacro(add_mcwamp_library name )

macro(add_mcwamp_executable name )
  if (CXXAMP_ENABLE_HSA_OKRA)
    add_definitions("-DCXXAMP_ENABLE_HSA_OKRA=1")
  endif (CXXAMP_ENABLE_HSA_OKRA)
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
