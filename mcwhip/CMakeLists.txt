####################
# HIP runtime (only available on HSA)
####################
if (HAS_HSA EQUAL 1)

  macro(add_hip_runtime_library name)
    CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hc ${CXXAMP_FLAGS}")
    include_directories(${MCWAMP_INC_DIR} "/opt/hsa/include/")
    add_library(${name} STATIC ${ARGN})
    add_dependencies(${name} llvm-link opt clang)
  endmacro(add_hip_runtime_library name)

  if(EXISTS "${PROJECT_SOURCE_DIR}/lib/hip/hip_hcc.cpp")
    add_hip_runtime_library(hip_runtime hip_hcc.cpp hc_am.cpp)
  else()
    add_hip_runtime_library(hip_runtime hip_runtime.cpp)
  endif()

  install(TARGETS hip_runtime
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)

  MESSAGE(STATUS "HSA available, going to build HIP Runtime")
else (HAS_HSA EQUAL 1)
  MESSAGE(STATUS "HSA NOT available, NOT going to build HIP Runtime")
endif (HAS_HSA EQUAL 1)

