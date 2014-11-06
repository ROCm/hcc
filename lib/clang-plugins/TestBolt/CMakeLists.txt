set (using_plugin 0)

  
# Build a test to showcase Bolt CLAMP code path is working
macro(test_clang_plugin name)
  set (srcs ${ARGN})
  
  # in clamp build mode
  set (COMPILER_BUILD_DIR ${PROJECT_BINARY_DIR}/compiler)

  # clamp headers path
  set (CLAMP_INC_DIR ${PROJECT_SOURCE_DIR}/include)

  set (INC_DIRS ${CLAMP_INC_DIR}
    ${LIBCXX_INC_DIR}
    ${BOLT_INC_DIRS} )

  # In build mode, we inherate from parent project. User can specify it
  list(APPEND INC_DIRS ${OPENCL_HEADER}/..)

  set (LIB_DIRS ${LIBCXX_LIB_DIR} 
    ${LIBCXXRT_LIB_DIR}
    ${COMPILER_BUILD_DIR}/lib 
    ${BOLT_LIB_DIRS} )

  include_directories (${INC_DIRS})
  link_directories (${LIB_DIRS})

  add_executable( ${name} ${srcs})

  foreach (user_lib ${USER_LIBS})
    target_link_libraries( ${name} ${user_lib} )
  endforeach()

  CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  
  if (using_plugin)
    set(CMAKE_CXX_FLAGS "-stdlib=libc++ -std=c++amp -Xclang -load -Xclang ${LIBRARY_OUTPUT_PATH}/libStmtRewriter.so -Xclang -plugin -Xclang StmtRewriter")
  else()
    set(CMAKE_CXX_FLAGS "-stdlib=libc++ -std=c++amp ")
  endif()

endmacro(test_clang_plugin)

set ( USER_LIBS
  mcwamp
  cxxrt
  c++ 
  dl
  OpenCL
)

list(APPEND USER_LIBS ampBolt.runtime.clang)

# For verification purpose, not for delivery
test_clang_plugin(TestBolt TestBolt.cpp)

install(TARGETS TestBolt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

