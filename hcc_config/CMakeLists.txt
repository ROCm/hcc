
if (HCC_RUNTIME_DEBUG)
  add_compile_options(-g -O0)
endif (HCC_RUNTIME_DEBUG)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../hcc_config)

set(CMAKE_CXX_FLAGS "-std=c++11" )
add_definitions(-DLIBCXX_HEADER=${LIBCXX_HEADER}) 

add_executable(hcc-config hcc_config.cpp)


add_custom_command(TARGET hcc-config POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ./hcc-config ./clamp-config
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

install(TARGETS hcc-config
    RUNTIME DESTINATION bin
)

install(FILES ${PROJECT_BINARY_DIR}/bin/clamp-config DESTINATION bin)


