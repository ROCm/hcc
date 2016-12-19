
if (HCC_RUNTIME_DEBUG)
  add_compile_options(-g -O0)
endif (HCC_RUNTIME_DEBUG)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../hcc_config)

set(CMAKE_CXX_FLAGS "-std=c++11" )
add_definitions(-DLIBCXX_HEADER=${LIBCXX_HEADER}) 

add_executable(hcc-config hcc_config.cpp)
add_executable(clamp-config hcc_config.cpp)

install(TARGETS clamp-config hcc-config
    RUNTIME DESTINATION bin
)
