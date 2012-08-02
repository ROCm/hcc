project(libc++)
cmake_minimum_required(VERSION 2.8)

include_directories(${PROJECT_SOURCE_DIR}/libcxxrt/src)

set(CMAKE_CXX_FLAGS "-DLIBCXXRT")

add_subdirectory(libcxxrt)
add_subdirectory(libcxx)

