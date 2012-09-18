project(libc++)
cmake_minimum_required(VERSION 2.8)

include_directories(${PROJECT_SOURCE_DIR}/libcxxrt/src)

set(CMAKE_CXX_FLAGS "-DLIBCXXRT")

add_subdirectory(libcxxrt)
add_subdirectory(libcxx)
FILE(GLOB libcxxrt_files "${PROJECT_BINARY_DIR}/libcxxrt/lib/lib*")
INSTALL(FILES ${libcxxrt_files}
  DESTINATION lib
)
set(CPACK_SOURCE_GENERATOR "TGZ;TZ")
include(CPack)
