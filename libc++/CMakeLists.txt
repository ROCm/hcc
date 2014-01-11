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
set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX "/opt/mcw")
set(CPACK_PACKAGE_NAME "libcxxamp")
set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "libcxxamp: libc++ for C++AMP to OpenCL compiler")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Ray I-Jui Sung <ray@multicorewareinc.com>")
set(CPACK_GENERATOR "DEB;TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")
include(CPack)
