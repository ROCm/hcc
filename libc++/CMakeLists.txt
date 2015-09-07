set(CMAKE_INSTALL_PREFIX "/opt/kalmar")
project(libcxx)
cmake_minimum_required(VERSION 2.8)

# define Kalmar version information
SET(KALMAR_VERSION_MAJOR "0")
SET(KALMAR_VERSION_MINOR "7")

# get date information
# use the last two digits of year + week number as KALMAR_VERSION_PATCH
execute_process(COMMAND date +%y%W
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE KALMAR_VERSION_PATCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# get commit information
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE KALMAR_DRIVER_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/libcxx
                OUTPUT_VARIABLE LIBCXX_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# set Kalmar version string
set(KALMAR_VERSION_STRING "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}-${KALMAR_DRIVER_COMMIT}-${LIBCXX_COMMIT}")

add_subdirectory(libcxx)

# add libcxxrt for non-OSX platforms
if(NOT APPLE)
include_directories(${PROJECT_SOURCE_DIR}/libcxxrt/src)
set(CMAKE_CXX_FLAGS "-DLIBCXXRT")
add_subdirectory(libcxxrt)
INSTALL(FILES ${PROJECT_BINARY_DIR}/libcxxrt/lib/libcxxrt.a
              ${PROJECT_BINARY_DIR}/libcxxrt/lib/libcxxrt.so
        DESTINATION lib
        COMPONENT libcxxrt
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_READ WORLD_READ) 
endif(NOT APPLE)

set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX "/opt/kalmar")
set(CPACK_PACKAGE_NAME "libcxxamp")
set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
set(CPACK_PACKAGE_VERSION ${KALMAR_VERSION_STRING})
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "libcxxamp: libc++ for Heterogeneous C++ to OpenCL/HSA compiler")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jack Chung <jack@multicorewareinc.com>")
set(CPACK_GENERATOR "DEB;TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")
include(CPack)
