# usage of this cmake script
# 1) build hcc first
# 2) run "make docs" under hcc build directory to get HTML files produced
# 3) go to docs/ under hcc build directory
# 4) cmake <hcc source>/doc
# 5) make package

cmake_minimum_required( VERSION 2.8 )
project (HCC-DOC)

# set default installation path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
  set(CMAKE_INSTALL_PREFIX "/opt/hcc")
else ()
  set(CMAKE_INSTALL_PREFIX "/opt/hcc")
endif ()
MESSAGE("Package installation path: ${CMAKE_INSTALL_PREFIX}")

#################
# Set up version information
#################

# define HCC version information
SET(KALMAR_VERSION_MAJOR "0")
SET(KALMAR_VERSION_MINOR "8")

# get date information
# use the last two digits of year + week number as KALMAR_VERSION_PATCH
execute_process(COMMAND date +%y%W
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..
                OUTPUT_VARIABLE KALMAR_VERSION_PATCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# get commit information
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/..
                OUTPUT_VARIABLE KALMAR_DRIVER_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../compiler/tools/clang
                OUTPUT_VARIABLE KALMAR_COMPILER_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# set HCC version string
set(KALMAR_VERSION_STRING "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}-${KALMAR_DRIVER_COMMIT}-${KALMAR_COMPILER_COMMIT}")

# show HCC version string
MESSAGE("========================================")
MESSAGE("HCC-DOC version: ${KALMAR_VERSION_STRING}")
MESSAGE("========================================")

# create documentation
configure_file(
  ${PROJECT_SOURCE_DIR}/../lib/doxygen_config.in
  ${PROJECT_BINARY_DIR}/lib/doxygen_config @ONLY)

add_custom_target(docs
    COMMAND doxygen "${PROJECT_BINARY_DIR}/lib/doxygen_config"
)

# install documentation
# optionally stored under doc/
install(DIRECTORY ${PROJECT_BINARY_DIR}/html
        DESTINATION doc
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT doc
        OPTIONAL)

set(CPACK_SET_DESTDIR TRUE)
set(CPACK_INSTALL_PREFIX "/opt/hcc")
set(CPACK_PACKAGE_NAME "hcc-doc")
set(CPACK_PACKAGE_VENDOR "MulticoreWare, Inc")
set(CPACK_PACKAGE_VERSION ${KALMAR_VERSION_STRING})
set(CPACK_PACKAGE_VERSION_MAJOR ${KALMAR_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${KALMAR_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${KALMAR_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME})
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "HCC Documentation")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jack Chung <jack@multicorewareinc.com>")
set(CPACK_GENERATOR "DEB;TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_BINARY_DEB "ON")
set(CPACK_BINARY_STGZ "OFF")
set(CPACK_SOURCE_TGZ "OFF")
set(CPACK_SOURCE_TZ "OFF")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_BINARY_TZ "OFF")

include (CPack)

