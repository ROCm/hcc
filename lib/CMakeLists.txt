
option(HCC_RUNTIME_DEBUG "Enable debug build for HCC Runtime" OFF)

if (HCC_RUNTIME_DEBUG)
  add_compile_options(-g -O0)
  add_definitions(-DKALMAR_DEBUG=1)
endif (HCC_RUNTIME_DEBUG)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(GNUInstallDirs)
set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )
set( CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/hcc )

####################
# C++AMP runtime (mcwamp)
####################
add_mcwamp_library(mcwamp mcwamp.cpp)
add_mcwamp_library(mcwamp_atomic mcwamp_atomic.cpp)

# Library interface to use runtime
add_library(hccrt INTERFACE)
target_compile_options(hccrt INTERFACE -hc)
target_link_libraries(hccrt INTERFACE "-hc -L $<TARGET_FILE_DIR:$<TARGET_NAME:mcwamp>> -Wl,-rpath $<TARGET_FILE_DIR:$<TARGET_NAME:mcwamp>>" "-Wl,--whole-archive" mcwamp "-Wl,--no-whole-archive")
target_link_libraries(hccrt INTERFACE dl m)
if (USE_LIBCXX)
  target_link_libraries(hccrt INTERFACE c++ c++abi)
endif (USE_LIBCXX)

# Library interface for building shared libraries
add_library(hccshared INTERFACE)
target_link_libraries(hccshared INTERFACE -fPIC -Wl,-Bsymbolic)

####################
# C++AMP tools
####################
install(PROGRAMS ${PROJECT_BINARY_DIR}/compiler/bin/clamp-device
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-embed
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-assemble
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-link
    ${PROJECT_BINARY_DIR}/compiler/bin/hc-kernel-assemble
    ${PROJECT_BINARY_DIR}/compiler/bin/hc-host-assemble
    DESTINATION compiler/bin)

####################
# C++AMP tools (HSA-specific)
####################
install(PROGRAMS ${PROJECT_BINARY_DIR}/bin/extractkernel
    DESTINATION bin)

####################
# CodeXL Activity Logger
####################
if (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

if (CODEXL_ACTIVITY_LOGGER_HEADER)
  include_directories(${CODEXL_ACTIVITY_LOGGER_HEADER})
endif (CODEXL_ACTIVITY_LOGGER_HEADER)

if (CODEXL_ACTIVITY_LOGGER_LIBRARY)
  include_directories(${CODEXL_ACTIVITY_LOGGER_LIBRARY})
endif (CODEXL_ACTIVITY_LOGGER_LIBRARY)

endif (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

####################
# add subdirectories
####################
add_subdirectory(hsa)
add_subdirectory(cpu)

####################
# install targets
####################
install(TARGETS mcwamp mcwamp_atomic hccrt hccshared
    EXPORT hcc-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

install( EXPORT hcc-targets
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR}
  NAMESPACE hcc::
)

include( CMakePackageConfigHelpers )

configure_package_config_file(
  hcc-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hcc-config.cmake
  INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
  PATH_VARS LIB_INSTALL_DIR INCLUDE_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hcc-config-version.cmake
  VERSION "${KALMAR_VERSION_MAJOR}.${KALMAR_VERSION_MINOR}.${KALMAR_VERSION_PATCH}"
  COMPATIBILITY SameMajorVersion
)


install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hcc-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hcc-config-version.cmake
  DESTINATION
    ${CONFIG_PACKAGE_INSTALL_DIR} )
