
option(HCC_RUNTIME_DEBUG "Enable debug build for HCC Runtime" OFF)

if (HCC_RUNTIME_DEBUG)
  add_compile_options(-g -O0)
  add_definitions(-DKALMAR_DEBUG=1)
endif (HCC_RUNTIME_DEBUG)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if (USE_LIBCXX)
  include_directories(${LIBCXX_HEADER})
endif (USE_LIBCXX)

####################
# C++AMP runtime (mcwamp)
####################
add_mcwamp_library(mcwamp mcwamp.cpp)
add_mcwamp_library(mcwamp_atomic mcwamp_atomic.cpp)

install(TARGETS mcwamp mcwamp_atomic
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

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
