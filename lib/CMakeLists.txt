# OpenCL/GMAC
set(GMAC_INC_DIR "${PROJECT_SOURCE_DIR}/gmac/src/include" "${PROJECT_BINARY_DIR}/gmac/src/include")
include_directories(${GMAC_INC_DIR} ${OPENCL_HEADER_DIR})

add_mcwamp_library(mcwamp mcwamp.cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mcwamp_executable(clamp-config mcwamp_main.cpp)
install(TARGETS mcwamp clamp-config
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-device
    DESTINATION bin)
