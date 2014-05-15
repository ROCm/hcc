# OpenCL/GMAC
set(GMAC_INC_DIR "${PROJECT_SOURCE_DIR}/gmac/src/include" "${PROJECT_BINARY_DIR}/gmac/src/include")
include_directories(${GMAC_INC_DIR} ${OPENCL_HEADER}/../)

add_mcwamp_library(mcwamp mcwamp.cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mcwamp_executable(clamp-config mcwamp_main.cpp)
install(TARGETS mcwamp clamp-config
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-device
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-spirify
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-embed
    DESTINATION bin)

if (CXXAMP_ENABLE_HSA)
add_mcwamp_library(hsacontext HSAContextKaveri.cpp)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    DESTINATION bin)
endif (CXXAMP_ENABLE_HSA)

if (CXXAMP_ENABLE_HSA_OKRA)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    DESTINATION bin)
endif (CXXAMP_ENABLE_HSA_OKRA)

install(FILES opencl_math.bc DESTINATION lib)
