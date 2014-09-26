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
    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-assemble
    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-link
    DESTINATION bin)

if (CXXAMP_ENABLE_HSA)
add_mcwamp_library(hsacontext HSAContextKaveri.cpp elf_utils.cpp)
install(TARGETS hsacontext
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsail
    DESTINATION bin)
endif (CXXAMP_ENABLE_HSA)

install(FILES opencl_math.bc DESTINATION lib)
