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

if (CXXAMP_ENABLE_HSA_OKRA)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    DESTINATION bin)
endif (CXXAMP_ENABLE_HSA_OKRA)

install(FILES opencl_math.bc DESTINATION lib)
