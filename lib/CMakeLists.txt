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

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsail
    DESTINATION bin)

if (HAS_HSA EQUAL 1)
add_mcwamp_library(hsacontext HSAContextKaveri.cpp elf_utils.cpp)
install(TARGETS hsacontext
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "HSA available, going to build HSA C++AMP Runtime")
else (HAS_HSA EQUAL 1)
MESSAGE(STATUS "HSA NOT available, NOT going to build HSA C++AMP Runtime")
endif (HAS_HSA EQUAL 1)

install(FILES opencl_math.bc DESTINATION lib)
