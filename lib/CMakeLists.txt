####################
# C++AMP runtime (mcwamp)
####################
add_mcwamp_library(mcwamp mcwamp.cpp)

####################
# C++AMP config (clamp-config)
####################
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mcwamp_executable(clamp-config mcwamp_main.cpp)
install(TARGETS mcwamp clamp-config
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

####################
# C++AMP tools
####################
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-device
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-spirify
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-embed
    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-assemble
    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-link
    DESTINATION bin)

####################
# C++AMP tools (HSA-specific)
####################
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsail
    DESTINATION bin)

####################
# Math builtin bitcode (SPIR)
####################
install(FILES opencl_math.bc DESTINATION lib)


####################
# OpenCL C double type detection
####################
install(FILES opencl_prefix.cl DESTINATION lib)

####################
# Dynamic memory allocation HSAIL logic (HSA)
####################
install(FILES new_delete.hsail DESTINATION lib)

####################
# add subdirectories
####################
add_subdirectory(opencl_12)
add_subdirectory(opencl_11)
add_subdirectory(hsa)

