####################
# C++AMP runtime (mcwamp)
####################
add_mcwamp_library(mcwamp mcwamp.cpp)
add_mcwamp_library(mcwamp_atomic mcwamp_atomic.cpp)

####################
# C++AMP config (clamp-config)
####################
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mcwamp_executable(clamp-config mcwamp_main.cpp)
add_mcwamp_executable(hcc-config mcwamp_main.cpp)
install(TARGETS mcwamp clamp-config hcc-config mcwamp_atomic
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
    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/hc-kernel-assemble
    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/hc-host-assemble
#    ${CMAKE_CURRENT_BINARY_DIR}/../compiler/bin/clamp-preprocess
    DESTINATION bin)

####################
# C++AMP tools (HSA-specific)
####################
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsatools
    ${CMAKE_CURRENT_BINARY_DIR}/clamp-hsail
    ${CMAKE_CURRENT_BINARY_DIR}/extractkernel
    DESTINATION bin)

####################
# builtin functions CL kernel (OpenCL C)
####################
INSTALL(FILES opencl_math.cl DESTINATION lib)

####################
# builtin functions bitcode (SPIR)
####################
install(FILES opencl_math.bc DESTINATION lib)

####################
# builtin functions bitcode (HSA)
####################
install(FILES hsa_math.bc DESTINATION lib)
install(FILES builtins-hsail.opt.bc DESTINATION lib)
install(FILES hsail-amdgpu-wrapper.ll DESTINATION lib)

####################
# OpenCL runtime version detection
####################
if (HAS_OPENCL EQUAL 1)
add_mcwamp_library_opencl(mcwamp_opencl_version opencl_version.cpp)
install(TARGETS mcwamp_opencl_version
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
endif (HAS_OPENCL EQUAL 1)

####################
# OpenCL C double type detection
####################
install(FILES opencl_prefix.cl DESTINATION lib)

####################
# Dynamic memory allocation HSAIL logic (HSA)
####################
install(FILES hsa_builtins.hsail DESTINATION lib)

####################
# add subdirectories
####################
add_subdirectory(opencl)
add_subdirectory(hsa)
add_subdirectory(cpu)

# Build plugins for clang
# FIXME: codes in this directory is obsolete, need to clean it up
#if (CXXAMP_ENABLE_BOLT)
#  add_subdirectory(clang-plugins)
#endif (CXXAMP_ENABLE_BOLT)
