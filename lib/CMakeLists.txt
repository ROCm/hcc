####################
# C++AMP config (clamp-config)
####################
add_config_executable(clamp-config mcwamp_main.cpp)
add_config_executable(hcc-config mcwamp_main.cpp)

####################
# C++AMP runtime (mcwamp)
####################
include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mcwamp_library(mcwamp mcwamp.cpp)
add_mcwamp_library(mcwamp_atomic mcwamp_atomic.cpp)

install(TARGETS clamp-config hcc-config mcwamp mcwamp_atomic
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

####################
# C++AMP tools
####################
install(PROGRAMS ${PROJECT_BINARY_DIR}/compiler/bin/clamp-device
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-spirify
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-embed
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-assemble
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-link
    ${PROJECT_BINARY_DIR}/compiler/bin/hc-kernel-assemble
    ${PROJECT_BINARY_DIR}/compiler/bin/hc-host-assemble
    DESTINATION compiler/bin)

####################
# C++AMP tools (HSA-specific)
####################
install(PROGRAMS ${PROJECT_BINARY_DIR}/compiler/bin/clamp-hsatools
    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-hsail
    DESTINATION compiler/bin)

install(PROGRAMS ${PROJECT_BINARY_DIR}/bin/extractkernel
    DESTINATION bin)

####################
# builtin functions bitcode (HSA)
####################
install(FILES hsa_math.bc DESTINATION lib)
install(FILES builtins-hsail.opt.bc DESTINATION lib)
install(FILES hsail-amdgpu-wrapper.ll DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY hsa_math.bc DESTINATION "${PROJECT_BINARY_DIR}/lib")
file(COPY builtins-hsail.opt.bc DESTINATION "${PROJECT_BINARY_DIR}/lib")
file(COPY hsail-amdgpu-wrapper.ll DESTINATION "${PROJECT_BINARY_DIR}/lib")

####################
# Dynamic memory allocation HSAIL logic (HSA)
####################
install(FILES hsa_builtins.hsail DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY hsa_builtins.hsail DESTINATION "${PROJECT_BINARY_DIR}/lib")

####################
# add subdirectories
####################
add_subdirectory(hsa)
add_subdirectory(cpu)

