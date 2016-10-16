
option(HCC_RUNTIME_DEBUG "Enable debug build for HCC Runtime" OFF)

if (HCC_RUNTIME_DEBUG)
  add_compile_options(-g -O0)
  add_definitions(-DKALMAR_DEBUG=1)
endif (HCC_RUNTIME_DEBUG)

####################
# C++AMP config (clamp-config)
####################
add_config_executable(clamp-config mcwamp_main.cpp)
add_config_executable(hcc-config mcwamp_main.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${LIBCXX_HEADER} )
target_link_libraries(clamp-config c++)
target_link_libraries(hcc-config c++)

####################
# C++AMP runtime (mcwamp)
####################
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
#    ${PROJECT_BINARY_DIR}/compiler/bin/clamp-preprocess
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
# builtin functions CL kernel (OpenCL C)
####################
INSTALL(FILES opencl_math.cl DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY opencl_math.cl DESTINATION "${PROJECT_BINARY_DIR}/lib")

####################
# builtin functions bitcode (SPIR)
####################
install(FILES opencl_math.bc DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY opencl_math.bc DESTINATION "${PROJECT_BINARY_DIR}/lib")

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
# Copy the file to directory matching the install directory
file(COPY opencl_prefix.cl DESTINATION "${PROJECT_BINARY_DIR}/lib")

####################
# Dynamic memory allocation HSAIL logic (HSA)
####################
install(FILES hsa_builtins.hsail DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY hsa_builtins.hsail DESTINATION "${PROJECT_BINARY_DIR}/lib")

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
