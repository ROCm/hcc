
if(NOT CMAKE_CXX_COMPILER MATCHES ".*hcc")
    set(CMAKE_CXX_COMPILER "${PROJECT_BINARY_DIR}/compiler/bin/clang++")
endif()

add_executable(cmake-test cmake-test.cpp)

# Explicitly set the GPU architecture so that
# cmake_test could be cross-compiled on a system
# without a GPU
get_target_property(cmake_test_link_flags cmake-test LINK_FLAGS)
if (cmake_test_link_flags)
  set(new_cmake_test_link_flags "${cmake_test_link_flags} --amdgpu-target=gfx803 --amdgpu-target=gfx901")
else()
  set(new_cmake_test_link_flags "--amdgpu-target=gfx803 --amdgpu-target=gfx901")
endif()
set_target_properties(cmake-test PROPERTIES LINK_FLAGS ${new_cmake_test_link_flags})

if(TARGET hccrt)
    target_link_libraries(cmake-test hccrt hc_am)
else()
    # Append default hcc installation
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hcc)
    target_link_libraries(cmake-test ${hcc_LIBRARIES})
endif()
