
if(COMMAND CMAKE_FORCE_CXX_COMPILER)
CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
endif()
add_executable(cmake-test cmake-test.cpp)
if(TARGET hccrt)
    target_link_libraries(cmake-test hccrt hc_am)
else()
    # Append default hcc installation
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hcc)
    target_link_libraries(cmake-test ${hcc_LIBRARIES})
endif()
