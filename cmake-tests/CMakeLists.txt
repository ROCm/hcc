
CMAKE_FORCE_CXX_COMPILER("${PROJECT_BINARY_DIR}/compiler/bin/clang++" MCWAMPCC)
add_executable(cmake-test cmake-test.cpp)
target_link_libraries(cmake-test hccrt)
