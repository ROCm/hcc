add_mcwamp_library(mcwamp_gtest gtest_main.cc gtest-all.cc)

include_directories( ${LIBCXX_HEADER} )
target_link_libraries(mcwamp_gtest c++)
