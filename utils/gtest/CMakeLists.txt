add_mcwamp_library(mcwamp_gtest gtest_main.cc gtest-all.cc)

if (USE_LIBCXX)
  include_directories( ${LIBCXX_HEADER} )
  target_link_libraries(mcwamp_gtest c++)
endif (USE_LIBCXX)
