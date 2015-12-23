include_directories(${GMAC_SRC_DIR})
add_mcwamp_library(mcwamp_gtest gtest_main.cc gtest-all.cc)

install(TARGETS mcwamp_gtest
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
install(FILES gtest.h DESTINATION include/gtest)



