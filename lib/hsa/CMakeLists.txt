####################
# HSA headers
####################
if (HAS_HSA EQUAL 1)
  include_directories(${HSA_HEADER} ${LIBHSAIL_HEADER_DIR} ${LIBHSAIL_HEADER_DIR}/generated)
endif (HAS_HSA EQUAL 1)

####################
# C++AMP runtime (HSA implementation)
####################
if (HAS_HSA EQUAL 1)
add_mcwamp_library_hsa(mcwamp_hsa mcwamp_hsa.cpp staging_buffer.cpp ../md5.cpp)
add_mcwamp_library_hsa(hc_am hc_am.cpp)
install(TARGETS mcwamp_hsa hc_am
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "HSA available, going to build HSA HCC Runtime")
else (HAS_HSA EQUAL 1)
MESSAGE(STATUS "HSA NOT available, NOT going to build HSA HCC Runtime")
endif (HAS_HSA EQUAL 1)
