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
# TBD install commands for HSA-specific files here
add_mcwamp_library_hsa(mcwamp_hsa mcwamp_hsa.cpp HSAContextKaveri.cpp)
install(TARGETS mcwamp_hsa
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "HSA available, going to build HSA C++AMP Runtime")
else (HAS_HSA EQUAL 1)
MESSAGE(STATUS "HSA NOT available, NOT going to build HSA C++AMP Runtime")
endif (HAS_HSA EQUAL 1)
