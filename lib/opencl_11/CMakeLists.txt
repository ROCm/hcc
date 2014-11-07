####################
# OpenCL headers
####################
if (HAS_OPENCL EQUAL 1)
  include_directories(${OPENCL_HEADER}/../)
endif (HAS_OPENCL EQUAL 1)

####################
# C++AMP runtime (OpenCL implementation)
####################
if ((HAS_OPENCL EQUAL 1) AND (HAS_OPENCL11 EQUAL 1))
add_mcwamp_library_opencl(mcwamp_opencl_11 mcwamp_opencl_11.cpp)
install(TARGETS mcwamp_opencl_11
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
MESSAGE(STATUS "OpenCL 1.1 available, going to build OpenCL (1.1) C++AMP Runtime")
else ((HAS_OPENCL EQUAL 1) AND (HAS_OPENCL11 EQUAL 1))
MESSAGE(STATUS "OpenCL 1.1 NOT available, NOT going to build OpenCL (1.1) C++AMP Runtime")
endif ((HAS_OPENCL EQUAL 1) AND (HAS_OPENCL11 EQUAL 1))

