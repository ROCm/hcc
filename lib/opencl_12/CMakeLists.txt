####################
# OpenCL headers
####################
if (HAS_OPENCL EQUAL 1)
  include_directories(${OPENCL_HEADER}/../)
endif (HAS_OPENCL EQUAL 1)

####################
# C++AMP runtime (OpenCL zero-copy implementation)
####################
if ((HAS_OPENCL EQUAL 1) AND (HAS_OPENCL12 EQUAL 1))
add_mcwamp_library_opencl(mcwamp_opencl_12 mcwamp_opencl_12.cpp)
install(TARGETS mcwamp_opencl_12
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
MESSAGE(STATUS "OpenCL 1.2 available, going to build OpenCL (zero-copy) C++AMP Runtime")
else ((HAS_OPENCL EQUAL 1) AND (HAS_OPENCL12 EQUAL 1))
MESSAGE(STATUS "OpenCL 1.2 NOT available, NOT going to build OpenCL (zero-copy)  C++AMP Runtime")
endif ((HAS_OPENCL EQUAL 1) AND (HAS_OPENCL12 EQUAL 1))

