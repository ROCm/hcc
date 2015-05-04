####################
# OpenCL headers
####################
if (HAS_OPENCL EQUAL 1)
  include_directories(${OPENCL_HEADER}/../)
endif (HAS_OPENCL EQUAL 1)

####################
# C++AMP runtime (OpenCL implementation)
####################
if (HAS_OPENCL EQUAL 1)
add_mcwamp_library_opencl(mcwamp_opencl mcwamp_opencl.cpp ../md5.cpp)
install(TARGETS mcwamp_opencl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
MESSAGE(STATUS "OpenCL available, going to build OpenCL C++AMP Runtime")
else (HAS_OPENCL EQUAL 1)
MESSAGE(STATUS "OpenCL NOT available, NOT going to build OpenCL C++AMP Runtime")
endif (HAS_OPENCL EQUAL 1)
