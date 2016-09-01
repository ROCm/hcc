####################
# HCC runtime (HSA implementation)
####################
if (HAS_ROCM EQUAL 1)
include_directories(${HSA_HEADER})
add_mcwamp_library_hsa(mcwamp_hsa mcwamp_hsa.cpp staging_buffer.cpp)
add_mcwamp_library_hsa(hc_am hc_am.cpp)
install(TARGETS mcwamp_hsa hc_am
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "ROCm available, going to build HSA HCC Runtime")
else (HAS_ROCM EQUAL 1)
MESSAGE(STATUS "ROCm NOT available, NOT going to build HSA HCC Runtime")
endif (HAS_ROCM EQUAL 1)
