####################
# HCC runtime (HSA implementation)
####################
if (HAS_ROCM EQUAL 1)
include_directories(${HSA_HEADER})
add_mcwamp_library_hsa(mcwamp_hsa mcwamp_hsa.cpp unpinned_copy_engine.cpp)
add_mcwamp_library_hsa(hc_am hc_am.cpp)
install(TARGETS mcwamp_hsa hc_am
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "ROCm available, going to build HSA HCC Runtime")
else (HAS_ROCM EQUAL 1)
MESSAGE(STATUS "ROCm NOT available, NOT going to build HSA HCC Runtime")
endif (HAS_ROCM EQUAL 1)

####################
# CodeXL Activity Logger Library
####################
if (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)
if (CODEXL_ACTIVITY_LOGGER_LIBRARY)
  target_link_libraries(mcwamp_hsa ${CODEXL_ACTIVITY_LOGGER_LIBRARY}/libCXLActivityLogger.so)
  target_link_libraries(hc_am ${CODEXL_ACTIVITY_LOGGER_LIBRARY}/libCXLActivityLogger.so)
endif (CODEXL_ACTIVITY_LOGGER_LIBRARY)
endif (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

