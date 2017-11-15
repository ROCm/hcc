####################
# C++AMP runtime (CPU implementation)
####################
add_mcwamp_library_cpu(mcwamp_cpu mcwamp_cpu.cpp)
install(TARGETS mcwamp_cpu
    EXPORT hcc-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
MESSAGE(STATUS "build HCC CPU Runtime")
