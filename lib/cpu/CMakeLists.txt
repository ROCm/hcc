####################
# C++AMP runtime (CPU implementation)
####################
add_mcwamp_library_cpu(mcwamp_cpu mcwamp_cpu.cpp)
install(TARGETS mcwamp_cpu
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
MESSAGE(STATUS "build CPU C++AMP Runtime")
