####################
# C++AMP runtime (CPU implementation)
####################
add_mcwamp_library_cpu(mcwamp_cpu mcwamp_cpu.cpp)
rocm_install_targets(
  TARGETS mcwamp_cpu
  PREFIX ${HCC_SUBDIR}
)
MESSAGE(STATUS "build HCC CPU Runtime")
