# OpenCL/GMAC
set(GMAC_INC_DIR "${PROJECT_SOURCE_DIR}/gmac/src/include")
include_directories(${GMAC_INC_DIR} ${OPENCL_HEADER_DIR})

add_mcwamp_library(mcwamp mcwamp.cpp)

add_mcwamp_executable(mcwexe mcwamp_main.cpp)
