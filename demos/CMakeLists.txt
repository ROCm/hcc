find_path(CUDA_BIN_PATH nvcc PATHS /usr/local/cuda/bin NO_DEFAULT_PATH)
if(CUDA_BIN_PATH)
  get_filename_component(CUDA_PATH ${CUDA_BIN_PATH} DIRECTORY)
  message("-- nvcc is found in " ${CUDA_BIN_PATH})
else()
  message("-- nvcc is NOT found, skipping 2dLB_CU target")
endif()

find_path(CLAMP_BIN_PATH clamp-config PATHS ${EXECUTABLE_OUTPUT_PATH} NO_DEFAULT_PATH)
if(CLAMP_BIN_PATH)
  message("-- clamp-config is NOT found, skipping 2dLB_gl target")
endif()

SET(DEMO_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/bin")
file(MAKE_DIRECTORY ${DEMO_BIN_PATH})

add_subdirectory(LB_Demo)

add_custom_target(demos
  DEPENDS LB_Demo
  COMMENT "Demos in ${DEMO_BIN_PATH}")
