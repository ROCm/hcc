SET(LB_DEMO_PATH "${DEMO_BIN_PATH}/LB_Demo")
file(MAKE_DIRECTORY ${LB_DEMO_PATH})

add_custom_target(2dLB_gl
  COMMAND test -f ${CLAMPCONFIG_BIN} && ${LLVM_ROOT}/bin/clang++ `${CLAMPCONFIG_BIN} --build --cxxflags --ldflags` -hc -o ${LB_DEMO_PATH}/2dLB_gl ${CMAKE_CURRENT_SOURCE_DIR}/2dLB_gl.cpp -lGLEW -lglut -lGL  || echo "clamp-config NOT found, skipping 2dLB GridLaunch"
  COMMENT "Building 2dLB GridLaunch")

if(CUDA_BIN_PATH)
  add_custom_target(2dLB_gl_CU
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/2dLB_gl.cpp ${CMAKE_CURRENT_BINARY_DIR}/2dLB_gl_CU.cu
    COMMAND ${CUDA_BIN_PATH}/nvcc -o ${LB_DEMO_PATH}/2dLB_gl_CU ${CMAKE_CURRENT_BINARY_DIR}/2dLB_gl_CU.cu -I${CUDA_PATH}/include -I${CUDA_PATH}/samples/common/inc -I${PROJECT_SOURCE_DIR}/include -DUSE_CUDA -L${CUDA_PATH}/samples/common/lib/linux/x86_64 -lGLEW -lglut -lGL
  COMMENT "Building 2dLB GridLaunch to CUDA")
else()
  add_custom_target(2dLB_gl_CU
    COMMENT WARNING "nvcc NOT found, skipping 2dLB GridLaunch to CUDA")
endif()

# Nvidia machines might not have glew.h in the standard /usr/include directory
# This path search supports only CUDA 5.0 and above.
find_path(GLEW_INCLUDE_DIR NAMES "GL/glew.h" PATHS ${CUDA_PATH}/samples/common/inc)
find_path(GLEW_LIB_DIR NAMES libGLEW.a PATHS ${CUDA_PATH}/samples/common/lib/linux/x86_64)

add_custom_target(2dLB_C
  COMMAND ${CMAKE_C_COMPILER} -o ${LB_DEMO_PATH}/2dLB_C -I${GLEW_INCLUDE_DIR} -L${GLEW_LIB_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/2dLB_C.c -lGLEW -lglut -lGL -lm
  COMMENT "Building 2dLB C")

if(CUDA_BIN_PATH)
  add_custom_target(2dLB_cuda
    COMMAND ${CUDA_BIN_PATH}/nvcc -o ${LB_DEMO_PATH}/2dLB_cuda ${CMAKE_CURRENT_SOURCE_DIR}/2dLB_cuda.cu -I${CUDA_PATH}/samples/common/inc -I${CUDA_PATH}/include -lcuda -lcudart -L${CUDA_PATH}/samples/common/lib/linux/x86_64 -lglut -lGL -lGLEW -m64
  COMMENT "Building 2dLB CUDA")
else()
  add_custom_target(2dLB_cuda
    COMMENT "nvcc NOT found, skipping 2dLB CUDA")
endif()

add_custom_target(LB_Demo
  DEPENDS 2dLB_gl 2dLB_gl_CU 2dLB_C 2dLB_cuda
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/cmap.dat ${LB_DEMO_PATH})
