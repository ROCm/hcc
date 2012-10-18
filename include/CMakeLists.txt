#install commands for headers
FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
INSTALL(FILES ${files} opencl_math.cl DESTINATION include)
