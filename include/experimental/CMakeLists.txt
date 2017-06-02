# PSTL headers
FILE(GLOB PSTL ${CMAKE_CURRENT_SOURCE_DIR}/algorithm
               ${CMAKE_CURRENT_SOURCE_DIR}/exception_list
               ${CMAKE_CURRENT_SOURCE_DIR}/execution_policy
               ${CMAKE_CURRENT_SOURCE_DIR}/numeric)
INSTALL(FILES ${PSTL} DESTINATION ${HCC_SUBDIR}/include/experimental)
# Copy the file to directory matching the install directory
file(COPY ${PSTL} DESTINATION "${PROJECT_BINARY_DIR}/include/experimental")

# PSTL internal headers
ADD_SUBDIRECTORY(impl)
