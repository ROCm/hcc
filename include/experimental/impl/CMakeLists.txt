# PSTL headers
FILE(GLOB PSTL ${CMAKE_CURRENT_SOURCE_DIR}/*.inl)
INSTALL(FILES ${PSTL} DESTINATION ${HCC_SUBDIR}/include/experimental/impl)
# Copy the file to directory matching the install directory
file(COPY ${PSTL} DESTINATION "${PROJECT_BINARY_DIR}/include/experimental/impl")

