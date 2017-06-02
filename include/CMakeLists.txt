#install commands for headers
FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
INSTALL(FILES ${files} DESTINATION ${HCC_SUBDIR}/include)
# Copy the file to directory matching the install directory
file(COPY ${files} DESTINATION "${PROJECT_BINARY_DIR}/include")
execute_process(COMMAND ln -sf hcc . WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/include)
INSTALL(FILES ${PROJECT_BINARY_DIR}/include/hcc DESTINATION ${HCC_SUBDIR}/include)

FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.inl)
INSTALL(FILES ${files} DESTINATION ${HCC_SUBDIR}/include)
# Copy the file to directory matching the install directory
file(COPY ${files} DESTINATION "${PROJECT_BINARY_DIR}/include")

# N4494 headers
FILE(GLOB N4494 ${CMAKE_CURRENT_SOURCE_DIR}/coordinate
                ${CMAKE_CURRENT_SOURCE_DIR}/array_view)
INSTALL(FILES ${N4494} DESTINATION ${HCC_SUBDIR}/include)
# Copy the file to directory matching the install directory
file(COPY ${N4494} DESTINATION "${PROJECT_BINARY_DIR}/include")

# PSTL headers
ADD_SUBDIRECTORY(experimental)
