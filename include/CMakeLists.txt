#install commands for headers
FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
INSTALL(FILES ${files} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${files} DESTINATION "${PROJECT_BINARY_DIR}/include")

FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.inl)
INSTALL(FILES ${files} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${files} DESTINATION "${PROJECT_BINARY_DIR}/include")

# N4494 headers
FILE(GLOB N4494 ${CMAKE_CURRENT_SOURCE_DIR}/coordinate
                ${CMAKE_CURRENT_SOURCE_DIR}/array_view)
INSTALL(FILES ${N4494} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${N4494} DESTINATION "${PROJECT_BINARY_DIR}/include")

# PSTL headers
ADD_SUBDIRECTORY(experimental)
