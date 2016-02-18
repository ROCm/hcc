#install commands for headers
FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
INSTALL(FILES ${files} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${files} DESTINATION "${PROJECT_BINARY_DIR}/include")

# short vector files
FILE(GLOB sv_files ${CMAKE_CURRENT_SOURCE_DIR}/*_short_vectors*)
INSTALL(FILES ${sv_files} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${sv_files} DESTINATION "${PROJECT_BINARY_DIR}/include")

# N4494 headers
FILE(GLOB N4494 ${CMAKE_CURRENT_SOURCE_DIR}/coordinate
                ${CMAKE_CURRENT_SOURCE_DIR}/array_view)
INSTALL(FILES ${N4494} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${N4494} DESTINATION "${PROJECT_BINARY_DIR}/include")

# PSTL headers
ADD_SUBDIRECTORY(experimental)
