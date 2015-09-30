#install commands for headers
FILE(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)
INSTALL(FILES ${files} DESTINATION include)

# N4494 headers
FILE(GLOB N4494 ${CMAKE_CURRENT_SOURCE_DIR}/coordinate
                ${CMAKE_CURRENT_SOURCE_DIR}/array_view)
INSTALL(FILES ${N4494} DESTINATION include)

# PSTL headers
ADD_SUBDIRECTORY(experimental)
