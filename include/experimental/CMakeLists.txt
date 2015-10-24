# PSTL headers
FILE(GLOB PSTL ${CMAKE_CURRENT_SOURCE_DIR}/algorithm
               ${CMAKE_CURRENT_SOURCE_DIR}/exception_list
               ${CMAKE_CURRENT_SOURCE_DIR}/execution_policy
               ${CMAKE_CURRENT_SOURCE_DIR}/numeric)
INSTALL(FILES ${PSTL} DESTINATION include/experimental)

# PSTL internal headers
ADD_SUBDIRECTORY(impl)
