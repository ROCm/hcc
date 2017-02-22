
if (HCC_RUNTIME_DEBUG)
  add_compile_options(-g -O0)
endif (HCC_RUNTIME_DEBUG)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../hcc_config)

set(CMAKE_CXX_FLAGS "-std=c++11" )

if (USE_LIBCXX)
  add_definitions(-DUSE_LIBCXX  -DLIBCXX_HEADER=${LIBCXX_HEADER}) 
endif (USE_LIBCXX)


####################
# CodeXL Activity Logger
####################
if (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)
add_definitions(-DCODEXL_ACTIVITY_LOGGER_ENABLED)
if (CODEXL_ACTIVITY_LOGGER_HEADER)
  add_definitions(-DCODEXL_ACTIVITY_LOGGER_HEADER=${CODEXL_ACTIVITY_LOGGER_HEADER})
endif (CODEXL_ACTIVITY_LOGGER_HEADER)
if (CODEXL_ACTIVITY_LOGGER_LIBRARY)
  add_definitions(-DCODEXL_ACTIVITY_LOGGER_LIBRARY=${CODEXL_ACTIVITY_LOGGER_LIBRARY})
endif (CODEXL_ACTIVITY_LOGGER_LIBRARY)
endif (USE_CODEXL_ACTIVITY_LOGGER EQUAL 1)

add_executable(hcc-config hcc_config.cpp)


add_custom_command(TARGET hcc-config POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink ./hcc-config ./clamp-config
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

install(TARGETS hcc-config
    RUNTIME DESTINATION bin
)

install(FILES ${PROJECT_BINARY_DIR}/bin/clamp-config DESTINATION bin)


